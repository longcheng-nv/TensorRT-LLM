#!/bin/bash
# tau(M=3) validation sweep for HLS Step 0 (op21 iter13).
# Re-measures M in {1,2,3,4} fp32 (+ fp16 262K spot) on THIS node so the
# tau(3) ratio is same-silicon; absolute us do not transfer across nodes.
# nsys cold-L2, median-of-3 per-launch Med(ns), matches run_sweep.sh method.
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/count_ge_multi_bench
OUT=nsys_m3; mkdir -p $OUT
CSV=results_m3.csv
echo "dtype,N,M,us_med,TT,VECW" > $CSV
NS="4096 8192 16384 32768 65536 131072 262144"
MS="1 2 3 4"
med_ns () {  # median-of-3 of the count_ge kernel Med(ns) col5
  vals=""
  for rep in 1 2 3; do
    r="$OUT/${1}_N${2}_M${3}_r${rep}"
    env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
      nsys profile -c cudaProfilerApi --capture-range-end=stop -f true -o "$r" \
      ./count_ge_multi $1 $2 $3 1 > "$r.out" 2>"$r.err"
    v=$(nsys stats --report cuda_gpu_kern_sum --format csv "$r.nsys-rep" 2>/dev/null \
        | grep -i "count_ge_multi" | head -1 | awk -F, '{printf "%.3f",$5/1000}')
    vals="$vals $v"
  done
  echo $vals | tr ' ' '\n' | sort -n | sed -n '2p'  # median of 3
}
GPU=${GPU:-0}
for dt in fp32; do for N in $NS; do for M in $MS; do
  us=$(med_ns $dt $N $M)
  info=$(grep -o "TT=[0-9]* VECW=[0-9]*" "$OUT/${dt}_N${N}_M${M}_r1.out" | head -1)
  tt=$(echo $info | grep -o "TT=[0-9]*" | cut -d= -f2)
  vw=$(echo $info | grep -o "VECW=[0-9]*" | cut -d= -f2)
  echo "$dt,$N,$M,$us,$tt,$vw" >> $CSV
  echo "[$dt N=$N M=$M] -> ${us}us (TT=$tt VECW=$vw)"
done; done; done
# fp16 spot cells at the regime-C anchor N (16-bit ladder relevance)
for N in 131072 262144; do for M in 1 2 3 4; do
  us=$(med_ns fp16 $N $M)
  info=$(grep -o "TT=[0-9]* VECW=[0-9]*" "$OUT/fp16_N${N}_M${M}_r1.out" | head -1)
  tt=$(echo $info | grep -o "TT=[0-9]*" | cut -d= -f2)
  vw=$(echo $info | grep -o "VECW=[0-9]*" | cut -d= -f2)
  echo "fp16,$N,$M,$us,$tt,$vw" >> $CSV
  echo "[fp16 N=$N M=$M] -> ${us}us (TT=$tt VECW=$vw)"
done; done
echo "=== M3 SWEEP DONE ==="; cat $CSV
