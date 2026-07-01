#!/bin/bash
# nsys cold-L2 sweep: dtype x N x M, BS=1, median-of-3 per-launch us.
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/count_ge_multi_bench
OUT=nsys; mkdir -p $OUT
CSV=results.csv
echo "dtype,N,M,us_med,TT,VECW" > $CSV
NS="4096 8192 16384 32768 65536 131072 262144"
MS="1 2 4 6 8"
DTS="fp32 fp16"
med_ns () {  # median-of-3 of the count_ge kernel Med(ns) col5
  vals=""
  for rep in 1 2 3; do
    r="$OUT/${1}_N${2}_M${3}_r${rep}"
    nsys profile -c cudaProfilerApi --capture-range-end=stop -f true -o "$r" \
      ./count_ge_multi $1 $2 $3 1 > "$r.out" 2>"$r.err"
    v=$(nsys stats --report cuda_gpu_kern_sum --format csv "$r.nsys-rep" 2>/dev/null \
        | grep -i "count_ge_multi" | head -1 | awk -F, '{printf "%.3f",$5/1000}')
    vals="$vals $v"
  done
  echo $vals | tr ' ' '\n' | sort -n | sed -n '2p'  # median of 3
}
for dt in $DTS; do for N in $NS; do for M in $MS; do
  us=$(med_ns $dt $N $M)
  # recover TT/VECW from one out file
  info=$(grep -o "TT=[0-9]* VECW=[0-9]*" "$OUT/${dt}_N${N}_M${M}_r1.out" | head -1)
  tt=$(echo $info | grep -o "TT=[0-9]*" | cut -d= -f2)
  vw=$(echo $info | grep -o "VECW=[0-9]*" | cut -d= -f2)
  echo "$dt,$N,$M,$us,$tt,$vw" >> $CSV
  echo "[$dt N=$N M=$M] -> ${us}us (TT=$tt VECW=$vw)"
done; done; done
echo "=== SWEEP DONE ==="; cat $CSV
