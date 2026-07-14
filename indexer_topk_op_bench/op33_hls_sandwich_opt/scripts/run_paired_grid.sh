#!/bin/bash
# CLEAN paired full grid: per cell, base+m3 BACK-TO-BACK on the SAME GPU (contention
# cancels in the ratio). Shard by cell across 8 GPUs. K512/1024 only (K2048=default).
cd "$(dirname "$0")/.."
OUT=results/paired_grid.csv; : > $OUT; echo "K,N,scen,base_ns,m3_ns,ratio" >> $OUT
NS=(4096 8192 16384 32768 65536 131072 262144 524288 1048576)
cells=(); for K in 512 1024; do for N in "${NS[@]}"; do for sc in best real worst; do cells+=("$K:$N:$sc"); done; done; done
run_cell () {  # gpu K N scen
  local g=$1 K=$2 N=$3 sc=$4
  local vals=()
  for m3 in 0 1; do
    local tag="pg_K${K}_N${N}_${sc}_m${m3}"
    env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g K=$K N=$N SCEN=$sc M3=$m3 \
      nsys profile -c cudaProfilerApi -t cuda -o results/$tag -f true --stats=false python3 scripts/paired_one.py >/dev/null 2>&1
    vals[$m3]=$(env -u GITHUB_TOKEN -u HF_TOKEN nsys stats --report cuda_gpu_kern_sum --format csv results/$tag.nsys-rep 2>/dev/null | grep -iE "kernel_cutlass|gvr|topk" | head -1 | awk -F',' '{print $4}')
    rm -f results/$tag.nsys-rep results/$tag.sqlite
  done
  local r=$(python3 -c "b=${vals[0]:-0};m=${vals[1]:-0};print(f'{b/m:.4f}' if m else '')" 2>/dev/null)
  echo "$K,$N,$sc,${vals[0]},${vals[1]},$r" >> $OUT
}
i=0
for c in "${cells[@]}"; do
  IFS=: read K N sc <<< "$c"; g=$((i%8))
  run_cell $g $K $N $sc &
  i=$((i+1))
  while [ "$(jobs -r|wc -l)" -ge 8 ]; do sleep 1; done
done
wait
echo PAIREDGRID_DONE
