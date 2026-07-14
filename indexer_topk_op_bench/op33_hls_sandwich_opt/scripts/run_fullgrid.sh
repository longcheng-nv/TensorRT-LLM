#!/bin/bash
# full op22rr fp32 BS=1 seqlen grid: base vs M=3, all 9 N × 3 scen × 3 K, 8-GPU shard.
cd "$(dirname "$0")/.."
OUT=results/fullgrid.csv; EX=results/fullgrid_exact.txt; : > $OUT; : > $EX
echo "cfg,K,dtype,N,scen,mean_ns" >> $OUT
NS=(4096 8192 16384 32768 65536 131072 262144 524288 1048576)
# build (cfg,K,N,scen) task list; K2048 lacks N4096 bundle -> skip
tasks=()
for cfg in base m3; do for K in 512 1024 2048; do for N in "${NS[@]}"; do for scen in best real worst; do
  [ "$K" = "2048" ] && [ "$N" = "4096" ] && continue
  tasks+=("$cfg:$K:$N:$scen")
done; done; done; done
n=${#tasks[@]}; g=0
for t in "${tasks[@]}"; do
  IFS=: read cfg K N scen <<< "$t"
  envq=""; [ "$cfg" = "m3" ] && [ "$K" != "2048" ] && envq="OP25_QFRACS=0.85,0.35"
  tag="fg_${cfg}_K${K}_N${N}_${scen}"
  ( env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g K=$K N=$N SCEN=$scen REPS=50 TAG=$tag EXFILE=$EX $envq \
      nsys profile -c cudaProfilerApi -t cuda -o results/$tag -f true --stats=false python3 scripts/nsys_cfg.py >/dev/null 2>&1
    m=$(env -u GITHUB_TOKEN -u HF_TOKEN nsys stats --report cuda_gpu_kern_sum --format csv results/$tag.nsys-rep 2>/dev/null | grep -iE "kernel_cutlass|gvr|topk" | head -1 | awk -F',' '{print $4}')
    echo "$cfg,$K,fp32,$N,$scen,$m" >> $OUT ) &
  g=$(( (g+1) % 8 ))
  # throttle: max 8 concurrent
  while [ "$(jobs -r | wc -l)" -ge 8 ]; do sleep 1; done
done
wait
find results -name "fg_*.nsys-rep" -delete 2>/dev/null; find results -name "fg_*.sqlite" -delete 2>/dev/null
echo FULLGRID_DONE
