#!/bin/bash
cd "$(dirname "$0")/.."
OUT=results/d3.csv; EX=results/d3_exact.txt; : > $OUT; : > $EX
echo "cfg,K,N,mean_ns" >> $OUT
CFGS=( "d3_qbins128|OP21_QBINS=128" "d3_qbins64|OP21_QBINS=64" )
cells=(); for K in 512 1024 2048; do for N in 8192 32768 65536; do cells+=("$K:$N"); done; done
gpu=0
for cfg in "${CFGS[@]}"; do
  lab="${cfg%%|*}"; env="${cfg#*|}"
  ( for c in "${cells[@]}"; do
      K="${c%%:*}"; N="${c##*:}"; tag="${lab}_K${K}_N${N}"
      env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$gpu K=$K N=$N SCEN=real REPS=60 TAG=$tag EXFILE=$EX $env \
        nsys profile -c cudaProfilerApi -t cuda -o results/d3_$tag -f true --stats=false python3 scripts/nsys_cfg.py >/dev/null 2>&1
      m=$(env -u GITHUB_TOKEN -u HF_TOKEN nsys stats --report cuda_gpu_kern_sum --format csv results/d3_$tag.nsys-rep 2>/dev/null | grep -iE "kernel_cutlass|gvr|topk" | head -1 | awk -F',' '{print $4}')
      echo "$lab,$K,$N,$m" >> $OUT
    done ) &
  gpu=7
done
wait
find results -name "d3_*.nsys-rep" -delete 2>/dev/null; find results -name "d3_*.sqlite" -delete 2>/dev/null
echo D3_DONE
