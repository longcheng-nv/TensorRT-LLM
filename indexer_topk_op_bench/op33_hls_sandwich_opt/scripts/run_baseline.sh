#!/bin/bash
# shard 15 cells (3K x 5N) across GPUs 1-5 by K; nsys each; parse kern mean.
cd "$(dirname "$0")/.."
declare -A GPU=( [512]=1 [1024]=2 [2048]=3 )
for K in 512 1024 2048; do
 g=${GPU[$K]}
 ( for N in 8192 16384 32768 65536 131072; do
     tag="bl_K${K}_N${N}"
     env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g K=$K N=$N SCEN=real REPS=60 \
       nsys profile -c cudaProfilerApi -t cuda -o results/$tag -f true --stats=false \
       python3 scripts/nsys_one.py >/dev/null 2>&1
     m=$(env -u GITHUB_TOKEN -u HF_TOKEN nsys stats --report cuda_gpu_kern_sum --format csv results/$tag.nsys-rep 2>/dev/null | grep -iE "kernel_cutlass|gvr|topk" | head -1 | awk -F',' '{print $4}')
     echo "K$K N$N gpu$g mean=${m}ns"
   done ) &
done
wait
find results -name "*.nsys-rep" -delete 2>/dev/null; find results -name "*.sqlite" -delete 2>/dev/null
echo BASELINE_DONE
