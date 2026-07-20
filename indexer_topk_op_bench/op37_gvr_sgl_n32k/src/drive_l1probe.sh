#!/bin/bash
# L1 probe: forced cs2/cs4 vs default (cs1) gvr_pr at the N=32771 rung.
# 2 workers on GPUs 2/3 (baseline owns 0/1). Usage: drive_l1probe.sh <W(0|1)>
set -u
W=$1
GPU=$((W + 2))
SRC=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op37_gvr_sgl_n32k/src
OUT=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op37_gvr_sgl_n32k/results/l1probe
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
i=0
printf 'flash 128k\npro 128k\nv32 32k\n' | while read -r m isl; do
  if [ $((i % 2)) -eq $W ]; then
    tag="real_${m}_bs_fp32_${isl}"
    if [ -f "$OUT/.done_${tag}" ]; then echo "SKIP: $tag"; i=$((i+1)); continue; fi
    echo "=== l1probe [gpu$GPU]: $tag ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$OUT/nsys_reps/${tag}" -f true \
         python3 "$SRC/sweep_op37.py" --family real --sweep bs --model "$m" \
           --dtype fp32 --isl "$isl" --ops gvr_pr,gvr_cs2,gvr_cs4 --out-root "$OUT" \
           --reps 20 --reps-warm 50 > "$OUT/${tag}.log" 2>&1; then
      touch "$OUT/.done_${tag}"
    else
      echo "!!! l1probe batch $tag FAILED"
    fi
  fi
  i=$((i+1))
done
echo "L1PROBE${W}DONE"
