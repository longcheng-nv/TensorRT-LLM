#!/bin/bash
# op37 baseline: <=2 concurrent nsys workers over the N>=32K real rungs.
# Usage: drive_op37.sh <W> <NW> <batchfile> [OUT]   (W = worker id, NW = #workers)
set -u
W=$1; NW=$2; BF=$3
SRC=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op37_gvr_sgl_n32k/src
OUT="${4:-/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op37_gvr_sgl_n32k/results/baseline}"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"
OPS="${OPS:-gvr_pr,sglang_v2}"
i=0
while read -r m isl; do
  if [ $((i % NW)) -eq $W ]; then
    tag="real_${m}_bs_fp32_${isl}"
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP: $tag"; i=$((i+1)); continue; fi
    echo "=== op37 [$W/gpu$W]: $tag ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$W \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 "$SRC/sweep_op37.py" --family real --sweep bs --model "$m" \
           --dtype fp32 --isl "$isl" --ops "$OPS" --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM" > "$OUT/${tag}.log" 2>&1; then
      touch "$done_m"
    else
      echo "!!! op37 batch $tag FAILED (see $OUT/${tag}.log)"
    fi
  fi
  i=$((i+1))
done < "$BF"
echo "OP37WORKER${W}DONE"
