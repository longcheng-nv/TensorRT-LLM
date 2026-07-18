#!/bin/bash
# op36 ship-verdict re-run: <=2 concurrent nsys on decisive batches.
# Usage: drive_verdict.sh <W(0|1)> <batchfile> [OUT]
set -u
W=$1; BF=$2
SRC=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op36_gvr_rival_7b/src
OUT="${3:-/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op36_gvr_rival_7b/results/a0_verdict}"
mkdir -p "$OUT/nsys_reps"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"
OPS="${OPS:-gvr_pr,sglang_v2,gvr_a0}"
i=0
while read -r m isl; do
  if [ $((i % 2)) -eq $W ]; then
    tag="real_${m}_bs_fp32_${isl}"
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP: $tag"; i=$((i+1)); continue; fi
    echo "=== verdict [$W/gpu$W]: $tag ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$W \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 "$SRC/sweep_op36.py" --family real --sweep bs --model "$m" \
           --dtype fp32 --isl "$isl" --ops "$OPS" --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM" > "$OUT/${tag}.log" 2>&1; then
      touch "$done_m"
    else
      echo "!!! verdict batch $tag FAILED"
    fi
  fi
  i=$((i+1))
done < "$BF"
echo "VERDICTWORKER${W}DONE"
