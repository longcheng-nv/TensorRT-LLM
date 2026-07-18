#!/bin/bash
# op36 Track-A screening driver. Usage: drive_op36_shard.sh <W> <GPU> <NW> [OUT]
# One nsys-rep per (model, isl) fp32 batch; arms = gvr_pr + sglang_v2 anchors
# + gvr_a0 (bundle-v2). Resumable (.done markers + cell-level jsonl).
# 8-way sharding = SCREENING axis only; ship verdicts re-run <=2 concurrent.
set -u
W=$1; GPU=$2; NW=$3
SRC=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op36_gvr_rival_7b/src
OUT="${4:-/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op36_gvr_rival_7b/results/a0_screen}"
mkdir -p "$OUT/nsys_reps"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"
OPS="${OPS:-gvr_pr,sglang_v2,gvr_a0}"

BATCHES=()
for isl in 4k 8k 16k 32k 64k 128k 256k 512k 1024k; do
  BATCHES+=("flash $isl" "pro $isl")
done
for isl in 4k 8k 16k 32k 64k 128k 256k; do
  BATCHES+=("v32 $isl")
done

i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % NW)) -eq $W ]; then
    read -r m isl <<< "$batch"
    tag="real_${m}_bs_fp32_${isl}"
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== op36 batch [$W/gpu$GPU]: $tag ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 "$SRC/sweep_op36.py" --family real --sweep bs --model "$m" \
           --dtype fp32 --isl "$isl" --ops "$OPS" --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM" > "$OUT/${tag}.log" 2>&1; then
      touch "$done_m"
    else
      echo "!!! batch $tag FAILED (see $OUT/${tag}.log; un-marked for resume)"
    fi
  fi
  i=$((i+1))
done
echo "OP36WORKER${W}DONE"
