#!/bin/bash
# §8b per-layer BS-scaling rival driver: 9 batches (model x 3 bench layers),
# arms = gvr_pr (launch contract) + radix/sglang_v2/flashinfer, 11-BS grid.
# Usage: drive_rival_bs_shard.sh <W> <GPU> <NW>
W=$1; GPU=$2; NW=$3
WD=/tmp/gvrlayers
LH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/layers_harness
OUT="$WD/rival_bs_results"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WD/cutlass450/nvidia_cutlass_dsl/python_packages:$WD/cutlass450${PYTHONPATH:+:$PYTHONPATH}"

BATCHES=("flash 10" "pro 14" "v32 14" "flash 22" "pro 30" "v32 34" "flash 34" "pro 46" "v32 54")
i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % NW)) -eq $W ]; then
    read -r m L <<< "$batch"
    tag="rival_bs_${m}_L${L}"
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== rival-bs batch [$W]: $tag ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" --force-overwrite=true \
         python3 "$LH/sweep_rival_layers.py" --model $m --bs-layer $L --out-root "$OUT"; then
      touch "$done_m"
    else
      echo "!! batch FAILED: $tag"
    fi
  fi
  i=$((i+1))
done
echo "shard $W done ($(date -u +%T))"
