#!/bin/bash
# Per-layer backfill shard driver. Usage: drive_layers_shard.sh <W> <GPU> <NW>
# One whole-batch nsys-rep per batch (cells in NVTX ranges); cell-resumable
# via the jsonl; .done marker skips finished batches. nsys-reps stay in /tmp
# (embed env tokens -> NEVER commit).
W=$1; GPU=$2; NW=$3
WD=/tmp/gvrlayers
LH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/layers_harness
OUT="$WD/layers_results"
mkdir -p "$OUT/nsys_reps"
# env: bypass NFS userbase except the pinned cutlass 4.5.0 (make_fragment);
# the container .pth resolves nvidia_cutlass_dsl from $WD/cutlass450 first.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WD/cutlass450/nvidia_cutlass_dsl/python_packages:$WD/cutlass450${PYTHONPATH:+:$PYTHONPATH}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"

mapfile -t BATCHES < <(python3 "$LH/batches_layers.py")
i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % NW)) -eq $W ]; then
    read -r sw m x <<< "$batch"
    if [ "$sw" = "seqlen" ]; then
      tag="real_seqlen_${m}_${x}"
      args="--sweep seqlen --model $m --isl $x"
    else
      tag="real_bs_${m}_L${x}"
      args="--sweep bs --model $m --layer $x"
    fi
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== layers batch [$W]: $tag -> $rep.nsys-rep ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" --force-overwrite=true \
         python3 "$LH/sweep_layers.py" $args --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM"; then
      touch "$done_m"
    else
      echo "!! batch FAILED: $tag (leaving undone for retry)"
    fi
  fi
  i=$((i+1))
done
echo "shard $W done ($(date -u +%T))"
