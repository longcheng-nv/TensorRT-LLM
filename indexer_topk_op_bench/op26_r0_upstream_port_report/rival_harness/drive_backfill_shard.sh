#!/bin/bash
# §8 rival full-ISL BS backfill shard driver. Usage: drive_backfill_shard.sh <W> <GPU> <NW>
# Same protocol as drive_rival_shard.sh (one nsys-rep per batch, NVTX c|/w|
# ranges, resumable via .done markers + cell-level jsonl _load_done), but:
#   - batches from batches_backfill.py (one per model x dtype x isl)
#   - arms restricted to op26_r0auto (anchor) + 3 external rivals
#   - rep is named EXACTLY like sweep_rival's jsonl stem (parse_rival prefers it)
#   - out dir rival_results_bf (kept separate from the 07-15 run)
W=$1; GPU=$2; NW=$3
WD=/tmp/gvrval1
RH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/rival_harness
OUT="$WD/rival_results_bf"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
C450=$WD/cutlass450/nvidia_cutlass_dsl/python_packages
[ -d "$C450" ] && export PYTHONPATH="$C450:$WD/cutlass450:$WD/fi_clean${PYTHONPATH:+:$PYTHONPATH}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"
OPS="op26_r0auto,radix_cutedsl,sglang_v2,flashinfer_topk"

mapfile -t BATCHES < <(python3 "$RH/batches_backfill.py")
i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % NW)) -eq $W ]; then
    read -r fam sw m dt isl <<< "$batch"
    tag="real_${m}_${sw}_${dt}_${isl}"        # == sweep_rival jsonl stem (with --isl)
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== backfill batch [$W]: $tag -> $rep.nsys-rep ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 "$RH/sweep_rival.py" --family real --sweep "$sw" --model "$m" \
           --dtype "$dt" --isl "$isl" --ops "$OPS" --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM" > "$OUT/${tag}.log" 2>&1; then
      touch "$done_m"
    else
      echo "!!! batch $tag FAILED (see $OUT/${tag}.log; left un-marked for resume)"
    fi
  fi
  i=$((i+1))
done
echo "BFWORKER${W}DONE"
