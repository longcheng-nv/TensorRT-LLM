#!/bin/bash
# qab shard driver — p4_exact_tail single-variable full-grid A/B.
# Usage: drive_qab_shard.sh <W> <GPU> <NW>
# Same protocol as drive_vsfull3_shard.sh: one whole-batch nsys-rep per
# (family, sweep, key, dtype); resumable via cell-level jsonl + .done marker.
W=$1; GPU=$2; NW=$3
WD=/tmp/gvrqab
RH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/qfracs_ab
OUT="$WD/qab_results"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
C450=$WD/cutlass450/nvidia_cutlass_dsl/python_packages
[ -d "$C450" ] && export PYTHONPATH="$C450:$WD/cutlass450:$WD/fi_clean${PYTHONPATH:+:$PYTHONPATH}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"

mapfile -t BATCHES < <(python3 "$RH/batches_qab.py")
i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % NW)) -eq $W ]; then
    read -r fam sw a b c <<< "$batch"
    if [ "$fam" = "synth" ]; then
      tag="synth_${sw}_${a}_K${b}_${c}"                  # a=scen b=K c=dtype
      args="--family synth --sweep $sw --scenario $a --K $b --dtype $c"
    else
      tag="real_${sw}_${a}_${b}"                          # a=model b=dtype
      args="--family real --sweep $sw --model $a --dtype $b"
    fi
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== qab batch [$W]: $tag -> $rep.nsys-rep ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 "$RH/sweep_qab.py" $args --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM" > "$OUT/${tag}.log" 2>&1; then
      touch "$done_m"
    else
      echo "!!! batch $tag FAILED (see $OUT/${tag}.log; left un-marked for resume)"
    fi
  fi
  i=$((i+1))
done
echo "QABWORKER${W}DONE"
