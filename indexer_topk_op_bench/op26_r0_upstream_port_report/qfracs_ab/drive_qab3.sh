#!/bin/bash
# qab phase-2 driver — pre-PR validation batches (16-bit / BS axis).
# Usage: BATCHFILE=batches_qab3.py drive_qab2.sh <GPU>
# Single worker, batch file selectable; same protocol/OUT as drive_qab_shard.sh.
GPU=$1
BATCHFILE="${BATCHFILE:-batches_qab3.py}"
WD=/tmp/gvrqab
RH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/qfracs_ab
OUT="$WD/qab3_results"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
C450=$WD/cutlass450/nvidia_cutlass_dsl/python_packages
[ -d "$C450" ] && export PYTHONPATH="$C450:$WD/cutlass450:$WD/fi_clean${PYTHONPATH:+:$PYTHONPATH}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"

mapfile -t BATCHES < <(python3 "$RH/$BATCHFILE")
for batch in "${BATCHES[@]}"; do
  read -r fam sw a b c <<< "$batch"
  if [ "$fam" = "synth" ]; then
    tag="synth_${sw}_${a}_K${b}_${c}"
    args="--family synth --sweep $sw --scenario $a --K $b --dtype $c"
  else
    tag="real_${sw}_${a}_${b}"
    args="--family real --sweep $sw --model $a --dtype $b"
  fi
  done_m="$OUT/.done_${tag}"
  rep="$OUT/nsys_reps/${tag}"
  if [ -f "$done_m" ]; then echo "SKIP done: $tag"; continue; fi
  echo "=== qab2 batch: $tag -> $rep.nsys-rep ($(date -u +%T)) ==="
  if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
       nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
       -o "$rep" -f true \
       python3 "$RH/sweep_qab3.py" $args --out-root "$OUT" \
         --reps "$REPS" --reps-warm "$REPS_WARM" > "$OUT/${tag}.log" 2>&1; then
    touch "$done_m"
  else
    echo "!!! batch $tag FAILED (see $OUT/${tag}.log; left un-marked for resume)"
  fi
done
echo "QAB3DONE:$BATCHFILE"
