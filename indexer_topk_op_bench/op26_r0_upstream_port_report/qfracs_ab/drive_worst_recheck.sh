#!/bin/bash
# Worst-cell noise re-check: repeat the synth-worst K2048 fp32 seq-len batch
# 3x into SEPARATE out-roots (fresh jsonl+rep each; no resume interference).
# Usage: drive_worst_recheck.sh <GPU>
GPU=$1
WD=/tmp/gvrqab
RH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/qfracs_ab
export PYTHONNOUSERSITE=1
C450=$WD/cutlass450/nvidia_cutlass_dsl/python_packages
[ -d "$C450" ] && export PYTHONPATH="$C450:$WD/cutlass450:$WD/fi_clean${PYTHONPATH:+:$PYTHONPATH}"
for r in 1 2 3; do
  OUT="$WD/worst_recheck_r$r"
  mkdir -p "$OUT/nsys_reps"
  [ -f "$OUT/.done" ] && { echo "SKIP r$r"; continue; }
  echo "=== worst recheck r$r ($(date -u +%T)) ==="
  if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
       nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
       -o "$OUT/nsys_reps/synth_seqlen_worst_K2048_fp32" -f true \
       python3 "$RH/sweep_qab.py" --family synth --sweep seqlen --scenario worst \
         --K 2048 --dtype fp32 --out-root "$OUT" \
         --reps 20 --reps-warm 50 > "$OUT/run.log" 2>&1; then
    touch "$OUT/.done"
  else
    echo "!!! recheck r$r FAILED"
  fi
done
echo "WORSTRECHECKDONE"
