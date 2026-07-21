#!/bin/bash
# [op37-esc] chain v2: battery v3 (post falsi-factor text) on GPU0 ->
# on all-PASS launch the 132-cell re-verdict (gvr_pr, sglang_v2, gvr_esc)
# on GPU0+GPU1 -> results/esc2.
set -u
D=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op37_gvr_sgl_n32k
BLOG="$D/logs/battery_esc_v3.log"
OUT="$D/results/esc2"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450

echo "[chain2] battery v3 starting ($(date -u +%FT%T))"
CUDA_VISIBLE_DEVICES=0 python3 "$D/variant/battery_esc.py" > "$BLOG" 2>&1
LINE=$(grep "BATTERY_ESC:" "$BLOG" | tail -1)
echo "[chain2] battery verdict: $LINE"
if ! echo "$LINE" | grep -qE "BATTERY_ESC: ([0-9]+)/\1 PASS"; then
  echo "[chain2] ABORT: battery v3 not all-PASS; NOT launching sweep"
  grep FAIL "$BLOG" | head -20
  exit 1
fi

echo "[chain2] launching 132-cell re-verdict -> $OUT ($(date -u +%FT%T))"
mkdir -p "$OUT"
OPS="gvr_pr,sglang_v2,gvr_esc" setsid "$D/src/drive_op37.sh" 0 2 "$D/src/batches_n32k.txt" "$OUT" > "$D/results/esc2_w0.out" 2>&1 &
P0=$!
OPS="gvr_pr,sglang_v2,gvr_esc" setsid "$D/src/drive_op37.sh" 1 2 "$D/src/batches_n32k.txt" "$OUT" > "$D/results/esc2_w1.out" 2>&1 &
P1=$!
wait $P0 $P1
echo "[chain2] sweep workers done ($(date -u +%FT%T))"
echo "CHAIN2DONE"
