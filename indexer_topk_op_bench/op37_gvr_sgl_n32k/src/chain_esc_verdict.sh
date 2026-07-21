#!/bin/bash
# [op37-esc] chain: wait for battery_esc all-PASS -> launch the 132-cell
# nsys verdict sweep (gvr_pr, sglang_v2, gvr_esc) on GPU0+GPU1.
# Usage: setsid src/chain_esc_verdict.sh >> results/esc_chain.log 2>&1
set -u
D=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op37_gvr_sgl_n32k
BLOG="$D/logs/battery_esc.log"
OUT="$D/results/esc"

echo "[chain] waiting for battery_esc verdict ($(date -u +%FT%T))"
for i in $(seq 1 240); do
  if grep -q "BATTERY_ESC:" "$BLOG" 2>/dev/null; then break; fi
  sleep 60
done
LINE=$(grep "BATTERY_ESC:" "$BLOG" | tail -1)
echo "[chain] battery verdict: $LINE"
if ! echo "$LINE" | grep -qE "BATTERY_ESC: ([0-9]+)/\1 PASS"; then
  echo "[chain] ABORT: battery not all-PASS (or timed out); NOT launching sweep"
  grep FAIL "$BLOG" | head -20
  exit 1
fi

echo "[chain] launching 132-cell verdict sweep -> $OUT ($(date -u +%FT%T))"
mkdir -p "$OUT"
OPS="gvr_pr,sglang_v2,gvr_esc" setsid "$D/src/drive_op37.sh" 0 2 "$D/src/batches_n32k.txt" "$OUT" > "$D/results/esc_w0.out" 2>&1 &
P0=$!
OPS="gvr_pr,sglang_v2,gvr_esc" setsid "$D/src/drive_op37.sh" 1 2 "$D/src/batches_n32k.txt" "$OUT" > "$D/results/esc_w1.out" 2>&1 &
P1=$!
wait $P0 $P1
echo "[chain] sweep workers done ($(date -u +%FT%T))"
grep -h "DONE\|FAILED" "$D/results/esc_w0.out" "$D/results/esc_w1.out" | tail -4
echo "CHAINDONE"
