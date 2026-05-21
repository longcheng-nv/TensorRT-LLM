#!/bin/bash
# V4 Pro convenience: synth all 3 beta cfgs × N ∈ V4-Pro production envelope.
# Recall N is POST-compress (V4 cr=4):
#   32K ISL → N=7557    64K ISL → N=14460    100K ISL → ~25100 (no real Pro 100K capture yet)
set -e

OUTDIR="${1:-./synth_out_v4pro}"
BS="${BS:-1}"
SEED="${SEED:-42}"
BENCH_FLAG=""
[ "${BENCH:-0}" = "1" ] && BENCH_FLAG="--bench"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for N in 4K 7557 14460 25100; do
  for CFG in beta_shallow beta_moderate beta_deep; do
    python3 "$SCRIPT_DIR/synth_temporal_data.py" \
      --N "$N" --cfg "$CFG" --bs "$BS" --seed "$SEED" \
      --outdir "$OUTDIR" $BENCH_FLAG
  done
done

echo
echo "All synth outputs under: $OUTDIR"
ls -1 "$OUTDIR"
