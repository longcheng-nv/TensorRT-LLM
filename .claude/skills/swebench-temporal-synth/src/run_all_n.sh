#!/bin/bash
# Convenience: synth all 3 beta cfgs × N ∈ {4K, 8K, 16K, 32K, 64K, 128K}
# and optionally bench each. Reproduces Q19-tempC sweep dimension.
set -e

OUTDIR="${1:-./synth_out_full}"
BS="${BS:-1}"
SEED="${SEED:-42}"
BENCH_FLAG=""
[ "${BENCH:-0}" = "1" ] && BENCH_FLAG="--bench"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for N in 4K 8K 16K 32K 64K 128K; do
  for CFG in beta_shallow beta_moderate beta_deep; do
    python3 "$SCRIPT_DIR/synth_temporal_data.py" \
      --N "$N" --cfg "$CFG" --bs "$BS" --seed "$SEED" \
      --outdir "$OUTDIR" $BENCH_FLAG
  done
done

echo
echo "All synth outputs under: $OUTDIR"
ls -1 "$OUTDIR"
