#!/bin/bash
# op16 Scheme X nsys A/B — remaining K/dtype grid (abX_ prefix), STRICTLY sequential
# (one nsys at a time; co-tenancy corrupts cold-L2). fp32/bf16/fp16 x K512/1024/2048,
# minus the two already done (abX_K512_fp32, abX_K2048_fp32).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="$(dirname "$HERE")"
GPU="${1:-0}"
OUT="$BUCKET/results/nsys_ab"; REP="$OUT/nsys_reps"
mkdir -p "$REP"

# (K, dtype) pairs still to run
PAIRS=("512 bf16" "512 fp16" "1024 fp32" "1024 bf16" "1024 fp16" "2048 bf16" "2048 fp16")

for p in "${PAIRS[@]}"; do
  set -- $p; K=$1; dt=$2
  tag="abX_K${K}_${dt}"
  echo "===== $(date -u +%H:%M:%S) nsys A/B $tag ====="
  CUDA_VISIBLE_DEVICES="$GPU" nsys profile -t cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "$REP/$tag" -f true \
    python3 "$HERE/nsys_ab.py" --K "$K" --dtype "$dt" \
      --out "$OUT/$tag.jsonl" \
    > "$OUT/$tag.runlog" 2>&1
  echo "  $tag EXIT=$? ($(wc -l < "$OUT/$tag.jsonl" 2>/dev/null) recs)"
done
echo "=== GRID REMAINING DONE ==="
