#!/bin/bash
# op16 nsys pure-kernel cold-L2 A/B driver (REPORT-IDENTICAL protocol).
# Runs nsys_ab.py under nsys for each (K,dtype), producing one .nsys-rep + jsonl
# per batch. Then parse_ab.py extracts per-op cold/warm us and compares to report.
#
# Usage: bash run_nsys_ab.sh [GPU] [aim_permille] [sample_size]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="$(dirname "$HERE")"
GPU="${1:-0}"
AIM="${2:-1150}"
NS="${3:-4096}"
OUTROOT="$BUCKET/results/nsys_ab"
REPDIR="$OUTROOT/nsys_reps"
mkdir -p "$REPDIR"

KS=(512 1024 2048)
DTS=(fp32 bf16 fp16)

for K in "${KS[@]}"; do
  for dt in "${DTS[@]}"; do
    jsonl="$OUTROOT/ab_K${K}_${dt}.jsonl"
    rep="$REPDIR/ab_K${K}_${dt}"
    echo "===== nsys A/B K=$K dt=$dt (aim=$AIM ns=$NS) ====="
    CUDA_VISIBLE_DEVICES="$GPU" nsys profile -t cuda,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      -o "$rep" -f true \
      python3 "$HERE/nsys_ab.py" --K "$K" --dtype "$dt" \
        --sample-size "$NS" --aim "$AIM" --out "$jsonl" \
        2>&1 | grep -v "FutureWarning\|warnings.warn\|pynvml\|UserWarning" | tail -12
  done
done
echo "=== ALL BATCHES DONE — parsing ==="
python3 "$HERE/parse_ab.py" "$OUTROOT"
