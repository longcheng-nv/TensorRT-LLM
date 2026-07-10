#!/usr/bin/env bash
# op27 iter1 — nsys driver for the small-N decomposition A/B.
# Usage: GPU=0 [SCENARIOS="best worst real"] ./drive_ab_decomp.sh
# nsys gotchas: env -u GITHUB_TOKEN -u HF_TOKEN (sqlite embeds process env);
# `nsys -c cudaProfilerApi` exits 143 on SUCCESS -> no `set -e`.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
SCENARIOS="${SCENARIOS:-best worst real}"
OUT="$HERE/results/nsys/ab_decomp${SUFFIX:-}"
mkdir -p "$OUT"
cd "$HERE"

echo "### drive_ab_decomp: host=$(hostname) GPU=$GPU start=$(date -u +%FT%TZ)"

for scen in $SCENARIOS; do
  rep="$OUT/ab_${scen}_fp32"
  if [ -f "$rep.done" ]; then echo "SKIP $scen (done)"; continue; fi
  echo "=== scenario $scen ==="
  rm -f "$rep.jsonl"
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o "$rep" \
    python3 ab_decomp.py --scenario "$scen" --out "$rep.jsonl" \
    > "$rep.out" 2>&1
  grep -q "AB BATCH DONE" "$rep.out" && touch "$rep.done" \
    || echo "WARN: $scen batch did not finish clean"
done
echo "### drive_ab_decomp done $(date -u +%FT%TZ)"
