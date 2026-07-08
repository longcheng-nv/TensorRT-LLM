#!/bin/bash
# P0 batch driver, GPU0 (anchored axis): item 4 (17-cell P0 grid @HEAD)
# then item 2 (high-BS 3-arm A/B x 3 scenarios).
set -e
cd "$(dirname "$0")/.."
OUT=results/nsys/p0batch

echo "=== [gpu0] P0 17-cell grid @HLS HEAD ==="
GPU=0 bash scripts/drive_nsys_iter2.sh

echo "=== [gpu0] high-BS 3-arm A/B ==="
for scen in best worst real; do
  rep="$OUT/hb3_${scen}_fp32"
  if [ -f "${rep}.done" ]; then echo "skip $rep"; continue; fi
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=0 \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o "$rep" \
    python3 scripts/ab_p0batch.py --scenario "$scen" --dtype fp32 \
      --cells highbs --with-orig --out "${rep}.jsonl" --reps 30 \
    2>&1 | tail -3
  touch "${rep}.done"
done
echo "GPU0 P0 BATCH ALL DONE"
