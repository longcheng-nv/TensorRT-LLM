#!/bin/bash
# P0 batch driver, GPU1: item 3 (fp32 tail 3-arm direct verdict) then
# item 1 (16-bit HLS A/B, also 3-arm) x 3 scenarios each.
set -e
cd "$(dirname "$0")/.."
OUT=results/nsys/p0batch

echo "=== [gpu1] fp32 tail 3-arm A/B ==="
for scen in best worst real; do
  rep="$OUT/ab3_${scen}_fp32"
  if [ -f "${rep}.done" ]; then echo "skip $rep"; continue; fi
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=1 \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o "$rep" \
    python3 scripts/ab_p0batch.py --scenario "$scen" --dtype fp32 \
      --cells tail --with-orig --out "${rep}.jsonl" --reps 30 \
    2>&1 | tail -3
  touch "${rep}.done"
done

echo "=== [gpu1] 16-bit tail 3-arm A/B ==="
for dt in bf16 fp16; do
  for scen in best worst real; do
    rep="$OUT/ab3_${scen}_${dt}"
    if [ -f "${rep}.done" ]; then echo "skip $rep"; continue; fi
    env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=1 \
      nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
      -f true -o "$rep" \
      python3 scripts/ab_p0batch.py --scenario "$scen" --dtype "$dt" \
        --cells tail --with-orig --out "${rep}.jsonl" --reps 30 \
      2>&1 | tail -3
    touch "${rep}.done"
  done
done
echo "GPU1 P0 BATCH ALL DONE"
