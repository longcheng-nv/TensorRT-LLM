#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op19 nsys pure-kernel validation, 16-bit dtypes: base vs auto-dispatch
# sandwich on representative winning cells (bf16/fp16). Cold-L2 flush inside
# the cudaProfilerApi window; pure-kernel time from cuda_gpu_kern_sum, gvr
# rows only. Tokens are stripped from the profiled env (nsys sqlite embeds
# the process environment); reports stay untracked.
set -e
cd "$(dirname "$0")/.."
OUT=results/nsys
mkdir -p "$OUT"
# K:dtype:N:BS:iters — top fullgrid winners per (dtype, cfg family)
CELLS="2048:bf16:262144:1:60 1024:bf16:8192:8:60 1024:bf16:16384:2048:20 \
2048:fp16:262144:1:60 1024:fp16:32768:4:60 1024:fp16:16384:2048:20"
for cell in $CELLS; do
  IFS=':' read -r K DT N BS IT <<< "$cell"
  for which in base auto; do
    rep="$OUT/${which}_k${K}_${DT}_n${N}_bs${BS}"
    if [ -f "${rep}.nsys-rep" ]; then echo "skip $rep"; continue; fi
    env -u GITHUB_TOKEN -u HF_TOKEN -u GH_TOKEN \
      nsys profile -c cudaProfilerApi --force-overwrite true -o "$rep" \
      python3 scripts/nsys_run.py "$which" "$K" "$DT" "$N" "$BS" "$IT" 2>&1 | tail -1
  done
done
echo "=== pure-kernel summary (gvr rows) ==="
for cell in $CELLS; do
  IFS=':' read -r K DT N BS IT <<< "$cell"
  for which in base auto; do
    rep="$OUT/${which}_k${K}_${DT}_n${N}_bs${BS}"
    nsys stats --report cuda_gpu_kern_sum --format csv --force-export true \
      "$rep.nsys-rep" 2>/dev/null | grep -i "gvr\|topk" | head -2 | \
      awk -F',' -v w="$which" -v k="$K" -v d="$DT" -v n="$N" -v b="$BS" \
        '{printf "%s K=%s %s N=%s BS=%s med_ns=%s name=%s\n", w, k, d, n, b, $6, substr($9,1,60)}'
  done
done
