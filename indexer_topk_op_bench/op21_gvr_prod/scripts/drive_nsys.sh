#!/bin/bash
# op21 iter1 nsys pure-kernel verdict: gvr_ms on the PLAN P0+P1 priority
# grid (fp32). Cold-L2 flush inside the cudaProfilerApi window; pure-kernel
# median from cuda_gpu_kern_sum, gvr rows only. Resumable (skips existing
# reps). Token hygiene: env -u GITHUB_TOKEN -u HF_TOKEN (nsys embeds env).
set -e
cd "$(dirname "$0")/.."
OUT=results/nsys
mkdir -p "$OUT"
# P0: K1024 largeN smallBS; P1: all K, midN, highBS. K:N:BS:iters
CELLS="\
1024:65536:1:60 1024:65536:4:60 1024:65536:8:60 1024:65536:16:60 \
1024:131072:1:60 1024:131072:4:60 1024:131072:8:60 1024:131072:16:60 \
1024:262144:1:60 1024:262144:4:60 1024:262144:8:60 1024:262144:16:60 \
1024:4096:64:40 1024:4096:256:30 1024:4096:1024:20 \
1024:8192:64:40 1024:8192:256:30 1024:8192:1024:20 \
1024:16384:64:40 1024:16384:256:30 1024:16384:1024:20 \
512:4096:64:40 512:4096:256:30 512:4096:1024:20 \
512:8192:64:40 512:8192:256:30 512:8192:1024:20 \
512:16384:64:40 512:16384:256:30 512:16384:1024:20 \
2048:8192:64:40 2048:8192:256:30 2048:8192:1024:20 \
2048:16384:64:40 2048:16384:256:30 2048:16384:1024:20"
for cell in $CELLS; do
  IFS=':' read -r K N BS IT <<< "$cell"
  rep="$OUT/ms_k${K}_fp32_n${N}_bs${BS}"
  if [ -f "${rep}.nsys-rep" ]; then echo "skip $rep"; continue; fi
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=${GPU:-0} \
    nsys profile -c cudaProfilerApi --force-overwrite true -o "$rep" \
    python3 scripts/nsys_run.py "$K" fp32 "$N" "$BS" "$IT" 2>&1 | tail -1
done
echo "=== pure-kernel median (gvr rows) ==="
for cell in $CELLS; do
  IFS=':' read -r K N BS IT <<< "$cell"
  rep="$OUT/ms_k${K}_fp32_n${N}_bs${BS}"
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export true \
    "$rep.nsys-rep" 2>/dev/null | grep -i "gvr" | head -1 | \
    awk -F',' -v k="$K" -v n="$N" -v b="$BS" \
      '{printf "K=%s N=%s BS=%s med_ns=%s\n", k, n, b, $6}'
done
