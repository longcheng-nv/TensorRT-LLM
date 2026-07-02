#!/bin/bash
# op19 nsys pure-kernel validation: base vs auto-dispatch sandwich on
# representative winning cells (fp32). Cold-L2 flush inside the
# cudaProfilerApi window; pure-kernel time from cuda_gpu_kern_sum, gvr rows
# only (evict uniform_ kernel excluded).
set -e
cd "$(dirname "$0")/.."
OUT=results/nsys
mkdir -p "$OUT"
# K:N:BS:iters — BS=1 single-CTA + cluster cells (60 iters), high-BS cells (30)
CELLS="512:16384:1:60 512:262144:1:60 1024:32768:1:60 1024:262144:16:60 2048:262144:16:60 512:262144:2048:20 1024:32768:2048:20 2048:262144:2048:20"
for cell in $CELLS; do
  IFS=':' read -r K N BS IT <<< "$cell"
  for which in base auto; do
    rep="$OUT/${which}_k${K}_fp32_n${N}_bs${BS}"
    if [ -f "${rep}.nsys-rep" ]; then echo "skip $rep"; continue; fi
    nsys profile -c cudaProfilerApi --force-overwrite true -o "$rep" \
      python3 scripts/nsys_run.py "$which" "$K" fp32 "$N" "$BS" "$IT" 2>&1 | tail -1
  done
done
echo "=== pure-kernel summary (gvr rows) ==="
for cell in $CELLS; do
  IFS=':' read -r K N BS IT <<< "$cell"
  for which in base auto; do
    rep="$OUT/${which}_k${K}_fp32_n${N}_bs${BS}"
    nsys stats --report cuda_gpu_kern_sum --format csv --force-export true \
      "$rep.nsys-rep" 2>/dev/null | grep -i "gvr\|topk" | head -2 | \
      awk -F',' -v w="$which" -v k="$K" -v n="$N" -v b="$BS" \
        '{printf "%s K=%s N=%s BS=%s med_ns=%s name=%s\n", w, k, n, b, $6, substr($9,1,60)}'
  done
done
