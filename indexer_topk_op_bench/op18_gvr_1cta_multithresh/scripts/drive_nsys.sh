#!/bin/bash
# op18 nsys pure-kernel validation: base vs auto-dispatch mt on representative
# cells. Cold-L2 flush inside the cudaProfilerApi window; pure-kernel time from
# cuda_gpu_kern_sum, gvr rows only (evict memset/uniform excluded).
set -e
cd "$(dirname "$0")/.."
OUT=results/nsys
mkdir -p "$OUT"
CELLS="512:16384 512:65536 512:262144 1024:32768 1024:262144 2048:65536 2048:262144"
for cell in $CELLS; do
  K=${cell%%:*}; N=${cell##*:}
  for which in base auto; do
    rep="$OUT/${which}_k${K}_fp32_n${N}"
    if [ -f "${rep}.nsys-rep" ]; then echo "skip $rep"; continue; fi
    nsys profile -c cudaProfilerApi --force-overwrite true -o "$rep" \
      python3 scripts/nsys_run.py "$which" "$K" fp32 "$N" 100 2>&1 | tail -1
  done
done
echo "=== pure-kernel summary (gvr rows) ==="
for cell in $CELLS; do
  K=${cell%%:*}; N=${cell##*:}
  for which in base auto; do
    rep="$OUT/${which}_k${K}_fp32_n${N}"
    nsys stats --report cuda_gpu_kern_sum --format csv --force-export true \
      "$rep.nsys-rep" 2>/dev/null | grep -i "gvr\|topk" | head -2 | \
      awk -F',' -v w="$which" -v k="$K" -v n="$N" '{printf "%s K=%s N=%s med_ns=%s name=%s\n", w, k, n, $6, $9}'
  done
done
