#!/bin/bash
# op21 iter13 — P0 no-regress spot A/B (iter11 precedent: 5-cell old/new
# nsys on the op21 synth data; fallback rarely fires there so this checks
# the fast path carries no tax from the seeding writes).
# Usage: GPU=1 ./scripts/ab_p0_spot_logfalsi.sh
set -u
cd "$(dirname "$0")/.."
OUT=results/nsys/iter13_p0_spot
mkdir -p "$OUT"
CELLS="512:fp32:262144:1 1024:fp32:65536:1 1024:fp32:262144:1 \
2048:fp32:262144:1 1024:bf16:262144:1"
for cell in $CELLS; do
  IFS=':' read -r K DT N BS <<< "$cell"
  for arm in old new; do
    [ "$arm" = old ] && LF=0 || LF=1
    rep="$OUT/${arm}_k${K}_${DT}_n${N}_bs${BS}"
    [ -f "${rep}.nsys-rep" ] && { echo "skip $rep"; continue; }
    OP21_FB_LOGFALSI=$LF env -u GITHUB_TOKEN -u HF_TOKEN \
      CUDA_VISIBLE_DEVICES=${GPU:-0} \
      nsys profile -c cudaProfilerApi --capture-range-end=stop -f true \
      -o "$rep" python3 scripts/nsys_run_auto.py "$K" "$DT" "$N" "$BS" 60 \
      > "$rep.out" 2>&1
    med=$(env -u GITHUB_TOKEN -u HF_TOKEN nsys stats --report cuda_gpu_kern_sum \
      --format csv "$rep.nsys-rep" 2>/dev/null | grep -i "gvr" | head -1 \
      | awk -F, '{printf "%.2f", $5/1000}')
    echo "P0SPOT $arm K=$K $DT N=$N BS=$BS med_us=$med"
  done
done
echo "P0 SPOT DONE"
