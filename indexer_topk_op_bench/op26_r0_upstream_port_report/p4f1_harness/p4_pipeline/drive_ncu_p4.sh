#!/bin/bash
# NCU instruction-level P4 accounting over representative cells.
#   bash drive_ncu_p4.sh [gpu]
# RUN ONLY AFTER the p4pipe sweep is finished (no probes during grid runs).
set -e
GPU=${1:-0}
NCU=/opt/nvidia/nsight-compute/2026.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
mkdir -p "$HERE/ncu_reps"
cd "$HERE"

CELLS=${CELLS:-"flash_4k_L02 pro_8k_L30 flash_128k_L02 v32_32k_L34 flash_256k_L02 pro_256k_L30 flash_512k_L02 pro_1024k_L30 v32_128k_L14 v32_256k_L34 pro_4k_L02"}

for cell in $CELLS; do
  rep="$HERE/ncu_reps/p4_${cell}"
  if [ -f "${rep}.ncu-rep" ]; then echo "skip $cell (exists)"; continue; fi
  echo "=== ncu $cell"
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    "$NCU" --set full --import-source no \
    -k "regex:gvr_topk_kernel_gvrpkgp4t" -s 10 -c 1 -f -o "$rep" \
    python3 "$HERE/ncu_p4_one.py" --cell "$cell" \
    > "$HERE/ncu_reps/p4_${cell}.log" 2>&1
  env -u GITHUB_TOKEN -u HF_TOKEN \
    "$NCU" --import "${rep}.ncu-rep" --page source --csv \
    > "$HERE/ncu_reps/p4_${cell}_source.csv" 2>/dev/null
  env -u GITHUB_TOKEN -u HF_TOKEN \
    "$NCU" --import "${rep}.ncu-rep" --page details --csv \
    > "$HERE/ncu_reps/p4_${cell}_details.csv" 2>/dev/null
  python3 "$HERE/parse_ncu_p4.py" --csv "$HERE/ncu_reps/p4_${cell}_source.csv" \
    --out "$HERE/ncu_reps/p4_${cell}_segs.json" || echo "PARSE FAIL $cell"
done
echo "ncu sweep done"
