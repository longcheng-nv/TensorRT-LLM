#!/bin/bash
# op25 — P0 no-regress grid (17 cells, gvr_ms_auto @ op25 ship config) into
# a fresh dir; verdict via OP21_NSYS_DIR + nsys_verdict.py msa fp32.
# Clone of op21 drive_nsys_iter2.sh (resumable, cold-L2, token hygiene).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP21="$HERE/../op21_gvr_prod"
OUT="$HERE/results/nsys/p0_ship"
mkdir -p "$OUT"
cd "$OP21"
CELLS="\
1024:65536:1:60 1024:65536:4:60 1024:65536:8:60 1024:65536:16:60 \
1024:131072:1:60 1024:131072:4:60 1024:131072:8:60 1024:131072:16:60 \
1024:262144:1:60 1024:262144:4:60 1024:262144:8:60 1024:262144:16:60 \
512:131072:1:60 512:262144:1:60 2048:131072:1:60 2048:262144:1:60 2048:262144:16:60"
for cell in $CELLS; do
  IFS=':' read -r K N BS IT <<< "$cell"
  rep="$OUT/msa_k${K}_fp32_n${N}_bs${BS}"
  if [ -f "${rep}.nsys-rep" ]; then echo "skip $rep"; continue; fi
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=${GPU:-1} \
    nsys profile -c cudaProfilerApi --force-overwrite true -o "$rep" \
    python3 scripts/nsys_run_auto.py "$K" fp32 "$N" "$BS" "$IT" 2>&1 | tail -1
done
echo "P0 grid done -> OP21_NSYS_DIR=$OUT python3 scripts/nsys_verdict.py msa fp32"
