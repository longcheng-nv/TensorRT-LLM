#!/bin/bash
# op21 iter8 nsys: gvr_ms_auto on the P0 grid, bf16 + fp16, cold-L2,
# resumable. Token hygiene: env -u (nsys embeds process env).
set -e
cd "$(dirname "$0")/.."
OUT=results/nsys
mkdir -p "$OUT"
CELLS="\
1024:65536:1:60 1024:65536:4:60 1024:65536:8:60 1024:65536:16:60 \
1024:131072:1:60 1024:131072:4:60 1024:131072:8:60 1024:131072:16:60 \
1024:262144:1:60 1024:262144:4:60 1024:262144:8:60 1024:262144:16:60 \
512:131072:1:60 512:262144:1:60 2048:131072:1:60 2048:262144:1:60 2048:262144:16:60"
for DT in bf16 fp16; do
  for cell in $CELLS; do
    IFS=':' read -r K N BS IT <<< "$cell"
    rep="$OUT/msa_k${K}_${DT}_n${N}_bs${BS}"
    if [ -f "${rep}.nsys-rep" ]; then echo "skip $rep"; continue; fi
    env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=${GPU:-0} \
      nsys profile -c cudaProfilerApi --force-overwrite true -o "$rep" \
      python3 scripts/nsys_run_auto.py "$K" "$DT" "$N" "$BS" "$IT" 2>&1 | tail -1
  done
done
