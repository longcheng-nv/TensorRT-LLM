#!/bin/bash
set -u
TAG=$1; GPU=$2; shift 2
B=$(dirname $(dirname $(readlink -f $0)))
cd $B
for CELL in "$@"; do
  M=results/nsys/${TAG}_${CELL}.done
  [ -f $M ] && { echo "[skip] $CELL"; continue; }
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU GVR_BSX_TP_BS=16 GVR_BSX_DENSE_BS=8 \
    nsys profile -o results/nsys/${TAG}_${CELL} --force-overwrite true \
    -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    python3 scripts/ab.py --cell $CELL --tag ${TAG}_${CELL} --bs 16,32,64,128,256,1024 \
      --reps-cold 10 --reps-warm 5 \
    > results/nsys/${TAG}_${CELL}.log 2>&1 \
  && touch $M && echo "[done] $CELL" || echo "[FAIL] $CELL"
done
