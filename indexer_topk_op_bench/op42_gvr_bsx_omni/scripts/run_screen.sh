#!/bin/bash
# op42 nsys screen: cells sharded over 4 GPUs, one nsys rep per cell.
# Usage: run_screen.sh <tag> <gpu> <cell> [<cell>...]
set -u
TAG=$1; GPU=$2; shift 2
B=$(dirname $(dirname $(readlink -f $0)))
cd $B
for CELL in "$@"; do
  M=results/nsys/${TAG}_${CELL}.done
  [ -f $M ] && { echo "[skip] $CELL"; continue; }
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    nsys profile -o results/nsys/${TAG}_${CELL} --force-overwrite true \
    -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    python3 scripts/ab.py --cell $CELL --tag ${TAG}_${CELL} \
      --reps-cold 10 --reps-warm 5 ${AB_EXTRA:-} \
      > results/nsys/${TAG}_${CELL}.log 2>&1 \
  && touch $M && echo "[done] $CELL" || echo "[FAIL] $CELL"
done
