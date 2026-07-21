#!/bin/bash
# nsys A/B driver: candidate dir -> per-cell cold/warm GPU-projected times.
#   bash run_nsys_ab.sh <cand_dir> [tag] [gpu] [cells]
set -e
CDIR=$1; TAG=${2:-t0}; GPU=${3:-6}; CELLS=${4:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
REP=$HERE/nsys_reps/ab_$TAG
mkdir -p $HERE/nsys_reps
env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -f true -o "$REP" \
  python3 "$HERE/nsys_ab.py" --cand "$CDIR" --tag "$TAG" --cells "$CELLS"
python3 "$HERE/parse_ab.py" --rep "$REP.nsys-rep" --tag "$TAG"
