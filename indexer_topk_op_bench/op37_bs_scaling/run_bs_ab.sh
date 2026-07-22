#!/bin/bash
# op37 BS-scaling nsys driver: one cell per nsys run (checkpointable).
#   bash run_bs_ab.sh <cell> [gpu] [bs_list]
set -e
CELL=$1; GPU=${2:-0}; BSL=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
REP=$HERE/nsys_reps/bs_$CELL
mkdir -p "$HERE/nsys_reps"
EXTRA=()
[ -n "$BSL" ] && EXTRA+=(--bs "$BSL")
env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --capture-range-end=stop -f true -o "$REP" \
  python3 "$HERE/bs_ab.py" --cell "$CELL" --tag "$CELL" "${EXTRA[@]}"
python3 "$HERE/parse_bs.py" --rep "$REP.nsys-rep" --cell "$CELL"
