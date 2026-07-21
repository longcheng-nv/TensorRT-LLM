#!/bin/bash
# 8-GPU sharded 865-cell phase-breakdown run under nsys.
#   bash drive_phases_shards.sh <tag> [ngpu]
set -e
TAG=${1:-full}; NG=${2:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
mkdir -p "$HERE/nsys_reps"
cd "$HERE"   # shared cwd -> shared cute PTX cache
for g in $(seq 0 $((NG-1))); do
  setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -f true -o "$HERE/nsys_reps/phases_${TAG}_g$g" \
    python3 "$HERE/measure_phases_full.py" --shard $g/$NG \
    --tag ${TAG}_g$g > "$HERE/shard_${TAG}_g$g.log" 2>&1 &
  echo "shard $g pid $!"
done
wait
echo "all shards done"
