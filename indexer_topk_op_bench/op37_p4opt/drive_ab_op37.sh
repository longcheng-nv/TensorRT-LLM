#!/bin/bash
# 2-shard paired nsys A/B (ship-verdict discipline: <=2 concurrent nsys).
#   bash drive_ab_op37.sh <tag>
set -e
TAG=${1:-ab}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
mkdir -p "$HERE/nsys_reps"
cd "$HERE"
for g in 0 1; do
  setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -f true -o "$HERE/nsys_reps/ab37_${TAG}_g$g" \
    python3 "$HERE/ab_op37.py" --shard $g/2 --tag ${TAG}_g$g \
    > "$HERE/ab_${TAG}_g$g.log" 2>&1 &
  echo "shard $g pid $!"
done
wait
echo done
