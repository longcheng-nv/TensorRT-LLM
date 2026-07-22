#!/bin/bash
# N-shard 865-cell grid, PR arm ONLY, explicit GPU list:
#   bash drive_grid_pr_gpulist.sh <tag> <gpu,gpu,...>
# GVRPKG_DIR env selects the package (default gvrpkg_04a0 = pinned head).
set -e
TAG=$1; IFS=',' read -ra GPUS <<< "$2"
NG=${#GPUS[@]}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
mkdir -p "$HERE/nsys_reps" "$HERE/grid_logs"
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}
  setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -f true -o "$HERE/nsys_reps/grid_${TAG}_g$i" \
    python3 "$HERE/nsys_ab.py" --arms gvr_pr --grid full --shard $i/$NG \
    --tag ${TAG}_g$i > "$HERE/grid_logs/${TAG}_g$i.log" 2>&1 &
  echo "shard $i -> GPU $g pid $!"
done
wait
echo "all shards done"
