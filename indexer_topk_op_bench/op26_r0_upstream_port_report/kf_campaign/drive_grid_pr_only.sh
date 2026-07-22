#!/bin/bash
# 8-GPU sharded 865-cell full-grid nsys, PR arm ONLY (denominator grid).
#   bash drive_grid_pr_only.sh <tag> [ngpu]
# GVRPKG_DIR env selects the package (default gvrpkg_04a0 = pinned head).
set -e
TAG=$1; NG=${2:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
mkdir -p "$HERE/nsys_reps" "$HERE/grid_logs"
for g in $(seq 0 $((NG-1))); do
  setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -f true -o "$HERE/nsys_reps/grid_${TAG}_g$g" \
    python3 "$HERE/nsys_ab.py" --arms gvr_pr --grid full --shard $g/$NG \
    --tag ${TAG}_g$g > "$HERE/grid_logs/${TAG}_g$g.log" 2>&1 &
  echo "shard $g pid $!"
done
wait
echo "all shards done"
