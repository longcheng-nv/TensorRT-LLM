#!/bin/bash
# N-shard 865-cell grid on an explicit GPU list: bash drive_grid_gpulist.sh <cand_dir> <tag> <gpu,gpu,...>
set -e
CDIR=$1; TAG=$2; IFS=',' read -ra GPUS <<< "$3"
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
    python3 "$HERE/nsys_ab.py" --cand "$CDIR" --grid full --shard $i/$NG \
    --tag ${TAG}_g$i > "$HERE/grid_logs/${TAG}_g$i.log" 2>&1 &
  echo "shard $i -> GPU $g pid $!"
done
wait
echo "all shards done"
