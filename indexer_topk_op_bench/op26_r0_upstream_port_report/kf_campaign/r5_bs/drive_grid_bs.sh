#!/bin/bash
# R5 sharded BS-grid nsys driver on an explicit GPU list.
#   bash drive_grid_bs.sh <tag> <gpu,gpu,...> [cand_dir] [extra nsys_bs args...]
# cand_dir omitted/'-' => PR arm only (denominator grid).
set -e
TAG=$1; IFS=',' read -ra GPUS <<< "$2"; CDIR=${3:--}
shift 3 2>/dev/null || shift $#
NG=${#GPUS[@]}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
mkdir -p "$HERE/nsys_reps" "$HERE/grid_logs"
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}
  if [ "$CDIR" = "-" ]; then
    ARGS="--arms gvr_pr"
  else
    ARGS="--cand $CDIR"
  fi
  setsid env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -f true -o "$HERE/nsys_reps/grid_${TAG}_g$i" \
    python3 "$HERE/nsys_bs.py" $ARGS --shard $i/$NG --tag ${TAG}_g$i "$@" \
    > "$HERE/grid_logs/${TAG}_g$i.log" 2>&1 &
  echo "shard $i -> GPU $g pid $!"
done
wait
echo "all shards done"
