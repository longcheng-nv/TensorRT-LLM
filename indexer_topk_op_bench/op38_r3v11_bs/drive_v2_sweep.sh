#!/bin/bash
# op38: 8-GPU sharded nsys sweep of candidate v2 over the 75-cell fp32 envelope.
#   bash drive_v2_sweep.sh <tag>
set -e
TAG=${1:-v2}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
mkdir -p "$HERE/nsys_reps" "$HERE/sweep_logs"
for g in 0 1 2 3 4 5 6 7; do
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -f true -o "$HERE/nsys_reps/${TAG}_s$g" \
    python3 "$HERE/bs38_nsys.py" --shard $g/8 --tag ${TAG}_s$g \
    > "$HERE/sweep_logs/${TAG}_s$g.log" 2>&1 &
done
wait
echo "[drive] all shards done"
