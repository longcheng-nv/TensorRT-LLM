#!/bin/bash
# op38: 8-GPU sharded variant-ladder probe of all v2-losing cases.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
for g in 0 1 2 3 4 5 6 7; do
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    python3 "$HERE/probe_v3.py" --shard $g/8 --tag v3 \
    > "$HERE/sweep_logs/v3_s$g.log" 2>&1 &
done
wait
echo "[drive] v3 probe all shards done"
