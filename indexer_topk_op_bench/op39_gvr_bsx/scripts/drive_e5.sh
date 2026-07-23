#!/bin/bash
# op39: 8-GPU sharded envelope sweep of production arm v2.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONNOUSERSITE=1
mkdir -p "$HERE/results/nsys" "$HERE/results/sweep_logs"
for g in 0 1 2 3 4 5 6 7; do
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -f true -o "$HERE/results/nsys/e5_s$g" \
    python3 "$HERE/scripts/bs39_nsys.py" --shard $g/8 --tag e5_s$g \
    > "$HERE/results/sweep_logs/e5_s$g.log" 2>&1 &
done
wait
echo "[drive] e1 all shards done"
