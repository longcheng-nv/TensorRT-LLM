#!/bin/bash
# op41: 8-GPU sharded nsys sweep of v3mt over the 75-cell fp32 envelope.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONNOUSERSITE=1
mkdir -p "$HERE/results/nsys" "$HERE/results/sweep_logs"
for g in 0 1 2 3 4 5 6 7; do
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -f true -o "$HERE/results/nsys/v3mt_s$g" \
    python3 "$HERE/scripts/bs41_nsys.py" --shard $g/8 --tag v3mt_s$g \
    > "$HERE/results/sweep_logs/v3mt_s$g.log" 2>&1 &
done
wait
echo "[drive] all shards done"
