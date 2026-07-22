#!/bin/bash
# op37 ship-verdict driver: 2 shards, one nsys rep per batch, <=2 concurrent
# nsys (discipline). Idempotent: a batch is skipped iff its csv marker exists.
#   bash drive_ab37_ship.sh            # both shards (backgrounded per shard)
#   bash drive_ab37_ship.sh <shard>    # run one shard inline (0 or 1)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
GPUS=(2 3)
mkdir -p "$HERE/ship/nsys_reps"
cd "$HERE"

run_shard() {
  local s=$1 g=${GPUS[$1]}
  local i=0
  while IFS= read -r spec; do
    if (( i % 2 == s )); then
      local tag=${spec// /_}
      if [[ -f "$HERE/ship/ship_${tag}.csv" ]]; then
        echo "[shard$s] skip $spec (done)"
      else
        echo "[shard$s] run  $spec on GPU$g"
        env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
          nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
          --capture-range-end=stop -f true \
          -o "$HERE/ship/nsys_reps/ship_${tag}" \
          python3 "$HERE/ab37_ship.py" --batch "$spec" \
          >> "$HERE/ship/shard${s}.log" 2>&1 \
          || echo "[shard$s] BATCH FAILED: $spec" >> "$HERE/ship/shard${s}.log"
      fi
    fi
    i=$((i+1))
  done < <(python3 "$HERE/ab37_ship.py" --list)
  echo "[shard$s] all batches done"
}

if [[ $# -ge 1 ]]; then
  run_shard "$1"
else
  for s in 0 1; do
    setsid bash "$0" "$s" > "$HERE/ship/driver${s}.log" 2>&1 &
    echo "shard $s pgid $!"
  done
fi
