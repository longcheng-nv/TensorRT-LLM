#!/bin/bash
# BS-scaling driver: 7 shards on GPUs 0-5,7 (GPU6 has resident memory).
# One nsys rep per (model, isl) batch; csv marker = resume skip.
#   bash drive_ab37_bs.sh            # launch all shards
#   bash drive_ab37_bs.sh <shard>    # one shard inline
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
GPUS=(0 1 2 3 4 5 7)
NS=${#GPUS[@]}
mkdir -p "$HERE/ship/nsys_reps"
cd "$HERE"

run_shard() {
  local s=$1 g=${GPUS[$1]}
  local i=0
  while IFS= read -r spec; do
    if (( i % NS == s )); then
      local tag=${spec// /_}
      if [[ -f "$HERE/ship/bs_${tag}.csv" ]]; then
        echo "[shard$s] skip $spec (done)"
      else
        echo "[shard$s] run  $spec on GPU$g"
        env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
          nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
          --capture-range-end=stop -f true \
          -o "$HERE/ship/nsys_reps/bs_${tag}" \
          python3 "$HERE/ab37_bs.py" --batch "$spec" \
          >> "$HERE/ship/bs_shard${s}.log" 2>&1 \
          || echo "[shard$s] BATCH FAILED: $spec" >> "$HERE/ship/bs_shard${s}.log"
      fi
    fi
    i=$((i+1))
  done < <(python3 "$HERE/ab37_bs.py" --list)
  echo "[shard$s] all batches done"
}

if [[ $# -ge 1 ]]; then
  run_shard "$1"
else
  for s in $(seq 0 $((NS-1))); do
    setsid bash "$0" "$s" > "$HERE/ship/bs_driver${s}.log" 2>&1 &
    echo "shard $s pid $!"
  done
fi
