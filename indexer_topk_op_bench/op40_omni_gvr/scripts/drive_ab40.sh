#!/bin/bash
# op40 A/B nsys driver: shards over GPUs (one nsys per GPU), idempotent:
# batch skipped iff its csv marker exists. GPU list via GPUS_LIST env.
#   GPUS_LIST="0 1 2 3 4 5 6 7" bash drive_ab40.sh <arms> <tagdir>   # all shards
#   bash drive_ab40.sh <arms> <tagdir> <shard-idx>                   # one shard inline
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OP40=$(dirname "$HERE")
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
GPUS=(${GPUS_LIST:-2 3})
NS=${#GPUS[@]}
ARMS=${1:?arms e.g. base or base,v1}
TAGDIR=${2:?tagdir e.g. bl0}
mkdir -p "$OP40/results/$TAGDIR/nsys_reps"

run_shard() {
  local s=$1 g=${GPUS[$1]}
  local i=0
  while IFS= read -r spec; do
    if (( i % NS == s )); then
      local tag=${spec// /_}
      if [[ -f "$OP40/results/$TAGDIR/${tag}.csv" ]]; then
        echo "[shard$s] skip $spec (done)"
      else
        echo "[shard$s] run  $spec on GPU$g"
        env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
          nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
          --capture-range-end=stop -f true \
          -o "$OP40/results/$TAGDIR/nsys_reps/${tag}" \
          python3 "$HERE/ab40.py" --arms "$ARMS" --tagdir "$TAGDIR" \
          --batch "$spec" \
          >> "$OP40/results/$TAGDIR/shard${s}.log" 2>&1 \
          || echo "[shard$s] BATCH FAILED: $spec" >> "$OP40/results/$TAGDIR/shard${s}.log"
      fi
    fi
    i=$((i+1))
  done < <(python3 "$HERE/ab40.py" --list)
  echo "[shard$s] all batches done"
}

if [[ $# -ge 3 ]]; then
  run_shard "$3"
else
  for s in $(seq 0 $((NS-1))); do
    GPUS_LIST="${GPUS[*]}" setsid bash "$0" "$ARMS" "$TAGDIR" "$s" > "$OP40/results/$TAGDIR/driver${s}.log" 2>&1 &
    echo "shard $s pid $!"
  done
fi
