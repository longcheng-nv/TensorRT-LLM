#!/bin/bash
# Pass-2 driver: local rival re-measure. 7 shards on GPUs 1-7 (GPU0 excluded).
#   bash drive_bs_rv.sh            # launch all shards
#   bash drive_bs_rv.sh <shard>    # one shard inline
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
GPUS=(1 2 3 4 5 6 7)
NS=${#GPUS[@]}
CELLS_PER_BATCH=44   # 11 BS x 4 arms
mkdir -p "$HERE/results_rv" "$HERE/nsys_reps_rv"
cd "$HERE"

run_shard() {
  local s=$1 g=${GPUS[$1]}
  local i=0
  while IFS= read -r spec; do
    if (( i % NS == s )); then
      local m=${spec%% *}; local rest=${spec#* }; local isl=${rest%% *}; local L=${rest##* }
      local jl="$HERE/results_rv/rv_${m}_${isl}_L${L}.jsonl"
      local n=0
      [[ -f "$jl" ]] && n=$(wc -l < "$jl")
      if (( n >= CELLS_PER_BATCH )); then
        echo "[shard$s] skip $spec (done, $n records)"
      else
        (( n > 0 )) && { echo "[shard$s] partial $spec ($n) -> rerun"; python3 -c "import os;os.remove('$jl')"; }
        echo "[shard$s] run  $spec on GPU$g"
        env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
          nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
          --capture-range-end=stop -f true \
          -o "$HERE/nsys_reps_rv/rv_${m}_${isl}_L${L}" \
          python3 "$HERE/bs_rv.py" --batch "$spec" \
          >> "$HERE/results_rv/shard${s}.log" 2>&1 \
          || echo "[shard$s] BATCH FAILED: $spec" >> "$HERE/results_rv/shard${s}.log"
      fi
    fi
    i=$((i+1))
  done < <(python3 "$HERE/bs_rv.py" --list 2>/dev/null | grep -E '^(flash|pro|v32) ')
  echo "[shard$s] all batches done" >> "$HERE/results_rv/shard${s}.log"
}

if [[ $# -ge 1 ]]; then
  run_shard "$1"
else
  for s in $(seq 0 $((NS-1))); do
    setsid bash "$0" "$s" > "$HERE/results_rv/driver${s}.log" 2>&1 &
    echo "shard $s pid $!"
  done
fi
