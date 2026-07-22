#!/bin/bash
# compB BS-scaling driver: 7 shards on GPUs 1-7 (GPU0 excluded: broken cooling
# on umbriel-b200-019, 70C+ idle). One nsys rep per (model, isl, L) batch.
# Batch-level resume: a COMPLETE jsonl (22 records) skips; a partial one is
# deleted and the whole batch re-runs (the nsys rep is per-batch, -f true).
#   bash drive_bs_kf.sh            # launch all shards (setsid, background)
#   bash drive_bs_kf.sh <shard>    # one shard inline
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
GPUS=(1 2 3 4 5 6 7)
NS=${#GPUS[@]}
CELLS_PER_BATCH=22   # 11 BS x 2 arms
cd "$HERE"

run_shard() {
  local s=$1 g=${GPUS[$1]}
  local i=0
  while IFS= read -r spec; do
    if (( i % NS == s )); then
      local tag=${spec// /_}; tag=${tag/_L/_L}
      local m=${spec%% *}; local rest=${spec#* }; local isl=${rest%% *}; local L=${rest##* }
      local jl="$HERE/results/bs_${m}_${isl}_L${L}.jsonl"
      local n=0
      [[ -f "$jl" ]] && n=$(wc -l < "$jl")
      if (( n >= CELLS_PER_BATCH )); then
        echo "[shard$s] skip $spec (done, $n records)"
      else
        (( n > 0 )) && { echo "[shard$s] partial $spec ($n) -> rerun"; rm -f "$jl"; }
        echo "[shard$s] run  $spec on GPU$g"
        env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
          nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
          --capture-range-end=stop -f true \
          -o "$HERE/nsys_reps/bs_${m}_${isl}_L${L}" \
          python3 "$HERE/bs_kf.py" --batch "$spec" \
          >> "$HERE/results/shard${s}.log" 2>&1 \
          || echo "[shard$s] BATCH FAILED: $spec" >> "$HERE/results/shard${s}.log"
      fi
    fi
    i=$((i+1))
  done < <(python3 "$HERE/bs_kf.py" --list | grep -E '^(flash|pro|v32) ')
  echo "[shard$s] all batches done" >> "$HERE/results/shard${s}.log"
}

if [[ $# -ge 1 ]]; then
  run_shard "$1"
else
  for s in $(seq 0 $((NS-1))); do
    setsid bash "$0" "$s" > "$HERE/results/driver${s}.log" 2>&1 &
    echo "shard $s pid $!"
  done
fi
