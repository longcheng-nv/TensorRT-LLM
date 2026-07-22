#!/bin/bash
# compB BS>1 extension minimal-validation driver: 6 cells, one per GPU 1-6
# (umbriel-b200-039, all GPUs healthy/idle). One nsys rep per cell; jsonl is
# the resume marker (complete cell = 24 records for A, 15 for B).
#   bash drive_bs_ext.sh            # launch all shards (setsid, background)
#   bash drive_bs_ext.sh <shard>    # one shard inline
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450
GPUS=(1 2 3 4 5 6)
cd "$HERE"

expected() { case "$1" in *512k*) echo 15 ;; *) echo 24 ;; esac; }

run_shard() {
  local s=$1 g=${GPUS[$1]}
  local i=0
  while IFS= read -r spec; do
    if (( i == s )); then
      local m=${spec%% *}; local rest=${spec#* }; local isl=${rest%% *}; local L=${rest##* }
      local tag="${m}_${isl}_L${L}"
      local jl="$HERE/results/ext_${tag}.jsonl"
      local want; want=$(expected "$tag")
      local n=0
      [[ -f "$jl" ]] && n=$(wc -l < "$jl")
      if (( n >= want )); then
        echo "[shard$s] skip $spec (done, $n records)"
      else
        (( n > 0 )) && { echo "[shard$s] partial $spec ($n) -> rerun"; rm -f "$jl"; }
        echo "[shard$s] run  $spec on GPU$g"
        env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$g \
          nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
          --capture-range-end=stop -f true \
          -o "$HERE/nsys_reps/ext_${tag}" \
          python3 "$HERE/bs_ext.py" --batch "$spec" \
          >> "$HERE/results/shard${s}.log" 2>&1 \
          || echo "[shard$s] CELL FAILED: $spec" >> "$HERE/results/shard${s}.log"
      fi
    fi
    i=$((i+1))
  done < <(python3 "$HERE/bs_ext.py" --list | grep -E '^(flash|pro|v32) ')
  echo "[shard$s] done" >> "$HERE/results/shard${s}.log"
}

if [[ $# -ge 1 ]]; then
  run_shard "$1"
else
  for s in 0 1 2 3 4 5; do
    setsid bash "$0" "$s" > "$HERE/results/driver${s}.log" 2>&1 &
    echo "shard $s pid $!"
  done
fi
