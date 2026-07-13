#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op26 backlog-2 edge-aim R1 nsys driver — one .nsys-rep per scenario,
# resumable via .done markers. Protocol clone of drive_nsys_qfracs.sh.
#
# Usage: OUT=results_b200_op26_r1aim_ab GPU=5 SCENARIOS="real" ./drive_nsys_r1aim.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:?set OUT=results_b200_op26_r1aim_ab}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$BENCH/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-50}"
SCENARIOS="${SCENARIOS:-real worst}"
K="${K:-2048}"
DT="${DT:-fp32}"
cd "$HERE"

echo "### drive_nsys_r1aim: host=$(hostname) GPU=$GPU OUT=$OUT rev=$(git rev-parse --short HEAD 2>/dev/null)"
for scen in $SCENARIOS; do
  SOUT="$OUT/$scen"
  REPDIR="$SOUT/nsys_reps"
  mkdir -p "$REPDIR"
  done_m="$SOUT/.done_K${K}_${DT}"
  jsonl="$SOUT/results_K${K}_${DT}.jsonl"
  rep="$REPDIR/r1aim_K${K}_${DT}"
  if [ -f "$done_m" ]; then echo "SKIP done: $scen K=$K dt=$DT"; continue; fi
  rm -f "$jsonl" "$rep.nsys-rep" "$rep.sqlite"
  echo "=== nsys batch: $scen K=$K dt=$DT -> $rep.nsys-rep  ($(date -u +%T)) ==="
  if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
       nsys profile -t cuda,nvtx \
       --capture-range=cudaProfilerApi --capture-range-end=stop \
       -o "$rep" -f true \
       python3 sweep_r1aim.py --scenario "$scen" \
         --K "$K" --dtype "$DT" --out-root "$OUT" \
         --reps "$REPS" --reps-warm "$REPS_WARM"; then
    touch "$done_m"
  else
    echo "!!! batch $scen K=$K dt=$DT FAILED (leaving un-marked for resume)"
  fi
done
echo "ALL R1AIM NSYS BATCHES DONE  end=$(date -u +%FT%TZ)"
