#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op26 backlog-3 kC-diet nsys driver — one .nsys-rep per scenario,
# resumable via .done markers. Protocol clone of drive_nsys_qfracs.sh.
#
# Usage: OUT=results_b200_op26_kcdiet_ab GPU=4 SCENARIOS="real" ./drive_nsys_kcdiet.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:?set OUT=results_b200_op26_kcdiet_ab}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$BENCH/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-50}"
SCENARIOS="${SCENARIOS:-real worst}"
cd "$HERE"

echo "### drive_nsys_kcdiet: host=$(hostname) GPU=$GPU OUT=$OUT rev=$(git rev-parse --short HEAD 2>/dev/null)"
for scen in $SCENARIOS; do
  SOUT="$OUT/$scen"
  REPDIR="$SOUT/nsys_reps"
  mkdir -p "$REPDIR"
  done_m="$SOUT/.done_K512"
  jsonl="$SOUT/results_K512.jsonl"
  rep="$REPDIR/kcdiet_K512"
  if [ -f "$done_m" ]; then echo "SKIP done: $scen"; continue; fi
  rm -f "$jsonl" "$rep.nsys-rep" "$rep.sqlite"
  echo "=== nsys batch: $scen kcdiet -> $rep.nsys-rep  ($(date -u +%T)) ==="
  if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
       nsys profile -t cuda,nvtx \
       --capture-range=cudaProfilerApi --capture-range-end=stop \
       -o "$rep" -f true \
       python3 sweep_kcdiet.py --scenario "$scen" --out-root "$OUT" \
         --reps "$REPS" --reps-warm "$REPS_WARM"; then
    touch "$done_m"
  else
    echo "!!! batch $scen kcdiet FAILED (leaving un-marked for resume)"
  fi
done
echo "ALL KCDIET NSYS BATCHES DONE  end=$(date -u +%FT%TZ)"
