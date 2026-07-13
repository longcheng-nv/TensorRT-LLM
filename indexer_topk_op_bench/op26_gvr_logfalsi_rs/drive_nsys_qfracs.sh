#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op26 backlog-1 qfracs A/B nsys driver — one .nsys-rep per (scenario,
# K, dtype) batch, resumable via .done markers. Protocol clone of
# drive_nsys_op22rr.sh pointed at sweep_qfracs.py.
#
# Usage: OUT=results_b200_op26_qfracs_ab GPU=4 ./drive_nsys_qfracs.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:?set OUT=results_b200_op26_qfracs_ab}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$BENCH/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-50}"
SCENARIOS="${SCENARIOS:-real worst}"
BATCHES="${BATCHES:-1024:fp32 2048:fp16}"
cd "$HERE"

echo "### drive_nsys_qfracs: host=$(hostname) GPU=$GPU OUT=$OUT reps=$REPS warm=$REPS_WARM rev=$(git rev-parse --short HEAD 2>/dev/null)"
for scen in $SCENARIOS; do
  SOUT="$OUT/$scen"
  REPDIR="$SOUT/nsys_reps"
  mkdir -p "$REPDIR"
  for kd in $BATCHES; do
    K="${kd%%:*}"; dt="${kd##*:}"
    done_m="$SOUT/.done_K${K}_${dt}"
    jsonl="$SOUT/results_K${K}_${dt}.jsonl"
    rep="$REPDIR/qfracs_K${K}_${dt}"
    if [ -f "$done_m" ]; then echo "SKIP done: $scen K=$K dt=$dt"; continue; fi
    rm -f "$jsonl" "$rep.nsys-rep" "$rep.sqlite"
    echo "=== nsys batch: $scen K=$K dt=$dt -> $rep.nsys-rep  ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
         nsys profile -t cuda,nvtx \
         --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 sweep_qfracs.py --scenario "$scen" \
           --K "$K" --dtype "$dt" --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM"; then
      touch "$done_m"
    else
      echo "!!! batch $scen K=$K dt=$dt FAILED (leaving un-marked for resume)"
    fi
  done
done
echo "ALL QFRACS NSYS BATCHES DONE  end=$(date -u +%FT%TZ)"
