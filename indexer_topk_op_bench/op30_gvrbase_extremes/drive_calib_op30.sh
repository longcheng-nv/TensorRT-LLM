#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op30 calibration nsys driver — one .nsys-rep per model shard.
# Usage: GPU=0 MODEL=v4flash ./drive_calib_op30.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
OUT="${OUT:-$BENCH/results_b200_op30_calib}"
GPU="${GPU:-0}"
MODEL="${MODEL:?set MODEL=v4flash|v4pro|v32}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-20}"
cd "$HERE"
mkdir -p "$OUT/nsys_reps"

done_m="$OUT/.done_calib_${MODEL}"
rep="$OUT/nsys_reps/calib_${MODEL}"
if [ -f "$done_m" ]; then echo "SKIP done: calib $MODEL"; exit 0; fi
rm -f "$OUT/calib_${MODEL}.jsonl" "$rep.nsys-rep" "$rep.sqlite"
echo "=== op30 calib batch: $MODEL GPU=$GPU -> $rep.nsys-rep ($(date -u +%T)) ==="
if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
     nsys profile -t cuda,nvtx \
     --capture-range=cudaProfilerApi --capture-range-end=stop \
     -o "$rep" -f true \
     python3 calib_op30.py --model "$MODEL" --out-root "$OUT" \
       --reps "$REPS" --reps-warm "$REPS_WARM"; then
  touch "$done_m"
  echo "calib $MODEL DONE"
else
  echo "!!! calib $MODEL FAILED (left un-marked)"
fi
