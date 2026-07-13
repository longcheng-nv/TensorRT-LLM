#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op22 REAL-capture chapter nsys driver — one .nsys-rep per (model, dtype)
# batch, resumable at BATCH granularity via .done_* markers. Clone of
# drive_nsys_op22rr.sh pointed at sweep_op22_real.py (13 report arms,
# real last-decode-step bundles). Split across GPUs with MODELS/DTYPES.
#
# Usage:
#   OUT=results_b200_op22real GPU=0 MODELS=flash DTYPES=fp32 ./drive_nsys_op22real.sh
#   OUT=results_b200_op22real GPU=7 MODELS=v32 DTYPES="bf16 fp16" ./drive_nsys_op22real.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:?set OUT=results_b200_op22real}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$BENCH/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-50}"
MODELS="${MODELS:-flash pro v32}"
DTYPES="${DTYPES:-fp32 bf16 fp16}"
cd "$HERE"

echo "### drive_nsys_op22real: host=$(hostname) GPU=$GPU OUT=$OUT reps=$REPS warm=$REPS_WARM rev=$(git rev-parse --short HEAD 2>/dev/null)"
echo "### models='$MODELS' dtypes='$DTYPES' start=$(date -u +%FT%TZ)"

REPDIR="$OUT/nsys_reps"
mkdir -p "$REPDIR" "$OUT/realcap_sweep"
for model in $MODELS; do
  for dt in $DTYPES; do
    done_m="$OUT/.done_realcap_${model}_${dt}"
    jsonl="$OUT/realcap_sweep/results_${model}_${dt}.jsonl"
    rep="$REPDIR/realcap_${model}_${dt}"
    if [ -f "$done_m" ]; then echo "SKIP done: $model $dt"; continue; fi
    rm -f "$jsonl" "$rep.nsys-rep" "$rep.sqlite"   # fresh, whole-batch measure
    echo "=== nsys batch: $model $dt -> $rep.nsys-rep  ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
         nsys profile -t cuda,nvtx \
         --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 sweep_op22_real.py --model "$model" --dtype "$dt" \
           --out-root "$OUT" --reps "$REPS" --reps-warm "$REPS_WARM" \
           ${BS:+--bs "$BS"} ${LAYERS:+--layers "$LAYERS"}; then
      touch "$done_m"
    else
      echo "!!! batch $model $dt FAILED (leaving un-marked for resume)"
    fi
  done
done
echo "ALL OP22REAL NSYS BATCHES DONE  end=$(date -u +%FT%TZ)"
