#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# nsys cold-L2 A/B: op15 gvr_smem vs baseline gvr_cutedsl_rs, small-N seqlen grid.
# Usage: OUT=results/ab GPU=1 ./run_ab.sh   (paths relative to op15 bucket)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:-results/ab}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$BUCKET/$OUT_REL";; esac
REPDIR="$OUT/nsys_reps"
GPU="${GPU:-1}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"
KS="${KS:-512 1024 2048}"; DTYPES="${DTYPES:-fp32 bf16 fp16}"
mkdir -p "$REPDIR"; cd "$HERE"
echo "### op15 AB nsys: host=$(hostname) GPU=$GPU OUT=$OUT reps=$REPS warm=$REPS_WARM start=$(date -u +%FT%TZ)"
for K in $KS; do
  for dt in $DTYPES; do
    done="$OUT/.done_seqlen_K${K}_${dt}"
    rep="$REPDIR/seqlen_K${K}_${dt}"
    jsonl="$OUT/seqlen_sweep/results_K${K}_${dt}.jsonl"
    if [ -f "$done" ]; then echo "SKIP done: K=$K dt=$dt"; continue; fi
    rm -f "$jsonl" "$rep.nsys-rep" "$rep.sqlite"
    echo "=== nsys batch K=$K dt=$dt -> $rep.nsys-rep ($(date -u +%T)) ==="
    if CUDA_VISIBLE_DEVICES="$GPU" nsys profile -t cuda,nvtx \
         --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python ab_nsys.py --K "$K" --dtype "$dt" --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM"; then
      touch "$done"
    else
      echo "!!! batch K=$K dt=$dt FAILED"
    fi
  done
done
echo "ALL AB BATCHES DONE end=$(date -u +%FT%TZ)"
