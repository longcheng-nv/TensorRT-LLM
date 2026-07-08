#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op22rr nsys driver — one .nsys-rep per (scenario, sweep, K, dtype) batch,
# resumable at BATCH granularity via .done_* markers. Clone of
# drive_nsys_op22.sh pointed at sweep_op22rr.py (5 arms, radix-relative
# scenario bundles). Split across GPUs with DTYPES/SCENARIOS/KS env vars.
#
# Usage:
#   OUT=results_b200_op22rr GPU=0 DTYPES="fp32" ./drive_nsys_op22rr.sh
#   OUT=results_b200_op22rr GPU=1 DTYPES="bf16 fp16" ./drive_nsys_op22rr.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:?set OUT=results_b200_op22rr}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$BENCH/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-50}"
SCENARIOS="${SCENARIOS:-real best worst}"
SWEEPS="${SWEEPS:-seqlen bs bs_hugeN}"
KS="${KS:-512 1024 2048}"
DTYPES="${DTYPES:-fp32 bf16 fp16}"
cd "$HERE"

echo "### drive_nsys_op22rr: host=$(hostname) GPU=$GPU OUT=$OUT reps=$REPS warm=$REPS_WARM rev=$(git rev-parse --short HEAD 2>/dev/null)"
echo "### scenarios='$SCENARIOS' sweeps='$SWEEPS' Ks='$KS' dtypes='$DTYPES' start=$(date -u +%FT%TZ)"

for scen in $SCENARIOS; do
  SOUT="$OUT/$scen"
  REPDIR="$SOUT/nsys_reps"
  mkdir -p "$REPDIR"
  for sw in $SWEEPS; do
    case "$sw" in
      seqlen) sub=seqlen_sweep;; bs) sub=bs_scaling;; bs_hugeN) sub=bs_hugeN;;
      *) echo "unknown sweep $sw"; exit 2;;
    esac
    for K in $KS; do
      for dt in $DTYPES; do
        done_m="$SOUT/.done_${sw}_K${K}_${dt}"
        jsonl="$SOUT/$sub/results_K${K}_${dt}.jsonl"
        rep="$REPDIR/${sw}_K${K}_${dt}"
        if [ -f "$done_m" ]; then echo "SKIP done: $scen $sw K=$K dt=$dt"; continue; fi
        rm -f "$jsonl" "$rep.nsys-rep" "$rep.sqlite"   # fresh, whole-batch measure
        echo "=== nsys batch: $scen $sw K=$K dt=$dt -> $rep.nsys-rep  ($(date -u +%T)) ==="
        if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
             nsys profile -t cuda,nvtx \
             --capture-range=cudaProfilerApi --capture-range-end=stop \
             -o "$rep" -f true \
             python3 sweep_op22rr.py --sweep "$sw" --scenario "$scen" \
               --K "$K" --dtype "$dt" --out-root "$SOUT" \
               --reps "$REPS" --reps-warm "$REPS_WARM"; then
          touch "$done_m"
        else
          echo "!!! batch $scen $sw K=$K dt=$dt FAILED (leaving un-marked for resume)"
        fi
      done
    done
  done
done
echo "ALL OP22RR NSYS BATCHES DONE  end=$(date -u +%FT%TZ)"
