#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op22 nsys driver — one .nsys-rep per (scenario, sweep, K, dtype) batch,
# resumable at BATCH granularity via .done_* markers (a batch must be measured
# in ONE nsys run; on resume any partial batch is deleted and redone fresh).
# Clone of harness/drive_nsys_full.sh; scenario-serial per PLAN section 4 W5.
#
# Usage:
#   OUT=results_b200_op22 GPU=0 [SCENARIOS="real best worst"] \
#     [SWEEPS="seqlen bs"] [KS="512 1024 2048"] [DTYPES="fp32 bf16 fp16"] \
#     [REPS=20] [REPS_WARM=50] ./drive_nsys_op22.sh
#
# nsys gotchas: env -u GITHUB_TOKEN -u HF_TOKEN (sqlite embeds process env);
# `nsys -c cudaProfilerApi` exits 143 on SUCCESS -> no `set -e`.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:?set OUT=results_b200_op22 (or results_b300_op22)}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$BENCH/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-50}"
SCENARIOS="${SCENARIOS:-real best worst}"
SWEEPS="${SWEEPS:-seqlen bs}"
KS="${KS:-512 1024 2048}"
DTYPES="${DTYPES:-fp32 bf16 fp16}"
cd "$HERE"

echo "### drive_nsys_op22: host=$(hostname) GPU=$GPU OUT=$OUT reps=$REPS warm=$REPS_WARM"
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
             python3 sweep_op22.py --sweep "$sw" --scenario "$scen" \
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
echo "ALL OP22 NSYS BATCHES DONE  end=$(date -u +%FT%TZ)"
