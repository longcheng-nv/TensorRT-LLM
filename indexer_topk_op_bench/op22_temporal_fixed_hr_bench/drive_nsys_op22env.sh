#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op22 ENV nsys driver — one .nsys-rep per (scenario, sweep, K) fp32 batch,
# resumable at BATCH granularity via .done_* markers. Clone of
# drive_nsys_op28.sh pointed at sweep_op22env.py (unified 9-arm envelope).
#
# Usage:
#   OUT=results_b200_op22env GPU=0 SCENARIOS="best" SWEEPS="seqlen bs bs_hugeN" \
#       KS="512 1024 2048" ./drive_nsys_op22env.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_REL="${OUT:?set OUT=results_b200_op22env}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$HERE/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-50}"
SCENARIOS="${SCENARIOS:-best worst}"
SWEEPS="${SWEEPS:-seqlen bs bs_hugeN}"
KS="${KS:-512 1024 2048}"
export SYNTH_POSITIONAL="${SYNTH_POSITIONAL:-1}"
cd "$HERE"

echo "### drive_nsys_op22env: host=$(hostname) GPU=$GPU OUT=$OUT reps=$REPS warm=$REPS_WARM rev=$(git rev-parse --short HEAD 2>/dev/null)"
echo "### scenarios='$SCENARIOS' sweeps='$SWEEPS' Ks='$KS' SYNTH_POSITIONAL=$SYNTH_POSITIONAL start=$(date -u +%FT%TZ)"

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
      done_m="$SOUT/.done_${sw}_K${K}_fp32"
      jsonl="$SOUT/$sub/results_K${K}_fp32.jsonl"
      rep="$REPDIR/${sw}_K${K}_fp32"
      if [ -f "$done_m" ]; then echo "SKIP done: $scen $sw K=$K"; continue; fi
      rm -f "$jsonl" "$rep.nsys-rep" "$rep.sqlite"   # fresh whole-batch measure
      echo "=== nsys batch: $scen $sw K=$K fp32 -> $rep.nsys-rep  ($(date -u +%T)) ==="
      if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
           SYNTH_POSITIONAL="$SYNTH_POSITIONAL" \
           nsys profile -t cuda,nvtx \
           --capture-range=cudaProfilerApi --capture-range-end=stop \
           -o "$rep" -f true \
           python3 sweep_op22env.py --sweep "$sw" --scenario "$scen" \
             --K "$K" --dtype fp32 --out-root "$SOUT" \
             --reps "$REPS" --reps-warm "$REPS_WARM"; then
        touch "$done_m"
      else
        echo "!!! batch $scen $sw K=$K FAILED (leaving un-marked for resume)"
      fi
    done
  done
done
echo "ALL OP22ENV NSYS BATCHES DONE  end=$(date -u +%FT%TZ)"
