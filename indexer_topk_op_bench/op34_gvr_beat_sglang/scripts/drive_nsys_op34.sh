#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op34 nsys driver — one .nsys-rep per (model, ISL) batch on a chosen arm set,
# resumable per batch via .done markers and per-(arm,layer) jsonl append.
# cold-L2 canonical, cudaProfilerApi window (identical protocol to op22 v4cap).
#
# Usage:
#   OUT=results ARMS="sglang_v2,op26_r0auto,op26_r0@kc1536" GPU=0 \
#     MODELS=flash ISLS="32k 256k" LAYERS="2,20,42" ./drive_nsys_op34.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMP="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:-results}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$CAMP/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"
ARMS="${ARMS:?set ARMS=comma,list}"
MODELS="${MODELS:-flash pro}"
ISLS="${ISLS:-4k 8k 16k 32k 64k 128k 256k 512k 1024k}"
TAG="${TAG:-probe}"
cd "$HERE"
REPDIR="$OUT/nsys_reps"; mkdir -p "$REPDIR" "$OUT/$TAG"
echo "### op34 nsys host=$(hostname) GPU=$GPU OUT=$OUT tag=$TAG arms=$ARMS rev=$(git rev-parse --short HEAD 2>/dev/null) start=$(date -u +%FT%TZ)"
for model in $MODELS; do
  for isl in $ISLS; do
    done_m="$OUT/.done_${TAG}_${model}_${isl}"
    rep="$REPDIR/${TAG}_${model}_${isl}"
    if [ -f "$done_m" ]; then echo "SKIP done: $model $isl"; continue; fi
    rm -f "$rep.nsys-rep" "$rep.sqlite"
    echo "=== nsys batch: $model $isl -> $rep.nsys-rep ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
         nsys profile -t cuda,nvtx \
         --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 nsys_op34.py --model "$model" --isl "$isl" --arms "$ARMS" \
           --out-root "$OUT/$TAG" --reps "$REPS" --reps-warm "$REPS_WARM" \
           ${LAYERS:+--layers "$LAYERS"}; then
      touch "$done_m"
    else
      echo "!!! batch $model $isl FAILED (un-marked for resume)"
    fi
  done
done
echo "ALL OP34 NSYS BATCHES DONE end=$(date -u +%FT%TZ)"
