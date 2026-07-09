#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op22rr radix-CUDA arm backfill — 8-GPU shard launcher (umbriel-b200-027).
# 81 batches = 3 scenarios x 3 dtypes x 3 K x 3 sweeps; arms =
# gvr_cutedsl anchor + radix_single_cuda + radix_multi_cuda.
# Load-balance: fp32 chains (slowest) get 9 batches each; the worst/fp16
# chain is K-sharded onto the three lighter bf16/fp16 GPUs (12 each).
# Idempotent via the driver's .done_* markers; each chain runs under
# setsid so it survives session death and is killable by PGID.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
OUT=results_b200_op22rr_radix027
ARMS="gvr_cutedsl,radix_single_cuda,radix_multi_cuda"

chain() {  # chain <gpu> <log> <spec>... ; spec = SCEN:DTYPES:KS
  local gpu="$1" log="$2"; shift 2
  (
    for spec in "$@"; do
      IFS=: read -r scen dts ks <<< "$spec"
      env OUT="$OUT" GPU="$gpu" OP22RR_ARMS="$ARMS" \
          SCENARIOS="$scen" DTYPES="$dts" KS="$ks" \
          ./drive_nsys_op22rr.sh
    done
    echo "CHAIN GPU$gpu ALL DONE $(date -u +%FT%TZ)"
  ) >> "$log" 2>&1
}

setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 0 radix027_gpu0.log 'real:fp32:512 1024 2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 1 radix027_gpu1.log 'real:bf16:512 1024 2048' 'worst:fp16:512'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 2 radix027_gpu2.log 'real:fp16:512 1024 2048' 'worst:fp16:1024'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 3 radix027_gpu3.log 'best:fp32:512 1024 2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 4 radix027_gpu4.log 'best:bf16:512 1024 2048' 'worst:fp16:2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 5 radix027_gpu5.log 'best:fp16:512 1024 2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 6 radix027_gpu6.log 'worst:fp32:512 1024 2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 7 radix027_gpu7.log 'worst:bf16:512 1024 2048'" &
echo "launched 8 chains; markers in ../$OUT/*/.done_*"
