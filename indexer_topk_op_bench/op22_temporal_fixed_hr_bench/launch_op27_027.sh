#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op22rr op27_hls arm backfill — 8-GPU shard launcher (umbriel-b200-027).
# Clone of launch_radix027.sh: 81 batches = 3 scenarios x 3 dtypes x 3 K x
# 3 sweeps; arms = gvr_cutedsl anchor + op27_hls (gvr_ms_auto @ op27 HEAD,
# K2048 tail ladder default-ON). Idempotent via .done_* markers; chains run
# under setsid (survive session death, killable by PGID).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
OUT=results_b200_op22rr_op27027
ARMS="gvr_cutedsl,op27_hls"

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

setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 0 op27_027_gpu0.log 'real:fp32:512 1024 2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 1 op27_027_gpu1.log 'real:bf16:512 1024 2048' 'worst:fp16:512'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 2 op27_027_gpu2.log 'real:fp16:512 1024 2048' 'worst:fp16:1024'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 3 op27_027_gpu3.log 'best:fp32:512 1024 2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 4 op27_027_gpu4.log 'best:bf16:512 1024 2048' 'worst:fp16:2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 5 op27_027_gpu5.log 'best:fp16:512 1024 2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 6 op27_027_gpu6.log 'worst:fp32:512 1024 2048'" &
setsid bash -c "$(declare -f chain); cd '$HERE'; OUT=$OUT; ARMS=$ARMS; chain 7 op27_027_gpu7.log 'worst:bf16:512 1024 2048'" &
echo "launched 8 chains; markers in ../$OUT/*/.done_*"
