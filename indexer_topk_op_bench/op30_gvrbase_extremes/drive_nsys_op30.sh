#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op30 nsys driver — DYNAMIC claim-queue over all (scenario, sweep, K, dtype)
# batches: each driver instance (one per GPU) atomically claims the next
# unclaimed batch via mkdir, so 8 drivers load-balance with no static shard
# collisions (op26 gotcha). One .nsys-rep per batch; .done_* markers give
# batch-granular resume (failed batches release their claim for retry).
#
# Usage (one per GPU):
#   GPU=0 ./drive_nsys_op30.sh
# Optional env: OUT (default ../results_b200_op30), REPS/REPS_WARM,
#   SCENARIOS/SWEEPS/KS/DTYPES to restrict the queue, OP30_ARMS/OP30_NS/
#   OP30_BS forwarded to sweep_op30.py.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
OUT_REL="${OUT:-results_b200_op30}"
case "$OUT_REL" in /*) OUT="$OUT_REL";; *) OUT="$BENCH/$OUT_REL";; esac
GPU="${GPU:-0}"
REPS="${REPS:-20}"
REPS_WARM="${REPS_WARM:-50}"
# heavy-first order: bs (84-cell) fp32 batches are the long pole
SCENARIOS="${SCENARIOS:-best worst}"
SWEEPS="${SWEEPS:-bs bs_hugeN seqlen}"
KS="${KS:-512 1024 2048}"
DTYPES="${DTYPES:-fp32 bf16 fp16}"
cd "$HERE"
mkdir -p "$OUT/claims"

echo "### drive_nsys_op30: host=$(hostname) GPU=$GPU OUT=$OUT reps=$REPS warm=$REPS_WARM rev=$(git rev-parse --short HEAD 2>/dev/null)"
echo "### scenarios='$SCENARIOS' sweeps='$SWEEPS' Ks='$KS' dtypes='$DTYPES' start=$(date -u +%FT%TZ)"

for sw in $SWEEPS; do
  case "$sw" in
    seqlen) sub=seqlen_sweep;; bs) sub=bs_scaling;; bs_hugeN) sub=bs_hugeN;;
    *) echo "unknown sweep $sw"; exit 2;;
  esac
  for dt in $DTYPES; do
    for scen in $SCENARIOS; do
      for K in $KS; do
        tag="${scen}_${sw}_K${K}_${dt}"
        SOUT="$OUT/$scen"
        done_m="$SOUT/.done_${sw}_K${K}_${dt}"
        [ -f "$done_m" ] && { echo "SKIP done: $tag"; continue; }
        # atomic claim — another GPU may own it
        if ! mkdir "$OUT/claims/$tag" 2>/dev/null; then
          echo "SKIP claimed: $tag"; continue
        fi
        echo "$(hostname) GPU=$GPU pid=$$ $(date -u +%FT%TZ)" \
          > "$OUT/claims/$tag/owner"
        REPDIR="$SOUT/nsys_reps"
        mkdir -p "$REPDIR"
        jsonl="$SOUT/$sub/results_K${K}_${dt}.jsonl"
        rep="$REPDIR/${sw}_K${K}_${dt}"
        rm -f "$jsonl" "$rep.nsys-rep" "$rep.sqlite"   # whole-batch measure
        echo "=== nsys batch: $tag GPU=$GPU -> $rep.nsys-rep ($(date -u +%T)) ==="
        if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES="$GPU" \
             nsys profile -t cuda,nvtx \
             --capture-range=cudaProfilerApi --capture-range-end=stop \
             -o "$rep" -f true \
             python3 sweep_op30.py --sweep "$sw" --scenario "$scen" \
               --K "$K" --dtype "$dt" --out-root "$SOUT" \
               --reps "$REPS" --reps-warm "$REPS_WARM"; then
          touch "$done_m"
        else
          echo "!!! batch $tag FAILED (claim released for retry)"
          rm -rf "$OUT/claims/$tag"
        fi
      done
    done
  done
done
echo "GPU=$GPU QUEUE DRAINED  end=$(date -u +%FT%TZ)"
