#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# BS-scaling fresh sweep at the CURRENT PR#16457 head (@e6fdbfac3d):
# REAL decode-capture BS grids (flash/pro/v32 x fp32/fp16/bf16 = 9 batches,
# 11 BS x all ISL rungs each), 3 arms same-run, nsys cold-L2.
# Usage: drive_canonical_bs.sh <GPU>   — batch i runs on GPU i%8
set -u
GPU=$1
WD=/tmp/gvrcanon_bs_e6fdbfac
RH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/headfull_harness
OUT="$WD/refresh_results"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
C450=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages
export PYTHONPATH="$C450:/tmp/gvrlayers/cutlass450${PYTHONPATH:+:$PYTHONPATH}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"

BATCHES=(
  "real bs flash fp32"
  "real bs pro fp32"
  "real bs v32 fp32"
  "real bs flash fp16"
  "real bs pro fp16"
  "real bs v32 fp16"
  "real bs flash bf16"
  "real bs pro bf16"
  "real bs v32 bf16"
)
i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % 8)) -eq "$GPU" ]; then
    read -r fam sw m dt <<< "$batch"
    tag="real_${sw}_${m}_${dt}"
    args="--family real --sweep $sw --model $m --dtype $dt"
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== bs batch [GPU$GPU]: $tag ($(date -u +%T)) ==="
    if env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
         nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o "$rep" -f true \
         python3 "$RH/sweep_refresh.py" $args --out-root "$OUT" \
           --reps "$REPS" --reps-warm "$REPS_WARM" > "$OUT/${tag}.log" 2>&1; then
      touch "$done_m"
    else
      echo "!!! batch $tag FAILED (see $OUT/${tag}.log; left un-marked for resume)"
    fi
  fi
  i=$((i+1))
done
echo "BSWORKER${GPU}DONE"
