#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Canonical-grid fresh sweep at the CURRENT PR#16457 head (@e6fdbfac3d):
# the 9 fp32 seqlen batches behind REPORT §9b "Canonical shipped-head tables"
# (synth best/worst x K512/1024/2048 + real flash/pro/v32), 3 arms same-run
# (gvr_base / gvr_pr=head / op26_r0auto anchor), nsys cold-L2.
# Usage: drive_canonical.sh <GPU>   — worker for one GPU; batch i runs on GPU i%8
set -u
GPU=$1
WD=/tmp/gvrcanon_e6fdbfac
RH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/headfull_harness
OUT="$WD/refresh_results"
mkdir -p "$OUT/nsys_reps"
export PYTHONNOUSERSITE=1
C450=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages
export PYTHONPATH="$C450:/tmp/gvrlayers/cutlass450${PYTHONPATH:+:$PYTHONPATH}"
REPS="${REPS:-20}"; REPS_WARM="${REPS_WARM:-50}"

BATCHES=(
  "synth seqlen best 512 fp32"
  "synth seqlen worst 512 fp32"
  "synth seqlen best 1024 fp32"
  "synth seqlen worst 1024 fp32"
  "synth seqlen best 2048 fp32"
  "synth seqlen worst 2048 fp32"
  "real seqlen flash fp32"
  "real seqlen pro fp32"
  "real seqlen v32 fp32"
)
i=0
for batch in "${BATCHES[@]}"; do
  if [ $((i % 8)) -eq "$GPU" ]; then
    read -r fam sw a b c <<< "$batch"
    if [ "$fam" = "synth" ]; then
      tag="synth_${sw}_${a}_K${b}_${c}"
      args="--family synth --sweep $sw --scenario $a --K $b --dtype $c"
    else
      tag="real_${sw}_${a}_${b}"
      args="--family real --sweep $sw --model $a --dtype $b"
    fi
    done_m="$OUT/.done_${tag}"
    rep="$OUT/nsys_reps/${tag}"
    if [ -f "$done_m" ]; then echo "SKIP done: $tag"; i=$((i+1)); continue; fi
    echo "=== canonical batch [GPU$GPU]: $tag ($(date -u +%T)) ==="
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
echo "CANONWORKER${GPU}DONE"
