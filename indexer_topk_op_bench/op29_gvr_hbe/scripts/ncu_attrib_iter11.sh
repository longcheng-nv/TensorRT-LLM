#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# iter11 L3: attribute the K2048 HBE +188us at 262144xBS1024 (scenario real).
# 2x2 grid {gvr29_hbe, sglang_v2} x {K512, K2048} + hbe K1024 reference.
# Attribution only — never quote these as performance baselines.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
OUT=results/iter11
mkdir -p "$OUT"

METRICS=gpu__time_duration.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__maximum_warps_per_active_cycle_pct,\
launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,\
launch__occupancy_limit_warps,launch__occupancy_limit_blocks,\
launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,\
launch__registers_per_thread,launch__grid_size,\
sm__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed

for spec in gvr29_hbe:512 gvr29_hbe:1024 gvr29_hbe:2048 sglang_v2:512 sglang_v2:2048; do
  op=${spec%%:*}; K=${spec##*:}
  tag="${op}_K${K}"
  echo "=== $tag ==="
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    ncu --kernel-name 'regex:topk_main_kernel|gvr29_hbe_kernel' -s 2 -c 3 \
    --metrics "$METRICS" --csv \
    python3 scripts/ncu_cell_op29.py --op "$op" --K "$K" --N 262144 \
    --BS 1024 --scenario real --launches 6 \
    > "$OUT/ncu_${tag}.csv" 2> "$OUT/ncu_${tag}.err" || {
      echo "FAILED $tag (see $OUT/ncu_${tag}.err)"; continue; }
  echo "ok -> $OUT/ncu_${tag}.csv"
done
echo "ALL DONE"
