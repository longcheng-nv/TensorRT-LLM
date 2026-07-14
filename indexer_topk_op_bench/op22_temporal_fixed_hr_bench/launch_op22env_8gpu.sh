#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op22 ENV — 8-GPU balanced launcher for the unified 9-arm envelope sweep.
# 18 batches = {best,worst} x {seqlen,bs,bs_hugeN} x {K512,K1024,K2048}.
# Each GPU runs a disjoint (scenario,sweep,K) chain via drive_nsys_op22env.sh;
# all write the SAME OUT root but distinct jsonl/.done/rep files -> no collision.
# setsid-detached (survives ssh drop) + resumable at batch granularity.
#
# Run ON the 8-GPU B200 node (NFS-shared checkout — files already present):
#   cd .../op22_temporal_fixed_hr_bench && ./launch_op22env_8gpu.sh
# Monitor:   tail -f envrun_gpu*.log ; ls results_b200_op22env/*/.done_* | wc -l
# Kill:      pkill -f sweep_op22env; pkill -f drive_nsys_op22env; pkill -f "nsys profile"
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
OUT="${OUT:-results_b200_op22env}"
export SYNTH_POSITIONAL=1

NG=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ') || NG=0
echo "### launch_op22env_8gpu: host=$(hostname) visible_gpus=$NG OUT=$OUT rev=$(git rev-parse --short HEAD 2>/dev/null)"
if [ "$NG" -lt 8 ]; then
  echo "!!! WARNING: only $NG GPUs visible; this launcher assumes 8. Edit the"
  echo "!!! shard table below (or run drive_nsys_op22env.sh manually) if <8."
fi

# 0) pre-generate all bundles once (deterministic, cheap; avoids 8 procs racing
#    the same skill import / file writes).
echo "=== pre-generating bundles_env (best+worst) ==="
python3 bundle_data_env.py best worst 2>&1 | tail -3

# gpu -> "SCEN|SWEEPS|KS"
run() {  # $1=gpu $2=scen $3=sweeps $4=ks $5=tag
  echo "  GPU$1: scen=$2 sweeps='$3' K='$4' -> envrun_gpu$1.log"
  setsid env OUT="$OUT" GPU="$1" SCENARIOS="$2" SWEEPS="$3" KS="$4" \
      ./drive_nsys_op22env.sh >"envrun_gpu$1.log" 2>&1 &
}

run 0 best  "bs seqlen"  "512"
run 1 best  "bs seqlen"  "1024"
run 2 best  "bs seqlen"  "2048"
run 3 worst "bs seqlen"  "512"
run 4 worst "bs seqlen"  "1024"
run 5 worst "bs seqlen"  "2048"
run 6 best  "bs_hugeN"   "512 1024 2048"
run 7 worst "bs_hugeN"   "512 1024 2048"

echo "### 8 chains launched (setsid). Total batches=18. Watch: tail -f envrun_gpu*.log"
echo "### progress: watch(1) 'ls $OUT/*/.done_* 2>/dev/null | wc -l'  (target 18)"
