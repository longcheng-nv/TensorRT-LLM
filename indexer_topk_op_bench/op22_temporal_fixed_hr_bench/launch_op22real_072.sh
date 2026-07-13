#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# op22 REAL-capture fleet launcher — umbriel-b200-072, 9 batches on 8 GPUs.
# Each driver is setsid-detached (survives session drops; stop with
# pkill -f drive_nsys_op22real). Logs: realcap_gpu<N>.log
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
OUT=results_b200_op22real

launch() {  # $1 gpu, $2 models, $3 dtypes
  setsid env OUT="$OUT" GPU="$1" MODELS="$2" DTYPES="$3" \
    bash drive_nsys_op22real.sh > "realcap_gpu$1.log" 2>&1 &
  echo "gpu$1: models='$2' dtypes='$3' pid=$!"
}

launch 0 flash fp32
launch 1 pro   fp32
launch 2 v32   fp32
launch 3 flash bf16
launch 4 flash fp16
launch 5 pro   bf16
launch 6 pro   fp16
launch 7 v32   "bf16 fp16"
echo "fleet launched $(date -u +%FT%TZ)"
