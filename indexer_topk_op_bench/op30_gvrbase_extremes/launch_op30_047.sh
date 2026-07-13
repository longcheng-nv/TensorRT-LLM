#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# op30 full-campaign launcher on umbriel-b200-047 — 8 dynamic-queue drivers,
# one per GPU, each under setsid (survives session death; stop with
# pkill -f sweep_op30 + pkill -f drive_nsys_op30, then re-check respawn).
# Resume-safe: re-running only picks up unclaimed/undone batches.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
for g in $GPUS; do
  setsid bash -c "GPU=$g ./drive_nsys_op30.sh > op30_gpu$g.log 2>&1" &
  echo "launched driver GPU=$g"
done
echo "all drivers launched $(date -u +%FT%TZ)"
