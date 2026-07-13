#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Footnote measurement: cost of the (untimed-in-sweep) sglang_v2 topk_plan
kernel, CUDA-event median over 200 reps, per BS. Production runs plan once
per step and reuses across ~61 indexer layers, so per-layer amortized cost
= plan/61."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))
from sglang_v2_op import plan  # noqa: E402

for BS in (1, 16, 64, 256, 512, 2048):
    sl = torch.full((BS,), 131072, dtype=torch.int32, device="cuda")
    md = torch.zeros((BS + 1, 2), dtype=torch.int32, device="cuda")
    for _ in range(20):
        plan(sl, md)
    torch.cuda.synchronize()
    ts = []
    for _ in range(200):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        plan(sl, md)
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    print(f"BS={BS}: plan median {ts[100]:.2f} us  (/61 layers = "
          f"{ts[100]/61:.3f} us/layer)")
