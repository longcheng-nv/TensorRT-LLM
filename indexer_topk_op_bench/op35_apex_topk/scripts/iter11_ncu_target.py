# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Single-shot kernel launcher for NCU: --case BSxN --mode {1,2,3}."""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../src"))
from apex_op import apex_topk, pick_config, workspace  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--bs", type=int, default=1024)
ap.add_argument("--n", type=int, default=65536)
ap.add_argument("--k", type=int, default=512)
ap.add_argument("--mode", type=int, default=1)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)
torch.manual_seed(a.n + a.bs)
x = torch.rand(a.bs, a.n, device="cuda") + 1.0
cfg = pick_config(a.bs, a.n, a.k)
ws = workspace(a.bs, a.k, cfg, x.device)
for _ in range(a.reps):
    apex_topk(x, a.k, cfg=cfg, ws=ws, mode=a.mode)
    torch.cuda.synchronize()
    ws["counts"].zero_(); ws["tickets"].zero_()
print("done", cfg)
