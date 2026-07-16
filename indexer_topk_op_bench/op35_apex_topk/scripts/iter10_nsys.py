# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter10 nsys: APEX v0 E2E span vs same-node pure-read anchor, canonical cells.
Order per cell: [pure-read k_stream_reduce] x reps, [apex] x reps (match by name)."""
import argparse
import os
import sys

import torch
import torch.cuda.profiler as prof
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../src"))
from apex_op import apex_topk, pick_config, workspace  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)

probe = load(name="floor_probe", sources=[os.path.join(HERE, "../src/floor_probe.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math",
                                "-gencode=arch=compute_100,code=sm_100"],
             build_directory=os.environ.get("BUILD_DIR", "/tmp/op35_build"), verbose=False)

EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
# (BS, N, K, read_cpr) — read_cpr matches iter0 anchor grid
CASES = [(1, 131072, 512, 148), (1, 262144, 512, 148), (1, 1048576, 512, 296),
         (32, 262144, 512, 8), (256, 262144, 512, 1), (1024, 65536, 512, 1)]

prof.start()
for BS, N, K, rcpr in CASES:
    x = torch.rand(BS, N, device="cuda") + 1.0
    red = torch.zeros(BS, device="cuda")
    tk = torch.zeros(BS, dtype=torch.int32, device="cuda")
    cfg = pick_config(BS, N, K)
    ws = workspace(BS, K, cfg, x.device)
    # warm both
    probe.stream_reduce(x, red, tk, rcpr)
    apex_topk(x, K, cfg=cfg, ws=ws)
    torch.cuda.synchronize()
    for _ in range(a.reps):
        EVICT.uniform_(); torch.cuda.synchronize()
        probe.stream_reduce(x, red, tk, rcpr)
        torch.cuda.synchronize()
    for _ in range(a.reps):
        EVICT.uniform_(); torch.cuda.synchronize()
        apex_topk(x, K, cfg=cfg, ws=ws)
        torch.cuda.synchronize()
prof.stop()
print("profiled")
