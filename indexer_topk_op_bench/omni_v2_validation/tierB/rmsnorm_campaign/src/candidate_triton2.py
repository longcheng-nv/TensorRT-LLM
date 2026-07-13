# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iter2 screening variant: fixed-config single-pass Triton RMSNorm (no autotune).

Env knobs (screening only; the shipped module hard-codes the winner):
    NW       num_warps (default 8)
    EV_LOAD  eviction policy for x load:  "" | evict_first | evict_last
    EV_STORE eviction policy for y store: "" | evict_first
Rationale: iter1 autotune benches in WARM L2 (its quick bench does no eviction)
-> config choice miscalibrated for the cold-L2 deployment axis; and streaming
cells (T=16384, 470 MB >> L2) may want evict_first.
"""
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_rmsnorm import EPS, get_adversarial_inputs, get_inputs, reference_fn  # noqa: F401,E402

NW = int(os.environ.get("NW", "8"))
EV_LOAD = os.environ.get("EV_LOAD", "") or None
EV_STORE = os.environ.get("EV_STORE", "") or None


@triton.jit
def _rmsnorm_kernel2(X, W, Y, N, eps, BLOCK: tl.constexpr,
                     EVL: tl.constexpr, EVS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0,
                eviction_policy=EVL).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask,
             eviction_policy=EVS)


def kernel_fn(x, w):
    t, n = x.shape
    y = torch.empty_like(x)
    _rmsnorm_kernel2[(t,)](x, w, y, n, EPS, BLOCK=8192,
                           EVL=EV_LOAD, EVS=EV_STORE, num_warps=NW)
    return y
