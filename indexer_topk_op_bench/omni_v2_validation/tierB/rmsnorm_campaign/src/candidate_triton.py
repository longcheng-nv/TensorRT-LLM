# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iter1 candidate: single-pass Triton RMSNorm, 1 CTA per row, fp32 accumulation.

Hypothesis: incumbent uses 128 threads/row; at small T the grid underfills the
GPU and per-CTA memory-level parallelism is the lever -> wider CTAs (more
warps) should close the copy-ceiling gap at T<=256 while matching at T>=4096.
Out-of-place, CUDA-graph compatible (allocation via caching allocator).
"""
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_rmsnorm import EPS, get_adversarial_inputs, get_inputs, reference_fn  # noqa: F401,E402


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["n_rows"],
)
@triton.jit
def _rmsnorm_kernel(X, W, Y, n_rows, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


def kernel_fn(x, w):
    t, n = x.shape
    y = torch.empty_like(x)
    _rmsnorm_kernel[(t,)](x, w, y, t, n, EPS, BLOCK=8192)
    return y
