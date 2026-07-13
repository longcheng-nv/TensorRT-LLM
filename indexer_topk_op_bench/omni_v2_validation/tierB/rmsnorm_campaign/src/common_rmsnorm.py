# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared problem definition for the rmsnorm_campaign impl modules.

Contract (omni-kernel v2 scripts): every impl module re-exports
    kernel_fn(x, w) -> out        (the timed/verified implementation)
    reference_fn(x, w) -> out     (eager fp32-upcast reference)
    get_inputs() -> [x, w]        (fresh tensors, per-cell seed policy)
    get_adversarial_inputs() -> list of [x, w]

Cell selection: TOKENS env var (default 4096). hidden=7168, bf16, eps=1e-6.
Seed policy: cell_seed = f(tokens) so different cells get different draws
(constant seeds across cells are banned — SKILL Phase 4.2).
randn is acceptable here (dense-norm class; the randn ban targets
low-precision *selection* inputs).
"""
import os

import torch

HIDDEN = 7168
EPS = 1e-6
DTYPE = torch.bfloat16
TOKEN_GRID = [1, 16, 256, 4096, 16384]


def tokens():
    return int(os.environ.get("TOKENS", "4096"))


def cell_seed(t):
    return 0x5EED + 31 * HIDDEN + 7 * t


def make_inputs(t=None, seed=None):
    t = tokens() if t is None else t
    g = torch.Generator(device="cuda")
    g.manual_seed(cell_seed(t) if seed is None else seed)
    x = torch.randn((t, HIDDEN), generator=g, device="cuda", dtype=torch.float32).to(DTYPE)
    w = (torch.rand((HIDDEN,), generator=g, device="cuda", dtype=torch.float32) + 0.5).to(DTYPE)
    return [x, w]


def get_inputs():
    return make_inputs()


def reference_fn(x, w):
    """Eager fp32-upcast RMSNorm (matches flashinfer.norm.rmsnorm semantics)."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + EPS) * w.float()).to(x.dtype)


def get_adversarial_inputs():
    """Dense-class adversarial track: magnitude extremes, degenerate rows."""
    t = tokens()
    cases = []
    # all-zero rows (rsqrt(eps) path)
    x0 = torch.zeros((t, HIDDEN), device="cuda", dtype=DTYPE)
    w = make_inputs(t)[1]
    cases.append([x0, w])
    # constant large-magnitude rows (fp32 sum-of-squares headroom)
    cases.append([torch.full((t, HIDDEN), 200.0, device="cuda", dtype=DTYPE), w])
    # tiny values (underflow risk in x^2)
    cases.append([torch.full((t, HIDDEN), 1e-4, device="cuda", dtype=DTYPE), w])
    # one huge outlier per row, rest small
    xo = torch.full((t, HIDDEN), 0.01, device="cuda", dtype=DTYPE)
    xo[:, 137] = 1e4
    cases.append([xo, w])
    # negative weights + mixed signs
    g = torch.Generator(device="cuda")
    g.manual_seed(cell_seed(t) ^ 0xAD5)
    xm = (torch.randn((t, HIDDEN), generator=g, device="cuda") * 40.0).to(DTYPE)
    wn = -make_inputs(t)[1]
    cases.append([xm, wn])
    return cases
