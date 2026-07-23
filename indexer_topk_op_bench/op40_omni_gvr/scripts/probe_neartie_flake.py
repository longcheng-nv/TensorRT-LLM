# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flake-rate probe for the nondeterministic neartie_K2048_N262144 (cs=8)
baseline failure observed in gate run 2 (passed run 1). 100 repeated launches
of the SAME compiled kernel + same input; per-launch exactness; also sweep
cs in {1,4,8,16} to localize the race to the cluster path.

  CUDA_VISIBLE_DEVICES=<g> python3 probe_neartie_flake.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(HERE))

from ab40 import compile_arm, exact_set, launch_cfg  # noqa: E402

DEV = "cuda"
K, cr, N = 2048, 1, 262144
REPS = 100


def make_case():
    g = torch.Generator(device=DEV).manual_seed(hash((K, N)) % (2**31))
    logits = torch.randn(1, N, generator=g, device=DEV)
    _ = torch.randn(1, N, generator=g, device=DEV)          # plateau draw
    _ = torch.rand(1, N, generator=g, device=DEV)           # plateau mask
    _ = torch.randn(1, N, generator=g, device=DEV)          # narrow draw
    base = torch.randn(1, N, generator=g, device=DEV)       # neartie base
    v = torch.topk(base[0], K + 64).values
    kth = float(v[K - 1])
    nt = base.clone()
    band = torch.arange(2 * K, device=DEV) % 3
    idx = torch.randperm(N, generator=g, device=DEV)[:2 * K]
    eps = (torch.tensor([0.0, 1.0, -1.0], device=DEV)[band]
           * 1.2e-7 * max(abs(kth), 1e-3))
    nt[0, idx] = kth + eps
    gg = torch.Generator(device=DEV).manual_seed(
        hash((f"neartie_K{K}_N{N}", 7)) % (2**31))
    noisy = nt[0] + 0.5 * torch.randn(N, generator=gg, device=DEV)
    pre = torch.topk(noisy, K).indices.to(torch.int32).reshape(1, K).contiguous()
    return nt, pre


def main():
    logits, pre = make_case()
    sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
    for cs in (1, 4, 8, 16):
        cfg = launch_cfg(logits, N)
        cfg["cluster_size"] = cs
        fn = compile_arm("base", K, cr, cfg)
        fails = 0
        for i in range(REPS):
            oi = torch.full((1, K), -7, dtype=torch.int32, device=DEV)
            fn(logits, pre, sl, None, oi, None)
            torch.cuda.synchronize()
            if not exact_set(oi, logits[0], K, N):
                fails += 1
        print(f"cs={cs:2d}: {fails}/{REPS} launches inexact", flush=True)


if __name__ == "__main__":
    main()
