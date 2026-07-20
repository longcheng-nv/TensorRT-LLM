# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ncu driver: launch the PR-head GVR kernel (gvrpkgprod2) on ONE real cell
a few times so Nsight Compute can profile it. Cold-L2 evict between launches.

Usage: ncu ... python3 ncu_cell_prod2.py --model pro --isl 512k [--layer 30]
"""
import argparse
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "harness"))

from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)

BENCH_L = {"flash": 22, "pro": 30, "v32": 34}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--isl", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    L = a.layer if a.layer is not None else BENCH_L[a.model]
    RD = RV32 if a.model == "v32" else RV4
    bd = RD.get_bundle(a.model, a.isl, L, "fp32")
    K, N, cr = int(bd["K"]), int(bd["N"]), int(bd["cr"])
    lg = bd["logits"].contiguous()
    pre = bd["preIdx"].contiguous()
    sl = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
    out = torch.empty((1, K), dtype=torch.int32, device="cuda")
    print(f"cell {a.model}/{a.isl}/L{L}: K={K} N={N} cr={cr} "
          f"cfg={GvrTopKKernel.pick_config(torch.float32, 1, N, N * cr)}",
          flush=True)
    evict = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    # warmup / JIT compile outside profiling
    GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr)
    torch.cuda.synchronize()
    for _ in range(a.reps):
        evict.random_()          # cold L2
        torch.cuda.synchronize()
        GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr)
        torch.cuda.synchronize()
    # sanity: exact
    idx = out[0].long()
    v = lg[0, :N].float()
    ok = idx.unique().numel() == K and torch.equal(
        v[idx].sort().values, torch.topk(v, K).values.sort().values)
    print("exact:", bool(ok), flush=True)


if __name__ == "__main__":
    main()
