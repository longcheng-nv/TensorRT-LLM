# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose: is the C:tie K2048 16-bit failure an op26 regression or an
inherited anchor property? Run the exact failing Suite-C construction on all
four arms (both anchors + both op26 arms)."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE / "src"))

from gvr_cutedsl_op import gvr_cutedsl as anchor_1cta  # noqa: E402
from gvr_multicta_cutedsl_op import gvr_multicta_cutedsl as anchor_mc  # noqa: E402
from gvr_op26_op import gvr_cutedsl_op26, gvr_multicta_op26  # noqa: E402

DEV = "cuda"
ARMS = [
    ("anchor_1cta", anchor_1cta),
    ("op26_1cta", gvr_cutedsl_op26),
    ("anchor_mc", anchor_mc),
    ("op26_mc", gvr_multicta_op26),
]


def report(name, out, logits, N, K):
    o = out[0]
    idx = o.long()
    bad = int(((idx < 0) | (idx >= N)).sum())
    if bad:
        print(f"  {name:12s}: FAIL  {bad} out-of-range/-1")
        return
    uniq = len(set(o.tolist()))
    if uniq != K:
        print(f"  {name:12s}: FAIL  dup indices uniq={uniq}")
        return
    sel = logits[0].gather(0, idx).float().sort().values
    ref = torch.topk(logits[0][:N].float(), K).values.sort().values
    if torch.equal(sel, ref):
        print(f"  {name:12s}: OK    exact value-set")
    else:
        d = (sel - ref).abs().max().item()
        print(f"  {name:12s}: FAIL  value-set mismatch maxdiff={d:.3e}")


torch.manual_seed(99)
K, cr = 2048, 1
for N in (16384, 131072):
    for dt in (torch.bfloat16, torch.float16):
        # exact Suite-C construction
        row = torch.rand(N) * 0.5
        plateau = torch.randperm(N)[: 5 * K]
        row[plateau] = 0.75
        winners = plateau[: K // 2]
        row[winners] = 0.9
        row = row.to(dt).cuda().view(1, N).contiguous()
        pre = torch.topk(row[0].float(), K).indices.int().view(1, K).contiguous()
        seq = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        print(f"C:tie K={K} {dt} N={N}:")
        for name, fn in ARMS:
            try:
                out = fn(row, pre, seq, K, compress_ratio=cr)
                torch.cuda.synchronize()
                report(name, out, row, N, K)
            except Exception as e:
                print(f"  {name:12s}: ERR   {type(e).__name__}: {str(e)[:100]}")
print("done")
