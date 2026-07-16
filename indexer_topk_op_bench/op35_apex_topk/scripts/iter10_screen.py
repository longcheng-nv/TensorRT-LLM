# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter10: APEX v0 E2E exactness screen (synth) — tie-aware value-multiset vs
torch.topk across grid modes, K, N incl. odd-N + repeated-call (self-clean)."""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../src"))
from apex_op import apex_topk, pick_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)


def check(x, K, N=None, tag=""):
    BS = x.size(0)
    n = N or x.size(1)
    idx = apex_topk(x, K, N=N).clone()
    torch.cuda.synchronize()
    ok = True
    for r in range(BS):
        got = x[r, :n][idx[r].long()]
        ref = torch.topk(x[r, :n].float(), K).values
        if not torch.equal(got.sort(descending=True).values, ref):
            ok = False
            break
        if idx[r].long().unique().numel() != K or int(idx[r].max()) >= n:
            ok = False
            break
    print(f"{tag}: BS={BS} N={n} K={K} -> {'OK' if ok else 'FAIL'}", flush=True)
    return ok


allok = True
g = torch.Generator(device="cuda")
CASES = [
    (1, 131072, 512), (1, 262144, 2048), (1, 1048576, 512), (1, 4096, 512),
    (2, 262144, 1024), (4, 65536, 512), (8, 131072, 2048),
    (16, 32768, 512), (32, 262144, 512), (256, 65536, 1024), (1024, 65536, 512),
]
for BS, N, K in CASES:
    g.manual_seed(BS * 1000003 + N + K)
    x = torch.randn(BS, N, device="cuda", generator=g)
    allok &= check(x, K, tag="synth")
    # repeat call on same workspace (self-clean check)
    allok &= check(x, K, tag="synth-rep")

# odd logical N with padded stride
for BS, N, K in [(1, 1027, 512), (4, 66666, 1024), (32, 131071, 2048)]:
    stride = (N + 63) // 64 * 64
    xp = torch.full((BS, stride), float("nan"), device="cuda")
    g.manual_seed(N)
    xp[:, :N] = torch.randn(BS, N, device="cuda", generator=g)
    allok &= check(xp, K, N=N, tag="oddN")

# tie plateaus (bf16-quantized values in fp32 container) + constant rows
for BS, N, K in [(1, 262144, 2048), (64, 65536, 512)]:
    g.manual_seed(7 + N)
    x = torch.randn(BS, N, device="cuda", generator=g).to(torch.bfloat16).float()
    allok &= check(x, K, tag="bf16-plateau")
x = torch.full((4, 65536), 1.25, device="cuda")
allok &= check(x, 512, tag="const-row")

print("ALL", "OK" if allok else "FAIL")
