# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Paired A/B: p4_exact_tail ON (new default) vs OFF — perf neutrality gate.

Single GPU, back-to-back arms per cell, cold L2 (256MB evict between timed
calls), CUDA-event timing repeated; nsys wrapper optional (cudaProfilerApi
window). Cells span [worst, real, best] axes + a 16-bit control (compiled
code must be byte-identical -> exactly noise) + the ambiguous real cell
(pro/512k fp32, where the repair actually RUNS - its cost is the price of
correctness there).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, "/tmp/gvrval1")

import real_data_v4cap as RV4  # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

DEV = "cuda"
_evict = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


def cold(call, reps=60, warm=15):
    ts = []
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(warm + reps):
            _evict.zero_()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            call()
            e1.record()
            e1.synchronize()
            if i >= warm:
                ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def synth_cell(K, N, scen, dt, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    lg = torch.randn(1, N, generator=g, device=DEV, dtype=torch.float32)
    noise = 0.25 if scen == "best" else 3.0
    pre = torch.topk(lg + noise * torch.randn(1, N, generator=g, device=DEV), K, dim=-1).indices.to(torch.int32)
    return lg.to(dt), pre


def run_cell(name, lg_row, pre_row, K, cr, BS, arms=(None, False)):
    """arms: p4_exact_tail values; None = new default, False = old behavior."""
    lg = lg_row.expand(BS, -1).contiguous()
    pre = pre_row.expand(BS, -1).contiguous()
    sl = torch.full((BS,), lg.shape[1] * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    res = {}
    for tail in arms:   # paired back-to-back
        call = lambda: GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr, p4_exact_tail=tail)
        call()
        torch.cuda.synchronize()
        res[tail] = cold(call)
    r = res[arms[0]] / res[arms[1]]
    print(f"{name:>44} | on {res[arms[0]]:8.2f}us off {res[arms[1]]:8.2f}us | on/off {r:.3f}")
    return name, r


rows = []
for K, N, scen in ((512, 16384, "best"), (512, 262144, "worst"), (1024, 131072, "best"),
                   (2048, 65536, "worst"), (2048, 262144, "best")):
    lg, pre = synth_cell(K, N, scen, torch.float32)
    for BS in (1, 256):
        rows.append(run_cell(f"synth/{scen}/K{K}/fp32/N{N}/BS{BS}", lg, pre, K, 1 if K == 2048 else 4, BS))
# 16-bit control (const-folded OFF both arms -> pure noise)
lg, pre = synth_cell(1024, 131072, "best", torch.bfloat16)
rows.append(run_cell("CONTROL synth/best/K1024/bf16/N131072/BS1", lg, pre, 1024, 4, 1))
# real cells: flash 128k (unambiguous), pro 512k (AMBIGUOUS - repair runs)
for model, isl, L, K, cr in (("flash", "128k", 22, 512, 4), ("pro", "512k", 30, 1024, 4)):
    b = RV4.get_bundle(model, isl, L, "fp32")
    for BS in (1, 256):
        rows.append(run_cell(f"real/{model}/{isl}/fp32/BS{BS}", b["logits"], b["preIdx"], K, cr, BS))

import math
neutral = [r for n, r in rows if "pro/512k" not in n]
gm = math.exp(sum(math.log(r) for _, r in [x for x in rows if "pro/512k" not in x[0]]) / len(neutral))
print(f"\nneutral-cell geomean on/off = {gm:.4f}  (pro/512k excluded: repair active there)")
