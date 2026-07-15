# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CRUX-C rung-2 proxy: does a multi-CTA fused collect + rank tail beat sglang
at large N? Bounds the op34 winning hypothesis WITHOUT the full CuTe build.

kernel1 (Triton, grid=C CTAs/row): each CTA scans its 1/C slice ONCE, collects
(val,idx) >= threshold into a global candidate list via block-level stream
compaction (tl.cumsum exclusive prefix + ONE atomic_add per block-iter to
reserve slots — cheap, not per-element). This is the fused count+collect pass.
tail: torch.topk on the M<=kC candidates (a realistic proxy for P4 rank-scatter).

Thresholds tested (bracket the tail cost):
  oracle : t = K-th largest value  -> count ~ K (minimal candidates, best case)
  hintish: t chosen so count ~ kC  -> worst-case tail load
Both are EXACT (count>=K => top-K subset of candidates; torch.topk picks the K).

Timing: run once in a cudaProfilerApi window; NCU sums kernel1 + torch.topk
kernel gpu_times = pure-kernel multi-CTA total, cold-L2. Compare to sglang NCU.
"""
import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
OPBENCH = HERE.parents[1]
sys.path.insert(0, str(OPBENCH / "harness"))
sys.path.insert(0, str(OPBENCH / "op26_gvr_logfalsi_rs" / "src"))
import real_data_v4cap as RD4

DEV = "cuda"
NEG = -3.0e38


@triton.jit
def _collect(x_ptr, N, C, thr, cand_val_ptr, cand_idx_ptr, ctr_ptr,
             CAP, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    chunk = (N + C - 1) // C
    start = c * chunk
    end = tl.minimum(start + chunk, N)
    off = start
    while off < end:
        idx = off + tl.arange(0, BLOCK)
        m = idx < end
        v = tl.load(x_ptr + idx, mask=m, other=-3.0e38)
        qual = m & (v >= thr)
        qi = qual.to(tl.int32)
        nq = tl.sum(qi)
        excl = tl.cumsum(qi, axis=0) - qi          # exclusive prefix in block
        base = tl.atomic_add(ctr_ptr, nq)          # one atomic per block-iter
        slot = base + excl
        sm = qual & (slot < CAP)
        tl.store(cand_val_ptr + slot, v, mask=sm)
        tl.store(cand_idx_ptr + slot, idx, mask=sm)
        off += BLOCK


def multi_cta_topk(x, N, K, thr, C, CAP=8192, BLOCK=1024):
    cand_val = torch.full((CAP,), NEG, dtype=torch.float32, device=DEV)
    cand_idx = torch.zeros((CAP,), dtype=torch.int32, device=DEV)
    ctr = torch.zeros(1, dtype=torch.int32, device=DEV)
    _collect[(C,)](x, N, C, thr, cand_val, cand_idx, ctr, CAP, BLOCK=BLOCK)
    M = int(ctr.item())
    Mc = min(M, CAP)
    kk = min(K, Mc)
    tv, ti = torch.topk(cand_val[:Mc], kk)
    out = cand_idx[ti]
    return out, M


def pick_threshold(row, K, kind):
    s = torch.sort(row, descending=True).values
    if kind == "oracle":
        return float(s[K - 1].item()), "count~K"
    if kind == "hintish":
        # threshold that admits ~kC=5120 candidates (worst-case tail)
        j = min(5120, row.numel() - 1)
        return float(s[j].item()), "count~5120"
    raise ValueError(kind)


def run_cell(model, isl, C, kind, verbose=True):
    layers = RD4.MODELS[model]["layers"]
    L = layers[len(layers) // 2]
    b = RD4.get_bundle(model, isl, L, "fp32")
    row = b["logits"][0, :b["N"]].contiguous()
    N, K = b["N"], b["K"]
    thr, note = pick_threshold(row, K, kind)
    out, M = multi_cta_topk(row, N, K, thr, C)
    # exactness
    ref = torch.topk(row, K).values.sort().values
    got = row[out.long()].sort().values
    vdiff = (got - ref).abs().max().item()
    if verbose:
        print(f"{model}/{isl} N={N} K={K} C={C} {kind}({note}): M={M} "
              f"vdiff={vdiff:.2e} {'EXACT' if vdiff==0 else 'MISS'}")
    return out, M, vdiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pro")
    ap.add_argument("--isl", default="256k")
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--kind", default="oracle", choices=["oracle", "hintish"])
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()

    if a.profile:
        layers = RD4.MODELS[a.model]["layers"]
        L = layers[len(layers) // 2]
        b = RD4.get_bundle(a.model, a.isl, L, "fp32")
        row = b["logits"][0, :b["N"]].contiguous()
        N, K = b["N"], b["K"]
        thr, _ = pick_threshold(row, K, a.kind)
        for _ in range(5):
            multi_cta_topk(row, N, K, thr, a.C)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        multi_cta_topk(row, N, K, thr, a.C)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print(f"profiled {a.model}/{a.isl} C={a.C} {a.kind}")
        return

    for kind in ("oracle", "hintish"):
        for isl in ("256k", "1024k"):
            for C in (16, 32, 64):
                run_cell(a.model, isl, C, kind)


if __name__ == "__main__":
    main()
