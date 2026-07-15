# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op34 iter4 — multi-CTA single-pass GVR top-K (the MLP lever).

The NCU CRUX (analysis/NCU_CRUX_048.md) showed the BS=1 top-K is LATENCY-bound
(<1% DRAM/SM peak) and sglang wins purely via 8-CTA memory-level parallelism.
This kernel exploits the 147 idle SMs at BS=1: it splits the row across C>8 CTAs
and does ONE fused count+collect pass at the GVR HINT threshold (the hint lets
GVR collect at a known threshold where sglang needs a separate histogram pass).

Skeleton (GVR threshold-method, parallelized — NOT a new algorithm):
  threshold t = min over the K prev-topK (hint) gathered values. count(>=t)>=K
  <=> t <= tau* (the true K-th value) <=> ALL true top-K are >= t <=> EXACT.
  (proof: >=K elements are >= t  =>  the K-th largest is >= t.)  hint.min is the
  weakest hint value, so it is <= tau* whenever ANY true top-K element is in the
  hint — which holds on 100% of the real v4cap cells (host-verified /tmp/happy).
  Fail-soft: if count < K (would only happen if hit_rate were 0), fall back to a
  full exact top-K (guaranteed correct).

kernel1 (grid=C): each CTA scans its 1/C slice once, stream-compacts elements
  >= t into a shared candidate list via block prefix-sum + one atomic/block-iter.
tail: exact top-K over the M<=N candidates (torch.topk here; a lean single-CTA
  rank-scatter is the CuTe productization — this module proves the algorithm).

Dispatch C by N (BS=1): more CTAs at large N where the scan dominates; 1 CTA at
tiny N where op26_r0's single-CTA phase-chain already wins (op32 wall).
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

DEV = "cuda"
_NEG = -3.0e38


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
        excl = tl.cumsum(qi, axis=0) - qi
        base = tl.atomic_add(ctr_ptr, nq)
        slot = base + excl
        sm = qual & (slot < CAP)
        tl.store(cand_val_ptr + slot, v, mask=sm)
        tl.store(cand_idx_ptr + slot, idx, mask=sm)
        off += BLOCK


def dispatch_C(N):
    if N >= 65536:
        return 64
    if N >= 16384:
        return 32
    if N >= 4096:
        return 8
    return 1


# reusable scratch (grown as needed) to avoid per-call alloc in the timed path
_SCRATCH = {}


def _scratch(CAP):
    key = CAP
    if key not in _SCRATCH:
        _SCRATCH[key] = (
            torch.empty((CAP,), dtype=torch.float32, device=DEV),
            torch.empty((CAP,), dtype=torch.int32, device=DEV),
            torch.zeros(1, dtype=torch.int32, device=DEV),
        )
    return _SCRATCH[key]


def mcta_topk(logits_row, preidx_row, N, K, out, C=None, BLOCK=1024,
              mode="hint", collect_only=False, t_override=None):
    """logits_row [1, Npad] fp32; preidx_row [1, K] int32; out [1, K] int32.
    mode: 'hint' = t=min(hint gathered) (exact, real); 'oracle' = t=true K-th
    value (UB best-case, cheats — for the double-lock feasibility bound only).
    collect_only: skip the tail (times the fused scan+collect alone).
    t_override: use this threshold (precompute oracle t OUTSIDE the timed loop)."""
    if C is None:
        C = dispatch_C(N)
    row = logits_row[0]
    pre = preidx_row[0].long()
    if t_override is not None:
        t = t_override
    elif mode == "oracle":
        t = float(torch.topk(row[:N], K).values.min().item())
    else:
        t = float(row[pre].min().item())
    CAP = ((N + BLOCK - 1) // BLOCK) * BLOCK
    cand_val, cand_idx, ctr = _scratch(CAP)
    ctr.zero_()
    _collect[(C,)](row, N, C, t, cand_val, cand_idx, ctr, CAP, BLOCK=BLOCK)
    if collect_only:
        # timing-only diagnostic: write in-range sentinel so a downstream
        # exactness probe cannot OOB-index (values are NOT the real top-K).
        out[0].copy_(torch.arange(K, device=DEV, dtype=torch.int32))
        return out
    M = int(ctr.item())
    Mc = min(M, CAP)
    if Mc >= K:
        tv, ti = torch.topk(cand_val[:Mc], K)
        out[0].copy_(cand_idx[ti])
    else:  # fail-soft (hit_rate==0 degenerate): full exact top-K
        _, gi = torch.topk(row[:N], K)
        out[0].copy_(gi.to(torch.int32))
    return out


if __name__ == "__main__":
    # smoke vs torch.topk on random rows (value-multiset)
    torch.manual_seed(0)
    for K, N in ((1024, 65539), (512, 16387), (1024, 262127)):
        row = torch.randn(1, N, dtype=torch.float32, device=DEV)
        noisy = row + 0.7 * row.std() * torch.randn_like(row)
        pre = torch.topk(noisy[0], K).indices.int().view(1, K)
        out = torch.empty((1, K), dtype=torch.int32, device=DEV)
        mcta_topk(row, pre, N, K, out)
        ref = torch.topk(row[0], K).values.sort().values
        got = row[0][out[0].long()].sort().values
        d = (got - ref).abs().max().item()
        print(f"K={K} N={N} C={dispatch_C(N)} vdiff={d:.2e} {'OK' if d == 0 else 'FAIL'}")
    print("op34_mcta smoke done")
