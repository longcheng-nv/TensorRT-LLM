#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""iter11 adversarial-logits gate: P4 path-C exactness on planted near-tie
clusters (the upstream ec04147502 failure mode, which op21's other gates
cannot see — they adversarialize preIdx, not logits collision structure).

Construction per row (fp32): strata A (clear winners, [2,3)), B (sparse
spreaders (1.001,1.9)), C (cluster of DISTINCT values spaced 2 ULP at 1.0,
straddling the K-th rank), D (spreaders (0.6,0.999)), E (bulk <= 0.01).
preIdx = half true-top-K + half D-pointers => wide data-driven band
[thr1, thr0) that contains C inside ONE coarse bin (cnt(b*) >> 32 => path
C) while the fine 256-split leaves multiple DISTINCT values per bin at the
cut => a fixed-depth stash-order emit picks a wrong subset (vd > 0).

Exactness check = tie-tolerant value multiset (value_metrics vd == 0,
nneg == 0, uniq == K), same convention as the other smokes.
"""
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))
from real_data_v2 import value_metrics  # noqa: E402
from gvr_ms_op import gvr_ms  # noqa: E402
from gvr_msc_op import gvr_msc  # noqa: E402

DEV = "cuda"


def make_row(K, N, seed, n_c=300, above_frac=0.55):
    """One adversarial row + preIdx. Returns (row[f32,N], preIdx[i32,K])."""
    g = torch.Generator().manual_seed(seed)
    n_b = 250
    n_d = 250
    r_above = int(n_c * above_frac)          # cluster members above the cut
    n_a = K - n_b - r_above                  # so the K-th rank falls inside C
    assert n_a > 0
    row = torch.rand(N, generator=g) * 0.01                     # E bulk
    perm = torch.randperm(N, generator=g)
    a_i = perm[:n_a]
    b_i = perm[n_a:n_a + n_b]
    c_i = perm[n_a + n_b:n_a + n_b + n_c]
    d_i = perm[n_a + n_b + n_c:n_a + n_b + n_c + n_d]
    row[a_i] = 2.0 + torch.rand(n_a, generator=g)               # A
    row[b_i] = 1.001 + torch.rand(n_b, generator=g) * 0.899     # B
    ulp = torch.finfo(torch.float32).eps                        # ~1.19e-7 @1.0
    # C: distinct near-ties, DESCENDING so "first-encountered" != "largest"
    row[c_i] = (1.0 + torch.arange(n_c, dtype=torch.float32).flip(0)
                * (2.0 * ulp))
    row[d_i] = 0.6 + torch.rand(n_d, generator=g) * 0.399       # D
    row = row.float()
    # preIdx: half the true top-K + half D pointers (wrong guesses) => wide
    # gathered-value spread => wide straddle band that contains C.
    true_top = torch.topk(row, K).indices
    half = K // 2
    pre = torch.empty(K, dtype=torch.int32)
    pre[:half] = true_top[:half].to(torch.int32)
    d_pool = d_i.repeat((half + n_d - 1) // n_d)[:half]
    pre[half:] = d_pool.to(torch.int32)
    pre = pre[torch.randperm(K, generator=g)]
    return row.to(DEV), pre.to(DEV).contiguous()


def check(out, lg, K):
    ref = torch.topk(lg[0].float(), K).indices
    vd, _rc, nn = value_metrics(out[:1], lg[:1].float(), ref, K)
    u = torch.unique(out[0][out[0] >= 0]).numel()
    return vd == 0 and nn == 0 and u == K, vd, u


def main():
    torch.manual_seed(0)
    ok = bad = 0
    fails = []
    CASES = []
    # cr follows the production conventions: K512/K1024 = V4 Flash/Pro
    # (cr=4, preIdx used directly, kernel offset 0); K2048 = V3.2 (cr=1,
    # caller passes prev_topk-1, kernel adds the +1 diagonal offset).
    for K, N, cr in ((1024, 262144, 4), (1024, 131072, 4), (512, 262144, 4),
                     (2048, 262144, 1)):
        for seed in (11, 22, 33):
            for above_frac in (0.4, 0.55):
                CASES.append((K, N, cr, seed, above_frac))
    for K, N, cr, seed, af in CASES:
        row, pre = make_row(K, N, seed, above_frac=af)
        if cr == 1:
            pre = (pre - 1).clamp(min=0)  # diagonal convention: kernel +1
        lg = row.unsqueeze(0).contiguous()
        pr = pre.unsqueeze(0).contiguous()
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        for tag, fn in (
            ("ms", lambda: gvr_ms(lg, pr, sl, K, compress_ratio=cr)),
            ("C4", lambda: gvr_msc(lg, pr, sl, K, cr, C=4)),
            ("C8", lambda: gvr_msc(lg, pr, sl, K, cr, C=8)),
        ):
            out = fn()
            torch.cuda.synchronize()
            good, vd, u = check(out, lg, K)
            ok += good
            bad += not good
            if not good:
                fails.append((tag, K, N, seed, af, float(vd), u))
                print(f"FAIL {tag} K{K} N{N} s{seed} af{af}: "
                      f"vdiff={vd:.3e} uniq={u}/{K}")
    print(f"adversarial-band: {ok} ok / {bad} fail")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
