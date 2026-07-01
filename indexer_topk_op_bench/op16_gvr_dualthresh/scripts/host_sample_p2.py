# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Host prototype: sampling-based quantile init for GVR Phase-2 (cheaper-P2 lever).

Idea (op13 unbuilt H2): instead of iterative full-N secant `count_ge` passes to
pin the threshold, read a small STRIDED subsample (n_s elements, N-independent),
estimate the K-th-value quantile from the sample, then do ONE full-N confirm
pass; only correct (secant) if it missed the accept window. The saving is
COMPUTE (fewer total element-comparisons), which is the real large-N P2 cost
(op14 showed HBM is moot — input fits L2). Distinct from op14 compaction.

Metric of merit: number of FULL-N `count_ge` passes (confirm + corrections).
Baseline secant does ~2-3. If sampling does ~1-1.5 AND stays exact, that's the
win. Also test whether sampling pins a TIGHT threshold (small kC => cand≈K,
unlocking the P4 collapse) cheaply.

This is host-only validation (searchsorted counts). Kernel build follows only if
the pass reduction is real across grid × 3 beta cfgs × seeds.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "op13_gvr_p2cand" / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
from p2_replay import (  # noqa: E402
    _prep_row, _count_ge, replay_row, SecantCfg, F32, NEG_FLT_MAX,
    _DTYPE_NAME, MAX_REFINE_ITERS)
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrParams  # noqa: E402


def sampled_quantile_replay(logits_row, pre_idx_row, N, K, cr, dtype,
                            n_sample=4096, aim_mult=1.35, kCC=None, kFTarget=None):
    """Sampling-init P2 replay. Returns (full_n_evals, cand, exact, thr, n_sample_eff).

    full_n_evals = # of full-N count_ge (confirm + corrections). The sample read
    (n_sample elements) is tracked separately (weight ~ n_sample/N of a pass).
    """
    gp = GvrParams.get(_DTYPE_NAME[dtype], K, cr)
    kK = K
    kCC = kCC if kCC is not None else gp.kC
    kFTarget = kFTarget if kFTarget is not None else gp.kFTarget

    prep = _prep_row(logits_row, pre_idx_row, N, K, cr, dtype)
    xs = prep["xs"]
    pmin, pmax = prep["pmin"], prep["pmax"]
    if pmax <= NEG_FLT_MAX or pmin >= pmax:
        return 0, min(kK, N), (N <= K), float(pmax), 0

    # full-precision fp32 row values (ascending sorted already in xs)
    x = xs  # ascending
    n_eff = min(n_sample, N)
    stride = max(1, N // n_eff)
    # strided sample from the UNSORTED order is what a kernel does; but for a
    # quantile estimate the order is irrelevant — sample by stride over sorted
    # is a *biased* even-spacing; use a pseudo-random-but-deterministic gather
    # over the original positions instead. We emulate original-order strided
    # sample: take every `stride`-th element of the (unsorted) row.
    # xs is sorted; to get an unbiased sample we instead subsample xs uniformly
    # (same empirical quantile as a random sample of the same size).
    idx = torch.linspace(0, N - 1, steps=n_eff, device=x.device).round().long()
    samp = x[idx]                       # ascending sample of size n_eff
    # aim: want count_ge(t0) ≈ aim_mult*K in full data (bias low → count≥K).
    aim_count = min(int(aim_mult * kK), kCC)
    r_s = max(1, int(round(aim_count * n_eff / N)))   # r_s-th largest in sample
    # t0 = value at rank r_s from top of the ascending sample
    t0 = float(samp[n_eff - r_s].item())

    # ---- confirm + secant correction using the standard bracket ----
    val_lo, val_hi = pmin, pmax
    cnt_lo = kK + (kK >> 2)
    cnt_hi = 1
    thr = F32(t0)
    done = 0
    full_evals = 1
    c = _count_ge(x, thr)
    if kK <= c <= kCC:
        done = 1
    elif c > kCC:
        val_lo, cnt_lo = thr, c
    else:
        val_hi, cnt_hi = thr, c

    it = 0
    while it < MAX_REFINE_ITERS and done == 0:
        vlo, vhi = val_lo, val_hi
        clo, chi = cnt_lo, cnt_hi
        rng = F32(vhi - vlo)
        if clo > chi and rng > F32(1e-10):
            f = F32(F32(clo - kFTarget) / F32(clo - chi))
            f = F32(max(F32(0.05), f)); f = F32(min(f, F32(0.95)))
            nv = F32(vlo + rng * f)
        else:
            nv = F32((vlo + vhi) * F32(0.5))
        if nv <= vlo:
            nv = F32(vlo + rng * F32(0.05))
        if nv >= vhi:
            nv = F32(vhi - rng * F32(0.05))
        thr = nv
        c = _count_ge(x, thr)
        full_evals += 1
        if kK <= c <= kCC:
            done = 1
        elif c > kCC:
            val_lo, cnt_lo = thr, c
        else:
            val_hi, cnt_hi = thr, c
        it += 1

    if done == 0:
        thr = val_lo if cnt_lo <= kCC * 2 else val_hi
    c_final = _count_ge(x, thr)
    cand = min(c_final, kCC)
    exact = (c_final >= kK) and (c_final <= kCC)
    return full_evals, cand, exact, float(thr), n_eff


def main():
    import argparse
    from synth_data import get_bundle
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--dt", nargs="+", default=["fp32", "bf16", "fp16"])
    ap.add_argument("--N", type=int, nargs="+", default=[4096, 16384, 65536, 131072, 262144])
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--cfgs", nargs="+", default=["beta_shallow", "beta_moderate", "beta_deep"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--nsample", type=int, default=4096)
    ap.add_argument("--aim", type=float, default=1.35)
    ap.add_argument("--kcc", type=int, default=None, help="tight kCC (None=default wide)")
    args = ap.parse_args()
    dtmap = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

    print(f"n_sample={args.nsample} aim_mult={args.aim} kCC={'default' if args.kcc is None else args.kcc}")
    print(f"{'K':>4} {'dt':>4} {'N':>7} | {'base_evals':>10} {'samp_evals':>10} {'base_cand':>9} {'samp_cand':>9} | {'samp_exact':>10} {'compute_ratio':>13}")
    agg = {}
    for K in args.K:
        cr = args.cr if K != 2048 else 1
        for dts in args.dt:
            dtype = dtmap[dts]
            for N in args.N:
                be, se, bc, sc, bx, sx = [], [], [], [], 0, 0
                ns_eff = 0
                ncell = 0
                for cfg in args.cfgs:
                    for seed in args.seeds:
                        b = get_bundle(K, dtype, N, cfg=cfg, seed=seed)
                        lg = b["logits"][0]
                        pi = b["preIdx"][0]
                        # baseline secant
                        base = replay_row(lg, pi, N, K, cr, dtype,
                                          SecantCfg(kCC=args.kcc))
                        # sampled-quantile
                        fev, cand, ex, thr, nse = sampled_quantile_replay(
                            lg, pi, N, K, cr, dtype, n_sample=args.nsample,
                            aim_mult=args.aim, kCC=args.kcc)
                        be.append(base.p2_evals); se.append(fev)
                        bc.append(base.cand_count); sc.append(cand)
                        bx += int(base.exact); sx += int(ex)
                        ns_eff = nse; ncell += 1
                m = lambda a: sum(a) / len(a)
                # compute ratio = (samp full-N passes + sample/N) / base full-N passes
                comp_ratio = (m(se) + ns_eff / N) / max(m(be), 1e-9)
                print(f"{K:>4} {dts:>4} {N:>7} | {m(be):>10.2f} {m(se):>10.2f} {m(bc):>9.0f} {m(sc):>9.0f} | {sx:>4}/{ncell:<5} {comp_ratio:>13.3f}")
                agg[(K, dts, N)] = (m(be), m(se), comp_ratio, sx == ncell)
    # summary
    ok = all(v[3] for v in agg.values())
    avg_ratio = sum(v[2] for v in agg.values()) / len(agg)
    print(f"\nALL sampled-init EXACT: {ok}   mean compute_ratio (samp/base): {avg_ratio:.3f}")


if __name__ == "__main__":
    main()
