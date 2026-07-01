# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Host validation of SECANT convergence-acceleration variants (op13 H2).

User constraint: stay within the "secant bracket-then-refine" structure; only
optimize the ITERATION (interpolation/init), introducing NO new pass and NO
sampling. Goal: cut the number of full-N count_ge evals (esp. large N /
K1024/K2048 where baseline needs 2.78-3.67), staying exact.

Variants (all reuse the same [val_lo,val_hi] bracket + accept window [K,kCC]):
  - base      : current kernel secant (linear, f clamped [0.05,0.95], iter0 cap 0.5)
  - illinois  : regula-falsi with Illinois stagnation damping (halve the retained
                endpoint's weight) — superlinear, no extra state
  - quad      : inverse-quadratic interpolation on last 3 (thr,count) pts, secant fallback
  - base+pq / illinois+pq : same, but init t0 = preIdx-value quantile (aim-th from top)

Metric: mean full-N count_ge evals + exact rate, across grid x 3 cfgs x seeds.
Only interpolation/init variants (kernel-cheap) are tested — NO sampling.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "op13_gvr_p2cand" / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
from p2_replay import (  # noqa: E402
    _prep_row, _count_ge, _init_thr, SecantCfg, F32, NEG_FLT_MAX,
    _DTYPE_NAME, MAX_REFINE_ITERS)
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrParams  # noqa: E402


def _run(logits_row, pre_idx_row, N, K, cr, dtype, variant, init_mode="mean",
         init_q=0.5, kCC=None, kFTarget=None):
    gp = GvrParams.get(_DTYPE_NAME[dtype], K, cr)
    kK = K
    kCC = kCC if kCC is not None else gp.kC
    kFT = kFTarget if kFTarget is not None else gp.kFTarget
    prep = _prep_row(logits_row, pre_idx_row, N, K, cr, dtype)
    xs = prep["xs"]; pmin, pmax = prep["pmin"], prep["pmax"]
    if pmax <= NEG_FLT_MAX or pmin >= pmax:
        return 0, min(kK, N), (N <= K)

    thr = _init_thr(prep, SecantCfg(init_mode=init_mode, init_q=init_q))
    val_lo, val_hi = pmin, pmax
    cnt_lo = kK + (kK >> 2); cnt_hi = 1
    # Illinois stagnation side memory
    last_side = 0  # -1 low retained, +1 high retained
    # history for quad
    hist = []  # (thr, count)

    def classify(t):
        c = _count_ge(xs, t)
        hist.append((t, c))
        return c

    done = 0
    evals = 1
    c = classify(thr)
    fa = None  # f at val_lo (count_lo - kFT) > 0 ; fb at val_hi < 0
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
        # ---- interpolation ----
        if variant == "quad" and len(hist) >= 3:
            # inverse quadratic on last 3 distinct points vs target kFT
            pts = hist[-3:]
            xs3 = [p[0] for p in pts]; ys3 = [p[1] - kFT for p in pts]
            nv = None
            if len(set(ys3)) == 3:
                x0, x1, x2 = xs3; y0, y1, y2 = ys3
                try:
                    nv = (x0*y1*y2/((y0-y1)*(y0-y2))
                          + x1*y0*y2/((y1-y0)*(y1-y2))
                          + x2*y0*y1/((y2-y0)*(y2-y1)))
                except ZeroDivisionError:
                    nv = None
            if nv is None or not (vlo < nv < vhi):
                nv = None
            if nv is None:
                variant_step = "secant"
            else:
                variant_step = "quad"; nvf = F32(nv)
        else:
            variant_step = "secant"

        if variant_step == "secant":
            if clo > chi and rng > F32(1e-10):
                f = F32(F32(clo - kFT) / F32(clo - chi))
                # Illinois damping: if the SAME endpoint was retained, halve its pull
                if variant in ("illinois", "illinois_pq"):
                    if last_side < 0:      # low endpoint kept -> pull toward high
                        f = F32(f / 2.0)
                    elif last_side > 0:    # high endpoint kept -> pull toward low
                        f = F32(1.0 - (1.0 - f) / 2.0)
                f = F32(max(F32(0.05), f)); f = F32(min(f, F32(0.95)))
                if it == 0 and variant not in ("illinois", "illinois_pq"):
                    f = F32(min(f, F32(0.5)))
                nvf = F32(vlo + rng * f)
            else:
                nvf = F32((vlo + vhi) * F32(0.5))
        if nvf <= vlo:
            nvf = F32(vlo + rng * F32(0.05))
        if nvf >= vhi:
            nvf = F32(vhi - rng * F32(0.05))
        thr = nvf
        c = classify(thr); evals += 1
        if kK <= c <= kCC:
            done = 1
        elif c > kCC:
            val_lo, cnt_lo = thr, c; last_side = -1
        else:
            val_hi, cnt_hi = thr, c; last_side = 1
        it += 1

    if done == 0:
        thr = val_lo if cnt_lo <= kCC * 2 else val_hi
    cf = _count_ge(xs, thr); cand = min(cf, kCC)
    exact = (cf >= kK) and (cf <= kCC)
    return evals, cand, exact


def main():
    import argparse
    from synth_data import get_bundle
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--dt", nargs="+", default=["fp32", "bf16"])
    ap.add_argument("--N", type=int, nargs="+", default=[16384, 65536, 131072, 262144])
    ap.add_argument("--cfgs", nargs="+", default=["beta_shallow", "beta_moderate", "beta_deep"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    dtmap = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    variants = [("base", "mean"), ("illinois", "mean"), ("quad", "mean"),
                ("base", "pquantile"), ("illinois", "pquantile")]

    tot = {v: [0, 0, 0] for v in [f"{a}/{b}" for a, b in variants]}  # evals_sum, exact, n
    print(f"{'K':>4} {'dt':>4} {'N':>7} | " + " ".join(f"{a[:4]}/{b[:2]:>2}" for a, b in variants))
    for K in args.K:
        cr = 1 if K == 2048 else 4
        for dts in args.dt:
            dtype = dtmap[dts]
            for N in args.N:
                if N <= 2 * K:
                    continue
                row_ev = {}
                for (var, im) in variants:
                    evs, exs, ns = [], 0, 0
                    for cfg in args.cfgs:
                        for seed in args.seeds:
                            b = get_bundle(K, dtype, N, cfg=cfg, seed=seed)
                            e, cand, ex = _run(b["logits"][0], b["preIdx"][0], N, K, cr,
                                               dtype, var, init_mode=im, init_q=0.35)
                            evs.append(e); exs += int(ex); ns += 1
                    key = f"{var}/{im}"
                    me = sum(evs)/len(evs)
                    tot[key][0] += sum(evs); tot[key][1] += exs; tot[key][2] += ns
                    row_ev[key] = (me, exs, ns)
                cells = " ".join(f"{row_ev[f'{a}/{b}'][0]:.2f}{'*' if row_ev[f'{a}/{b}'][1]<row_ev[f'{a}/{b}'][2] else ' '}"
                                 for a, b in variants)
                print(f"{K:>4} {dts:>4} {N:>7} | {cells}")
    print("\n=== overall mean evals (lower=better; ! = has inexact cell) ===")
    for (a, b) in variants:
        k = f"{a}/{b}"
        s, ex, n = tot[k]
        flag = "" if ex == n else f"  !INEXACT {ex}/{n}"
        print(f"  {k:20s}: {s/n:.3f} evals   exact {ex}/{n}{flag}")


if __name__ == "__main__":
    main()
