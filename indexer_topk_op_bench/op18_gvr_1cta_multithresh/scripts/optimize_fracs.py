# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# CDF-aware round-1 placement optimizer (multi-seed): for each (K, N, M), pick
# M fracs lambda in [0,1] (thr = pmin + lambda*(pmax-pmin)) so that, across
# seeds, (a) at least one threshold lands count in [K, kC] (no done=2), and
# (b) the tightest count >= K is minimized. Uses a per-seed lambda->count curve
# (GPU searchsorted); targets a geometric count ladder mapped through the
# median curve, then validates across seeds.
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
import synth_data  # noqa: E402

KC = {512: 5120, 1024: 5120, 2048: 6144}
CR = {512: 4, 1024: 4, 2048: 1}
SEEDS = (42, 0, 1, 2, 3)


def row_state(K, N, seed):
    b = synth_data.get_bundle(K, torch.float32, N, seed=seed)
    logits, pre = b["logits"][0].float(), b["preIdx"][0].long()
    off = 1 if CR[K] == 1 else 0
    idx = pre + off
    idx = idx[(idx >= 0) & (idx < logits.shape[0])]
    v = logits[idx]
    pmin, pmax = v.min().item(), v.max().item()
    sv_neg = torch.sort(-logits).values  # ascending -v for searchsorted
    return pmin, pmax, sv_neg


def count_at(sv_neg, thr):
    return int(torch.searchsorted(sv_neg, -thr, right=True).item())


def lam_curve(pmin, pmax, sv_neg, grid=512):
    lams = [i / (grid - 1) for i in range(grid)]
    cnts = [count_at(sv_neg, pmin + l * (pmax - pmin)) for l in lams]
    return lams, cnts


def lam_for_target(lams, cnts, target):
    # counts non-increasing in lambda; find lambda whose count is closest <= target
    best_l, best_c = 0.0, cnts[0]
    for l, c in zip(lams, cnts):
        if abs(c - target) < abs(best_c - target):
            best_l, best_c = l, c
    return best_l


def eval_fracs(states, fracs, K):
    """Across seeds: tightest count>=K per seed; None if no thr has count>=K
    (falls back to pmin anchor... caller must include 0.0). done2 if tightest>kC."""
    tight, done2 = [], 0
    for pmin, pmax, sv in states:
        cnts = [count_at(sv, pmin + f * (pmax - pmin)) for f in fracs]
        ge = [c for c in cnts if c >= K]
        t = min(ge) if ge else count_at(sv, pmin)
        if t > KC[K]:
            done2 += 1
        tight.append(t)
    return tight, done2


def optimize(K, N, M, ladder_scales):
    states = [row_state(K, N, s) for s in SEEDS]
    # median lambda->count curve
    lams, _ = lam_curve(*states[0])
    curves = []
    for st in states:
        _, c = lam_curve(*st)
        curves.append(c)
    med = [sorted(col)[len(col) // 2] for col in zip(*curves)]
    fracs = [0.0]  # pmin safety anchor (count>=K guaranteed)
    for s in ladder_scales[: M - 1]:
        fracs.append(lam_for_target(lams, med, int(K * s)))
    fracs = sorted(set(round(f, 4) for f in fracs))
    tight, done2 = eval_fracs(states, fracs, K)
    return fracs, tight, done2


if __name__ == "__main__":
    LADDERS = {
        2: [1.35],
        3: [1.25, 2.2],
        4: [1.2, 1.8, 3.2],
        6: [1.15, 1.5, 2.2, 3.4, 5.0],
        8: [1.1, 1.3, 1.6, 2.1, 2.9, 4.2, 6.0],
    }
    out = {}
    for K in (512, 1024, 2048):
        for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            if N <= 2 * K:
                continue
            for M in (2, 3, 4, 6, 8):
                fracs, tight, done2 = optimize(K, N, M, LADDERS[M])
                key = f"{K}_{N}_{M}"
                out[key] = {"fracs": fracs, "tight": tight, "done2": done2}
                print(f"K={K:4d} N={N:6d} M={M}: fracs={['%.3f' % f for f in fracs]} "
                      f"tight={tight} done2={done2}", flush=True)
    with open(_HERE.parent / "results" / "fracs_table.json", "w") as f:
        json.dump(out, f, indent=1)
