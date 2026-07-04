# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""iter8a: host-replay A/B of cheaper P2 root-finders.

For each K and each candidate window (baseline kCC, kc2x, kc3x), compare the
interp modes {linear, logcount, illinois, logillinois} on P2 eval count at
100% exactness + convergence. kFTarget is swept per (window, mode) and the
eval-optimal value picked (same protocol as kcc_host_prepass.py).

Success bar (from ITERATIONS.md iter-8 plan):
  * narrow-window evals <= baseline-linear evals (the +1 eval tax -> 0), esp.
    at large N (131K/262K) where each eval is a full-N scan; and/or
  * baseline-window evals reduced (K2048 baseline is 3.0-3.25).

Usage: python3 scripts/rootfinder_sweep.py [--dt fp32] [--K 512 1024 2048]
Writes results/rootfinder_sweep_<dt>.json + prints a per-N table.
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
from p2_replay import SecantCfg, replay_row  # noqa: E402
from synth_data import get_bundle             # noqa: E402

_CR = {512: 4, 1024: 4, 2048: 1}
CFGS = ["beta_shallow", "beta_moderate", "beta_deep"]
N_BY_K = {
    512: [4096, 8192, 16384, 32768, 65536, 131072, 262144],
    1024: [4096, 8192, 16384, 32768, 65536, 131072, 262144],
    2048: [8192, 16384, 32768, 65536, 131072, 262144],
}
SEEDS = [0, 1, 2, 3]
_DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
MODES = ["linear", "logcount", "illinois", "logillinois"]

# iter-3 eval-optimal (kCC, kFT) winners for the LINEAR reference
_LIN_NARROW = {
    512: {"kc2x": (1024, 1024), "kc3x": (1536, 1280)},
    1024: {"kc2x": (2048, 2048), "kc3x": (3072, 2560)},
    2048: {"kc2x": (4096, 3686), "kc3x": (6144, 3686)},
}

_BUN = {}


def bun(K, dt, N, cfg, s):
    key = (K, dt, N, cfg, s)
    if key not in _BUN:
        _BUN[key] = get_bundle(K, dt, N, cfg=cfg, seed=s)
    return _BUN[key]


def eval_cfg(K, dt, cfg):
    """-> (perN mean evals, perN mean cand, exact, converged, tot)."""
    cr = _CR[K]
    perN_ev, perN_cand = {}, {}
    exact = conv = tot = 0
    for N in N_BY_K[K]:
        evs, cds = [], []
        for c in CFGS:
            for s in SEEDS:
                b = bun(K, dt, N, c, s)
                lg = b["logits"].to(dt).contiguous()
                pre = b["preIdx"].contiguous()
                r = replay_row(lg[0], pre[0], N, K, cr, dt, cfg)
                evs.append(r.p2_evals)
                cds.append(r.cand_count)
                exact += int(r.exact)
                conv += int(r.converged)
                tot += 1
        perN_ev[N] = st.mean(evs)
        perN_cand[N] = st.mean(cds)
    return perN_ev, perN_cand, exact, conv, tot


def ft_candidates(K, kcc):
    """kFTarget sweep grid: arithmetic multiples + the geometric mid of [K,kcc]."""
    cands = {min(int(m * K), kcc) for m in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0)}
    cands.add(int(math.sqrt(K * kcc)))          # geometric mid (natural log-mode aim)
    cands.add(int(0.8 * kcc))
    return sorted(c for c in cands if K <= c <= kcc)


def best_ft(K, dt, kcc, mode):
    """Eval-optimal kFTarget for (window kcc, interp mode); require 100% exact+conv.
    Tie-break: fewer large-N evals, then smaller mean cand."""
    best = None
    for ft in ft_candidates(K, kcc):
        cfg = SecantCfg(kCC=kcc, kFTarget=ft, interp_mode=mode)
        pe, pc, ex, cv, tot = eval_cfg(K, dt, cfg)
        if ex != tot or cv != tot:
            continue
        ns = N_BY_K[K]
        meanev = st.mean(list(pe.values()))
        bigev = st.mean([pe[n] for n in ns if n >= 131072])
        meancand = st.mean(list(pc.values()))
        key = (meanev, bigev, meancand)
        if best is None or key < best[0]:
            best = (key, ft, pe, pc)
    return best


def fmt_row(tag, pe, pc, K, ns):
    ev = " ".join(f"{pe[n]:4.2f}" for n in ns)
    cd = " ".join(f"{pc[n] / K:4.2f}" for n in ns)
    return f"  {tag:<28s} ev[{ev}]  cand/K[{cd}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", default="fp32", choices=list(_DT))
    ap.add_argument("--K", type=int, nargs="+", default=[512, 1024, 2048])
    args = ap.parse_args()
    dt = _DT[args.dt]

    out = {}
    for K in args.K:
        ns = N_BY_K[K]
        gp_defaults = SecantCfg()  # default kCC/kFT
        print(f"\n===== K={K} {args.dt}  (N: {' '.join(str(n) for n in ns)}) =====")
        rows = {}

        # ---- baseline window (default kCC), all modes ----
        pe, pc, ex, cv, tot = eval_cfg(K, dt, gp_defaults)
        assert ex == tot and cv == tot, f"baseline linear not exact?! {ex}/{tot}"
        print(fmt_row("base-lin (default kFT)", pe, pc, K, ns))
        rows["base-lin"] = dict(kft="default", ev=pe, cand=pc)

        from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrParams
        gp = GvrParams.get(
            {torch.float32: "float32", torch.bfloat16: "bfloat16",
             torch.float16: "float16"}[dt], K, _CR[K])
        for mode in MODES[1:]:
            b = best_ft(K, dt, gp.kC, mode)
            if b is None:
                print(f"  base-{mode}: NO exact+conv kFT")
                continue
            _, ft, pe, pc = b
            print(fmt_row(f"base-{mode} ft={ft}", pe, pc, K, ns))
            rows[f"base-{mode}"] = dict(kft=ft, ev=pe, cand=pc)

        # ---- narrow windows ----
        for wname, (kcc, lin_ft) in _LIN_NARROW[K].items():
            cfg = SecantCfg(kCC=kcc, kFTarget=lin_ft, interp_mode="linear")
            pe, pc, ex, cv, tot = eval_cfg(K, dt, cfg)
            ok = "" if (ex == tot and cv == tot) else f"  !!exact {ex}/{tot} conv {cv}/{tot}"
            print(fmt_row(f"{wname}-lin ft={lin_ft}", pe, pc, K, ns) + ok)
            rows[f"{wname}-lin"] = dict(kft=lin_ft, ev=pe, cand=pc, ok=ok == "")
            for mode in MODES[1:]:
                b = best_ft(K, dt, kcc, mode)
                if b is None:
                    print(f"  {wname}-{mode}: NO exact+conv kFT")
                    continue
                _, ft, pe, pc = b
                print(fmt_row(f"{wname}-{mode} ft={ft}", pe, pc, K, ns))
                rows[f"{wname}-{mode}"] = dict(kft=ft, ev=pe, cand=pc)
        out[K] = rows

    res = _HERE.parent / "results" / f"rootfinder_sweep_{args.dt}.json"
    res.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {res}")


if __name__ == "__main__":
    main()
