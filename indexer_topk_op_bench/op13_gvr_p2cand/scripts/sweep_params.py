# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Sweep P2 secant params on the validated host replay.

For each candidate SecantCfg, replay over the full grid and aggregate per
(K, dtype) group:
    p2_evals (mean/median/max), cand_count (mean as xK), exact%, converged%.
A config is ADMISSIBLE only if exact==100% AND converged==100% (no P2 fallback)
across EVERY cell in the group (and we also enforce per-N to expose large-N risk).

Prints, per (K,dtype): baseline vs each admissible config, sorted by mean cand.
"""
import sys
import statistics as st
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
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def candidate_cfgs(K):
    cfgs = [SecantCfg(init_mode="mean")]  # baseline first
    for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        cfgs.append(SecantCfg(init_mode="lerp", init_alpha=a))
    for q in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        cfgs.append(SecantCfg(init_mode="pquantile", init_q=q))
    # narrowed acceptance window (cap candidates), with default mean init
    for kc in (int(1.25 * K), int(1.5 * K), int(2 * K), int(3 * K)):
        cfgs.append(SecantCfg(init_mode="mean", kCC=kc))
    # best-guess combos: good init + narrowed cap + tightened target
    for q in (0.1, 0.2, 0.3):
        for kc in (int(1.5 * K), int(2 * K)):
            cfgs.append(SecantCfg(init_mode="pquantile", init_q=q, kCC=kc, kFTarget=int(1.2 * K)))
    for a in (0.3, 0.4, 0.5):
        for kc in (int(1.5 * K), int(2 * K)):
            cfgs.append(SecantCfg(init_mode="lerp", init_alpha=a, kCC=kc, kFTarget=int(1.2 * K)))
    return cfgs


# Precompute bundles once
_BUN = {}
def bundle(K, dt, N, cfg, s):
    key = (K, dt, N, cfg, s)
    if key not in _BUN:
        _BUN[key] = get_bundle(K, dt, N, cfg=cfg, seed=s)
    return _BUN[key]


def eval_cfg(K, dt, cfg):
    cr = _CR[K]
    evals, cands = [], []
    exact = conv = tot = 0
    # track per-N worst to expose large-N fallback risk
    perN = {}
    for N in N_BY_K[K]:
        for c in CFGS:
            for s in SEEDS:
                b = bundle(K, dt, N, c, s)
                lg = b["logits"].to(dt).contiguous()
                pre = b["preIdx"].contiguous()
                r = replay_row(lg[0], pre[0], N, K, cr, dt, cfg)
                evals.append(r.p2_evals); cands.append(r.cand_count)
                exact += int(r.exact); conv += int(r.converged); tot += 1
                d = perN.setdefault(N, [0, 0, 0])
                d[0] += int(r.exact); d[1] += int(r.converged); d[2] += 1
    return dict(
        ev_mean=st.mean(evals), ev_med=st.median(evals), ev_max=max(evals),
        cand_mean=st.mean(cands), cand_med=st.median(cands), cand_max=max(cands),
        exact=exact, conv=conv, tot=tot, perN=perN,
    )


def main():
    for K in (512, 1024, 2048):
        for dt in DTYPES:
            dn = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[dt]
            print(f"\n================ K={K} {dn} (cr={_CR[K]}) ================")
            rows = []
            base = None
            for cfg in candidate_cfgs(K):
                r = eval_cfg(K, dt, cfg)
                r["tag"] = cfg.tag()
                if cfg.init_mode == "mean" and cfg.kCC is None and cfg.kFTarget is None:
                    base = r
                rows.append(r)
            bcand = base["cand_mean"]
            bev = base["ev_mean"]
            print(f"  BASELINE: p2ev_mean={bev:.2f} cand_mean={bcand:.0f} "
                  f"({bcand/K:.2f}xK) exact={base['exact']}/{base['tot']} conv={base['conv']}/{base['tot']}")
            # admissible = 100% exact + 100% converged
            adm = [r for r in rows if r["exact"] == r["tot"] and r["conv"] == r["tot"]]
            adm.sort(key=lambda r: r["cand_mean"])
            print(f"  -- admissible configs (100% exact + no fallback), sorted by cand_mean --")
            print(f"     {'tag':38s} {'p2ev(mean/max)':>14s} {'cand_mean(xK)':>16s} {'cand_max':>9s}")
            for r in adm[:12]:
                print(f"     {r['tag']:38s} {r['ev_mean']:6.2f}/{r['ev_max']:<4.0f}    "
                      f"{r['cand_mean']:7.0f} ({r['cand_mean']/K:.2f}x)   {r['cand_max']:7.0f}")
            # also show best non-admissible (to see how far we COULD go)
            inadm = [r for r in rows if not (r["exact"] == r["tot"] and r["conv"] == r["tot"])]
            if inadm:
                inadm.sort(key=lambda r: r["cand_mean"])
                r = inadm[0]
                print(f"  (lowest-cand INADMISSIBLE: {r['tag']} cand={r['cand_mean']:.0f} "
                      f"exact={r['exact']}/{r['tot']} conv={r['conv']}/{r['tot']})")


if __name__ == "__main__":
    main()
