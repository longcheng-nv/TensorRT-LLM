# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pareto (p2_evals vs cand_count) extraction on the host replay.

Focus: find configs that cut cand WITHOUT adding P2 evals (init-only lever),
and expose the per-N eval breakdown (large-N is eval-dominated, so any extra
eval there is expensive). Restricted to admissible (100% exact, no fallback).
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
DN = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}


def cfgs_for(K):
    out = [("baseline", SecantCfg(init_mode="mean"))]
    # init-only (no kCC narrowing) — candidate reduction at ~baseline evals
    for a in (0.2, 0.3, 0.4, 0.5, 0.6):
        out.append((f"lerp_a{a}", SecantCfg(init_mode="lerp", init_alpha=a)))
    for q in (0.1, 0.2, 0.3, 0.4):
        out.append((f"pq_q{q}", SecantCfg(init_mode="pquantile", init_q=q)))
    # init + mild kCC narrowing (moderate cand cut, moderate eval cost)
    for a in (0.4, 0.5):
        for kc in (int(2 * K), int(3 * K)):
            out.append((f"lerp_a{a}_kc{kc}", SecantCfg(init_mode="lerp", init_alpha=a, kCC=kc)))
    # aggressive kCC narrowing (max cand cut, high eval cost)
    for kc in (int(1.25 * K), int(1.5 * K)):
        out.append((f"kc{kc}", SecantCfg(init_mode="mean", kCC=kc)))
        out.append((f"lerp_a0.5_kc{kc}", SecantCfg(init_mode="lerp", init_alpha=0.5, kCC=kc)))
    return out


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
    perN_ev = {N: [] for N in N_BY_K[K]}
    perN_cand = {N: [] for N in N_BY_K[K]}
    for N in N_BY_K[K]:
        for c in CFGS:
            for s in SEEDS:
                b = bundle(K, dt, N, c, s)
                lg = b["logits"].to(dt).contiguous(); pre = b["preIdx"].contiguous()
                r = replay_row(lg[0], pre[0], N, K, cr, dt, cfg)
                evals.append(r.p2_evals); cands.append(r.cand_count)
                perN_ev[N].append(r.p2_evals); perN_cand[N].append(r.cand_count)
                exact += int(r.exact); conv += int(r.converged); tot += 1
    return dict(ev=st.mean(evals), cand=st.mean(cands),
                exact=exact, conv=conv, tot=tot,
                perN_ev={N: st.mean(v) for N, v in perN_ev.items()},
                perN_cand={N: st.mean(v) for N, v in perN_cand.items()})


for K in (512, 1024, 2048):
    for dt in DTYPES:
        print(f"\n===== K={K} {DN[dt]} =====")
        results = []
        for name, cfg in cfgs_for(K):
            r = eval_cfg(K, dt, cfg); r["name"] = name
            r["adm"] = (r["exact"] == r["tot"] and r["conv"] == r["tot"])
            results.append(r)
        base = next(r for r in results if r["name"] == "baseline")
        print(f"  {'config':20s} {'p2ev':>5s} {'cand/K':>7s} {'adm':>4s}  per-N p2ev [4K 8K 16K 32K 64K 128K 256K]")
        for r in results:
            ns = sorted(r["perN_ev"])
            evs = " ".join(f"{r['perN_ev'][n]:.1f}" for n in ns)
            mark = "ok" if r["adm"] else "XX"
            star = " *FREE" if (r["adm"] and r["ev"] <= base["ev"] + 0.3 and r["cand"] < base["cand"] * 0.9) else ""
            print(f"  {r['name']:20s} {r['ev']:5.2f} {r['cand']/K:6.2f}x {mark:>4s}  [{evs}]{star}")
