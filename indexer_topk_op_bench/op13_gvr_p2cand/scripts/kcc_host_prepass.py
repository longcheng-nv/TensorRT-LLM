# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Host pre-pass: for each kCC target, find the kFTarget minimizing P2 evals
while staying 100% exact + no fallback. Reports per-N (evals, cand/K) for the
chosen kFTarget, so the kernel sweep only measures eval-optimal configs.
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
DN = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
_BUN = {}


def bun(K, dt, N, cfg, s):
    key = (K, dt, N, cfg, s)
    if key not in _BUN:
        _BUN[key] = get_bundle(K, dt, N, cfg=cfg, seed=s)
    return _BUN[key]


def eval_cfg(K, dt, cfg):
    cr = _CR[K]
    perN_ev, perN_cand = {}, {}
    exact = conv = tot = 0
    for N in N_BY_K[K]:
        evs, cds = [], []
        for c in CFGS:
            for s in SEEDS:
                b = bun(K, dt, N, c, s)
                lg = b["logits"].to(dt).contiguous(); pre = b["preIdx"].contiguous()
                r = replay_row(lg[0], pre[0], N, K, cr, dt, cfg)
                evs.append(r.p2_evals); cds.append(r.cand_count)
                exact += int(r.exact); conv += int(r.converged); tot += 1
        perN_ev[N] = st.mean(evs); perN_cand[N] = st.mean(cds)
    return perN_ev, perN_cand, exact, conv, tot


for K in (512, 1024, 2048):
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        print(f"\n===== K={K} {DN[dt]} =====")
        # baseline
        be, bc, ex, cv, tot = eval_cfg(K, dt, SecantCfg(init_mode="mean"))
        ns = sorted(be)
        print(f"  baseline kFT=default: ev[{' '.join(f'{be[n]:.1f}' for n in ns)}] "
              f"cand/K[{' '.join(f'{bc[n]/K:.1f}' for n in ns)}] exact={ex}/{tot}")
        for mult in (1.5, 2.0, 3.0):
            kcc = int(mult * K)
            # try several kFTarget; pick min-eval that is 100% exact + converged
            best = None
            for ftm in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5):
                ft = min(int(ftm * K), kcc)
                cfg = SecantCfg(init_mode="mean", kCC=kcc, kFTarget=ft)
                pe, pc, ex, cv, tot = eval_cfg(K, dt, cfg)
                ok = (ex == tot and cv == tot)
                meanev = st.mean(list(pe.values()))
                if ok and (best is None or meanev < best[0]):
                    best = (meanev, ft, pe, pc)
            if best is None:
                print(f"  kCC={kcc} ({mult}xK): NO exact+converged kFTarget found")
                continue
            _, ft, pe, pc = best
            print(f"  kCC={kcc} ({mult:.1f}xK) bestFT={ft}: "
                  f"ev[{' '.join(f'{pe[n]:.1f}' for n in ns)}] "
                  f"cand/K[{' '.join(f'{pc[n]/K:.1f}' for n in ns)}]")
