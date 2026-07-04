# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Validate op13 p2_replay (baseline cfg) vs harness/count_gvr_iters on the grid.

Checks that SecantCfg(init_mode="mean") reproduces the kernel-faithful baseline
p2_evals and cand_count (within tie noise) and that the baseline is EXACT (the
candidate superset contains the true top-K) for every (dtype, K, N, cfg, seed).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
from p2_replay import SecantCfg, replay_row  # noqa: E402
from count_gvr_iters import count_iters       # noqa: E402
from synth_data import get_bundle             # noqa: E402

_CR = {512: 4, 1024: 4, 2048: 1}
CFGS = ["beta_shallow", "beta_moderate", "beta_deep"]
N_BY_K = {
    512: [4096, 8192, 16384, 32768, 65536, 131072, 262144],
    1024: [4096, 8192, 16384, 32768, 65536, 131072, 262144],
    2048: [8192, 16384, 32768, 65536, 131072, 262144],
}
SEEDS = [0, 1, 2, 3]

base = SecantCfg(init_mode="mean")
n_tot = n_cand_ok = n_eval_ok = n_exact = 0
worst = []
for K in (512, 1024, 2048):
    cr = _CR[K]
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for N in N_BY_K[K]:
            for cfg in CFGS:
                for s in SEEDS:
                    b = get_bundle(K, dt, N, cfg=cfg, seed=s)
                    lg = b["logits"].to(dt).contiguous()
                    pre = b["preIdx"].contiguous()
                    # reference: kernel-faithful host replay
                    ref = count_iters(lg[0], pre[0], N, K, cr, dt, want_idx=False)
                    got = replay_row(lg[0], pre[0], N, K, cr, dt, base)
                    n_tot += 1
                    # p2_evals must match exactly (same control flow)
                    eval_ok = (got.p2_evals == ref.p2_evals)
                    # cand_count may differ by tie placement at boundary; allow small slack
                    cand_ok = abs(got.cand_count - ref.cand_count) <= 2
                    n_eval_ok += int(eval_ok)
                    n_cand_ok += int(cand_ok)
                    n_exact += int(got.exact)
                    if not (eval_ok and cand_ok):
                        worst.append((K, str(dt), N, cfg, s,
                                      got.p2_evals, ref.p2_evals,
                                      got.cand_count, ref.cand_count))

print(f"cells={n_tot}")
print(f"  p2_evals match : {n_eval_ok}/{n_tot}")
print(f"  cand_count ~eq : {n_cand_ok}/{n_tot}")
print(f"  baseline exact : {n_exact}/{n_tot}")
if worst:
    print("\nMISMATCHES (K,dt,N,cfg,seed | got_ev/ref_ev got_cand/ref_cand):")
    for w in worst[:30]:
        print("  ", w)
else:
    print("\nALL MATCH — replay is baseline-faithful.")
