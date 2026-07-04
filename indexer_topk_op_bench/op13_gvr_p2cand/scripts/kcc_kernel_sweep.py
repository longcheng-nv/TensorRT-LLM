# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Kernel-measure the host-recommended kCC sweet-spot configs (eval-optimal
kFTarget) vs baseline on the REAL snap kernel. Reports per-N P1-P4 + ΔTOT.

Configs (per K) come from kcc_host_prepass: kCC=2xK and kCC=3xK with their
min-eval exact kFTarget. Goal: confirm the small/mid-N net win and large-N
neutrality predicted by the +1-eval / big-cand-cut tradeoff.
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from phase_ab import run  # noqa: E402  (build+measure on real kernel)

# per-K: list of (label, kCC, kFTarget); host-prepass eval-optimal picks
CONFIGS = {
    512:  [("kc2x", 1024, 1024), ("kc3x", 1536, 1280)],
    1024: [("kc2x", 2048, 2048), ("kc3x", 3072, 2560)],
    2048: [("kc2x", 4096, 3686), ("kc3x", 6144, 3686)],
}
N_BY_K = {
    512: [4096, 8192, 16384, 32768, 65536, 262144],
    1024: [4096, 8192, 16384, 32768, 65536, 262144],
    2048: [8192, 16384, 32768, 65536, 262144],
}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--dt", default="fp32")
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    K = args.K
    cr_val = 4 if K in (512, 1024) else 1
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dt]
    cfgs = CONFIGS[K]
    print(f"=== K={K} {args.dt} cr={cr_val} : kCC sweet-spot vs baseline (real snap kernel) ===")
    hdr = f"{'N':>8} | {'variant':8s} | {'tot':>6s} {'P1':>5s} {'P2':>6s} {'P3':>6s} {'P4':>6s} | {'dTOT':>7s} {'dP2':>6s} {'dP3+P4':>7s}"
    print(hdr)
    for N in N_BY_K[K]:
        _, base = run(K, dtype, N, cr_val, kC_ovr=None, kFT_ovr=None, reps=args.reps)
        bt = sum(base.values())
        print(f"{N:>8} | {'baseline':8s} | {bt:6.2f} {base['P1']:5.2f} {base['P2']:6.2f} {base['P3']:6.2f} {base['P4']:6.2f} |")
        for label, kcc, kft in cfgs:
            _, v = run(K, dtype, N, cr_val, kC_ovr=kcc, kFT_ovr=kft, reps=args.reps)
            vt = sum(v.values())
            d_tot = vt - bt
            d_p2 = v['P2'] - base['P2']
            d_p34 = (v['P3']+v['P4']) - (base['P3']+base['P4'])
            tag = "WIN" if d_tot < -0.3 else ("loss" if d_tot > 0.3 else "~")
            print(f"{'':>8} | {label:8s} | {vt:6.2f} {v['P1']:5.2f} {v['P2']:6.2f} {v['P3']:6.2f} {v['P4']:6.2f} | "
                  f"{d_tot:+7.2f} {d_p2:+6.2f} {d_p34:+7.2f}  {tag}")
