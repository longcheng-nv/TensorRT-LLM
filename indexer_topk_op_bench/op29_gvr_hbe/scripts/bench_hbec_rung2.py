#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""HBE-C rung-2 microbench driver (DESIGN §6): per-round latency of the
cluster reduce patterns at BS=1 (one 8-CTA cluster), via the slope of
t(rounds) — launch overhead and pool-entry cost cancel in the fit.

Modes (see hbec_rung2.cu):
  0 sync-only | 1 dense 1024-bin all-reduce (stock TopKCluster Phase 1.5)
  2 dense 4096-bin | 3 M×8 scalar reduce | 4 = 3 + sparse mini-hist

Decision output: chain saving per row = t(1) - (t(3)+minihist part of
t(4)) compared against the BS=1 cell totals (10.9-33.3us, results_b200_op29).

Usage: CUDA_VISIBLE_DEVICES=<g> python3 bench_hbec_rung2.py [--json out]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src" / "gvr29"
_BUILD = _HERE.parents[1] / "_build" / "hbec_rung2"
_BUILD.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

MOD = load(
    name="hbec_rung2",
    sources=[str(_SRC / "hbec_rung2.cu")],
    extra_cuda_cflags=["-O3", "-std=c++20",
                       "-gencode=arch=compute_100f,code=sm_100f"],
    extra_cflags=["-O3", "-std=c++20"],
    build_directory=str(_BUILD),
    verbose=False,
)

ROUNDS = (16, 64, 256)
REPS = 200
MODES = {0: "sync_only", 1: "dense_1024bin", 2: "dense_4096bin",
         3: "scalars_Mx8", 4: "scalars+minihist", 5: "scalars+localMH"}
# cand_per_cta sweeps (x8 CTAs = row totals 8K/16K/32K)
CANDS = {4: (1024, 2048, 4096), 5: (1024, 2048, 4096)}


def time_one(mode, rounds, cand, out):
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(20):
        MOD.run(mode, rounds, cand, out)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        st.record()
        MOD.run(mode, rounds, cand, out)
        en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en) * 1e3)   # us
    return float(np.median(ts))


def slope_us(mode, cand=0):
    out = torch.zeros(8, dtype=torch.uint32, device="cuda")
    pts = [(r, time_one(mode, r, cand, out)) for r in ROUNDS]
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    a, b = np.polyfit(x, y, 1)
    return a, b, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    torch.cuda.init()
    res = {}
    print(f"{'mode':>18} {'cand/CTA':>8} {'us/round':>9} {'intercept':>9} "
          f"{'raw t(16,64,256) us':>28}")
    for mode, name in MODES.items():
        for cand in CANDS.get(mode, (0,)):
            a, b, pts = slope_us(mode, cand)
            key = name if not cand else f"{name}_c{cand}"
            res[key] = {"us_per_round": round(a, 4),
                        "intercept_us": round(b, 2),
                        "raw": [(r, round(t, 1)) for r, t in pts]}
            print(f"{name:>18} {cand:>8} {a:>9.3f} {b:>9.1f}   "
                  f"{[round(t,1) for _, t in pts]}")

    sync = res["sync_only"]["us_per_round"]
    d1024 = res["dense_1024bin"]["us_per_round"]
    sc = res["scalars_Mx8"]["us_per_round"]
    print("\n== net (minus sync-only baseline) ==")
    print(f"  dense 1024-bin all-reduce : {d1024-sync:6.3f} us")
    print(f"  dense 4096-bin all-reduce : "
          f"{res['dense_4096bin']['us_per_round']-sync:6.3f} us")
    print(f"  M x 8 scalar reduce       : {sc-sync:6.3f} us")
    for c in CANDS[4]:
        mh = res[f"scalars+minihist_c{c}"]["us_per_round"]
        print(f"  +minihist {c*8:>5} atomics   : {mh-sync:6.3f} us "
              f"(minihist part {mh-sc:6.3f})")
    for c in CANDS[5]:
        mh = res[f"scalars+localMH_c{c}"]["us_per_round"]
        print(f"  +localMH  {c*8:>5} cands     : {mh-sync:6.3f} us "
              f"(localMH part {mh-sc:6.3f})")
    print(f"\n  chain saving (dense1024 -> Mx8 scalars): "
          f"{d1024-sc:6.3f} us/row")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
