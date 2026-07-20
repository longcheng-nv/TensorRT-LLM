#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op37 L-J go/no-go replay — multi-rung tight-bracket band sizes on real cells.

For each real decode cell: build the R0-style hint ladder (quantiles of the
gathered preIdx hint values), evaluate an M-column rung set, count row
elements >= each rung (torch), find the tightest bracketing pair around K,
and report band = cnt_lo - cnt_hi plus the sure set cnt_hi.

Compares against today's admission: cand = count at the SINGLE admitted rung
(first rung with K <= cnt <= kC, else 'miss'). P4 work today ~ cand; under
L-J ~ band. Prints per-cell and summary ratios.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OPBENCH = HERE.parents[1]
sys.path.insert(0, str(OPBENCH / "harness"))
import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

KC = {512: 3072, 1024: 3072, 2048: 5120}   # kC per K (approx: K512 kernel 3072)

CELLS = [(m, isl) for m in ("flash", "pro") for isl in
         ("128k", "256k", "512k", "1024k")] + \
        [("v32", isl) for isl in ("32k", "64k", "128k", "256k")]
LAYER = {"flash": 22, "pro": 30, "v32": 34}

# candidate rung ladders (quantiles of GATHERED hint values, high->low)
LADDERS = {
    "ship(M2+vseed)": None,                       # baseline marker
    "M4": (0.95, 0.85, 0.6, 0.35),
    "M6": (0.97, 0.9, 0.8, 0.65, 0.5, 0.35),
    "M8": (0.98, 0.94, 0.88, 0.8, 0.7, 0.58, 0.45, 0.3),
}


def main():
    print(f"{'cell':22s} {'K':>5} {'N':>7} {'hit':>5} | ladder  cand_today  sure  band  band/cand")
    agg = {}
    for m, isl in CELLS:
        RD = RV32 if m == "v32" else RV4
        try:
            b = RD.get_bundle(m, isl, LAYER[m], "fp32")
        except Exception as e:
            print(f"{m}/{isl}: SKIP {e}")
            continue
        row = b["logits"][0, :b["N"]].float().cuda()
        pre = b["preIdx"][0].long().cuda().clamp(min=0)
        K, N = b["K"], b["N"]
        gathered = row[pre]                       # R0 hint values
        pmean = gathered.mean()                   # vseed column analogue
        # today's admission: rungs = quantiles (0.85,[0.35]) + vseed pmean
        qs = (0.85,) if K in (512, 1024) else (0.6, 0.35)
        rungs_today = [torch.quantile(gathered, q).item() for q in qs] + [pmean.item()]
        cnts = sorted(int((row >= t).sum()) for t in rungs_today)
        cand_today = None
        for c in cnts:
            if K <= c <= KC[K]:
                cand_today = c
                break
        miss = cand_today is None
        if miss:
            cand_today = max(c for c in cnts if c) if cnts else N  # fallback scale
        for name, qs2 in LADDERS.items():
            if qs2 is None:
                continue
            rungs = [torch.quantile(gathered, q).item() for q in qs2] + [pmean.item()]
            pairs = sorted((int((row >= t).sum())) for t in rungs)
            lo = min((c for c in pairs if c >= K), default=None)
            hi = max((c for c in pairs if c < K), default=0)
            if lo is None:
                lo = N  # all rungs above K-th value -> band = full miss
            band = lo - hi
            key = name
            agg.setdefault(key, []).append((band, cand_today))
            print(f"{m+'/'+isl:22s} {K:>5} {N:>7} {b['hit_rate']:.2f} | {name:6s} "
                  f"{cand_today:>9}{'*' if miss else ' '} {hi:>5} {band:>6} "
                  f"{band/max(cand_today,1):>8.2f}")
    print("\n== summary (median band / median cand_today, lower better)")
    import statistics
    for k, v in agg.items():
        bands = [x[0] for x in v]
        cands = [x[1] for x in v]
        print(f"{k:6s} med_band={statistics.median(bands):7.0f} "
              f"med_cand={statistics.median(cands):7.0f} "
              f"cells_band<=1.5K={sum(1 for b2 in bands if b2 <= 1.5*2048)}/{len(bands)}")


if __name__ == "__main__":
    main()
