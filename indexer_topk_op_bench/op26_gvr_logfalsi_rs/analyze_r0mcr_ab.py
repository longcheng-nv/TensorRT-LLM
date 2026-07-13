# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26_r0mcr (mc-port leader rank-scatter P4, iter7) A/B verdict vs op26_r0mc.

Input = results_b200_op26_r0mcr_ab (batch-paired 2-arm nsys sweep,
worst+real x seqlen/bs/bs_hugeN x 3K x 3dtype). Per-cell metric =
us_cold(r0mc) / us_cold(r0mcr)  (>1 = rank-scatter faster). Views:

  - gm by (dtype, K) over the mc DISPATCH REGION (N>=65536 & BS<=64,
    the only cells op26_r0auto ever routes to mc) — the ship gate;
  - gm by (dtype, K) over ALL cells (context);
  - worst-axis focus (the P1b-tax band the 1cta r0f probe targeted);
  - loss cells < 0.98 listed.

Decision rule (mirror of the 1cta dispatch_p4rs_mc_op26 verdict): gate ON
per dtype iff mc-region gm >= 1.005 with no systematic loss band.

Usage: python3 analyze_r0mcr_ab.py [<root>]
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op26_r0mcr_ab"
SUBS = [("seqlen", "seqlen_sweep"), ("bs", "bs_scaling"),
        ("bs_hugeN", "bs_hugeN")]


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def load():
    cells = {}
    for scen_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for _sw, sub in SUBS:
            f = scen_dir / sub / "results.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if "us_cold" not in r:
                    continue
                k = (r["scenario"], r["sweep"], r["K"], r["dtype"],
                     r["N"], r["BS"])
                cells.setdefault(k, {})[r["op"]] = r["us_cold"]
    return cells


def main():
    cells = load()
    pairs = {k: v["op26_r0mc"] / v["op26_r0mcr"] for k, v in cells.items()
             if "op26_r0mc" in v and "op26_r0mcr" in v}
    print(f"paired cells: {len(pairs)}")

    def view(name, sel):
        b = defaultdict(list)
        for k, sp in pairs.items():
            if sel(k):
                b[(k[3], k[2])].append(sp)
        print(f"\n== {name} ==")
        allv = []
        for kk in sorted(b):
            v = b[kk]
            allv += v
            print(f"  {kk[0]:5} K{kk[1]:<5} n={len(v):3} gm={gm(v):.4f} "
                  f"min={min(v):.3f} max={max(v):.3f}")
        if allv:
            print(f"  ALL: n={len(allv)} gm={gm(allv):.4f}")

    view("mc dispatch region (N>=65536 & BS<=64) — SHIP GATE",
         lambda k: k[4] >= 65536 and k[5] <= 64)
    view("mc dispatch region, WORST axis only",
         lambda k: k[0] == "worst" and k[4] >= 65536 and k[5] <= 64)
    view("all cells (context)", lambda k: True)

    losses = sorted((sp, k) for k, sp in pairs.items()
                    if sp < 0.98 and k[4] >= 65536 and k[5] <= 64)
    print(f"\nmc-region loss cells <0.98: {len(losses)}")
    for sp, k in losses[:20]:
        print(f"  {sp:.3f}  {k}")


if __name__ == "__main__":
    main()
