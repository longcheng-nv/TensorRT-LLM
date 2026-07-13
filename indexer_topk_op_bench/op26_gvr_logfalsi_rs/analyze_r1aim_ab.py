# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 backlog-2 verdict: K2048 fp32 edge-aim R1 paired nsys A/B.

Input = results_b200_op26_r1aim_ab (sweep_r1aim.py batches). Per-cell
metric = us_cold(center) / us_cold(edge)  (>1 = edge faster). R1 only
fires on R0 miss, so real-axis deltas should be noise; worst axis is the
band with potential meat. Ship bar mirrors the other mc ablations:
gm >= 1.005 in a coherent band with no <0.98 loss cells.

Usage: python3 analyze_r1aim_ab.py [<root>]
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op26_r1aim_ab"


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def main():
    cells = {}
    for scen_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for jf in sorted(scen_dir.glob("results_K*.jsonl")):
            kd = jf.stem[len("results_K"):]
            rep = scen_dir / "nsys_reps" / f"r1aim_K{kd}.nsys-rep"
            kern = parse_rep(rep) if rep.exists() else {}
            for line in jf.read_text().splitlines():
                r = json.loads(line)
                if "error" in r:
                    print(f"  ERROR cell: {r['op']} N{r['N']} BS{r['BS']}: "
                          f"{r['error']}")
                    continue
                us = kern.get(r["range_cold"])
                if us is None:
                    continue
                k = (r["scenario"], r["K"], r["dtype"], r["N"], r["BS"])
                cells.setdefault(k, {})[r["op"]] = us
                if r.get("exact") == "FAIL":
                    print(f"  EXACT FAIL: {k} {r['op']}")

    pairs = {k: v["center"] / v["edge"] for k, v in cells.items()
             if "center" in v and "edge" in v}
    print(f"paired cells: {len(pairs)}")
    by_scen = defaultdict(list)
    for k, sp in sorted(pairs.items()):
        by_scen[k[0]].append(sp)
        print(f"  {k[0]:5} K{k[1]} {k[2]:5} N{k[3]:>7} BS{k[4]:<3} "
              f"center/edge={sp:.4f}")
    print()
    for sc, v in sorted(by_scen.items()):
        print(f"{sc:5}: n={len(v)} gm={gm(v):.4f} "
              f"min={min(v):.3f} max={max(v):.3f}")
    allv = [sp for v in by_scen.values() for sp in v]
    if allv:
        print(f"ALL  : n={len(allv)} gm={gm(allv):.4f}")


if __name__ == "__main__":
    main()
