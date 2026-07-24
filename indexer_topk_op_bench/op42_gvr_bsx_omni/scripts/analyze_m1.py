# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M1 82-cell screen analysis: parse m1_*.nsys-rep -> results/m1_data.csv
(FRESH file, no dedup vs ab_data.csv), then gm rollups vs the 1.40 bar.

Usage: python3 scripts/analyze_m1.py [--reparse]"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP42 = HERE.parent
BENCH = OP42.parent
sys.path.insert(0, str(BENCH / "report"))

CSV = OP42 / "results" / "m1_data.csv"
FIELDS = ["cell", "BS", "arm", "cold_us", "warm_us", "exact"]


def reparse():
    from parse_nsys_full import parse_rep
    rows = []
    reps = sorted((OP42 / "results" / "nsys").glob("m1_*.nsys-rep"))
    for rep in reps:
        kern = parse_rep(str(rep))
        per = {}
        for rng, us in kern.items():
            parts = rng.split("|")
            if len(parts) != 4:
                continue
            mode, arm, cell, bstag = parts
            per.setdefault((cell, int(bstag[2:]), arm), {})[mode] = us
        exact = {}
        for ep in (OP42 / "results").glob(f"exact_m1_{rep.stem.replace('m1_', '')}*.json"):
            exact.update(json.load(open(ep)))
        for (cell, bs, arm), d in sorted(per.items()):
            rows.append(dict(cell=cell, BS=bs, arm=arm,
                             cold_us=d.get("c"), warm_us=d.get("w"),
                             exact=exact.get(f"{cell}|BS{bs}|{arm}", "")))
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[reparse] {len(reps)} reps -> {len(rows)} rows -> {CSV}")


def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def npad_band(cell):
    sz = cell.split("_")[1]
    v = int(sz[:-1]) * 1024
    if v < 32768:
        return "a:<32k"
    if v < 131072:
        return "b:32-128k"
    if v < 524288:
        return "c:128-512k"
    return "d:>=512k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reparse", action="store_true")
    args = ap.parse_args()
    if args.reparse or not CSV.exists():
        reparse()

    cold = defaultdict(dict)
    for r in csv.DictReader(open(CSV)):
        if r["cold_us"]:
            cold[(r["cell"], int(r["BS"]))][r["arm"]] = float(r["cold_us"])

    sp = {}  # (cell, bs) -> speedup pr/bsx
    for k, d in cold.items():
        if "gvr_pr" in d and "bsx" in d:
            sp[k] = d["gvr_pr"] / d["bsx"]

    cells = sorted({c for c, _ in sp})
    bss = sorted({b for _, b in sp})
    print(f"cells={len(cells)} pairs={len(sp)}  OVERALL gm={gm(sp.values()):.4f} "
          f"min={min(sp.values()):.3f} max={max(sp.values()):.3f}  (bar 1.40)")

    print("\n== gm by BS ==")
    for b in bss:
        v = [s for (c, bb), s in sp.items() if bb == b]
        lo = min(v)
        print(f"  BS{b:<5} gm={gm(v):.3f} min={lo:.3f} n={len(v)}")

    print("\n== gm by model x BS band ==")
    for m in ("flash", "pro", "v32"):
        v = [s for (c, b), s in sp.items() if c.startswith(m)]
        if v:
            print(f"  {m:<6} gm={gm(v):.3f} min={min(v):.3f} n={len(v)}")

    print("\n== gm by npad band x BS ==")
    for band in sorted({npad_band(c) for c in cells}):
        row = [f"  {band:<11}"]
        for b in bss:
            v = [s for (c, bb), s in sp.items() if bb == b and npad_band(c) == band]
            row.append(f"{gm(v):5.2f}" if v else "    -")
        print(" ".join(row))

    weak = sorted((s, c, b) for (c, b), s in sp.items() if s < 0.95)
    print(f"\n== weak pairs (<0.95): {len(weak)}/{len(sp)} ==")
    for s, c, b in weak[:40]:
        print(f"  {c:<22} BS{b:<5} {s:.3f}")
    if len(weak) > 40:
        print(f"  ... +{len(weak) - 40} more")


if __name__ == "__main__":
    main()
