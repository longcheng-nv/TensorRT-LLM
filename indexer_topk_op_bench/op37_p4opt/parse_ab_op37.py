# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate op37 A/B nsys reps -> per-cell per-arm us + base/arm speedups.

  python3 parse_ab_op37.py --tag ab1
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

ARMS = ["base", "d2a", "d2b", "d2ab", "d1a", "all"]


def rung(cs, N):
    if cs == 1:
        return "cs1-small" if N <= 8448 else "cs1-mid"
    return f"cs{cs}"


def gm(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ab1")
    args = ap.parse_args()

    us = {}
    meta = {}
    for g in (0, 1):
        rep = HERE / "nsys_reps" / f"ab37_{args.tag}_g{g}.nsys-rep"
        if rep.exists():
            for rng, t in parse_rep(str(rep)).items():
                _, arm, uuid = rng.split("|", 2)
                us[(uuid, arm)] = t
        for r in csv.DictReader(open(HERE / f"ab37_{args.tag}_g{g}.csv")):
            meta[r["uuid"]] = (int(r["cs"]), int(r["N"]), int(r["K"]))

    cells = sorted({u for u, _ in us})
    rows = []
    print(f"{'cell':24s} {'cs':>3s} " +
          " ".join(f"{a:>7s}" for a in ARMS) +
          "   speedup base/arm: " + " ".join(f"{a:>6s}" for a in ARMS[1:]))
    for u in cells:
        cs, N, K = meta.get(u, (0, 0, 0))
        t = {a: us.get((u, a)) for a in ARMS}
        if not t["base"]:
            continue
        sp = {a: (t["base"] / t[a]) if t[a] else None for a in ARMS[1:]}
        rows.append(dict(uuid=u, cs=cs, N=N, K=K, rung=rung(cs, N),
                         **{f"us_{a}": round(t[a], 3) if t[a] else None
                            for a in ARMS},
                         **{f"sp_{a}": round(sp[a], 4) if sp[a] else None
                            for a in ARMS[1:]}))
        print(f"{u:24s} {cs:>3d} " +
              " ".join(f"{t[a]:7.2f}" if t[a] else "      -" for a in ARMS) +
              "                     " +
              " ".join(f"{sp[a]:6.3f}" if sp[a] else "     -" for a in ARMS[1:]))

    with open(HERE / f"ab37_{args.tag}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\ngeomean base/arm by rung:")
    by = defaultdict(list)
    for r in rows:
        by[r["rung"]].append(r)
        by["ALL"].append(r)
    for rg in ["cs1-small", "cs1-mid", "cs4", "cs8", "ALL"]:
        g = by.get(rg)
        if not g:
            continue
        line = " ".join(
            f"{a}={gm([r[f'sp_{a}'] for r in g]):.4f}" for a in ARMS[1:]
            if gm([r[f'sp_{a}'] for r in g]))
        worst = {a: min((r[f"sp_{a}"] for r in g if r[f"sp_{a}"]), default=None)
                 for a in ARMS[1:]}
        print(f" {rg:9s} n={len(g):2d}  {line}")
        print(f"           worst: " + " ".join(
            f"{a}={worst[a]:.3f}" for a in ARMS[1:] if worst[a]))


if __name__ == "__main__":
    main()
