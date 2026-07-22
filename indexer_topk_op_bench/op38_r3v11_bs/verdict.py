# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op38 verdict: join sharded nsys candidate times with the §7b report pr.

Outputs v2_data.csv (cell, K, N, BS, cand_us, pr_us, speedup, exact) and a
summary: BS>1 geomean / arith-mean / min / #<1.0 (goal: mean >= 1.8, all >= 1.0),
plus BS=1 col for reference."""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()

    kern, exact = {}, {}
    for g in range(args.shards):
        rep = HERE / "nsys_reps" / f"{args.tag}_s{g}.nsys-rep"
        if rep.exists():
            kern.update(parse_rep(str(rep)))
        ep = HERE / f"exact_{args.tag}_s{g}.json"
        if ep.exists():
            exact.update(json.load(open(ep)))

    pr = {}
    meta = {}
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" /
                                 "bs_real_layers.csv")):
        cname = f"{r['model']}_{r['isl']}_L{int(r['L']):02d}"
        pr[(cname, int(r["BS"]))] = float(r["pr"])
        meta[cname] = (int(r["N"]), r["model"])

    rows = []
    for rng, us in kern.items():
        parts = rng.split("|")
        if len(parts) != 4 or parts[0] != "c":
            continue
        cname, bs = parts[2], int(parts[3][2:])
        p = pr.get((cname, bs))
        ex = exact.get(f"{cname}|BS{bs}", "?")
        rows.append(dict(cell=cname, N=meta.get(cname, (0, "?"))[0],
                         model=meta.get(cname, (0, "?"))[1], BS=bs,
                         cand_us=round(us, 3), pr_us=p,
                         speedup=round(p / us, 4) if p else None,
                         exact="OK" if ex == "OK" else ex))
    rows.sort(key=lambda r: (r["model"], r["N"], r["cell"], r["BS"]))
    with open(HERE / f"{args.tag}_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gt1 = [r for r in rows if r["BS"] > 1 and r["speedup"]]
    sp = [r["speedup"] for r in gt1]
    n_bad = sum(1 for x in sp if x < 1.0)
    n_inexact = sum(1 for r in rows if r["exact"] != "OK")
    print(f"cells parsed: {len({r['cell'] for r in rows})}, "
          f"BS>1 cases: {len(gt1)}, inexact: {n_inexact}")
    print(f"BS>1 geomean {statistics.geometric_mean(sp):.4f}  "
          f"mean {statistics.mean(sp):.4f}  min {min(sp):.4f}  "
          f"<1.0: {n_bad}/{len(sp)}")
    bs1 = [r["speedup"] for r in rows if r["BS"] == 1 and r["speedup"]]
    if bs1:
        print(f"BS=1 (vs report pr, ref only) geomean "
              f"{statistics.geometric_mean(bs1):.4f} min {min(bs1):.4f}")
    print("\nworst 25 BS>1 cases:")
    for r in sorted(gt1, key=lambda r: r["speedup"])[:25]:
        print(f"  {r['cell']:22s} BS{r['BS']:5d} cand={r['cand_us']:9.2f} "
              f"pr={r['pr_us']:9.2f} x{r['speedup']:.3f}")
    # per-BS breakdown
    print("\nper-BS geomean/min:")
    for bs in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        v = [r["speedup"] for r in gt1 if r["BS"] == bs]
        if v:
            print(f"  BS{bs:5d}: gm {statistics.geometric_mean(v):.3f} "
                  f"min {min(v):.3f} <1.0: {sum(1 for x in v if x < 1)}/{len(v)}")


if __name__ == "__main__":
    main()
