# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iter9 verdict: parse iter9ab3_*.nsys-rep (fresh CSV, no ab_data dedup),
compare per re-measured cell vs the M1/iter8 state, and project the full
82-cell grid gm (M1 base, iter8b patch, iter9 patch — freshest wins).

Usage: python3 scripts/analyze_i9.py [--reparse]"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP42 = HERE.parent
BENCH = OP42.parent
sys.path.insert(0, str(BENCH / "report"))

CSV = OP42 / "results" / "i9_data.csv"
FIELDS = ["cell", "BS", "arm", "cold_us", "warm_us", "exact"]


def reparse_tag(tag):
    from parse_nsys_full import parse_rep
    rows = []
    for rep in sorted((OP42 / "results" / "nsys").glob(f"{tag}_*.nsys-rep")):
        kern = parse_rep(str(rep))
        per = defaultdict(dict)
        for rng, us in kern.items():
            parts = rng.split("|")
            if len(parts) != 4:
                continue
            mode, arm, cell, bstag = parts
            per[(cell, int(bstag[2:]))][f"{mode}|{arm}"] = us
        for (cell, bs), d in sorted(per.items()):
            for arm in ("gvr_pr", "bsx"):
                if f"c|{arm}" in d:
                    rows.append(dict(cell=cell, BS=bs, arm=arm,
                                     cold_us=d[f"c|{arm}"],
                                     warm_us=d.get(f"w|{arm}", ""), exact=1))
    return rows


def sp_table(rows):
    t = defaultdict(dict)
    for r in rows:
        t[(r["cell"], int(r["BS"]))][r["arm"]] = float(r["cold_us"])
    return {k: v["gvr_pr"] / v["bsx"] for k, v in t.items()
            if "gvr_pr" in v and "bsx" in v}


def gm(vals):
    vals = list(vals)
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reparse", action="store_true")
    args = ap.parse_args()

    if args.reparse or not CSV.exists():
        rows = reparse_tag("iter9ab3")
        with open(CSV, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[i9] parsed {len(rows)} rows -> {CSV}")

    m1 = sp_table(list(csv.DictReader(open(OP42 / "results" / "m1_data.csv"))))
    i8 = sp_table(reparse_tag("iter8b"))
    i9 = sp_table(list(csv.DictReader(open(CSV))))
    b5 = sp_table(reparse_tag("iter9b5"))  # B5 delta reps override iter9ab3

    proj = dict(m1)
    proj.update(i8)
    pre_i9 = dict(proj)          # iter8-projected state (the 1.3531 table)
    proj.update(i9)
    if b5:
        cells_b5 = sorted({c for c, _ in b5})
        print(f"\n== iter9b5 delta cells (override AB3): {cells_b5}")
        for c in cells_b5:
            pairs = sorted(bs for cc, bs in b5 if cc == c)
            print(f"  {c:18s} " + " ".join(
                f"BS{bs}:{proj.get((c, bs), float('nan')):.2f}->{b5[(c, bs)]:.2f}"
                for bs in pairs))
        i9.update(b5)
        proj.update(b5)

    print(f"\n== iter9 re-measured cells (nsys cold, vs pre-iter9 state) ==")
    cells = sorted({c for c, _ in i9})
    for c in cells:
        pairs = sorted(bs for cc, bs in i9 if cc == c)
        old = [pre_i9.get((c, bs)) for bs in pairs]
        new = [i9[(c, bs)] for bs in pairs]
        og = gm([v for v in old if v]) if any(old) else float("nan")
        print(f"  {c:18s} gm {og:6.3f} -> {gm(new):6.3f}   " +
              " ".join(f"BS{bs}:{pre_i9.get((c, bs), float('nan')):.2f}->"
                       f"{i9[(c, bs)]:.2f}" for bs in pairs))

    for name, t in (("pre-iter9 (M1+iter8b)", pre_i9), ("iter9 projected", proj)):
        weak = sum(1 for v in t.values() if v < 0.95)
        print(f"\n== {name}: pairs={len(t)} gm={gm(t.values()):.4f} "
              f"min={min(t.values()):.3f} weak(<0.95)={weak}")
    by_bs = defaultdict(list)
    for (c, bs), v in proj.items():
        by_bs[bs].append(v)
    print("  by BS: " + " ".join(f"{bs}:{gm(v):.2f}"
                                 for bs, v in sorted(by_bs.items())))


if __name__ == "__main__":
    main()
