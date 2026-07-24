# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse op42 nsys reps -> results/ab_data.csv.

Range names (house protocol): {c,w}|<arm>|<cell>|BS<n>.
us = summed in-range kernel time / instances, evict kernels excluded."""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP42 = HERE.parent
BENCH = OP42.parent
sys.path.insert(0, str(BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

FIELDS = ["cell", "BS", "arm", "cold_us", "warm_us", "exact", "speedup_cold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, nargs="+")
    ap.add_argument("--csv", default=str(OP42 / "results" / "ab_data.csv"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    seen = set()
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            seen.add((r["cell"], r["BS"], r["arm"]))
    write_header = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if write_header:
        w.writeheader()

    for rep in args.rep:
        kern = parse_rep(rep)  # {range: us_per_call}
        per = {}
        for rng, us in kern.items():
            parts = rng.split("|")
            if len(parts) != 4:
                continue
            mode, arm, cell, bstag = parts
            per.setdefault((cell, int(bstag[2:])), {})[f"{mode}|{arm}"] = us
        cells = {c for c, _ in per}
        exact = {}
        for c in cells:
            for ep in (OP42 / "results").glob(f"exact_*{c}*.json"):
                exact.update(json.load(open(ep)))
        n = 0
        for (cell, bs) in sorted(per):
            d = per[(cell, bs)]
            sp = (d.get("c|gvr_pr") / d.get("c|bsx")
                  if d.get("c|gvr_pr") and d.get("c|bsx") else None)
            for arm in ("gvr_pr", "bsx"):
                if f"c|{arm}" not in d or (cell, str(bs), arm) in seen:
                    continue
                ok = exact.get(f"{cell}|BS{bs}|{arm}", [None])[0]
                w.writerow(dict(cell=cell, BS=bs, arm=arm,
                                cold_us=round(d[f"c|{arm}"], 3),
                                warm_us=round(d.get(f"w|{arm}", 0), 3),
                                exact=ok,
                                speedup_cold=(round(sp, 4)
                                              if arm == "bsx" and sp else "")))
                n += 1
        print(f"[parse] {Path(rep).name}: {n} rows")
    f.close()


if __name__ == "__main__":
    main()
