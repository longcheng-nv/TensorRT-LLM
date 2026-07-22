# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse one op37 bs_<cell>.nsys-rep -> append rows to bs_data.csv.

Range names: {c,w}|{gvr_pr,champion}|<cell>|BS<n>.
us = summed in-range kernel time / instances (house nvtx_kern_sum protocol,
evict kernels excluded)."""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

from bs_ab import CELLS  # noqa: E402

FIELDS = ["K", "N", "cell", "BS", "arm", "cold_us", "warm_us", "exact",
          "speedup_cold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--csv", default=str(HERE / "bs_data.csv"))
    args = ap.parse_args()

    kern = parse_rep(args.rep)          # {range: us_per_call}
    cell = CELLS[args.cell]
    exact = {}
    ep = HERE / f"exact_{args.cell}.json"
    if ep.exists():
        exact = json.load(open(ep))

    per_bs = {}
    for rng, us in kern.items():
        parts = rng.split("|")
        if len(parts) != 4 or parts[2] != args.cell:
            continue
        mode, arm, _, bstag = parts
        per_bs.setdefault(int(bstag[2:]), {})[f"{mode}|{arm}"] = us

    csv_path = Path(args.csv)
    seen = set()
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            seen.add((r["cell"], r["BS"], r["arm"]))
    new = csv_path.exists() is False
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for bs in sorted(per_bs):
            d = per_bs[bs]
            sp = (d.get("c|gvr_pr") / d.get("c|champion")
                  if d.get("c|gvr_pr") and d.get("c|champion") else None)
            for arm in ("gvr_pr", "champion"):
                if (args.cell, str(bs), arm) in seen:
                    continue
                ok = exact.get(f"{args.cell}|BS{bs}|{arm}", [None])[0]
                w.writerow(dict(K=cell["K"], N=cell["N"], cell=args.cell,
                                BS=bs, arm=arm,
                                cold_us=round(d.get(f"c|{arm}", 0), 3),
                                warm_us=round(d.get(f"w|{arm}", 0), 3),
                                exact=ok,
                                speedup_cold=(round(sp, 4)
                                              if arm == "champion" and sp
                                              else "")))
            if sp:
                print(f"{args.cell} BS{bs:5d} pr={d.get('c|gvr_pr', 0):9.2f} "
                      f"champ={d.get('c|champion', 0):9.2f} x{sp:6.3f}")


if __name__ == "__main__":
    main()
