# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 envelope verdict: join e1 sharded nsys arm times with report pr AND
op38 v3 (own prior); compute per-case best-of(arm, v3) combined-dispatch
projection vs the campaign bars (mean >= 1.8, all >= 1.0)."""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def main():
    kern, exact = {}, {}
    for g in range(8):
        rep = HERE.parent / "results" / "nsys" / f"e2_s{g}.nsys-rep"
        if rep.exists():
            kern.update(parse_rep(str(rep)))
        ep = HERE.parent / "results" / f"exact_e2_s{g}.json"
        if ep.exists():
            exact.update(json.load(open(ep)))
    pr, v3 = {}, {}
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" /
                                 "bs_real_layers.csv")):
        pr[(f"{r['model']}_{r['isl']}_L{int(r['L']):02d}",
            int(r["BS"]))] = float(r["pr"])
    for r in csv.DictReader(open(BENCH / "op38_r3v11_bs" / "v3_data.csv")):
        if r["speedup"]:
            v3[(r["cell"], int(r["BS"]))] = float(r["speedup"])
    rows = []
    for rng, us in kern.items():
        parts = rng.split("|")
        if len(parts) != 4 or parts[1] != "arm":
            continue
        cname, bs = parts[2], int(parts[3][2:])
        p = pr.get((cname, bs))
        if not p or bs == 1:
            continue
        ex = exact.get(f"{cname}|BS{bs}", "?")
        rows.append(dict(cell=cname, BS=bs, arm_us=round(us, 3),
                         pr_us=p, x_arm=round(p / us, 4),
                         x_v3=v3.get((cname, bs)),
                         exact="OK" if ex == "OK" else ex))
    rows.sort(key=lambda r: (r["cell"], r["BS"]))
    with open(HERE.parent / "results" / "e2_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    xa = [r["x_arm"] for r in rows]
    comb = [max(r["x_arm"], r["x_v3"] or 0) for r in rows]
    n_inex = sum(1 for r in rows if r["exact"] != "OK")
    print(f"cases: {len(rows)}  inexact: {n_inex}")
    print(f"ARM alone      : gm {statistics.geometric_mean(xa):.4f} "
          f"mean {statistics.mean(xa):.4f} min {min(xa):.4f} "
          f"<1.0 {sum(1 for x in xa if x < 1)}/{len(xa)}")
    print(f"BEST(arm, v3)  : gm {statistics.geometric_mean(comb):.4f} "
          f"mean {statistics.mean(comb):.4f} min {min(comb):.4f} "
          f"<1.0 {sum(1 for x in comb if x < 1)}/{len(comb)}")
    print("\nper-BS arm gm / combined gm:")
    for bs in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        a = [r["x_arm"] for r in rows if r["BS"] == bs]
        c = [max(r["x_arm"], r["x_v3"] or 0) for r in rows if r["BS"] == bs]
        if a:
            print(f"  BS{bs:5d}: arm {statistics.geometric_mean(a):.3f} "
                  f"(min {min(a):.3f}) | comb {statistics.geometric_mean(c):.3f}")
    print("\narm wins over v3 (x_arm > x_v3, x_arm >= 1.0):",
          sum(1 for r in rows if r["x_v3"] and r["x_arm"] > r["x_v3"]
              and r["x_arm"] >= 1))


if __name__ == "__main__":
    main()
