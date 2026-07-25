# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41 e7 verdict: three-arm combined dispatch BEST(arm_e6, v3mt, champ7)
vs report pr — the official envelope after KF-champion integration."""
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP41 = HERE.parent
BENCH = OP41.parent
sys.path.insert(0, str(BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def main():
    kern, exact = {}, {}
    for g in range(8):
        rep = OP41 / "results" / "nsys" / f"champ7_s{g}.nsys-rep"
        if rep.exists():
            kern.update(parse_rep(str(rep)))
        ep = OP41 / "results" / f"exact_champ7_s{g}.json"
        if ep.exists():
            exact.update(json.load(open(ep)))
    pr = {}
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" /
                                 "bs_real_layers.csv")):
        pr[(f"{r['model']}_{r['isl']}_L{int(r['L']):02d}",
            int(r["BS"]))] = float(r["pr"])
    rows = []
    for rng, us in kern.items():
        parts = rng.split("|")
        if len(parts) != 4 or parts[0] != "c":
            continue
        cname, bs = parts[2], int(parts[3][2:])
        p = pr.get((cname, bs))
        ex = exact.get(f"{cname}|BS{bs}", "?")
        rows.append(dict(cell=cname, BS=bs, champ_us=round(us, 3), pr_us=p,
                         speedup=round(p / us, 4) if p else None,
                         exact="OK" if ex == "OK" else ex))
    rows.sort(key=lambda r: (r["cell"], r["BS"]))
    with open(OP41 / "results" / "champ7_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_inexact = sum(1 for r in rows if r["exact"] != "OK")
    ch = {(r["cell"], r["BS"]): r["speedup"] for r in rows if r["speedup"]}
    mt, arm = {}, {}
    for r in csv.DictReader(open(OP41 / "results" / "v3mt_data.csv")):
        if r["speedup"]:
            mt[(r["cell"], int(r["BS"]))] = float(r["speedup"])
    for r in csv.DictReader(open(BENCH / "op39_gvr_bsx" / "results" /
                                 "e6_data.csv")):
        arm[(r["cell"], int(r["BS"]))] = float(r["x_arm"])
    keys = sorted(arm)
    print(f"cases: {len(keys)}  champ inexact: {n_inexact}")

    def summ(name, vals):
        print(f"{name}: gm {statistics.geometric_mean(vals):.4f} "
              f"mean {statistics.mean(vals):.4f} min {min(vals):.4f} "
              f"<1.0 {sum(1 for x in vals if x < 1.0)}/{len(vals)}")
    summ("champ alone         ", [ch[k] for k in keys])
    summ("combined 2-arm (rec)", [max(arm[k], mt[k]) for k in keys])
    summ("COMBINED 3-arm (e7) ", [max(arm[k], mt[k], ch[k]) for k in keys])
    served = sum(1 for k in keys if ch[k] > max(arm[k], mt[k])
                 and ch[k] >= 1.0)
    print(f"champ-served cells (wins over both arms, >=1.0): {served}")
    top = sorted(((ch[k] - max(arm[k], mt[k]), k) for k in keys),
                 reverse=True)[:12]
    print("top champ upgrades:")
    for d, k in top:
        if d <= 0:
            break
        print(f"  {k[0]:20s} BS{k[1]:5d}: {max(arm[k], mt[k]):.2f} -> "
              f"{ch[k]:.2f}")


if __name__ == "__main__":
    main()
