# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41 envelope verdict: (1) v3mt alone vs report pr -> v3mt_data.csv,
(2) op39 combined dispatch re-run: BEST(arm_e6, v3mt) vs BEST(arm_e6, v3) —
does swapping v3 -> v3mt move the 750-cell record (e6: gm 1.3179 /
mean 1.3564 / min 0.7665 / wins n/a)?"""
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
        rep = OP41 / "results" / "nsys" / f"v3mt_s{g}.nsys-rep"
        if rep.exists():
            kern.update(parse_rep(str(rep)))
        ep = OP41 / "results" / f"exact_v3mt_s{g}.json"
        if ep.exists():
            exact.update(json.load(open(ep)))

    pr, meta = {}, {}
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
    with open(OP41 / "results" / "v3mt_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_inexact = sum(1 for r in rows if r["exact"] != "OK")
    mt = {(r["cell"], r["BS"]): r["speedup"] for r in rows if r["speedup"]}
    v3 = {}
    for r in csv.DictReader(open(BENCH / "op38_r3v11_bs" / "v3_data.csv")):
        if r["speedup"]:
            v3[(r["cell"], int(r["BS"]))] = float(r["speedup"])
    arm = {}
    for r in csv.DictReader(open(BENCH / "op39_gvr_bsx" / "results" /
                                 "e6_data.csv")):
        arm[(r["cell"], int(r["BS"]))] = float(r["x_arm"])

    # envelope keys = the 750 e6 cases (BS 2..1024)
    keys = sorted(arm.keys())
    print(f"envelope cases: {len(keys)}, v3mt inexact: {n_inexact}")

    def summ(name, vals):
        print(f"{name}: gm {statistics.geometric_mean(vals):.4f} "
              f"mean {statistics.mean(vals):.4f} min {min(vals):.4f} "
              f"<1.0 {sum(1 for x in vals if x < 1.0)}/{len(vals)}")

    summ("v3   alone (record)", [v3[k] for k in keys])
    summ("v3mt alone         ", [mt[k] for k in keys])
    per_k = {k: (v3[k], mt[k]) for k in keys}
    ratio = [b / a for a, b in per_k.values()]
    print(f"v3mt/v3 per-case: gm {statistics.geometric_mean(ratio):.4f} "
          f"min {min(ratio):.4f} max {max(ratio):.4f} "
          f"<0.98: {sum(1 for x in ratio if x < 0.98)}")
    summ("COMBINED e6 (record)", [max(arm[k], v3[k]) for k in keys])
    summ("COMBINED arm+v3mt   ", [max(arm[k], mt[k]) for k in keys])
    summ("COMBINED arm+v3+v3mt", [max(arm[k], v3[k], mt[k]) for k in keys])
    wins = sum(1 for k in keys if arm[k] >= mt[k] and arm[k] >= 1.0)
    print(f"arm wins over v3mt: {wins}")


if __name__ == "__main__":
    main()
