# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KF candidate vs REPORT rival arms (sglang_v2 / radix_cutedsl / flashinfer_topk).

Joins a KF full-grid CSV (grid_<tag>.csv: pr_cold/cand_cold, this campaign's
nsys cold-L2 runs) with the op26 REPORT per-cell rival sweep
(rival_layers_full.csv: absolute us) on the 865-cell uuid grid.

Per-cell ratio is NORMALIZED via the PR arm measured in both campaigns:
    cand_vs_rival = (rival_us / cand_cold) * (pr_cold / pr_report)
which cancels per-cell anchor drift between the two measurement sessions
(median pr_cold/pr_report = 1.010 over 865 cells at time of writing).

  python3 compare_rivals.py --tag r2c2g [--ops sglang_v2,radix_cutedsl]
"""
import argparse
import collections
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="grid_<tag>.csv of the KF candidate")
    ap.add_argument("--ops", default="sglang_v2,radix_cutedsl")
    args = ap.parse_args()

    riv = collections.defaultdict(dict)
    for r in csv.DictReader(open(REPORT / "rival_layers_full.csv")):
        uuid = f"{r['model']}_{r['isl']}_L{int(r['L']):02d}"
        riv[r["op"]][uuid] = float(r["us"])
    rep_pr = {}
    for r in csv.DictReader(open(REPORT / "real_3arm_layers_full.csv")):
        uuid = f"{r['model']}_{r['isl']}_L{int(r['layer']):02d}"
        rep_pr[uuid] = float(r["pr"])
    grid = {r["uuid"]: (float(r["pr_cold"]), float(r["cand_cold"]), int(r["N"]))
            for r in csv.DictReader(open(HERE / f"grid_{args.tag}.csv"))}

    cal = sorted(pk / rep_pr[u] for u, (pk, _, _) in grid.items() if u in rep_pr)
    print(f"pr calibration med {cal[len(cal)//2]:.3f} p5 {cal[len(cal)//20]:.3f} "
          f"p95 {cal[-len(cal)//20]:.3f}")

    for op in args.ops.split(","):
        vals, byN = {}, collections.defaultdict(list)
        for u, (pk, ck, n) in grid.items():
            if u in riv[op] and u in rep_pr:
                v = (riv[op][u] / ck) * (pk / rep_pr[u])
                vals[u] = v
                byN[n].append(v)
        vs = sorted(vals.values())
        n_ = len(vs)
        gm = math.exp(sum(map(math.log, vs)) / n_)
        worst = min(vals.items(), key=lambda x: x[1])
        print(f"\n{args.tag} vs {op}: geomean {gm:.3f}  win {sum(1 for v in vs if v > 1)}/{n_}  "
              f"min {vs[0]:.3f} ({worst[0]})  p5 {vs[n_//20]:.3f}  max {vs[-1]:.2f}")
        for n in sorted(byN):
            v = byN[n]
            g = math.exp(sum(map(math.log, v)) / len(v))
            print(f"  N={n:>7}: gm {g:.3f}  win {sum(1 for x in v if x > 1)}/{len(v)}")


if __name__ == "__main__":
    main()
