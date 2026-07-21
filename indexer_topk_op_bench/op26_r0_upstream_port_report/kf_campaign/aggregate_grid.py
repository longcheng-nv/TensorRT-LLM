# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate full-grid shard reps -> verdict CSV + summary.

  python3 aggregate_grid.py --tag <tag> [--ngpu 8]
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ngpu", type=int, default=8)
    args = ap.parse_args()
    cells = defaultdict(dict)
    exact = {}
    for g in range(args.ngpu):
        rep = HERE / "nsys_reps" / f"grid_{args.tag}_g{g}.nsys-rep"
        if not rep.exists():
            print(f"WARN missing {rep}")
            continue
        for rng, us in parse_rep(str(rep)).items():
            mode, arm, uuid = rng.split("|", 2)
            cells[uuid][f"{mode}|{arm}"] = us
        ep = HERE / f"exact_{args.tag}_g{g}.json"
        if ep.exists():
            exact.update(json.load(open(ep)))
    meta = {f"{r['model']}_{r['isl']}_L{int(r['layer']):02d}": r
            for r in csv.DictReader(
                open(HERE.parent / "real_3arm_layers_full.csv"))}
    rows, ratios = [], []
    n_reg, n_bad = 0, 0
    for uuid, d in sorted(cells.items()):
        m = meta.get(uuid, {})
        rc = (d.get("c|gvr_pr") / d["c|kf_cand"]
              if d.get("c|kf_cand") and d.get("c|gvr_pr") else None)
        ok = exact.get(f"{uuid}|kf_cand", [None])[0]
        if rc is not None:
            ratios.append(rc)
            if rc < 1.0:
                n_reg += 1
        if ok is False:
            n_bad += 1
        rows.append(dict(uuid=uuid, model=m.get("model"), isl=m.get("isl"),
                         layer=m.get("layer"), N=m.get("N"), K=m.get("K"),
                         hit=m.get("hit"),
                         pr_cold=d.get("c|gvr_pr"), cand_cold=d.get("c|kf_cand"),
                         speedup_cold=rc,
                         pr_warm=d.get("w|gvr_pr"), cand_warm=d.get("w|kf_cand"),
                         cand_exact=ok))
    out = HERE / f"grid_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    gm = statistics.geometric_mean(ratios)
    srt = sorted(ratios)
    print(f"cells {len(rows)}  geomean COLD {gm:.4f}  "
          f"min {srt[0]:.3f}  p5 {srt[len(srt)//20]:.3f}  max {srt[-1]:.3f}")
    print(f"regressions(<1.0) {n_reg}  inexact {n_bad}")
    print("worst 15:")
    worst = sorted((r for r in rows if r["speedup_cold"]),
                   key=lambda r: r["speedup_cold"])[:15]
    for r in worst:
        print(f"  {r['uuid']:22s} N={r['N']:>7} K={r['K']:>4} hit={r['hit']} "
              f"x{r['speedup_cold']:.3f}")
    print(f"csv -> {out}")


if __name__ == "__main__":
    main()
