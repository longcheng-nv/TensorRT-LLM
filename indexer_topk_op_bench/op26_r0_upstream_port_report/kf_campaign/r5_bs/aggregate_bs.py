# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate R5 BS-grid shard reps -> grid_<tag>.csv (+ baselines from a
pr-only denominator run).

  python3 aggregate_bs.py --tag r5pr --ngpu 3 [--baselines]
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ngpu", type=int, default=3)
    ap.add_argument("--baselines", action="store_true")
    args = ap.parse_args()
    cells = {}
    exact = {}
    for g in range(args.ngpu):
        rep = HERE / "nsys_reps" / f"grid_{args.tag}_g{g}.nsys-rep"
        if not rep.exists():
            print(f"WARN missing {rep}")
            continue
        for rng, us in parse_rep(str(rep)).items():
            mode, arm, cuuid = rng.split("|", 2)
            cells.setdefault(cuuid, {})[f"{mode}|{arm}"] = us
        ep = HERE / f"exact_{args.tag}_g{g}.json"
        if ep.exists():
            exact.update(json.load(open(ep)))
    meta = {m["uuid"]: m for m in csv.DictReader(open(HERE / "cells_meta_bs.csv"))}
    rows, ratios = [], []
    for cuuid, d in sorted(cells.items()):
        uuid, bs = cuuid.rsplit("_bs", 1)
        m = meta.get(uuid, {})
        rc = (d.get("c|gvr_pr") / d["c|kf_cand"]
              if d.get("c|kf_cand") and d.get("c|gvr_pr") else None)
        if rc is not None:
            ratios.append(rc)
        rows.append(dict(cuuid=cuuid, uuid=uuid, bs=int(bs),
                         model=m.get("model"), isl=m.get("isl"),
                         layer=m.get("layer"), N=m.get("N"), K=m.get("K"),
                         hit=m.get("hit"),
                         pr_cold=d.get("c|gvr_pr"), cand_cold=d.get("c|kf_cand"),
                         speedup_cold=rc,
                         pr_warm=d.get("w|gvr_pr"), cand_warm=d.get("w|kf_cand"),
                         cand_exact=exact.get(f"{cuuid}|kf_cand", [None])[0]))
    out = HERE / f"grid_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cases -> {out}")
    if ratios:
        gm = statistics.geometric_mean(ratios)
        nreg = sum(1 for x in ratios if x < 1.0)
        print(f"geomean COLD {gm:.4f}  min {min(ratios):.3f}  "
              f"max {max(ratios):.3f}  regs {nreg}/{len(ratios)}")
        worst = sorted((r for r in rows if r["speedup_cold"]),
                       key=lambda r: r["speedup_cold"])[:12]
        for r in worst:
            print(f"  {r['cuuid']:26s} N={r['N']:>7} K={r['K']:>4} "
                  f"hit={r['hit']} x{r['speedup_cold']:.3f}")
    if args.baselines:
        wl = [json.loads(l) for l in open(HERE / "workload_bs.jsonl")]
        bl = []
        r4 = {r["uuid"]: float(r["pr_cold"])
              for r in csv.DictReader(open(HERE.parent / "grid_r4pr2.csv"))
              if r["pr_cold"]}
        for w_ in wl:
            d = cells.get(w_["uuid"])
            if d and d.get("c|gvr_pr"):
                us = d["c|gvr_pr"]
            elif w_["uuid"].endswith("_bs1"):
                us = r4[w_["uuid"][:-4]]  # BS=1 guard: R4 denominator grid
            else:
                raise AssertionError(f"no pr time for {w_['uuid']}")
            bl.append(json.dumps({"uuid": w_["uuid"],
                                  "execution_time_ms": round(us / 1000, 6)}))
        (HERE / "baselines_bs.jsonl").write_text("\n".join(bl) + "\n")
        print(f"baselines_bs.jsonl: {len(bl)} rows")


if __name__ == "__main__":
    main()
