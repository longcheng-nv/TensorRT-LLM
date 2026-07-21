# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate phase-breakdown shards -> phase_full.csv (one row per cell,
wide: per-phase cycles/frac/us) + validation summary.

  python3 aggregate_phases.py --tag full [--ngpu 8]

us_est[phase] = frac[phase] * nsys_prod_us (cold-L2, per-launch mean from
nvtx_kern_sum of the pristine prod arm). overhead = nsys timed/prod - 1.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]
sys.path.insert(0, str(REPORT.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

PHASES = ["p1_gather_stats", "smem_stage", "p1b_rungs", "p2_count_admission",
          "p3_collect", "p4_select", "epilogue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--ngpu", type=int, default=8)
    args = ap.parse_args()

    nsys = {}
    cells = {}
    for g in range(args.ngpu):
        rep = HERE / "nsys_reps" / f"phases_{args.tag}_g{g}.nsys-rep"
        if rep.exists():
            for rng, us in parse_rep(str(rep)).items():
                mode, arm, uuid = rng.split("|", 2)
                nsys[(uuid, arm)] = us
        else:
            print(f"WARN missing {rep}")
        for p in [HERE / f"phases_{args.tag}_g{g}.json",
                  HERE / f"phases_{args.tag}_g{g}.jsonl"]:
            if not p.exists():
                continue
            recs = (json.load(open(p)) if p.suffix == ".json"
                    else [json.loads(l) for l in open(p) if l.strip()])
            for r in recs:
                r["gpu"] = g
                cells[r["uuid"]] = r
            break

    rows, n_err, n_inexact, n_nonmono, n_ovh = [], 0, 0, 0, 0
    for uuid, r in sorted(cells.items()):
        if r.get("error"):
            n_err += 1
            print(f"ERROR cell {uuid}: {r['error']}")
            continue
        up = nsys.get((uuid, "prod"))
        ut = nsys.get((uuid, "timed"))
        ovh = (ut / up - 1.0) if (up and ut) else None
        if not (r["exact_prod"] and r["exact_timed"]):
            n_inexact += 1
        if not r["mono"]:
            n_nonmono += 1
        if ovh is not None and abs(ovh) > 0.07:
            n_ovh += 1
        row = dict(uuid=uuid, model=r["model"], isl=r["isl"], layer=r["layer"],
                   K=r["K"], N=r["N"], cr=r["cr"], hit=r["hit"],
                   cs=r["cs"], T=r["T"], v256=r["v256"], mbpm=r["mbpm"],
                   wpr=r["wpr"], gpu=r["gpu"],
                   us_prod_nsys=round(up, 3) if up else None,
                   us_timed_nsys=round(ut, 3) if ut else None,
                   overhead=round(ovh, 4) if ovh is not None else None,
                   window_cyc=r["window_cyc"],
                   exact=(r["exact_prod"] and r["exact_timed"]),
                   mono=r["mono"], csv_pr_us=r.get("csv_pr_us"))
        for ph in PHASES:
            row[f"cyc_{ph}"] = round(r["cyc"][ph], 1)
            row[f"frac_{ph}"] = round(r["frac"][ph], 5)
            row[f"us_{ph}"] = (round(r["frac"][ph] * up, 4) if up else None)
        rows.append(row)

    out = HERE / f"phase_full_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"cells {len(rows)} (errors {n_err})  inexact {n_inexact}  "
          f"non-mono {n_nonmono}  |overhead|>7% {n_ovh}")
    ovhs = sorted(r["overhead"] for r in rows if r["overhead"] is not None)
    if ovhs:
        print(f"overhead med {ovhs[len(ovhs)//2]:+.3f}  "
              f"p95 {ovhs[int(len(ovhs)*0.95)]:+.3f}  max {ovhs[-1]:+.3f}")
    print(f"csv -> {out}")


if __name__ == "__main__":
    main()
