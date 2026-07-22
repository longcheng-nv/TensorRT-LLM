# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate sub-P4 pipeline shards -> p4pipe_full.csv (one row per cell,
wide: per-phase + per-sub-P4-stage cycles/frac/us) + validation summary.

  python3 aggregate_p4pipe.py --tag full [--ngpu 8]

us_est[stage] = frac[stage] * nsys_prod_us (cold-L2, per-launch mean from
nvtx_kern_sum of the pristine prod arm). overhead = nsys timed/prod - 1.
Extra gate vs §9e: p4_select frac drift per cell vs phase_full_full.csv
(the 8-stamp run) is reported so the added [p4sub] stamps can be shown not
to distort the top-level composition.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P4F1 = HERE.parent
REPORT = HERE.parents[1]
sys.path.insert(0, str(REPORT.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

PHASES = ["p1_gather_stats", "smem_stage", "p1b_rungs", "p2_count_admission",
          "p3_collect", "p4_select", "epilogue"]
SUBP4 = ["p4_peer_wait", "p4_dsmem_gather", "p4_minmax", "p4_coarse_hist",
         "p4_coarse_search", "p4_fine", "p4_scatter", "p4_tail"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--ngpu", type=int, default=8)
    args = ap.parse_args()

    # §9e reference p4_select fractions (stamp-count drift gate)
    ref = {}
    refcsv = P4F1 / "phase_breakdown_ptime" / "phase_full_full.csv"
    if refcsv.exists():
        for r in csv.DictReader(open(refcsv)):
            ref[r["uuid"]] = float(r["frac_p4_select"])

    nsys = {}
    cells = {}
    for g in range(args.ngpu):
        rep = HERE / "nsys_reps" / f"p4pipe_{args.tag}_g{g}.nsys-rep"
        if rep.exists():
            for rng, us in parse_rep(str(rep)).items():
                mode, arm, uuid = rng.split("|", 2)
                nsys[(uuid, arm)] = us
        else:
            print(f"WARN missing {rep}")
        for p in [HERE / f"p4pipe_{args.tag}_g{g}.json",
                  HERE / f"p4pipe_{args.tag}_g{g}.jsonl"]:
            if not p.exists():
                continue
            recs = (json.load(open(p)) if p.suffix == ".json"
                    else [json.loads(l) for l in open(p) if l.strip()])
            for r in recs:
                r["gpu"] = g
                cells[r["uuid"]] = r
            break

    rows, n_err, n_inexact, n_nonmono, n_ovh = [], 0, 0, 0, 0
    drifts = []
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
        drift = (r["frac"]["p4_select"] - ref[uuid]) if uuid in ref else None
        if drift is not None:
            drifts.append(drift)
        row = dict(uuid=uuid, model=r["model"], isl=r["isl"], layer=r["layer"],
                   K=r["K"], N=r["N"], cr=r["cr"], hit=r["hit"],
                   cs=r["cs"], T=r["T"], v256=r["v256"], mbpm=r["mbpm"],
                   wpr=r["wpr"], gpu=r["gpu"],
                   us_prod_nsys=round(up, 3) if up else None,
                   us_timed_nsys=round(ut, 3) if ut else None,
                   overhead=round(ovh, 4) if ovh is not None else None,
                   window_cyc=r["window_cyc"],
                   exact=(r["exact_prod"] and r["exact_timed"]),
                   mono=r["mono"], csv_pr_us=r.get("csv_pr_us"),
                   p4frac_drift_vs_9e=(round(drift, 5) if drift is not None
                                       else None))
        for ph in PHASES + SUBP4:
            row[f"cyc_{ph}"] = round(r["cyc"][ph], 1)
            row[f"frac_{ph}"] = round(r["frac"][ph], 5)
            row[f"us_{ph}"] = (round(r["frac"][ph] * up, 4) if up else None)
        # sub-P4 share *of P4* (not of kernel), the pipeline view
        p4c = r["cyc"]["p4_select"]
        for st in SUBP4:
            row[f"p4share_{st}"] = (round(r["cyc"][st] / p4c, 5) if p4c else None)
        rows.append(row)

    out = HERE / f"p4pipe_{args.tag}.csv"
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
    if drifts:
        print(f"p4frac drift vs §9e med {statistics.median(drifts):+.4f}  "
              f"maxabs {max(abs(d) for d in drifts):.4f}  n={len(drifts)}")
    print(f"csv -> {out}")


if __name__ == "__main__":
    main()
