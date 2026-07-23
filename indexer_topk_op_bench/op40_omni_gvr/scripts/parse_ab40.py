# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate op40 A/B batches -> per-cell us (+ paired ratios when 2 arms).

Verdict contract (PLAN.md): gm over 865 cells, zero-regression band = paired
ratio >= 0.97 per cell, exactness green on every cell x arm.

  python3 parse_ab40.py <tagdir>                 # e.g. bl0 or ab_v1
  python3 parse_ab40.py <tagdir> --ref base --var v1
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
sys.path.insert(0, str(OP40.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def gm(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def isl_band(N, cr):
    isl = N * cr
    return "32k-1M" if isl >= 32768 else "4k-32k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tagdir")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--var", default=None)
    args = ap.parse_args()
    D = OP40 / "results" / args.tagdir

    us, meta, inexact = {}, {}, []
    batches = sorted(D.glob("real865_*.csv"))
    print(f"[parse] {len(batches)} completed batches in {D}")
    for c in batches:
        rep = D / "nsys_reps" / f"{c.stem}.nsys-rep"
        if rep.exists():
            for rng, t in parse_rep(str(rep)).items():
                _, arm, uuid, _dt = rng.split("|", 3)
                us[(uuid, arm)] = t
        else:
            print(f"[parse] WARN missing rep for {c.stem}")
        for r in csv.DictReader(open(c)):
            if r["arm"] == "ERROR":
                inexact.append((r["uuid"], "ERROR"))
                continue
            meta.setdefault(r["uuid"], (int(r["cs"]), int(r["N"]), int(r["K"])))
            if r["exact"] != "True":
                inexact.append((r["uuid"], r["arm"]))

    model_cr = {"flash": 4, "pro": 4, "v32": 1}
    rows = []
    for uuid, (cs, N, K) in sorted(meta.items()):
        model = uuid.split("_")[0]
        row = dict(uuid=uuid, model=model, cs=cs, N=N, K=K,
                   band=isl_band(N, model_cr[model]))
        tref = us.get((uuid, args.ref))
        row[f"{args.ref}_us"] = round(tref, 3) if tref else None
        if args.var:
            tvar = us.get((uuid, args.var))
            row[f"{args.var}_us"] = round(tvar, 3) if tvar else None
            row["ratio"] = round(tref / tvar, 4) if (tref and tvar) else None
        rows.append(row)
    with open(D / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} cells · inexact/error: {inexact or 'none'}")
    if args.var:
        rr = [r for r in rows if r.get("ratio")]
        ratios = [r["ratio"] for r in rr]
        reg = [r for r in rr if r["ratio"] < 0.97]
        worst = min(rr, key=lambda r: r["ratio"]) if rr else None
        print(f"\n== paired {args.ref} vs {args.var}: n={len(rr)} "
              f"gm={gm(ratios):.4f} · regressions(<0.97) {len(reg)} · "
              f"worst {worst['ratio']:.4f} @ {worst['uuid']}" if rr else "no pairs")
        for key in ("model", "band"):
            by = defaultdict(list)
            for r in rr:
                by[r[key]].append(r["ratio"])
            for k in sorted(by):
                v = by[k]
                print(f"   {key}={k:8s} n={len(v):3d} gm={gm(v):.4f} "
                      f"min={min(v):.4f} max={max(v):.4f}")
        if reg:
            print("   regressed cells:", [(r['uuid'], r['ratio']) for r in
                                          sorted(reg, key=lambda r: r['ratio'])][:20])
        (D / "verdict.json").write_text(json.dumps(dict(
            n=len(rr), gm=gm(ratios), regressions=len(reg),
            worst=worst and {"uuid": worst["uuid"], "ratio": worst["ratio"]},
            inexact=inexact), indent=1))
    else:
        by = defaultdict(list)
        for r in rows:
            if r.get(f"{args.ref}_us"):
                by[(r["model"], r["band"])].append(r[f"{args.ref}_us"])
        print("\n-- baseline us by (model, band):")
        for k in sorted(by):
            v = by[k]
            print(f"   {k[0]:6s} {k[1]:8s} n={len(v):3d} "
                  f"gm={gm(v):8.2f}us min={min(v):8.2f} max={max(v):9.2f}")


if __name__ == "__main__":
    main()
