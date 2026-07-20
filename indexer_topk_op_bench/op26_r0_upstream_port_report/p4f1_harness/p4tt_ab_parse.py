#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse the p4tt tiny-tie A/B: per-cell x3-median slow/fast ratio
(>1 = FAST wins), plus cross-node anchor vs real_3arm.csv (REPORT §4)."""
import csv
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPORT = _HERE.parent
sys.path.insert(0, str(_REPORT.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gvrlayers/newpr_ab")


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


ref = {}
with open(_REPORT / "real_3arm.csv") as f:
    for r in csv.DictReader(f):
        if r.get("pr"):
            ref[(r["model"], r["isl"])] = float(r["pr"])

rows = []
inexact = []
for f in sorted(ROOT.glob("ab_*.jsonl")):
    m = f.stem.split("_")[1]
    kern = parse_rep(ROOT / "nsys_reps" / f"ab_{m}.nsys-rep")
    cells = {}
    for l in f.read_text().splitlines():
        r = json.loads(l)
        if r.get("exact") is False and r["arm"] == "fast":
            inexact.append((m, r["isl"], r["rnd"]))
        us = kern.get(r["range_cold"])
        if us:
            cells.setdefault(r["isl"], {}).setdefault(r["arm"], {})[r["rnd"]] = us
    for isl, d in cells.items():
        if "slow" in d and "fast" in d:
            rnds = sorted(set(d["slow"]) & set(d["fast"]))
            ratios = sorted(d["slow"][r] / d["fast"][r] for r in rnds)
            old_m = sorted(d["slow"].values())[len(d["slow"]) // 2]
            new_m = sorted(d["fast"].values())[len(d["fast"]) // 2]
            rows.append(dict(model=m, isl=isl, slow=old_m, fast=new_m,
                             ratio=ratios[len(ratios) // 2],
                             rmin=ratios[0], rmax=ratios[-1],
                             ref=ref.get((m, isl))))

rows.sort(key=lambda r: (r["model"], int(r["isl"][:-1])))
print(f"{'cell':16s} {'slow µs':>8s} {'fast µs':>8s} {'slo/fas':>8s} "
      f"{'(x3 range)':>15s} {'REPORT pr µs':>12s} {'slow/REPT':>10s}")
for r in rows:
    anch = f"{r['slow'] / r['ref']:.3f}" if r["ref"] else "-"
    refs = f"{r['ref']:.2f}" if r["ref"] else "-"
    print(f"{r['model']}/{r['isl']:7s} {r['slow']:8.2f} {r['fast']:8.2f} "
          f"{r['ratio']:8.3f} [{r['rmin']:.3f}-{r['rmax']:.3f}] {refs:>12s} {anch:>10s}")
per_m = {}
for r in rows:
    per_m.setdefault(r["model"], []).append(r["ratio"])
print()
for m, xs in per_m.items():
    print(f"{m}: gm {gm(xs):.4f}  min {min(xs):.3f}  max {max(xs):.3f}  n={len(xs)}")
allr = [r["ratio"] for r in rows]
print(f"ALL: NEW vs REPORT-baseline gm {gm(allr):.4f} "
      f"(>1 = latest PR faster)  min {min(allr):.3f} max {max(allr):.3f}")
anchs = [r["slow"] / r["ref"] for r in rows if r["ref"]]
print(f"anchor old(027)/REPORT(094): med {sorted(anchs)[len(anchs)//2]:.3f} "
      f"(cross-node context for absolute comparisons)")
print("new-arm inexact records:", inexact if inexact else "NONE")
