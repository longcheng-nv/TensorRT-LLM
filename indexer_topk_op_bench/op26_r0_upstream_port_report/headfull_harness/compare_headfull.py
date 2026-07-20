#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Full-coverage comparison: PR head @e6fdbfac3d (this sweep, b200-027) vs
REPORT.html numbers (refresh @018251950f grids: synth_3arm/real_3arm/
bs_synth/bs_real, b200-094).

Anchor gate: op26_r0auto measured in BOTH runs (rival_long.csv op26 rows for
seqlen; this sweep's op26 rows) — per-batch drift med/p95 gates cross-node
comparability, and ratios are additionally reported anchor-NORMALIZED.

Usage: python3 compare_headfull.py [results_dir]
       default /tmp/gvrheadfull/refresh_results (after parse_refresh.py)
"""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPORT = _HERE.parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gvrheadfull/refresh_results")


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---- load this sweep -------------------------------------------------------
rows = [json.loads(l) for l in (ROOT / "results.jsonl").read_text().splitlines()
        if l.strip()]
new = defaultdict(dict)   # key -> {op: us}
for r in rows:
    if "error" in r or "us" not in r:
        continue
    if r["family"] == "synth":
        k = ("synth", r["sweep"], r["scenario"], int(r["K"]), r["dtype"],
             int(r["N"]), int(r["BS"]))
    else:
        k = ("real", r["sweep"], r["model"], r["dtype"], r["isl"], int(r["BS"]))
    new[k][r["op"]] = r["us"]

# ---- REPORT reference grids -------------------------------------------------
def rcsv(name):
    p = _REPORT / name
    return list(csv.DictReader(open(p))) if p.exists() else []

ref = {}
for r in rcsv("synth_3arm.csv"):     # synth seqlen fp32
    ref[("synth", "seqlen", r["scen"], int(r["K"]), "fp32", int(r["N"]), 1)] = \
        dict(base=fnum(r["base"]), pr=fnum(r["pr"]), op26=fnum(r["op26"]))
for r in rcsv("bs_synth.csv"):
    ref[("synth", "bs", r["scen"], int(r["K"]), r["dtype"], int(r["N"]), int(r["BS"]))] = \
        dict(base=fnum(r["base"]), pr=fnum(r["pr"]), op26=fnum(r["op26"]))
for r in rcsv("real_3arm.csv"):
    ref[("real", "seqlen", r["model"], "fp32", r["isl"], 1)] = \
        dict(base=fnum(r["base"]), pr=fnum(r["pr"]), op26=fnum(r["op26"]))
for r in rcsv("bs_real.csv"):
    ref[("real", "bs", r["model"], r["dtype"], r["isl"], int(r["BS"]))] = \
        dict(base=fnum(r["base"]), pr=fnum(r["pr"]), op26=fnum(r["op26"]))

# ---- anchor drift (op26 both runs) ------------------------------------------
anch = []
for k, d in new.items():
    rr = ref.get(k)
    if rr and d.get("op26_r0auto") and rr.get("op26"):
        anch.append(d["op26_r0auto"] / rr["op26"])
anch.sort()
amed = st.median(anch) if anch else float("nan")
ap95 = anch[min(len(anch) - 1, int(0.95 * len(anch)))] if anch else float("nan")
print(f"== anchor op26 new(027)/REPORT: n={len(anch)} med {amed:.3f} p95 {ap95:.3f}")

# ---- per-axis comparison -----------------------------------------------------
AXES = [
    ("synth seqlen fp32 (§3)", lambda k: k[0] == "synth" and k[1] == "seqlen"),
    ("synth BS grid fp32 (§7)", lambda k: k[0] == "synth" and k[1] == "bs" and k[4] == "fp32"),
    ("synth BS grid 16-bit (§7)", lambda k: k[0] == "synth" and k[1] == "bs" and k[4] != "fp32"),
    ("real seqlen fp32 BS=1 (§4)", lambda k: k[0] == "real" and k[1] == "seqlen"),
    ("real BS grid fp32 (§7)", lambda k: k[0] == "real" and k[1] == "bs" and k[3] == "fp32"),
    ("real BS grid 16-bit (§7)", lambda k: k[0] == "real" and k[1] == "bs" and k[3] != "fp32"),
]
print(f"\n{'axis':28s} {'n':>5s} | {'pr: REPORT/new':>14s} {'norm':>6s} {'min':>6s} {'max':>6s} | "
      f"{'base: REPORT/new':>16s} {'norm':>6s}")
worst_cells = []
for name, sel in AXES:
    prr, brr, cells = [], [], []
    for k, d in new.items():
        if not sel(k):
            continue
        rr = ref.get(k)
        if not rr:
            continue
        if d.get("gvr_pr") and rr.get("pr"):
            v = rr["pr"] / d["gvr_pr"]
            prr.append(v)
            cells.append((v, k, rr["pr"], d["gvr_pr"]))
        if d.get("gvr_base") and rr.get("base"):
            brr.append(rr["base"] / d["gvr_base"])
    if not prr:
        print(f"{name:28s} {'0':>5s} | (no overlap)")
        continue
    g, b = gm(prr), gm(brr)
    print(f"{name:28s} {len(prr):5d} | {g:14.3f} {g/amed:6.3f} {min(prr):6.3f} {max(prr):6.3f} | "
          f"{b:16.3f} {b/amed:6.3f}")
    cells.sort()
    worst_cells += cells[:3]

print("\n== worst pr cells per axis (REPORT/new < 1 = head SLOWER than REPORT) ==")
for v, k, a, b in sorted(worst_cells)[:12]:
    print(f"  {v:6.3f}  {k}  REPORT {a:.2f}us -> head {b:.2f}us")
print("\n(norm = anchor-normalized: value/anchor_med; >1 = head faster than "
      "REPORT after removing node bias)")
