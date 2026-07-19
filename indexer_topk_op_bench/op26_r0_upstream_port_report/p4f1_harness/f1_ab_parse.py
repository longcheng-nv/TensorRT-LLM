#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse F1 Gate C nsys reps -> per-cell ON/OFF cold ratios. Gate: any cell
with ON/OFF > 1.025 blocks; report gm/min/max split bench-vs-fixture."""
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gvrlayers/f1ab")
BENCH_L = {"flash": 22, "pro": 30, "v32": 34}


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


rows = []
for f in sorted(ROOT.glob("f1ab_*.jsonl")):
    m = f.stem.split("_")[1]
    kern = parse_rep(ROOT / "nsys_reps" / f"f1ab_{m}.nsys-rep")
    cells = {}
    for l in f.read_text().splitlines():
        r = json.loads(l)
        us = kern.get(r["range_cold"])
        if us:
            cells.setdefault((r["isl"], r["L"], r["N"]), {}).setdefault(
                r["arm"], {})[r.get("rnd", 0)] = us
    for (isl, L, N), d in sorted(cells.items(), key=lambda t: t[0][2]):
        if "off" in d and "on" in d:
            rnds = sorted(set(d["off"]) & set(d["on"]))
            ratios = sorted(d["on"][r] / d["off"][r] for r in rnds)
            med = ratios[len(ratios) // 2]
            off_m = sorted(d["off"][r] for r in rnds)[len(rnds) // 2]
            on_m = sorted(d["on"][r] for r in rnds)[len(rnds) // 2]
            rows.append(dict(model=m, isl=isl, L=L, N=N, off=off_m, on=on_m,
                             ratio=med, rmin=ratios[0], rmax=ratios[-1],
                             fixture=L != BENCH_L[m]))

print(f"{'cell':22s} {'off µs':>8s} {'on µs':>8s} {'on/off':>7s} {'class':>8s}")
worst = None
for r in rows:
    tag = "FIXTURE" if r["fixture"] else "bench"
    flag = "  << REGRESSION" if r["ratio"] > 1.025 else ""
    print(f"{r['model']}/{r['isl']}/L{r['L']:<4d} {r['off']:8.2f} {r['on']:8.2f} "
          f"{r['ratio']:7.3f} [{r.get('rmin',0):.3f}-{r.get('rmax',0):.3f}] {tag:>8s}{flag}")
    if worst is None or r["ratio"] > worst["ratio"]:
        worst = r
bench = [r["ratio"] for r in rows if not r["fixture"]]
fixt = [r["ratio"] for r in rows if r["fixture"]]
nreg = sum(1 for r in rows if r["ratio"] > 1.025)
print(f"\nbench (99% path, n={len(bench)}): gm {gm(bench):.4f} "
      f"min {min(bench):.3f} max {max(bench):.3f}")
if fixt:
    print(f"fixture (recursion fires, n={len(fixt)}): gm {gm(fixt):.4f} "
          f"min {min(fixt):.3f} max {max(fixt):.3f}")
print(f"cells > 1.025: {nreg}  -> Gate C {'PASS' if nreg == 0 else 'FAIL'}")
