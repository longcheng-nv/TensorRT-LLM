# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 backlog-1 verdict: qfracs UH4 vs M2D paired nsys A/B.

Input = results_b200_op26_qfracs_ab (sweep_qfracs.py batches). Per-cell
metric = us_cold(m2d) / us_cold(uh4)  (>1 = uh4 faster). Views:
  - mc port, mc DISPATCH REGION (N>=65536) by (dtype, K) — the ship gate;
  - mc port, win-guard band (N in 8-32K);
  - 1cta port (16-bit small-N prod band);
  - loss cells < 0.98 listed.
Decision rule (RESUME_POST_ITER7.md section 2): only discuss a default
switch if the mc region shows a >=1.01 positive band AND no <0.98 loss
band anywhere.

Usage: python3 analyze_qfracs_ab.py [<root>]
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op26_qfracs_ab"


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def load():
    cells = {}
    for scen_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for jf in sorted(scen_dir.glob("results_K*.jsonl")):
            kd = jf.stem[len("results_K"):]
            rep = scen_dir / "nsys_reps" / f"qfracs_K{kd}.nsys-rep"
            kern = parse_rep(rep) if rep.exists() else {}
            for line in jf.read_text().splitlines():
                r = json.loads(line)
                if "error" in r:
                    print(f"  ERROR cell: {r['port']} {r['op']} K{r['K']} "
                          f"{r['dtype']} N{r['N']} BS{r['BS']}: {r['error']}")
                    continue
                us = kern.get(r["range_cold"])
                if us is None:
                    continue
                k = (r["scenario"], r["port"], r["K"], r["dtype"],
                     r["N"], r["BS"])
                cells.setdefault(k, {})[r["op"]] = us
                if r.get("exact") == "FAIL":
                    print(f"  EXACT FAIL: {k} {r['op']}")
    return cells


def main():
    cells = load()
    pairs = {k: v["m2d"] / v["uh4"] for k, v in cells.items()
             if "m2d" in v and "uh4" in v}
    print(f"paired cells: {len(pairs)}")

    def view(name, sel):
        b = defaultdict(list)
        for k, sp in pairs.items():
            if sel(k):
                b[(k[0], k[3], k[2])].append(sp)
        print(f"\n== {name} ==")
        allv = []
        for kk in sorted(b):
            v = b[kk]
            allv += v
            print(f"  {kk[0]:5} {kk[1]:5} K{kk[2]:<5} n={len(v):3} "
                  f"gm={gm(v):.4f} min={min(v):.3f} max={max(v):.3f}")
        if allv:
            print(f"  ALL: n={len(allv)} gm={gm(allv):.4f}")

    view("mc port, mc dispatch region (N>=65536) — SHIP GATE",
         lambda k: k[1] == "mc" and k[4] >= 65536)
    view("mc port, win-guard band (N in 8-32K)",
         lambda k: k[1] == "mc" and k[4] < 65536)
    view("1cta port (16-bit small-N prod band)",
         lambda k: k[1] == "r1cta")

    print("\n== loss cells (uh4 slower, ratio < 0.98) ==")
    losses = sorted((sp, k) for k, sp in pairs.items() if sp < 0.98)
    for sp, k in losses:
        print(f"  {sp:.3f}  {k}")
    if not losses:
        print("  none")

    print("\n== per-cell detail (mc dispatch region) ==")
    for k in sorted(k for k in pairs if k[1] == "mc" and k[4] >= 65536):
        print(f"  {k[0]:5} K{k[2]} {k[3]:5} N{k[4]:>7} BS{k[5]:<3} "
              f"m2d/uh4={pairs[k]:.4f}")


if __name__ == "__main__":
    main()
