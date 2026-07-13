# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 backlog-3 verdict: kC-diet K512@1536 paired nsys A/B.

Input = results_b200_op26_kcdiet_ab (sweep_kcdiet.py batches). Per-cell
metric = us_cold(kc5120) / us_cold(kc1536)  (>1 = diet faster). Views:
  - 1cta 16-bit band (fp16 16-32K, the small-N prod band where the
    occupancy story should bite);
  - 1cta BS>=128 route (fp32 large-N high-BS);
  - mc spot guard (expected insensitive: latency-bound).
Ship bar: coherent band gm >= 1.01 with no <0.98 loss cells anywhere
(diet raises fb_fix pressure on ~12.5% of real cells — a win must beat
that tax, not just tie). Shipping additionally requires the FULL 582
gate re-run per RESUME_POST_ITER7.md section 4.

Usage: python3 analyze_kcdiet_ab.py [<root>]
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
    HERE.parents[0] / "results_b200_op26_kcdiet_ab"


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def main():
    cells = {}
    for scen_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        rep = scen_dir / "nsys_reps" / "kcdiet_K512.nsys-rep"
        kern = parse_rep(rep) if rep.exists() else {}
        for jf in sorted(scen_dir.glob("results_K*.jsonl")):
            for line in jf.read_text().splitlines():
                r = json.loads(line)
                if "error" in r:
                    print(f"  ERROR cell: {r['port']} {r['op']} {r['dtype']} "
                          f"N{r['N']} BS{r['BS']}: {r['error']}")
                    continue
                us = kern.get(r["range_cold"])
                if us is None:
                    continue
                k = (r["scenario"], r["port"], r["dtype"], r["N"], r["BS"])
                cells.setdefault(k, {})[r["op"]] = us
                if r.get("exact") == "FAIL":
                    print(f"  EXACT FAIL: {k} {r['op']}")

    pairs = {k: v["kc5120"] / v["kc1536"] for k, v in cells.items()
             if "kc5120" in v and "kc1536" in v}
    print(f"paired cells: {len(pairs)}")

    def view(name, sel):
        b = defaultdict(list)
        for k, sp in pairs.items():
            if sel(k):
                b[(k[0], k[2])].append(sp)
        print(f"\n== {name} ==")
        allv = []
        for kk in sorted(b):
            v = b[kk]
            allv += v
            print(f"  {kk[0]:5} {kk[1]:5} n={len(v):3} gm={gm(v):.4f} "
                  f"min={min(v):.3f} max={max(v):.3f}")
        if allv:
            print(f"  ALL: n={len(allv)} gm={gm(allv):.4f}")

    view("1cta 16-bit band (fp16 16-32K)",
         lambda k: k[1] == "r1cta" and k[2] == "fp16")
    view("1cta BS>=128 route (fp32 large-N)",
         lambda k: k[1] == "r1cta" and k[2] == "fp32")
    view("mc spot guard (fp32)", lambda k: k[1] == "mc")

    print("\n== per-cell detail ==")
    for k in sorted(pairs):
        print(f"  {k[0]:5} {k[1]:5} {k[2]:5} N{k[3]:>7} BS{k[4]:<4} "
              f"kc5120/kc1536={pairs[k]:.4f}")

    losses = sorted((sp, k) for k, sp in pairs.items() if sp < 0.98)
    print(f"\nloss cells <0.98: {len(losses)}")
    for sp, k in losses:
        print(f"  {sp:.3f}  {k}")


if __name__ == "__main__":
    main()
