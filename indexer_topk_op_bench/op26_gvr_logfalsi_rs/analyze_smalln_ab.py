# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-N R0-ladder gate A/B verdict: op26_1cta (plain iter5 P2) vs
op26_r0 (R0 ladder v0.2) batch-paired at N<=32768.

Input = results_b200_op26_smalln_ab (3-arm batch-paired nsys sweep,
real/best/worst x bs x 3K x 3dtype, OP22RR_NS=4096..32768). Per-cell
metric = us_cold(op26_r0) / us_cold(op26_1cta)  (>1 = plain wins = the
R0 ladder is a net tax at that cell). The anchor arm (gvr_cutedsl) is
context only.

Views:
  - gm by (dtype, N) pooled over scenario/K/BS  — the gate axis;
  - gm by (dtype, K, N)                          — band check;
  - gm by (scenario, dtype, N)                   — worst-axis focus;
  - cells where R0 wins >=1.02 inside a candidate OFF region (would-be
    regression list if the gate turns R0 off there).

Decision rule (mirror of dispatch_p4rs_mc_op26 verdict style): per dtype,
N_R0_MIN = smallest N where plain/r0 gm <= 1.0 (R0 at least washes) and
stays <=1.0 for all larger N; require no systematic (K, N) band inside
the OFF region where R0 wins.

Usage: python3 analyze_smalln_ab.py [<root>]
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op26_smalln_ab"
SUBS = [("bs", "bs_scaling")]


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def load():
    cells = {}
    for scen_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for _sw, sub in SUBS:
            f = scen_dir / sub / "results.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if "us_cold" not in r:
                    continue
                k = (r["scenario"], r["sweep"], r["K"], r["dtype"],
                     r["N"], r["BS"])
                cells.setdefault(k, {})[r["op"]] = r["us_cold"]
    return cells


def show(title, groups):
    print(f"\n== {title}")
    for k in sorted(groups):
        v = groups[k]
        print(f"  {str(k):>28}: n={len(v):>3} gm {gm(v):.3f} "
              f"min {min(v):.3f} max {max(v):.3f}")


def main():
    cells = load()
    pairs = {}       # plain-over-r0 ratio: us_cold(r0)/us_cold(1cta)
    anchor = {}      # context: 1cta vs anchor
    for k, v in cells.items():
        if "op26_r0" in v and "op26_1cta" in v:
            pairs[k] = v["op26_r0"] / v["op26_1cta"]
        if "op26_1cta" in v and "gvr_cutedsl" in v:
            anchor[k] = v["gvr_cutedsl"] / v["op26_1cta"]
    print(f"paired cells: {len(pairs)}  (anchor-context cells: {len(anchor)})")

    by_dt_n = defaultdict(list)
    by_dt_k_n = defaultdict(list)
    by_scen_dt_n = defaultdict(list)
    for (scen, _sw, K, dt, N, BS), r in pairs.items():
        by_dt_n[(dt, N)].append(r)
        by_dt_k_n[(dt, K, N)].append(r)
        by_scen_dt_n[(scen, dt, N)].append(r)

    show("plain/r0 gm by (dtype, N)  [>1 = R0 ladder is net tax]", by_dt_n)
    show("plain/r0 gm by (dtype, K, N)", by_dt_k_n)
    show("plain/r0 gm by (scenario, dtype, N)", by_scen_dt_n)

    # candidate OFF regions and their would-be regressions (R0 wins >=1.02
    # inside the region, i.e. plain/r0 <= 1/1.02)
    for dt in ("fp32", "bf16", "fp16"):
        for nmin in (16384, 32768, 65536):
            reg = [(k, r) for k, r in pairs.items()
                   if k[3] == dt and k[4] < nmin]
            if not reg:
                continue
            vals = [r for _k, r in reg]
            losers = [(k, r) for k, r in reg if r <= 1 / 1.02]
            print(f"\n-- OFF region candidate: {dt} N<{nmin}: "
                  f"n={len(vals)} gm {gm(vals):.3f} "
                  f"(R0-wins>=1.02 cells: {len(losers)})")
            for k, r in sorted(losers, key=lambda t: t[1])[:12]:
                scen, _sw, K, _dt, N, BS = k
                print(f"     R0 wins {1/r:.3f}: {scen} K={K} N={N} BS={BS}")


if __name__ == "__main__":
    main()
