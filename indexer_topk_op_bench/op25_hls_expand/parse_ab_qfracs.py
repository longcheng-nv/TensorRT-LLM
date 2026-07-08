# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 Step 2 — verdict table for the S1a ladder nsys A/B.

Reads results/nsys/ab_qfracs/ab_<scen>_fp32.{nsys-rep,jsonl}; joins NVTX
ranges (c|base|<cell> / c|ship|<cell> / c|radix|<cell>, median pure-kernel
us) and prints base/ship + radix/ship per cell, per-scenario geomeans, and
the radix flip count (cells where ship>=radix but base<radix and v.v.).

Usage: python3 parse_ab_qfracs.py [best worst real]
"""
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

OUT = HERE / "results" / "nsys" / ("ab_qfracs" + os.environ.get("SUFFIX", ""))


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main():
    scens = sys.argv[1:] or ["best", "worst", "real"]
    for scen in scens:
        rep = OUT / f"ab_{scen}_fp32.nsys-rep"
        jl = OUT / f"ab_{scen}_fp32.jsonl"
        if not rep.exists() or not jl.exists():
            print(f"-- {scen}: missing rep/jsonl, skip")
            continue
        kern = parse_rep(rep)
        print(f"\n== scenario {scen} ==  ({len(kern)} nvtx ranges)")
        print(f"{'cell':>22} {'base_us':>8} {'ship_us':>8} {'radix_us':>9} "
              f"{'base/ship':>9} {'radix/ship':>10}  flip exact")
        r_bs, r_rs, flips = [], [], []
        env_bs, env_rs = [], []
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            cell = f"K{r['K']} N{r['N']} BS{r['BS']}"
            if "error" in r:
                print(f"{cell:>22} ERROR {r['error']}")
                continue
            tag = (f"{r['scenario']}|{r['K']}|{r['dtype']}|{r['N']}"
                   f"|{r['BS']}")
            ub = kern.get(f"c|base|{tag}")
            us = kern.get(f"c|ship|{tag}")
            ur = kern.get(f"c|radix|{tag}")
            if not (ub and us and ur):
                print(f"{cell:>22} missing nvtx median")
                continue
            bs_, rs_ = ub / us, ur / us
            r_bs.append(bs_)
            r_rs.append(rs_)
            flip = ""
            if ur / ub < 1.0 <= rs_:
                flip = "WIN"     # base lost to radix, ship beats it
                flips.append(cell)
            elif ur / ub >= 1.0 > rs_:
                flip = "LOSS"
            ex = f"{r.get('exact_base')}/{r.get('exact_ship')}" \
                 f"/{r.get('exact_radix')}"
            print(f"{cell:>22} {ub:8.2f} {us:8.2f} {ur:9.2f} "
                  f"{bs_:9.3f} {rs_:10.3f}  {flip:4s} {ex}")
        print(f"-- {scen} gm base/ship = {gm(r_bs):.3f}  "
              f"radix/ship = {gm(r_rs):.3f}  "
              f"ship-beats-radix {sum(x >= 1 for x in r_rs)}"
              f"/{len(r_rs)}  flips->WIN: {len(flips)}")


if __name__ == "__main__":
    main()
