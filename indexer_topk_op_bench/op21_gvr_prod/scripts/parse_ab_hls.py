# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op21 iter13 — verdict table for the HLS log-falsi nsys A/B.

Reads results/nsys/iter13_ab_hls/ab_<scen>_fp32.{nsys-rep,jsonl}; joins the
paired NVTX ranges (c|old|<cell> / c|new|<cell>, median pure-kernel us via
report/parse_nsys_full.parse_rep) and prints old/new per cell + geomeans.

Usage: python3 scripts/parse_ab_hls.py [best worst real]
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP21 = HERE.parents[0]
sys.path.insert(0, str(OP21.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

import os
OUT = OP21 / "results" / "nsys" / ("iter13_ab_hls"
                                   + os.environ.get("SUFFIX", ""))


def main():
    scens = sys.argv[1:] or ["best", "worst", "real"]
    all_ratios = []
    for scen in scens:
        rep = OUT / f"ab_{scen}_fp32.nsys-rep"
        jl = OUT / f"ab_{scen}_fp32.jsonl"
        if not rep.exists() or not jl.exists():
            print(f"-- {scen}: missing rep/jsonl, skip")
            continue
        kern = parse_rep(rep)
        print(f"\n== scenario {scen} ==  ({len(kern)} nvtx ranges)")
        print(f"{'cell':>28} {'ms_path':>8} {'old_us':>8} {'new_us':>8} "
              f"{'old/new':>8}  exact")
        ratios = []
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            if "error" in r:
                print(f"{r['K']}/{r['N']}/BS{r['BS']}: ERROR {r['error']}")
                continue
            base = (f"{r['scenario']}|{r['K']}|{r['dtype']}|{r['N']}"
                    f"|{r['BS']}")
            uo = kern.get(f"c|old|{base}")
            un = kern.get(f"c|new|{base}")
            cell = f"K{r['K']} N{r['N']} BS{r['BS']}"
            ex = f"{r.get('exact_old','?')}/{r.get('exact_new','?')}"
            if uo is None or un is None or un <= 0:
                print(f"{cell:>28} {str(r.get('ms_path')):>8} "
                      f"{'—' if uo is None else f'{uo:.2f}':>8} "
                      f"{'—' if un is None else f'{un:.2f}':>8} "
                      f"{'—':>8}  {ex}")
                continue
            ratio = uo / un
            ratios.append(ratio)
            all_ratios.append(ratio)
            print(f"{cell:>28} {str(r.get('ms_path')):>8} {uo:>8.2f} "
                  f"{un:>8.2f} {ratio:>8.3f}  {ex}")
        if ratios:
            gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            print(f"-- {scen}: gm old/new = {gm:.3f} "
                  f"(win(new) {sum(1 for x in ratios if x > 1.0)}"
                  f"/{len(ratios)}, max {max(ratios):.3f}, "
                  f"min {min(ratios):.3f})")
    if all_ratios:
        gm = math.exp(sum(math.log(x) for x in all_ratios) / len(all_ratios))
        print(f"\n== ALL: gm old/new = {gm:.3f} ({len(all_ratios)} cells) ==")


if __name__ == "__main__":
    main()
