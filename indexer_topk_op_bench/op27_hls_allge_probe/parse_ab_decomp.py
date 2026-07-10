# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op27 iter1 — verdict table for the small-N decomposition A/B.

Reads results/nsys/ab_decomp/ab_<scen>_fp32.{nsys-rep,jsonl}; joins NVTX
ranges (c|<arm>|<cell>, median pure-kernel us) and prints per-cell times
normalized to the plain gvr_cutedsl floor, plus the decomposition ratios:
    w3a tax   = w3a_s1 / stock_s1      (K512/K1024)
    slot2 tax = ship / w3a_s1          (K512/K1024)
    tail gain = stock_s1 / tail_s1     (K2048)
    slot2 tax = ship / stock_s1        (K2048)

Usage: python3 parse_ab_decomp.py [best worst real]
"""
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

OUT = HERE / "results" / "nsys" / ("ab_decomp" + os.environ.get("SUFFIX", ""))
ARMS_LO = ("plain", "stock_s1", "w3a_s1", "ship")
ARMS_HI = ("plain", "stock_s1", "ship", "tail_s1")


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main():
    scens = sys.argv[1:] or ["best", "worst", "real"]
    agg = {}
    for scen in scens:
        rep = OUT / f"ab_{scen}_fp32.nsys-rep"
        jl = OUT / f"ab_{scen}_fp32.jsonl"
        if not rep.exists() or not jl.exists():
            print(f"-- {scen}: missing rep/jsonl, skip")
            continue
        kern = parse_rep(rep)
        print(f"\n== scenario {scen} ==  ({len(kern)} nvtx ranges)")
        print(f"{'cell':>16} {'plain':>7} {'stock':>7} {'w3a':>7} {'ship':>7} "
              f"{'tail':>7} | {'w3a/stk':>7} {'shp/w3a':>7} {'stk/tail':>8} "
              f"{'plain/shp':>9} exact")
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            cell = f"K{r['K']} N{r['N']}"
            if "error" in r:
                print(f"{cell:>16} ERROR {r['error'][:60]}")
                continue
            tag = (f"{r['scenario']}|{r['K']}|{r['dtype']}|{r['N']}|{r['BS']}")
            arms = ARMS_HI if r["K"] == 2048 else ARMS_LO
            us = {a: kern.get(f"c|{a}|{tag}") for a in arms}
            if any(v is None for v in us.values()):
                print(f"{cell:>16} missing nvtx median {us}")
                continue
            w3a_stk = (us.get("w3a_s1") or float("nan")) / us["stock_s1"]
            shp_w3a = us["ship"] / us.get("w3a_s1", us["stock_s1"])
            stk_tail = us["stock_s1"] / us["tail_s1"] if "tail_s1" in us else float("nan")
            pl_shp = us["plain"] / us["ship"]
            ex = "/".join(str(r.get(f"exact_{a}", "?"))[:4] for a in arms)
            key = (scen, r["K"])
            agg.setdefault(key, {"w3a": [], "slot": [], "tail": [], "flr": []})
            if r["K"] != 2048:
                agg[key]["w3a"].append(w3a_stk)
                agg[key]["slot"].append(shp_w3a)
            else:
                agg[key]["tail"].append(stk_tail)
                agg[key]["slot"].append(us["ship"] / us["stock_s1"])
            agg[key]["flr"].append(pl_shp)
            print(f"{cell:>16} {us['plain']:7.2f} {us['stock_s1']:7.2f} "
                  f"{us.get('w3a_s1', float('nan')):7.2f} {us['ship']:7.2f} "
                  f"{us.get('tail_s1', float('nan')):7.2f} | {w3a_stk:7.3f} "
                  f"{shp_w3a:7.3f} {stk_tail:8.3f} {pl_shp:9.3f} {ex}")
    print("\n== geomeans (>1 = second arm SLOWER for tax cols; "
          ">1 = ship faster than plain for floor col) ==")
    print(f"{'scen':>6} {'K':>5} {'w3a tax':>8} {'slot2 tax':>9} "
          f"{'tail gain':>9} {'plain/ship':>10}")
    for (scen, K), d in sorted(agg.items()):
        print(f"{scen:>6} {K:>5} {gm(d['w3a']):8.3f} {gm(d['slot']):9.3f} "
              f"{gm(d['tail']):9.3f} {gm(d['flr']):10.3f}")


if __name__ == "__main__":
    main()
