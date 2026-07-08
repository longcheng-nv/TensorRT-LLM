# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op21 P0-batch — verdict tables for the multi-arm nsys A/B (ab_p0batch.py).

Reads results/nsys/p0batch/<prefix>_<scen>_<dtype>.{nsys-rep,jsonl}; joins the
paired NVTX ranges (c|<arm>|<cell>, median pure-kernel us via
report/parse_nsys_full.parse_rep) and prints per-cell arm times + pairwise
ratios + geomeans.

Usage: python3 scripts/parse_p0batch.py <prefix> <dtype> [scen ...]
  e.g.  python3 scripts/parse_p0batch.py ab3 fp32
        python3 scripts/parse_p0batch.py hb3 fp32 best worst
        python3 scripts/parse_p0batch.py ab3 bf16
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP21 = HERE.parents[0]
sys.path.insert(0, str(OP21.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

OUT = OP21 / "results" / "nsys" / "p0batch"


def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ab3"
    dtype = sys.argv[2] if len(sys.argv) > 2 else "fp32"
    scens = sys.argv[3:] or ["best", "worst", "real"]
    # pairwise ratios reported as time(a)/time(b) -> >1 means b faster
    pairs = [("orig", "shipped"), ("legacy", "shipped"), ("orig", "legacy")]
    agg = defaultdict(list)
    for scen in scens:
        rep = OUT / f"{prefix}_{scen}_{dtype}.nsys-rep"
        jl = OUT / f"{prefix}_{scen}_{dtype}.jsonl"
        if not rep.exists() or not jl.exists():
            print(f"-- {scen}/{dtype}: missing rep/jsonl, skip")
            continue
        kern = parse_rep(rep)
        print(f"\n== {prefix} scenario {scen} {dtype} ==  "
              f"({len(kern)} nvtx ranges)")
        hdr = (f"{'cell':>24} {'path':>8} {'orig':>8} {'legacy':>8} "
               f"{'shipped':>8} | {'o/s':>6} {'l/s':>6} {'o/l':>6}  exact")
        print(hdr)
        ratios = defaultdict(list)
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            if "error" in r:
                print(f"{r['K']}/{r['N']}/BS{r['BS']}: ERROR {r['error']}")
                continue
            base = f"{r['scenario']}|{r['K']}|{r['dtype']}|{r['N']}|{r['BS']}"
            t = {a: kern.get(f"c|{a}|{base}") for a in r["arms"]}
            cell = f"K{r['K']} N{r['N']} BS{r['BS']}"
            ex = "/".join((r.get(f"exact_{a}", "?") or "?")[:2]
                          for a in r["arms"])
            def fmt(v):
                return f"{v:8.2f}" if v else f"{'-':>8}"
            rr = {}
            for a, b in pairs:
                if t.get(a) and t.get(b):
                    rr[(a, b)] = t[a] / t[b]
                    ratios[(a, b)].append(t[a] / t[b])
                    agg[(scen, a, b)].append(t[a] / t[b])
            def fr(p):
                return f"{rr[p]:6.3f}" if p in rr else f"{'-':>6}"
            print(f"{cell:>24} {r.get('ms_path','-'):>8} {fmt(t.get('orig'))} "
                  f"{fmt(t.get('legacy'))} {fmt(t.get('shipped'))} | "
                  f"{fr(pairs[0])} {fr(pairs[1])} {fr(pairs[2])}  {ex}")
        for p in pairs:
            if ratios[p]:
                w = sum(x > 1 for x in ratios[p])
                print(f"-- {scen}: gm {p[0]}/{p[1]} = {gm(ratios[p]):.3f} "
                      f"(win({p[1]}) {w}/{len(ratios[p])})")
    print("\n== aggregate over scenarios ==")
    for (scen, a, b), xs in sorted(agg.items()):
        pass
    for p in pairs:
        xs = [x for (s, a, b), v in agg.items() if (a, b) == p for x in v]
        if xs:
            print(f"ALL {p[0]}/{p[1]}: gm={gm(xs):.3f} "
                  f"win({p[1]})={sum(x>1 for x in xs)}/{len(xs)}")


if __name__ == "__main__":
    main()
