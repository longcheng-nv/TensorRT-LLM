# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op27 verify — verdict for the 16-bit K2048 tail same-node A/B."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

OUT = HERE / "results" / "nsys" / "ab_tail16"


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


for dt in ("bf16", "fp16"):
    rep = OUT / f"real_{dt}.nsys-rep"
    jl = OUT / f"real_{dt}.jsonl"
    if not rep.exists():
        print(f"missing {rep}")
        continue
    kern = parse_rep(rep)
    ratios = []
    print(f"\n== real {dt} ==  ({len(kern)} ranges)")
    for line in jl.read_text().splitlines():
        r = json.loads(line)
        tag = f"{r['scenario']}|{r['K']}|{r['dtype']}|{r['N']}|{r['BS']}"
        a = kern.get(f"c|notail|{tag}")
        b = kern.get(f"c|tail|{tag}")
        if not (a and b):
            print(f"  N{r['N']:>8}: missing range")
            continue
        ratios.append(a / b)
        print(f"  N{r['N']:>8}: notail {a:7.2f}  tail {b:7.2f}  "
              f"notail/tail {a/b:5.3f}  exact {r.get('exact_notail')}/"
              f"{r.get('exact_tail')}")
    print(f"  gm notail/tail = {gm(ratios):.3f}  (>1 = tail faster)")
