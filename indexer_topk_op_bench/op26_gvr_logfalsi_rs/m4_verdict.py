#!/usr/bin/env python3
"""M4 verdict for iter6b: op26_r0mc (cluster R0) on the gap cells.

Grid: K1024 fp32 + K2048 fp16, sweeps seqlen/bs/bs_hugeN, scenarios
real/best/worst, arms = gvr_multicta_cutedsl (mc anchor) + op26_r0mc +
op26_r0 (1cta) + radix_cutedsl, all batch-internal pairs.

Gap-cell focus (RESUME_ITER6B_PROMPT §2.3): low-BS >=131K + 16-bit
large N + hugeN — where the 1cta arm hits the row-parallelism wall.

Usage: python3 m4_verdict.py [<mcab_root>]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
R = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results_b200_op26_iter6b_mcab"
SUBS = ["seqlen_sweep", "bs_scaling", "bs_hugeN"]

rows = defaultdict(dict)
for sd in sorted(R.iterdir()):
    if not sd.is_dir():
        continue
    for sub in SUBS:
        f = sd / sub / "results.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "us" not in r:
                continue
            rows[(r["scenario"], r["sweep"], r["K"], r["dtype"],
                  r["N"], r["BS"])][r["op"]] = r["us"]


def gm(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def table(pred, label):
    print(f"== {label} ==")
    print(f"  {'grp':22s} {'vs mc-anchor':>12s} {'vs 1cta-r0':>12s} "
          f"{'vs radix':>10s} {'cells':>6s}")
    for kd in ((1024, "fp32"), (2048, "fp16")):
        for scen in ("real", "best", "worst"):
            sel = [o for k, o in rows.items()
                   if k[0] == scen and (k[2], k[3]) == kd and pred(k)
                   and "op26_r0mc" in o]
            va = gm([o["gvr_multicta_cutedsl"] / o["op26_r0mc"]
                     for o in sel if "gvr_multicta_cutedsl" in o])
            v1 = gm([o["op26_r0"] / o["op26_r0mc"]
                     for o in sel if "op26_r0" in o])
            vr = gm([o["radix_cutedsl"] / o["op26_r0mc"]
                     for o in sel if "radix_cutedsl" in o])
            print(f"  {scen:6s} K={kd[0]:4d} {kd[1]:5s}  {va:12.3f} "
                  f"{v1:12.3f} {vr:10.3f} {len(sel):6d}")
    print()


table(lambda k: True, "ALL cells")
table(lambda k: k[4] >= 131072 and k[5] <= 16 and k[4] <= 262144,
      "GAP: N in [131K, 262K], BS<=16")
table(lambda k: k[4] >= 524288, "hugeN >= 512K")
table(lambda k: 8192 <= k[4] <= 262144, "core 8K-262K")

# worst cells vs the mc anchor (any hole?)
print("== op26_r0mc vs mc-anchor: (scen,K,dtype,N)-groups gm<0.95 ==")
agg = defaultdict(list)
for k, o in rows.items():
    if "gvr_multicta_cutedsl" in o and "op26_r0mc" in o:
        agg[(k[0], k[2], k[3], k[4])].append(
            o["gvr_multicta_cutedsl"] / o["op26_r0mc"])
n = 0
for kk, v in sorted(agg.items(), key=lambda x: gm(x[1])):
    g = gm(v)
    if g < 0.95:
        print(f"  gm={g:.3f} {kk} cells={len(v)}")
        n += 1
print(f"  total: {n} groups")
