#!/usr/bin/env python3
"""M3 verdict for iter6 (PLAN_ITER6): op26_r0 over the full 81-batch grid.

Criteria:
  (1) vs radix_cutedsl full-grid gm >= 1.0
  (2) 8K-262K win10 rate (cells with ratio >= 1.10) clearly above the
      iter5-1cta baseline on the same cells
  (3) vs anchor (gvr_cutedsl): no NEW <0.9 hole (worst-axis ~0.94 known)

Usage: python3 m3_verdict.py [<iter6_root>] [<iter5_root>]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
R6 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results_b200_op26_iter6grid"
R5 = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results_b200_op26a_iter5"
SUBS = ["seqlen_sweep", "bs_scaling", "bs_hugeN"]


def load(root):
    rows = defaultdict(dict)
    for scen_dir in sorted(root.iterdir()):
        if not scen_dir.is_dir():
            continue
        for sub in SUBS:
            f = scen_dir / sub / "results.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if "us" not in r:
                    continue
                k = (r["scenario"], r["sweep"], r["K"], r["dtype"],
                     r["N"], r["BS"])
                rows[k][r["op"]] = r["us"]
    return rows


def gm(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


rows6, rows5 = load(R6), load(R5)
CORE = lambda n: 8192 <= n <= 262144  # deployment battleground

# (1) vs radix
for label, pred in (("full-grid", lambda n: True), ("8K-262K", CORE),
                    ("hugeN 512K-1M", lambda n: n >= 524288)):
    rs = [o["radix_cutedsl"] / o["op26_r0"] for k, o in rows6.items()
          if pred(k[4]) and "radix_cutedsl" in o and "op26_r0" in o]
    w10 = sum(1 for x in rs if x >= 1.10)
    print(f"(1) vs radix [{label:14s}]: gm={gm(rs):.3f} cells={len(rs)} "
          f"win10={w10}/{len(rs)} ({100*w10/max(len(rs),1):.1f}%)")

# (2) win10 uplift vs iter5-1cta on shared 8K-262K cells. The iter5 ref
# grid carries only [gvr_cutedsl, op26_1cta] — recover its implied
# radix ratio via anchor transfer:
#   (radix/1cta)_5 ~= (radix/anchor)_6 * (anchor/1cta)_5
sh6, sh5 = [], []
for k, o in rows6.items():
    if not CORE(k[4]):
        continue
    o5 = rows5.get(k)
    if (o5 and "radix_cutedsl" in o and "op26_r0" in o
            and "gvr_cutedsl" in o and "gvr_cutedsl" in o5
            and "op26_1cta" in o5):
        sh6.append(o["radix_cutedsl"] / o["op26_r0"])
        sh5.append((o["radix_cutedsl"] / o["gvr_cutedsl"])
                   * (o5["gvr_cutedsl"] / o5["op26_1cta"]))
w6 = sum(1 for x in sh6 if x >= 1.10)
w5 = sum(1 for x in sh5 if x >= 1.10)
print(f"(2) 8K-262K shared cells={len(sh6)}: win10 iter6={w6} "
      f"({100*w6/max(len(sh6),1):.1f}%) vs iter5-1cta(implied)={w5} "
      f"({100*w5/max(len(sh5),1):.1f}%); gm iter6={gm(sh6):.3f} "
      f"iter5={gm(sh5):.3f}")

# (3) anchor holes <0.9 by (scenario, K, dtype, band)
BANDS = [(4096, 4096), (8192, 32768), (65536, 131072), (262144, 262144),
         (524288, 1048576)]
holes = []
agg = defaultdict(list)
for k, o in rows6.items():
    if "gvr_cutedsl" in o and "op26_r0" in o:
        band = next((b for b in BANDS if b[0] <= k[4] <= b[1]), None)
        if band:
            agg[(k[0], k[2], k[3], band)].append(
                o["gvr_cutedsl"] / o["op26_r0"])
for kk, v in sorted(agg.items()):
    g = gm(v)
    if g < 0.90:
        holes.append((g, kk, len(v)))
print(f"(3) vs anchor <0.90 holes (scen,K,dtype,band): {len(holes)}")
for g, kk, n in sorted(holes):
    print(f"    gm={g:.3f} {kk} cells={n}")
