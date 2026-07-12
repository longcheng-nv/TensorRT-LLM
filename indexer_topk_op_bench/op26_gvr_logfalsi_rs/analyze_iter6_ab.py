# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 iter6 — judge the R0 ladder A/B batches.

Reads the merged results.jsonl of the iter6 root (run parse_op22_cached.py
first) and prints, per (scenario, K, dtype) batch and per N band:
  - op26_r0 vs gvr_cutedsl (same-batch anchor ratio; >1 = r0 faster)
  - op26_r0 vs radix_cutedsl (the campaign objective, bar = 1.10)
  - op26_r0 vs sglang_streaming (fp32 K<=1024 batches)
  - op26_r0 vs op26_1cta (cross-root via anchor transfer: the iter5 grid's
    1cta/anchor ratio on the same cell, both node-local)

Usage: python3 analyze_iter6_ab.py [<iter6_root>] [<iter5_root>]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R6 = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op26_iter6"
R5 = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    HERE.parents[0] / "results_b200_op26a_iter5"

SUBS = ["seqlen_sweep", "bs_scaling", "bs_hugeN"]


def load(root):
    rows = defaultdict(dict)   # cellkey -> {op: us_cold}
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


rows6 = load(R6)
rows5 = load(R5)
print(f"iter6 cells: {len(rows6)}   iter5 ref cells: {len(rows5)}")

BANDS = [(8192, 32768), (65536, 131072), (262144, 262144),
         (524288, 1048576), (4096, 4096)]


def band_of(N):
    for lo, hi in BANDS:
        if lo <= N <= hi:
            return f"{lo//1024}K-{hi//1024}K" if lo != hi else f"{lo//1024}K"
    return "other"


by_batch = defaultdict(lambda: defaultdict(list))
loss_cells = []
for k, ops in sorted(rows6.items()):
    if "op26_r0" not in ops or "gvr_cutedsl" not in ops:
        continue
    scen, sweep, K, dt, N, BS = k
    r0 = ops["op26_r0"]
    batch = (scen, K, dt)
    band = band_of(N)
    by_batch[batch][("anchor", band)].append(ops["gvr_cutedsl"] / r0)
    if "radix_cutedsl" in ops:
        rr = ops["radix_cutedsl"] / r0
        by_batch[batch][("radix", band)].append(rr)
        if N >= 8192 and rr < 1.10:
            loss_cells.append((k, rr))
    if "sglang_streaming" in ops:
        by_batch[batch][("sglang", band)].append(ops["sglang_streaming"] / r0)
    o5 = rows5.get(k, {})
    if "op26_1cta" in o5 and "gvr_cutedsl" in o5:
        t5 = o5["gvr_cutedsl"] / o5["op26_1cta"]     # 1cta vs anchor (iter5)
        t6 = ops["gvr_cutedsl"] / r0                 # r0 vs anchor (iter6)
        by_batch[batch][("vs_1cta", band)].append(t6 / t5)

for batch in sorted(by_batch):
    scen, K, dt = batch
    print(f"\n== {scen} K={K} {dt} ==")
    rivals = sorted({r for (r, _) in by_batch[batch]})
    bands = ["4K", "8K-32K", "64K-128K", "256K", "512K-1024K"]
    hdr = f"  {'rival':8s}" + "".join(f" {b:>12s}" for b in bands) + f" {'ALL':>9s}"
    print(hdr)
    for rv in rivals:
        cells_all = []
        line = f"  {rv:8s}"
        for b in bands:
            v = by_batch[batch].get((rv, b), [])
            cells_all += v
            line += f" {gm(v):12.3f}" if v else f" {'-':>12s}"
        line += f" {gm(cells_all):9.3f}"
        print(line)

print(f"\n-- cells vs radix below the 1.10 bar (N>=8K): {len(loss_cells)}")
grp = defaultdict(list)
for (scen, sweep, K, dt, N, BS), rr in loss_cells:
    grp[(scen, K, dt, N)].append(rr)
for g in sorted(grp, key=lambda g: gm(grp[g])):
    print(f"  {g[0]:5s} K={g[1]:5d} {g[2]:4s} N={g[3]:7d}: gm={gm(grp[g]):.3f} "
          f"cells={len(grp[g])}")
