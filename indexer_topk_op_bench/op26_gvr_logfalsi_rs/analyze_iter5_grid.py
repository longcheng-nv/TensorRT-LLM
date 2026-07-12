# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 iter5 — headline judgment over the full-grid re-test root.

Pairs op26_1cta vs gvr_cutedsl per (scenario, sweep, K, dtype, N, BS) on
canonical cold-L2 us from the merged results.jsonl (run parse_op22_cached.py
first). Prints overall/per-dtype/per-scenario gm + win rate, the iter5d
pruned-region check, and any remaining loss cells (<0.95).

Usage: python3 analyze_iter5_grid.py [<root>]   default ../results_b200_op26a_iter5
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op26a_iter5"

SUBS = ["seqlen_sweep", "bs_scaling", "bs_hugeN"]
rows = defaultdict(dict)   # cellkey -> {op: us}
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

pairs = {k: (v["gvr_cutedsl"], v["op26_1cta"]) for k, v in rows.items()
         if "gvr_cutedsl" in v and "op26_1cta" in v}
print(f"paired cells: {len(pairs)}  (root={root.name})")


def gm(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def report(label, sel):
    ratios = [a / o for (a, o) in sel.values()]
    if not ratios:
        print(f"{label:34s}  (no cells)")
        return
    wins = sum(r > 1.0 for r in ratios)
    print(f"{label:34s} gm={gm(ratios):.4f}  win={wins}/{len(ratios)}"
          f" ({100*wins/len(ratios):.0f}%)")


report("OVERALL", pairs)
for dt in ("fp32", "bf16", "fp16"):
    report(f"  dtype={dt}", {k: v for k, v in pairs.items() if k[3] == dt})
for sc in ("real", "best", "worst"):
    report(f"  scenario={sc}", {k: v for k, v in pairs.items() if k[0] == sc})
for K in (512, 1024, 2048):
    report(f"  K={K}", {k: v for k, v in pairs.items() if k[2] == K})

print("\n-- iter5d pruned-region check --")
report("K2048 16-bit N=524288 (->stock)",
       {k: v for k, v in pairs.items()
        if k[2] == 2048 and k[3] in ("bf16", "fp16") and k[4] == 524288})
report("K2048 16-bit N=262144 (secant2)",
       {k: v for k, v in pairs.items()
        if k[2] == 2048 and k[3] in ("bf16", "fp16") and k[4] == 262144})
report("K1024 bf16 N=16384 (->stock)",
       {k: v for k, v in pairs.items()
        if k[2] == 1024 and k[3] == "bf16" and k[4] == 16384})
report("K1024 16-bit N in [32K,64K] (center)",
       {k: v for k, v in pairs.items()
        if k[2] == 1024 and k[3] in ("bf16", "fp16")
        and 32768 <= k[4] <= 65536})
report("K1024 fp32 N=131072 (pruned iter5b)",
       {k: v for k, v in pairs.items()
        if k[2] == 1024 and k[3] == "fp32" and k[4] == 131072})

print("\n-- remaining loss cells grouped (cell gm<0.95) --")
grp = defaultdict(list)
for k, (a, o) in pairs.items():
    grp[(k[2], k[3], k[4])].append(a / o)
bad = {g: gm(rs) for g, rs in grp.items() if gm(rs) < 0.95}
for g in sorted(bad, key=bad.get):
    print(f"  K={g[0]:5d} {g[1]:4s} N={g[2]:7d}  gm={bad[g]:.3f} "
          f"(cells={len(grp[g])})")
if not bad:
    print("  none")
