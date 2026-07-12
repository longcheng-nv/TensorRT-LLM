# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 fin-root negative-cell map + row-parallelism scaling analysis.

Input = op22rr_op26r_raw.csv (same-node op26_r0auto / gvr anchor / radix
triplets from results_b200_op26_iter6final). Three views:

1. negative-cell buckets vs radix by (scenario, K, dtype, N-band);
2. gm + win%% by (N-band, BS) — exposes the bimodal structure the
   aggregate geomeans hide (losses concentrate at BS<=8, wins at
   saturation);
3. per-row latency scaling op26r vs radix over BS — shows both ops are
   latency-bound flat at BS 1-16, so the low-BS gap is the per-row
   critical-path ratio (radix row-internal CTAs scale with N, GVR cs
   caps at 4).

Usage: python3 analyze_fin_negatives.py [<raw_csv>]
       default ../op22_temporal_fixed_hr_bench/op22rr_op26r_raw.csv
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "op22_temporal_fixed_hr_bench" / "op22rr_op26r_raw.csv"


def band(n):
    n = int(n)
    if n < 8192:
        return "smallN"
    if n <= 262144:
        return "core"
    return "hugeN"


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def main():
    rows = list(csv.DictReader(open(RAW)))

    print("== 1. negative cells vs radix by (scen,K,dtype,band) ==")
    neg = [r for r in rows if r["speedup_vs_radix_same_node_cold"]
           and float(r["speedup_vs_radix_same_node_cold"]) < 1.0]
    print(f"total {len(rows)} cells, negative: {len(neg)} "
          f"({100 * len(neg) / len(rows):.1f}%)")
    b = defaultdict(list)
    for r in neg:
        b[(r["scenario"], r["K"], r["dtype"], band(r["N"]))].append(
            float(r["speedup_vs_radix_same_node_cold"]))
    for k in sorted(b, key=lambda k: (k[3], k[0], k[1], k[2])):
        v = b[k]
        print(f"  {k[0]:6} K{k[1]:5} {k[2]:5} {k[3]:7} n={len(v):3} "
              f"gm={gm(v):.3f} min={min(v):.3f}")

    print("\n== 2. gm + win% vs radix by (band, BS) ==")
    b = defaultdict(list)
    for r in rows:
        sp = r["speedup_vs_radix_same_node_cold"]
        if sp:
            b[(band(r["N"]), int(r["BS"]))].append(float(sp))
    for k in sorted(b):
        v = b[k]
        win = 100 * sum(1 for x in v if x >= 1) / len(v)
        print(f"  {k[0]:8}BS={k[1]:5} n={len(v):4} gm={gm(v):.3f} "
              f"win={win:3.0f}%")

    print("\n== 3. per-row latency scaling (real/bs sweep) ==")
    d = defaultdict(dict)
    for r in rows:
        if r["scenario"] != "real" or r["sweep"] != "bs":
            continue
        x = r["radix_cold_us_local"]
        d[(r["K"], r["dtype"], int(r["N"]))][int(r["BS"])] = (
            float(r["op26r_cold_us_local"]), float(x) if x else None)
    for key in (("1024", "bf16", 65536), ("1024", "bf16", 262144),
                ("1024", "bf16", 1048576)):
        if key not in d:
            continue
        print(f"  K{key[0]} {key[1]} N={key[2]}: "
              f"BS -> op26r us (us/row) | radix us (us/row)")
        for bs in sorted(d[key]):
            o, x = d[key][bs]
            tail = f"| {x:9.1f} ({x / bs:8.2f})" if x else ""
            print(f"    BS={bs:5}: {o:9.1f} ({o / bs:8.2f}) {tail}")


if __name__ == "__main__":
    main()
