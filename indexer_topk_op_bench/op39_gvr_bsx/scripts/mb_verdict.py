# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 nsys-axis verdict for the fused mb screen: join NVTX ranges
c|fused|<cell>|BS<n> with report pr (bs_real_layers.csv). Median over reps."""
import argparse
import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default=str(HERE.parent / "results" / "nsys" / "f1.nsys-rep"))
    args = ap.parse_args()
    kern = parse_rep(args.rep)
    pr = {}
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" /
                                 "bs_real_layers.csv")):
        pr[(f"{r['model']}_{r['isl']}_L{int(r['L']):02d}", int(r["BS"]))] = float(r["pr"])
    rows = []
    for rng, us in kern.items():
        parts = rng.split("|")
        if len(parts) != 4 or parts[1] != "fused":
            continue
        cname, bs = parts[2], int(parts[3][2:])
        p = pr.get((cname, bs))
        rows.append((cname, bs, us, p, p / us if p else None))
    rows.sort(key=lambda r: (r[0], r[1]))
    sp = [r[4] for r in rows if r[4]]
    for cname, bs, us, p, x in rows:
        print(f"{cname:20s} BS{bs:5d} fused={us:8.2f} pr={p or 0:8.2f} x={x or 0:6.3f}")
    print(f"\nnsys-axis: {len(sp)} cases  gm "
          f"{statistics.geometric_mean(sp):.4f}  mean {statistics.mean(sp):.4f}  "
          f"min {min(sp):.4f}  <1.0: {sum(1 for v in sp if v < 1)}/{len(sp)}")


if __name__ == "__main__":
    main()
