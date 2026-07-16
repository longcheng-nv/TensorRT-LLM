# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse op35 L2 nsys A/B rounds -> per-cell median-of-3 verdict CSV.

Reads /tmp/op35_nsys/ab_r{1..3}_s{0..3}.nsys-rep. Per cell/arm/round the
nvtx_kern_sum us; verdict per cell = median over rounds of (base/var).
Writes results/nsys_ab_verdict.csv + prints 3-axis summary.
"""
import csv
import math
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OP35 = _HERE.parent
sys.path.insert(0, str(_OP35.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def main():
    per = {}   # (cell, arm) -> {round: us}
    for sh in range(8):
        rep = Path(f"/tmp/op35_nsys/ab_s{sh}.nsys-rep")
        if not rep.exists():
            continue
        for rng, us in parse_rep(rep).items():
            parts = rng.split("|")
            if len(parts) != 4 or parts[0] != "c":
                continue
            per.setdefault((parts[1], parts[2]), {})[int(parts[3][1:])] = us
    cells = sorted({c for c, _ in per})
    rows = []
    for c in cells:
        b = per.get((c, "base"), {})
        v = per.get((c, "var"), {})
        rounds = sorted(set(b) & set(v))
        if not rounds:
            continue
        ratios = [b[r] / v[r] for r in rounds]
        rows.append(dict(cell=c,
                         base_us=round(st.median(b[r] for r in rounds), 2),
                         var_us=round(st.median(v[r] for r in rounds), 2),
                         ratio_med=round(st.median(ratios), 4),
                         ratio_min=round(min(ratios), 4),
                         ratio_max=round(max(ratios), 4),
                         n_rounds=len(rounds)))
    out = _OP35 / "results" / "nsys_ab_verdict.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
    for axis, pred in (("synth-best", lambda c: c.startswith("synth_best")),
                       ("synth-worst", lambda c: c.startswith("synth_worst")),
                       ("real", lambda c: c.startswith("real")),
                       ("K2048", lambda c: "K2048" in c or c.startswith("real_v32")),
                       ("ALL", lambda c: True)):
        sub = [r for r in rows if pred(r["cell"])]
        if not sub:
            continue
        rs = [r["ratio_med"] for r in sub]
        lose = [r for r in sub if r["ratio_med"] < 0.97]
        print(f"{axis:12s} n={len(sub):2d} geomean={gm(rs):.4f} "
              f"min={min(rs):.3f} max={max(rs):.3f} cells<0.97: {len(lose)}")
    for r in sorted(rows, key=lambda x: x["ratio_med"])[:6]:
        print("  worst:", r["cell"], r["ratio_med"], f"({r['base_us']}->{r['var_us']}us)")
    for r in sorted(rows, key=lambda x: -x["ratio_med"])[:6]:
        print("  best :", r["cell"], r["ratio_med"], f"({r['base_us']}->{r['var_us']}us)")
    print("wrote", out)


if __name__ == "__main__":
    main()
