# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse the op35 nsys oracle reps into a per-cell phase-block decomposition.

Reads /tmp/op35_nsys/oracle_<g>.nsys-rep (g=0..3), NVTX ranges 'c|<cell>|<arm>'
via report/parse_nsys_full.parse_rep (evict-filtered kernel-sum per call).
Writes results/nsys_oracle_decomp.csv:
  cell, N, K, cs, base, p3, p4, floor,
  t_P3 = base - p3            (P3 collect scan)
  t_P4blk = base - p4         (handoff2 + P4 + writeback)
  t_floor = floor             (P1 + launch + emit)
  t_mid = p4 - floor - (identity-emit ~= writeback, ignored)
        = P1b + P2 + falsi recounts + handoff1  [since p4-arm ran P1..P3]
        minus t_P3 (p4 arm still ran P3) -> t_mid = (p4 - floor) - t_P3
"""
import csv
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OP35 = _HERE.parent
sys.path.insert(0, str(_OP35.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

def main():
    kern = {}
    for g in range(4):
        rep = Path(f"/tmp/op35_nsys/oracle_{g}.nsys-rep")
        if rep.exists():
            kern.update(parse_rep(rep))
    cells = {}
    for rng, us in kern.items():
        parts = rng.split("|")
        if len(parts) != 3 or parts[0] != "c":
            continue
        _, cid, arm = parts
        cells.setdefault(cid, {})[arm] = us
    rows = []
    for cid, arms in sorted(cells.items()):
        if not all(a in arms for a in ("base", "p3", "p4", "floor")):
            continue
        b, p3, p4, fl = arms["base"], arms["p3"], arms["p4"], arms["floor"]
        t_p3 = b - p3
        t_p4 = b - p4
        t_mid = (p4 - fl) - t_p3
        rows.append(dict(cell=cid, base=round(b, 2), p3arm=round(p3, 2),
                         p4arm=round(p4, 2), floor=round(fl, 2),
                         t_P3=round(t_p3, 2), t_P4blk=round(t_p4, 2),
                         t_mid=round(t_mid, 2),
                         fl_pct=round(fl / b * 100), P3_pct=round(t_p3 / b * 100),
                         P4_pct=round(t_p4 / b * 100), mid_pct=round(t_mid / b * 100)))
    out = _OP35 / "results" / "nsys_oracle_decomp.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['cell']:34s} base={r['base']:6.2f} floor={r['fl_pct']:3d}% "
              f"P3={r['P3_pct']:3d}% P4blk={r['P4_pct']:3d}% mid={r['mid_pct']:3d}%")
    gm = lambda xs: math.exp(sum(math.log(max(x, 1e-9)) for x in xs) / len(xs))
    ub_p4 = gm([r['base'] / (r['base'] - r['t_P4blk']) for r in rows if r['base'] > r['t_P4blk']])
    ub_p34 = gm([r['base'] / max(r['base'] - r['t_P4blk'] - r['t_P3'], 0.3) for r in rows])
    print(f"\ncells={len(rows)}  UB(zero P4blk)={ub_p4:.3f}  UB(zero P3+P4blk)={ub_p34:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
