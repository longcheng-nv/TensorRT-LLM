# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R4 parity gate: gvrpkg_04a0 vs gvrpkg_head, two same-GPU nsys runs.

Pairs per-cell cold/warm kernel-time medians from ab_parity04a0 and
ab_parityhead reps; gate = |geomean(head/04a0) - 1| <= 2%.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def cells_of(tag):
    kern = parse_rep(str(HERE / "nsys_reps" / f"ab_{tag}.nsys-rep"))
    out = {}
    for rng, us in kern.items():
        mode, arm, uuid = rng.split("|", 2)
        out.setdefault(uuid, {})[mode] = us
    return out


def main():
    a = cells_of("parity04a0")   # new pinned-head pkg
    b = cells_of("parityhead")   # old validated pkg
    rows, rc, rw = [], [], []
    for uuid in sorted(a):
        if uuid not in b:
            print(f"{uuid}: MISSING in head run")
            continue
        c = b[uuid]["c"] / a[uuid]["c"]
        w = b[uuid]["w"] / a[uuid]["w"]
        rc.append(c)
        rw.append(w)
        rows.append(dict(uuid=uuid, head_cold=b[uuid]["c"], a04_cold=a[uuid]["c"],
                         ratio_cold=c, ratio_warm=w))
        print(f"{uuid:22s} cold head={b[uuid]['c']:8.2f} 04a0={a[uuid]['c']:8.2f} "
              f"r={c:6.3f}  warm r={w:6.3f}")
    gc = statistics.geometric_mean(rc)
    gw = statistics.geometric_mean(rw)
    drift = abs(gc - 1.0)
    print(f"\nCOLD geomean head/04a0 = {gc:.4f}  WARM = {gw:.4f}  "
          f"drift {drift*100:.2f}% -> {'PASS' if drift <= 0.02 else 'FAIL'} (gate 2%)")
    (HERE / "ab_parity_r4.json").write_text(json.dumps(
        dict(cold_geomean_head_over_04a0=gc, warm_geomean=gw,
             drift=drift, gate="PASS" if drift <= 0.02 else "FAIL",
             cells=rows), indent=1))


if __name__ == "__main__":
    main()
