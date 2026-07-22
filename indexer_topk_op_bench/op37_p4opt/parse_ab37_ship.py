# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate op37 ship-verdict batches -> per-cell base/all ratios + the
3-axis verdict table [worst(synth-worst), real(865 fp32 + realdt), best].

Ship rule (NOTES.md): real geomean improves AND every axis regression-free
at the worst-cell >= 0.975 level AND exactness green on all cells/arms.

  python3 parse_ab37_ship.py            # verdict from all completed batches
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP = HERE / "ship"
sys.path.insert(0, str(HERE.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def gm(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def rung(cs, N):
    if cs == 1:
        return "cs1-small" if N <= 8448 else "cs1-mid"
    return f"cs{cs}"


def axis_of(tag, uuid, dt):
    if uuid.startswith("synth_worst"):
        return "worst"
    if uuid.startswith("synth_best"):
        return "best"
    return "real" if dt == "fp32" else f"real-{dt}"


def main():
    us = {}       # (uuid, dt, arm) -> us
    meta = {}     # (uuid, dt) -> (cs, N, K, exact_base, exact_all)
    inexact = []
    batches = sorted(SHIP.glob("ship_*.csv"))
    print(f"[parse] {len(batches)} completed batches")
    for c in batches:
        tag = c.stem[len("ship_"):]
        rep = SHIP / "nsys_reps" / f"ship_{tag}.nsys-rep"
        if rep.exists():
            for rng, t in parse_rep(str(rep)).items():
                _, arm, uuid, dt = rng.split("|", 3)
                us[(uuid, dt, arm)] = t
        else:
            print(f"[parse] WARN missing rep for {tag}")
        for r in csv.DictReader(open(c)):
            if r["arm"] == "ERROR":
                inexact.append((r["uuid"], r["dt"], "ERROR"))
                continue
            meta.setdefault((r["uuid"], r["dt"]),
                            [int(r["cs"]), int(r["N"]), int(r["K"])])
            if r["exact"] != "True":
                inexact.append((r["uuid"], r["dt"], r["arm"]))

    rows = []
    for (uuid, dt), (cs, N, K) in sorted(meta.items()):
        tb = us.get((uuid, dt, "base"))
        ta = us.get((uuid, dt, "all"))
        if not (tb and ta):
            continue
        rows.append(dict(uuid=uuid, dt=dt, cs=cs, N=N, K=K,
                         rung=rung(cs, N), axis=axis_of("", uuid, dt),
                         base_us=round(tb, 3), all_us=round(ta, 3),
                         ratio=round(tb / ta, 4)))
    with open(SHIP / "ship_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- verdict tables --------------------------------------------------
    print(f"\n{len(rows)} paired cells · inexact/error: {inexact or 'none'}")
    by_axis = defaultdict(list)
    for r in rows:
        by_axis[r["axis"]].append(r)

    verdict = {}
    for ax in ("worst", "real", "best", "real-bf16", "real-fp16"):
        rs = by_axis.get(ax)
        if not rs:
            continue
        ratios = [r["ratio"] for r in rs]
        worst_cell = min(rs, key=lambda r: r["ratio"])
        wins = sum(1 for x in ratios if x >= 1.0)
        ge0975 = sum(1 for x in ratios if x >= 0.975)
        verdict[ax] = dict(n=len(rs), gm=round(gm(ratios), 4),
                           worst=round(worst_cell["ratio"], 4),
                           worst_cell=f"{worst_cell['uuid']}/{worst_cell['dt']}",
                           wins=wins, ge0975=ge0975)
        print(f"\n== axis {ax}: n={len(rs)} gm={gm(ratios):.4f} "
              f"win {wins}/{len(rs)} · >=0.975 {ge0975}/{len(rs)} · "
              f"worst {worst_cell['ratio']:.4f} @ {worst_cell['uuid']}/{worst_cell['dt']}")
        by_rung = defaultdict(list)
        for r in rs:
            by_rung[r["rung"]].append(r["ratio"])
        for rg in sorted(by_rung):
            v = by_rung[rg]
            print(f"   {rg:10s} n={len(v):3d} gm={gm(v):.4f} "
                  f"min={min(v):.4f} max={max(v):.4f}")

    # per-K breakdown on synth axes (dtype interaction)
    for ax in ("worst", "best"):
        rs = by_axis.get(ax, [])
        by_kdt = defaultdict(list)
        for r in rs:
            by_kdt[(r["K"], r["dt"])].append(r["ratio"])
        if by_kdt:
            print(f"\n-- {ax} by (K, dtype):")
            for k in sorted(by_kdt):
                v = by_kdt[k]
                print(f"   K={k[0]:<5d} {k[1]:5s} n={len(v)} gm={gm(v):.4f} "
                      f"min={min(v):.4f}")

    ship_ok = (verdict.get("real", {}).get("gm", 0) > 1.0
               and all(v["worst"] >= 0.975 for v in verdict.values())
               and not inexact)
    print(f"\nSHIP RULE: {'PASS' if ship_ok else 'FAIL/PENDING'} "
          f"(real gm>1: {verdict.get('real', {}).get('gm')} · "
          f"all-axis worst>=0.975: "
          f"{ {a: v['worst'] for a, v in verdict.items()} } · "
          f"exactness: {'green' if not inexact else inexact[:5]})")
    (SHIP / "ship_verdict.json").write_text(
        json.dumps(dict(verdict=verdict, inexact=inexact, ship=ship_ok),
                   indent=1))


if __name__ == "__main__":
    main()
