# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate BS-scaling batches -> per (model, isl, BS) base/opt us + ratio,
plus an anchor drift check of the base arm (flags OFF = PR#16457 behavior)
against REPORT SS7's bs_real.csv `pr` column (fp32 rows, same layers).

  python3 parse_ab37_bs.py
Writes ship/bs_cells.csv + prints per-model BS x ISL speedup matrices.
"""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP = HERE / "ship"
REPORT_CSV = (HERE.parent / "op26_r0_upstream_port_report" / "bs_real.csv")
sys.path.insert(0, str(HERE.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def gm(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def main():
    us, meta, inexact = {}, {}, []
    batches = sorted(SHIP.glob("bs_*.csv"))
    batches = [b for b in batches if "shard" not in b.name
               and "cells" not in b.name]
    print(f"[parse] {len(batches)} completed batches")
    for c in batches:
        tag = c.stem[len("bs_"):]
        rep = SHIP / "nsys_reps" / f"bs_{tag}.nsys-rep"
        if rep.exists():
            for rng, t in parse_rep(str(rep)).items():
                _, arm, uuid, dt = rng.split("|", 3)
                us[(uuid, arm)] = t
        else:
            print(f"[parse] WARN missing rep for {tag}")
        for r in csv.DictReader(open(c)):
            if r["arm"] == "ERROR":
                inexact.append((r["uuid"], "ERROR"))
                continue
            meta[r["uuid"]] = (r["model"], r["isl"], int(r["BS"]),
                               int(r["cs"]), int(r["N"]), int(r["K"]))
            if r["exact"] != "True":
                inexact.append((r["uuid"], r["arm"]))

    rows = []
    for uuid, (model, isl, BS, cs, N, K) in sorted(meta.items()):
        tb, to = us.get((uuid, "base")), us.get((uuid, "opt"))
        if not (tb and to):
            continue
        rows.append(dict(uuid=uuid, model=model, isl=isl, BS=BS, cs=cs,
                         N=N, K=K, base_us=round(tb, 3), opt_us=round(to, 3),
                         ratio=round(tb / to, 4)))
    with open(SHIP / "bs_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} paired cells · inexact/error: {inexact or 'none'}")

    # ---- anchor drift vs REPORT bs_real.csv (pr column, fp32) ----------
    rep_pr = {}
    if REPORT_CSV.exists():
        for r in csv.DictReader(open(REPORT_CSV)):
            if r["dtype"] == "fp32":
                rep_pr[(r["model"], r["isl"], int(r["BS"]))] = float(r["pr"])
    drifts = []
    for r in rows:
        k = (r["model"], r["isl"], r["BS"])
        if k in rep_pr:
            drifts.append(r["base_us"] / rep_pr[k])
    if drifts:
        qs = statistics.quantiles(drifts, n=10)
        print(f"\nanchor drift base_now/report_pr (n={len(drifts)}): "
              f"median {statistics.median(drifts):.4f} "
              f"p10 {qs[0]:.4f} p90 {qs[-1]:.4f}")
    else:
        print("\nanchor drift: no fp32 overlap rows found in bs_real.csv")

    # ---- per-model BS x ISL ratio matrices -----------------------------
    isls_order = ["4k", "8k", "16k", "32k", "64k", "128k", "256k",
                  "512k", "1024k"]
    for model in ("flash", "pro", "v32"):
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        bss = sorted({r["BS"] for r in sub})
        isls = [i for i in isls_order if any(r["isl"] == i for r in sub)]
        print(f"\n== {model} speedup base/opt (rows=BS, cols=ISL) ==")
        print("BS    " + " ".join(f"{i:>7s}" for i in isls) + "      gm")
        for BS in bss:
            vals = {r["isl"]: r for r in sub if r["BS"] == BS}
            cells = [vals[i]["ratio"] if i in vals else None for i in isls]
            g = gm([c for c in cells if c])
            line = f"{BS:<5d} " + " ".join(
                f"{c:7.3f}" if c else f"{'--':>7s}" for c in cells)
            print(line + f" {g:7.3f}")
        print(f"model gm {gm([r['ratio'] for r in sub]):.4f} "
              f"min {min(r['ratio'] for r in sub):.4f} "
              f"max {max(r['ratio'] for r in sub):.4f}")

    allr = [r["ratio"] for r in rows]
    print(f"\nALL: n={len(allr)} gm={gm(allr):.4f} min={min(allr):.4f} "
          f"max={max(allr):.4f} · wins {sum(1 for x in allr if x >= 1.0)}"
          f"/{len(allr)} · >=0.975 {sum(1 for x in allr if x >= 0.975)}"
          f"/{len(allr)}")
    (SHIP / "bs_verdict.json").write_text(json.dumps(
        dict(n=len(allr), gm=gm(allr), min=min(allr), max=max(allr),
             inexact=inexact), indent=1))


if __name__ == "__main__":
    main()
