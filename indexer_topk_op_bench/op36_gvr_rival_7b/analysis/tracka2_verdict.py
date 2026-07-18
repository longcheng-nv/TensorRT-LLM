#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op36 Track A2 (distP4) verdict analysis.

Inputs: results/a2_verdict (9 routed-region batches, <=2-way, arms
gvr_pr/sglang_v2/gvr_a2) + results/b_screen (full grid, for the non-routed
cells of the composite). All ratios are within-batch (anchor-consistent);
cross-run mixing only happens at the composite level via per-cell ratios.

Outputs:
 1. anchor drift for the 9 batches vs b200-081 baseline
 2. a2/pr attribution per cell (excluding cs1_no_dp4 cells where a2 == pr
    by construction) — the pure distP4 effect
 3. zero-regression check (ship rule: a2 replaces pr in the R1-routed region
    iff a2 >= pr on every routed cell, tolerance 2%)
 4. composite gm vs sglang: Track-B ship table with pr -> a2 on routed cells

Usage: python3 tracka2_verdict.py [--no-parse]
"""
import csv
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from math import exp, log
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OP36 = _HERE.parent
_RH = _OP36.parents[0] / "op26_r0_upstream_port_report" / "rival_harness"

NO_PARSE = "--no-parse" in sys.argv
A2ROOT = _OP36 / "results" / "a2_verdict"
BROOT = _OP36 / "results" / "b_screen"


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return exp(sum(log(x) for x in xs) / len(xs)) if xs else float("nan")


def load(root, parse=False):
    if parse:
        subprocess.run([sys.executable, str(_RH / "parse_rival.py"), str(root)],
                       check=True)
    cells = defaultdict(dict)
    meta = {}
    for l in (root / "results.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("dtype") != "fp32":
            continue
        u = (r.get("us_span") or r.get("us")) \
            if r["op"] in ("sglang_v2", "sgl_bx") else r.get("us")
        k = (r["model"], r["isl"], r["BS"])
        if u:
            cells[k][r["op"]] = u
        meta.setdefault(k, {})[r["op"]] = r
    return cells, meta


def main():
    a2c, a2m = load(A2ROOT, parse=not NO_PARSE)
    bc, _ = load(BROOT, parse=False)

    # ---- 1. anchors --------------------------------------------------------
    base = {}
    with open(_OP36 / "results" / "baseline_real_bs.csv") as f:
        for r in csv.DictReader(f):
            if r["dtype"] == "fp32":
                base[(r["model"], r["isl"], int(r["BS"]))] = r
    dr = [a2c[k]["gvr_pr"] / float(base[k]["gvr_pr"]) for k in a2c
          if k in base and a2c[k].get("gvr_pr") and base[k].get("gvr_pr")]
    ds = [a2c[k]["sglang_v2"] / float(base[k]["sglang_v2"]) for k in a2c
          if k in base and a2c[k].get("sglang_v2") and base[k].get("sglang_v2")]
    print(f"== 1. anchors: pr med {st.median(dr):.3f} p95 "
          f"{sorted(dr)[int(0.95 * len(dr))]:.3f} | sgl med {st.median(ds):.3f} "
          f"p95 {sorted(ds)[int(0.95 * len(ds))]:.3f} (n={len(dr)})")

    # ---- 2. a2/pr attribution ---------------------------------------------
    print("\n== 2. distP4 attribution (pr/a2 speedup; dp4-active cells only) ==")
    print(f"{'cell':26s} {'cs':>10s} {'pr us':>8s} {'a2 us':>8s} {'pr/a2':>6s} {'vs sgl':>7s}")
    att, exact_bad = [], []
    for k in sorted(a2c):
        o = a2c[k]
        ra2 = a2m[k].get("gvr_a2", {})
        if ra2.get("exact") is False:
            exact_bad.append(k)
        if not all(o.get(x) for x in ("gvr_pr", "gvr_a2", "sglang_v2")):
            continue
        dp4 = ra2.get("flags") == "dist_p4"
        sp = o["gvr_pr"] / o["gvr_a2"]
        if dp4:
            att.append((k, sp))
        m, isl, bs = k
        routed = 32 <= bs <= 128
        tag = f"{m}/{isl}/BS{bs}" + ("*" if routed else "")
        print(f"{tag:26s} {ra2.get('launch_cfg', '?'):>10s} {o['gvr_pr']:8.2f} "
              f"{o['gvr_a2']:8.2f} {sp:6.3f} {o['sglang_v2'] / o['gvr_a2']:7.3f}"
              + ("" if dp4 else "  [cs1: a2==pr]"))
    if exact_bad:
        print(f"!! EXACT FAILURES: {exact_bad}")
    if att:
        sps = [s for _, s in att]
        print(f"\n  dp4-active: gm {gm(sps):.3f}  med {st.median(sps):.3f}  "
              f"min {min(sps):.3f}  max {max(sps):.3f}  n={len(sps)}")
        ro = [s for (m, i, b), s in att if 32 <= b <= 128]
        if ro:
            print(f"  routed (BS32-128): gm {gm(ro):.3f}  min {min(ro):.3f}  n={len(ro)}")

    # ---- 3+4. ship rule + composite ---------------------------------------
    print("\n== 3. ship rule + composite (Track-B table, pr -> best(pr,a2) routed) ==")
    reg = [(k, a2c[k]["gvr_a2"] / a2c[k]["gvr_pr"]) for k in a2c
           if 32 <= k[2] <= 128
           and all(a2c[k].get(x) for x in ("gvr_pr", "gvr_a2"))
           and a2c[k]["gvr_a2"] > 1.02 * a2c[k]["gvr_pr"]]
    print(f"  routed-cell regressions a2 vs pr (>2%): {len(reg)} "
          f"{[(f'{m}/{i}/BS{b}', round(v, 3)) for (m, i, b), v in reg]}")
    vals = []
    n_a2 = 0
    for k, o in bc.items():
        m, isl, bs = k
        if not all(o.get(x) for x in ("gvr_pr", "sgl_bx", "sglang_v2")):
            continue
        # N >= 65536 test via the b_screen N (isl->N mapping differs per model);
        # routed == the R1 rule evaluated on b_screen data:
        n = None
        for kk in (k,):
            pass
        # R1 used N>=65536 & BS 32-128; recover N from a2 meta or b jsonl is
        # overkill — the 9 a2 batches ARE the N>=65536 set, so membership =
        # (m, isl) in the a2 batch list.
        routed = (k in a2c) and (32 <= bs <= 128)
        if routed and all(a2c[k].get(x) for x in
                          ("gvr_pr", "gvr_a2", "sglang_v2")):
            ak = a2c[k]
            t = min(ak["gvr_pr"], ak["gvr_a2"])
            n_a2 += ak["gvr_a2"] < ak["gvr_pr"]
            vals.append(ak["sglang_v2"] / t)     # within-run ratio
        elif routed:
            vals.append(o["sglang_v2"] / o["gvr_pr"])
        else:
            vals.append(o["sglang_v2"] / o["sgl_bx"])
    print(f"  composite gm vs sglang (275 cells): {gm(vals):.3f} "
          f"(a2 faster than pr on {n_a2} routed cells)")


if __name__ == "__main__":
    main()
