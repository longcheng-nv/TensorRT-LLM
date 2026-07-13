#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""HBE-C rung-0 aggregator — GO-line verdict from replay_hbec_ladder.py jsonl.

Policy space evaluated offline from the recorded per-rung counts:
  collect rung c in {0 (loosest, DESIGN default), 1 (middle)}:
    stored candidates = row elems >= rung_c;  usable iff r* >= c;
    MISS = lt_K  OR  r* < c  OR  cnt_c > cap  (cap in {8,16,32}xK).
  E[passes] = 1 + 2*miss  (miss rows redo stock cluster Phase1+Phase2).

GO line (DESIGN §6, real axis): E[passes] <= ~1.2 and miss <= ~10%.

Usage: python3 parse_hbec_rung0.py results/hbec_rung0/rung0.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

CAPS_XK = (8, 16, 32)
COLLECTS = (0, 1)


def policy_miss(rec, c, cap_xk):
    if rec.get("miss_mode") == "no_valid_hint" or rec["lt_K"]:
        return True
    if rec["r_star"] < c:
        return True
    return rec["cnts"][c] > cap_xk * rec["K"]


def q(a, f):
    return float(np.quantile(np.array(a, dtype=float), f)) if a else float("nan")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "results/hbec_rung0/rung0.jsonl"
    recs = [json.loads(ln) for ln in open(path)
            if ln.strip() and not ln.startswith("#")]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["src"], r["arm"], r["scenario"], r["K"])].append(r)

    hdr = (f"{'src':>7} {'arm':>5} {'scen':>6} {'K':>5} {'rows':>4} | "
           f"{'br0/1/2':>9} {'ltK':>4} | {'cand*xK':>13} {'candAllxK':>13} | "
           + " ".join(f"c{c}cap{x}K" for c in COLLECTS for x in CAPS_XK))
    print(hdr)
    for key in sorted(groups):
        rs = groups[key]
        n = len(rs)
        br = [sum(1 for r in rs if r["r_star"] == m) for m in (0, 1, 2)]
        ltk = sum(1 for r in rs if r["lt_K"])
        cstar = [r["cand_rstar_xK"] for r in rs if r["cand_rstar_xK"]]
        call = [r["cand_all_xK"] for r in rs]
        miss_cols = " ".join(
            f"{sum(policy_miss(r, c, x) for r in rs)/n:7.2f}"
            for c in COLLECTS for x in CAPS_XK)
        print(f"{key[0]:>7} {key[1]:>5} {key[2]:>6} {key[3]:>5} {n:>4} | "
              f"{br[0]}/{br[1]}/{br[2]:>2} {ltk:>4} | "
              f"med {q(cstar,.5):4.1f} p90 {q(cstar,.9):5.1f} | "
              f"med {q(call,.5):4.1f} p90 {q(call,.9):5.1f} | {miss_cols}")

    # ---- GO line: real axis = rr real (cluster N) + realcap ----
    go_arms = sorted({r["arm"] for r in recs} - {"stock"})
    for arm in go_arms:
        print(f"\n== GO line ({arm} arm, real axis) ==")
        _go(recs, arm)


def _go(recs, arm):
    for label, sel in (
        ("rr-real (all N)", lambda r: r["src"] == "rr"
         and r["scenario"] == "real"),
        ("rr-real N>=131072", lambda r: r["src"] == "rr"
         and r["scenario"] == "real" and r["N"] >= 131072),
        ("realcap (all rows)", lambda r: r["src"] == "realcap"),
        ("REAL AXIS rr-real+realcap", lambda r: r["src"] == "realcap"
         or (r["src"] == "rr" and r["scenario"] == "real")),
        ("worst-guard rr-worst", lambda r: r["src"] == "rr"
         and r["scenario"] == "worst"),
        ("best-guard rr-best", lambda r: r["src"] == "rr"
         and r["scenario"] == "best"),
    ):
        rs = [r for r in recs if r["arm"] == arm and sel(r)]
        if not rs:
            continue
        n = len(rs)
        line = f"  {label:>26} rows {n:>3}: "
        for c in COLLECTS:
            for x in CAPS_XK:
                miss = sum(policy_miss(r, cx, x) for r, cx in
                           ((r, c) for r in rs)) / n
                ep = 1 + 2 * miss
                line += f"c{c}/{x}K miss {miss*100:4.1f}% E[p] {ep:4.2f} | "
        print(line)


if __name__ == "__main__":
    main()
