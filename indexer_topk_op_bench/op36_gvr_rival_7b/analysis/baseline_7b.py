#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op36 baseline — extract the op26-report §8 backfilled real-BS rival grid
(embedded RIVAL json, 2026-07-16 b200-081 single-node run) and compute the
campaign target arithmetic: PR vs sglang_v2 geomean on real fp32 BS x ISL,
loss map, and the strategic upper bounds that gate the 1.10 goal.

Output: results/baseline_real_bs.csv + analysis/BASELINE.md (stdout dump).
Timing column = us_span (honest wall-clock; avoids double-counting sglang's
overlapped 2-kernel PDL path).
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1] / "op26_r0_upstream_port_report" / "REPORT.html"
OUT_CSV = HERE.parent / "results" / "baseline_real_bs.csv"

ARMS = ["gvr_base", "gvr_pr", "op26_r0auto", "radix_cutedsl", "sglang_v2",
        "flashinfer_topk"]


def load_rival(report=REPORT):
    h = report.read_text()
    m = re.search(r"RIVAL\s*=", h)
    j = m.end()
    start = h.index("[", j)
    depth = 0
    for k in range(start, len(h)):
        if h[k] in "[{":
            depth += 1
        elif h[k] in "]}":
            depth -= 1
            if depth == 0:
                return json.loads(h[start:k + 1])
    raise ValueError("unterminated RIVAL block")


def gmean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def main():
    data = load_rival()
    rows = [d for d in data if d["family"] == "real" and d["sweep"] == "bs"]
    cell = defaultdict(dict)
    meta = {}
    for d in rows:
        key = (d["model"], d["isl"], d["dtype"], d["BS"])
        cell[key][d["op"]] = d["us_span"]
        meta[key] = (d["N"], d["hit"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w") as f:
        f.write("model,isl,dtype,BS,N,hit," + ",".join(ARMS) + "\n")
        for key in sorted(cell):
            m_, i_, dt, bs = key
            n, hit = meta[key]
            vals = ",".join(f"{cell[key].get(a, '')}" for a in ARMS)
            f.write(f"{m_},{i_},{dt},{bs},{n},{hit},{vals}\n")
    print(f"wrote {OUT_CSV} ({len(cell)} cells)")

    pairs = [(k, o) for k, o in cell.items()
             if "gvr_pr" in o and "sglang_v2" in o]
    r_pr = {k: o["sglang_v2"] / o["gvr_pr"] for k, o in pairs}

    def best(o, arms):
        ts = [o[a] for a in arms if a in o]
        return o["sglang_v2"] / min(ts)

    intree = ["gvr_pr", "op26_r0auto", "radix_cutedsl", "flashinfer_topk"]
    r_best = {k: best(o, intree) for k, o in pairs}

    print(f"\n== op36 target arithmetic (n={len(pairs)} fp32 real cells; "
          f">1 = we are faster than sglang_v2) ==")
    print(f"A  gvr_pr as-is                    gm {gmean(list(r_pr.values())):.3f}")
    for up in (1.15, 1.25):
        print(f"B  gvr_pr x{up} lever stack        gm "
              f"{gmean([r * up for r in r_pr.values()]):.3f}")
    print(f"C  best exact dispatch (in-tree+FI) gm {gmean(list(r_best.values())):.3f}")
    print(f"D  C + sglang-parity path at losses gm "
          f"{gmean([max(r, 1.0) for r in r_best.values()]):.3f}")
    print(f"E  D + 25% levers on won cells      gm "
          f"{gmean([max(r * 1.25 if r >= 1 else r, 1.0) for r in r_best.values()]):.3f}")
    print("TARGET = 1.10  ->  requires BEATING sglang per-cell on its home "
          "turf (small/mid-ISL fp32) — see PLAN.md feasibility gate.")

    for band, sel in [("ISL 4-16k", lambda k: k[1] in ("4k", "8k", "16k")),
                      ("ISL 32-128k", lambda k: k[1] in ("32k", "64k", "128k")),
                      ("ISL 256k-1M", lambda k: k[1] in ("256k", "512k", "1024k"))]:
        v = [r for k, r in r_pr.items() if sel(k)]
        vb = [r for k, r in r_best.items() if sel(k)]
        print(f"  {band:12s} n={len(v):3d}  pr gm {gmean(v):.3f}   "
              f"best-dispatch gm {gmean(vb):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
