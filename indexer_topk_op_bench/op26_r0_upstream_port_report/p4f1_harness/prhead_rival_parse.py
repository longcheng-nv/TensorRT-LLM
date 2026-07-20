#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Table: PR head vs op26_r0auto / sglang_v2 (span) / flashinfer, BS=1 fp32.
Median over 3 in-process rounds; ratio = t(arm)/t(prhead), >1 = PR head faster."""
import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gvrlayers/pr4")


def parse_rep_span(rep):
    out = subprocess.run(["nsys", "stats", "--report", "nvtx_gpu_proj_sum",
                          "--format", "csv", "--force-export=true", str(rep)],
                         capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr = next((i for i, r in enumerate(rows)
                if r and r[0] in ("Range", "NVTX Range", "Name")), None)
    if hdr is None:
        return {}
    cols = rows[hdr]
    i_inst = next(i for i, c in enumerate(cols) if "Instances" in c)
    i_tot = next(i for i, c in enumerate(cols) if "Total" in c)
    res = {}
    for r in rows[hdr + 1:]:
        if not r or "|" not in r[0]:
            continue
        try:
            n = int(r[i_inst]); t = float(r[i_tot])
        except (ValueError, IndexError):
            continue
        if n:
            res[r[0].lstrip(":")] = t / n / 1e3
    return res


def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


cells = {}
for f in sorted(ROOT.glob("pr4_*.jsonl")):
    m = f.stem.split("_")[1]
    rep = ROOT / "nsys_reps" / f"pr4_{m}.nsys-rep"
    kern = parse_rep(rep)
    span = parse_rep_span(rep)
    for l in f.read_text().splitlines():
        r = json.loads(l)
        if "error" in r:
            continue
        src = span if r["arm"] == "sglang_v2" else kern
        us = src.get(r["range_cold"]) or kern.get(r["range_cold"])
        if us:
            cells.setdefault((m, r["isl"], r["N"], round(r["hit"], 3)),
                             {}).setdefault(r["arm"], []).append(us)

ARMS = ["gvr_prhead", "op26_r0auto", "sglang_v2", "flashinfer_topk"]
LBL = {"gvr_prhead": "PRhead", "op26_r0auto": "op26", "sglang_v2": "sglang(span)",
       "flashinfer_topk": "FI"}
print(f"{'cell':14s} {'N':>7s} {'hit':>5s} | " +
      " ".join(f"{LBL[a]:>13s}" for a in ARMS) + " | ratio t(X)/t(PR): op26  sgl   FI")
rats = {a: [] for a in ARMS[1:]}
per_m = {}
order = {"flash": 0, "pro": 1, "v32": 2}
for (m, isl, N, hit), d in sorted(cells.items(), key=lambda t: (order[t[0][0]], t[0][2])):
    med = {a: sorted(v)[len(v) // 2] for a, v in d.items() if v}
    if "gvr_prhead" not in med:
        continue
    p = med["gvr_prhead"]
    line = f"{m + '/' + isl:14s} {N:7d} {hit:5.2f} | " + \
        " ".join(f"{med.get(a, float('nan')):10.2f}µs " for a in ARMS) + "|"
    for a in ARMS[1:]:
        if med.get(a):
            rr = med[a] / p
            rats[a].append(rr)
            per_m.setdefault((m, a), []).append(rr)
            line += f" {rr:5.3f}"
        else:
            line += "     -"
    print(line)
print("\nGEOMEAN t(arm)/t(PRhead)  (>1 = PR head FASTER):")
for a in ARMS[1:]:
    per = "  ".join(f"{mm}={gm(per_m.get((mm, a), [])):.3f}"
                    for mm in ("flash", "pro", "v32"))
    print(f"  vs {LBL[a]:13s}: ALL {gm(rats[a]):.3f}   ({per})")
