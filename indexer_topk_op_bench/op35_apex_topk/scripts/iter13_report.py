# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter13 report: join apex sweep spans with rival_long.csv frontier (fp32).
Prints per-regime geomeans of frontier/apex (speedup; >1 = apex wins)."""
import csv
import io
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "results/iter13"
RIVAL = HERE.parents[1] / "op26_r0_upstream_port_report/rival_long.csv"
rep = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "apex_fp32_run1.nsys-rep")
DT = sys.argv[2] if len(sys.argv) > 2 else "fp32"
JSONL = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / f"apex_{DT}.jsonl")


def parse_rep_span(rep):
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_gpu_proj_sum", "--format", "csv",
         "--force-export=true", str(rep)], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr = next((i for i, r in enumerate(rows)
                if r and r[0] in ("Range", "NVTX Range", "Name")), None)
    cols = rows[hdr]
    i_inst = next(i for i, c in enumerate(cols) if "Instances" in c)
    i_tot = next(i for i, c in enumerate(cols) if "Total" in c)
    res = {}
    for r in rows[hdr + 1:]:
        if not r or "|" not in r[0]:
            continue
        try:
            ninst = int(r[i_inst]); total_ns = float(r[i_tot])
        except (ValueError, IndexError):
            continue
        if ninst:
            res[r[0].lstrip(":")] = total_ns / ninst / 1e3
    return res


span = parse_rep_span(rep)

# frontier: per-cell best us_span across 6 arms, fp32 only
frontier = defaultdict(lambda: float("inf"))
best_arm = {}
for r in csv.DictReader(open(RIVAL)):
    if r["dtype"] != DT or r.get("us_span") in (None, ""):
        continue
    if r["family"] == "synth":
        key = ("synth", r["scenario"], int(r["K"]), int(r["N"]), int(r["BS"]))
    else:
        key = ("real", r["model"], r["isl"], int(r["N"]), int(r["BS"]))
    v = float(r["us_span"])
    if v < frontier[key]:
        frontier[key] = v
        best_arm[key] = r["op"]

recs = [json.loads(l) for l in open(JSONL)]
rows_out = []
for r in recs:
    if "error" in r:
        continue
    us = span.get(r["range_cold"])
    if us is None:
        continue
    if r["family"] == "synth":
        key = ("synth", r["scenario"], r["K"], r["N"], r["BS"])
    else:
        key = ("real", r["model"], r["isl"], r["N"], r["BS"])
    fr = frontier.get(key)
    if fr is None or fr == float("inf"):
        continue
    rows_out.append(dict(r, us_span=us, frontier=fr, arm=best_arm[key],
                         speedup=fr / us))

with open(ROOT / f"joined_{DT}.json", "w") as f:
    json.dump(rows_out, f)


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def bucket_bs(bs):
    return "BS1" if bs == 1 else ("BS2-8" if bs <= 8 else ("BS16-64" if bs <= 64 else "BS128-1024"))


def bucket_n(n):
    return "N<=16k" if n <= 16384 else ("N32-65k" if n <= 65536 else
                                        ("N131-262k" if n <= 262144 else "N>=512k"))


print(f"cells joined: {len(rows_out)}  (exact: {sum(1 for r in rows_out if r['exact'])})")
print(f"\nOVERALL geomean frontier/apex: {gm([r['speedup'] for r in rows_out]):.3f}")
for fam in ("synth", "real"):
    sel = [r for r in rows_out if r["family"] == fam]
    print(f"  {fam}: {gm([r['speedup'] for r in sel]):.3f} (n={len(sel)})")
print("\nper-regime (all fams):")
groups = defaultdict(list)
for r in rows_out:
    groups[(bucket_bs(r["BS"]), bucket_n(r["N"]))].append(r["speedup"])
for k in sorted(groups):
    xs = sorted(groups[k])
    print(f"  {str(k):28} gm {gm(xs):5.2f}  p10 {xs[int(.1*len(xs))]:5.2f} "
          f"p90 {xs[int(.9*len(xs))]:5.2f}  n={len(xs)}")
print("\nworst 12 cells:")
for r in sorted(rows_out, key=lambda r: r["speedup"])[:12]:
    tag = r.get("scenario") or f"{r.get('model')}/{r.get('isl')}"
    print(f"  {r['family']}/{tag} K{r['K']} N{r['N']} BS{r['BS']}: "
          f"apex {r['us_span']:.1f} vs frontier {r['frontier']:.1f} ({r['arm']}) = {r['speedup']:.2f}x")
print("\nbest 8 cells:")
for r in sorted(rows_out, key=lambda r: -r["speedup"])[:8]:
    tag = r.get("scenario") or f"{r.get('model')}/{r.get('isl')}"
    print(f"  {r['family']}/{tag} K{r['K']} N{r['N']} BS{r['BS']}: "
          f"apex {r['us_span']:.1f} vs frontier {r['frontier']:.1f} ({r['arm']}) = {r['speedup']:.2f}x")
