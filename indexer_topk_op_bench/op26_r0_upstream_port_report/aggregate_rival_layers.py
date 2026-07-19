# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse + aggregate the §8b per-layer rival sweep into rival_layers.csv.

Source: /tmp/gvrlayers/rival_layers_results/{rival_seqlen_*.jsonl, nsys_reps/}.
us = cold-L2 kernel-sum in NVTX range; us_span = projected NVTX GPU range
(canonical for sglang_v2's PDL 2-kernel path). Anchor gate printed: op26 rows
vs rival_long.csv (07-15 b200-044 run, op26 code unchanged).
"""
import csv
import io
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gvrlayers/rival_layers_results")
sys.path.insert(0, str(HERE.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def parse_rep_span(rep):
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_gpu_proj_sum", "--format", "csv",
         "--force-export=true", str(rep)], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr = next((i for i, r in enumerate(rows)
                if r and r[0] in ("Range", "NVTX Range", "Name")), None)
    if hdr is None:
        return {}
    cols = rows[hdr]
    try:
        i_inst = next(i for i, c in enumerate(cols) if "Instances" in c)
        i_tot = next(i for i, c in enumerate(cols) if "Total" in c)
    except StopIteration:
        return {}
    res = {}
    for r in rows[hdr + 1:]:
        if not r or "|" not in r[0]:
            continue
        try:
            ninst = int(r[i_inst]); tot = float(r[i_tot])
        except (ValueError, IndexError):
            continue
        if ninst:
            res[r[0].lstrip(":")] = tot / ninst / 1e3
    return res


out = ["model,isl,N,K,L,hit,op,us,us_span,exact"]
n_err = 0
for batch in sorted(SRC.glob("rival_seqlen_*.jsonl")):
    rep = SRC / "nsys_reps" / f"{batch.stem}.nsys-rep"
    kern = parse_rep(rep) if rep.exists() else {}
    span = parse_rep_span(rep) if rep.exists() else {}
    for line in batch.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            n_err += 1
            continue
        uc = kern.get(r["range_cold"])
        uw = kern.get(r["range_warm"])
        us = uc if uc is not None else uw
        if us is None:
            n_err += 1
            continue
        sc = span.get(r["range_cold"])
        out.append(f"{r['model']},{r['isl']},{r['N']},{r['K']},{r['L']},"
                   f"{round(r['hit'], 3) if r.get('hit') is not None else ''},"
                   f"{r['op']},{round(us, 4)},"
                   f"{round(sc, 4) if sc is not None else ''},{r.get('exact', '')}")
    print(f"  {batch.stem}: ranges={len(kern)}")
dst = HERE / "rival_layers.csv"
dst.write_text("\n".join(out) + "\n")
print(f"wrote {dst}: {len(out) - 1} rows ({n_err} omitted)")

# anchor: op26 rows vs rival_long.csv (same op code, 07-15 b200-044)
ref = {}
with open(HERE / "rival_long.csv") as f:
    for r in csv.DictReader(f):
        if (r["family"] == "real" and r["sweep"] == "seqlen"
                and r["dtype"] == "fp32" and r["op"] == "op26_r0auto"
                and r["BS"] == "1"):
            ref[(r["model"], r["isl"])] = float(r["us"])
pairs = []
for ln in out[1:]:
    c = ln.split(",")
    if c[6] == "op26_r0auto" and (c[0], c[1]) in ref:
        pairs.append(float(c[7]) / ref[(c[0], c[1])])
if pairs:
    pairs.sort()
    print(f"anchor op26(027)/rival_long(044): n={len(pairs)} "
          f"med {st.median(pairs):.3f} p95 {pairs[min(len(pairs)-1, int(0.95*len(pairs)))]:.3f}")
