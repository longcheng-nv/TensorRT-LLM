# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the per-layer backfill sweep (layers_harness, launch contract,
2026-07-19 umbriel-b200-027) into the report CSVs:

  real_3arm_layers_full.csv : §4b — BS=1 fp32, ALL captured GVR-active layers
      (flash 21 / pro 30 / v32 58), arms base/pr/op26, launch contract.
      SUPERSEDES real_3arm_layers.csv (07-15 frozen-cfg, 3 layers) as the
      §4b data source.
  bs_real_layers.csv        : §7b — fp32 11-BS grid x all ISL rungs x the 3
      GVR-active bench layers per model.

Anchor gates printed (not enforced): new L22/L30/L34 BS=1 rows vs
real_3arm.csv (07-16 b200-094 refresh) and vs bs_real.csv fp32 BS rows —
cross-node drift med/p95.

Usage: python3 aggregate_layers.py [results_jsonl]
       default /tmp/gvrlayers/layers_results/results.jsonl
"""
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gvrlayers/layers_results/results.jsonl"
ISL_ORD = {s: i for i, s in enumerate(
    ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"])}
MODEL_ORD = {"flash": 0, "pro": 1, "v32": 2}
LMAP_REFRESH = {"flash": 22, "pro": 30, "v32": 34}


def r3(a, b):
    try:
        return round(float(a) / float(b), 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


rows, n_err = [], 0
for line in open(SRC):
    if not line.strip():
        continue
    r = json.loads(line)
    if "error" in r or "us" not in r:
        n_err += 1
        continue
    rows.append(r)
print(f"layers results: {len(rows)} rows kept, {n_err} omitted")

# pivot: (sweep, model, isl, L, BS) -> {op: us, meta}
piv = defaultdict(dict)
for r in rows:
    d = piv[(r["sweep"], r["model"], r["isl"], int(r["L"]), int(r["BS"]))]
    d[r["op"]] = r["us"]
    d["N"], d["K"], d["hit"] = r["N"], r["K"], r.get("hit")
    if r["op"] == "gvr_pr":
        d["pr_exact"] = r.get("exact")
        d["cs"] = r.get("cluster_size", "")
        d["cfg"] = r.get("launch_cfg", "")
    if r["op"] == "gvr_base":
        d["base_exact"] = r.get("exact")

# ---- §4b full-layer CSV -----------------------------------------------------
hdr = ("model,isl,N,K,layer,hit,cs,base,pr,op26,pr_vs_base,pr_vs_op26,"
       "pr_exact,base_exact")
out = [hdr]
keys = sorted((k for k in piv if k[0] == "seqlen"),
              key=lambda k: (MODEL_ORD[k[1]], ISL_ORD[k[2]], k[3]))
for (_, m, isl, L, BS) in keys:
    d = piv[("seqlen", m, isl, L, BS)]
    b, p, o = d.get("gvr_base"), d.get("gvr_pr"), d.get("op26_r0auto")
    out.append(f"{m},{isl},{d['N']},{d['K']},{L},"
               f"{round(d['hit'], 3) if d.get('hit') is not None else ''},"
               f"{d.get('cs', '')},"
               f"{round(b, 2) if b else ''},{round(p, 2) if p else ''},"
               f"{round(o, 2) if o else ''},{r3(b, p)},{r3(o, p)},"
               f"{d.get('pr_exact', '')},{d.get('base_exact', '')}")
open(os.path.join(HERE, "real_3arm_layers_full.csv"), "w").write("\n".join(out) + "\n")
print(f"real_3arm_layers_full.csv: {len(out) - 1} rows")

# ---- §7b per-layer BS CSV ---------------------------------------------------
hdr2 = ("model,isl,L,dtype,N,cs,BS,hit,base,pr,op26,pr_vs_base,op26_vs_pr,"
        "pr_exact")
out2 = [hdr2]
keys2 = sorted((k for k in piv if k[0] == "bs"),
               key=lambda k: (MODEL_ORD[k[1]], k[3], ISL_ORD[k[2]], k[4]))
for (_, m, isl, L, BS) in keys2:
    d = piv[("bs", m, isl, L, BS)]
    b, p, o = d.get("gvr_base"), d.get("gvr_pr"), d.get("op26_r0auto")
    out2.append(f"{m},{isl},{L},fp32,{d['N']},{d.get('cs', '')},{BS},"
                f"{round(d['hit'], 3) if d.get('hit') is not None else ''},"
                f"{b or ''},{p or ''},{o or ''},{r3(b, p)},{r3(o, p)},"
                f"{d.get('pr_exact', '')}")
open(os.path.join(HERE, "bs_real_layers.csv"), "w").write("\n".join(out2) + "\n")
print(f"bs_real_layers.csv: {len(out2) - 1} rows")

# ---- exactness summary ------------------------------------------------------
ex_bad = [(k, piv[k].get("pr_exact")) for k in piv if piv[k].get("pr_exact") is False]
n_pr = sum(1 for k in piv if piv[k].get("pr_exact") is not None)
base_bad = sum(1 for k in piv if piv[k].get("base_exact") is False)
print(f"pr exact: {n_pr - len(ex_bad)}/{n_pr} (base inexact cells: {base_bad})")
if ex_bad:
    print("!! PR EXACT FAILURES:", ex_bad[:10])

# ---- anchor drift vs the 07-16 refresh --------------------------------------
import csv as _csv

def _gate(pairs, tag):
    if not pairs:
        print(f"anchor {tag}: no overlap")
        return
    dr = sorted(a / b for a, b in pairs)
    print(f"anchor {tag}: n={len(dr)} med {st.median(dr):.3f} "
          f"p95 {dr[min(len(dr) - 1, int(0.95 * len(dr)))]:.3f}")

with open(os.path.join(HERE, "real_3arm.csv")) as f:
    ref = {(r["model"], r["isl"]): r for r in _csv.DictReader(f)}
pairs = []
for (sw, m, isl, L, BS), d in piv.items():
    if sw != "seqlen" or L != LMAP_REFRESH[m] or not d.get("gvr_pr"):
        continue
    rr = ref.get((m, isl))
    if rr and rr.get("pr"):
        pairs.append((d["gvr_pr"], float(rr["pr"])))
_gate(pairs, "seqlen pr(027)/refresh(094)")

with open(os.path.join(HERE, "bs_real.csv")) as f:
    refb = {(r["model"], r["isl"], int(r["BS"])): r
            for r in _csv.DictReader(f) if r["dtype"] == "fp32"}
pairs = []
for (sw, m, isl, L, BS), d in piv.items():
    if sw != "bs" or L != LMAP_REFRESH[m] or not d.get("gvr_pr"):
        continue
    rr = refb.get((m, isl, BS))
    if rr and rr.get("pr"):
        pairs.append((d["gvr_pr"], float(rr["pr"])))
_gate(pairs, "bs pr(027)/refresh(094)")
