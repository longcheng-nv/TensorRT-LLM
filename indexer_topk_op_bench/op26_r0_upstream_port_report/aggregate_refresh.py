# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the PR-contract refresh sweep (launch()/pick_config-driven GVR
arms, branch HEAD 018251950f) into the report CSVs:

  synth_3arm.csv  (§3, synth seqlen fp32 BS=1)     scen,K,N,cs,base,pr,op26,pr_vs_base,my_vs_op26,exact
  real_3arm.csv   (§4, real  seqlen fp32 BS=1)     model,isl,N,K,hit_rate,cs,base,pr,op26,pr_vs_base,pr_vs_op26,exact,base_exact
  bs_synth.csv    (§7, synth bs — FULL 9-N grid)   K,dtype,N,scen,cs,BS,base,pr,op26,pr_vs_base,op26_vs_pr,pr_exact
  bs_real.csv     (§7, real  bs — ALL ISL rungs)   model,isl,L,dtype,N,cs,BS,hit,base,pr,op26,pr_vs_base,op26_vs_pr,pr_exact
  rival_long.csv  (§8) — gvr_base/gvr_pr/op26_r0auto rows REPLACED with the
                  refresh rows (external rival rows untouched, old run).

Anchor-drift gate: new op26 vs old rival_long op26 on overlapping cells
(median + p95 printed; op26 code unchanged, so drift = node/day noise).
Ratio conventions preserved from the original aggregators:
  synth_3arm.my_vs_op26 = pr/op26 ; real_3arm.pr_vs_op26 = op26/pr ;
  bs_*.op26_vs_pr = op26/pr ; *_vs_base = base/pr.
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gvrval1/refresh_results/results.jsonl"
LMAP = {"flash": 22, "pro": 30, "v32": 34}


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _r3(a, b):
    try:
        return round(float(a) / float(b), 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


rows = []
n_err = 0
for line in open(SRC):
    if not line.strip():
        continue
    r = json.loads(line)
    if "error" in r or "us" not in r:
        n_err += 1
        continue
    rows.append(r)
print(f"refresh results: {len(rows)} rows kept, {n_err} omitted")


def _ex(r):
    return r.get("exact") is True or r.get("exact") == "True"


def _cs(d, N):
    return d.get("cs", 1 if N < 65536 else 4)


# ---- pivot ------------------------------------------------------------------
syn_seq = defaultdict(dict)   # (scen,K,N)                -> arms
real_seq = defaultdict(dict)  # (model,isl,N,K)           -> arms
syn_bs = defaultdict(dict)    # (K,dt,N,scen,BS)          -> arms
real_bs = defaultdict(dict)   # (model,isl,dt,N,BS)       -> arms
for r in rows:
    u = _fnum(r["us"])
    if u is None:
        continue
    if r["family"] == "synth":
        key_seq = (r["scenario"], int(r["K"]), int(r["N"]))
        d = syn_seq[key_seq] if (r["sweep"] == "seqlen" and r["dtype"] == "fp32") \
            else syn_bs[(int(r["K"]), r["dtype"], int(r["N"]), r["scenario"], int(r["BS"]))]
    else:
        if r["sweep"] == "seqlen" and r["dtype"] == "fp32":
            d = real_seq[(r["model"], r["isl"], int(r["N"]), int(r["K"]))]
        else:
            d = real_bs[(r["model"], r["isl"], r["dtype"], int(r["N"]), int(r["BS"]))]
        d["hit"] = r.get("hit", "")
    d[r["op"]] = u
    if r["op"] == "gvr_pr":
        d["pr_exact"] = _ex(r)
        d["cs"] = r.get("cluster_size", "")
        d["cfg"] = r.get("launch_cfg", "")
    if r["op"] == "gvr_base":
        d["base_exact"] = _ex(r)

# ---- §3 synth_3arm.csv --------------------------------------------------------
sh = ["scen,K,N,cs,base,pr,op26,pr_vs_base,my_vs_op26,exact"]
for (scen, K, N) in sorted(syn_seq, key=lambda k: (k[0], k[1], k[2])):
    d = syn_seq[(scen, K, N)]
    b, p, o = d.get("gvr_base"), d.get("gvr_pr"), d.get("op26_r0auto")
    sh.append(f"{scen},{K},{N},{_cs(d, N)},{round(b, 2) if b else ''},"
              f"{round(p, 2) if p else ''},{round(o, 2) if o else ''},"
              f"{_r3(b, p)},{_r3(p, o)},{d.get('pr_exact', '')}")
open(os.path.join(HERE, "synth_3arm.csv"), "w").write("\n".join(sh) + "\n")

# ---- §4 real_3arm.csv ---------------------------------------------------------
ISL_ORD = {s: i for i, s in enumerate(
    ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"])}
rh = ["model,isl,N,K,hit_rate,cs,base,pr,op26,pr_vs_base,pr_vs_op26,exact,base_exact"]
for (model, isl, N, K) in sorted(real_seq, key=lambda k: (k[0], ISL_ORD.get(k[1], 99))):
    d = real_seq[(model, isl, N, K)]
    b, p, o = d.get("gvr_base"), d.get("gvr_pr"), d.get("op26_r0auto")
    hit = d.get("hit")
    rh.append(f"{model},{isl},{N},{K},{round(float(hit), 3) if hit not in ('', None) else ''},"
              f"{_cs(d, N)},{round(b, 2) if b else ''},{round(p, 2) if p else ''},"
              f"{round(o, 2) if o else ''},{_r3(b, p)},{_r3(o, p)},"
              f"{d.get('pr_exact', '')},{d.get('base_exact', '')}")
open(os.path.join(HERE, "real_3arm.csv"), "w").write("\n".join(rh) + "\n")

# ---- §7 bs_synth.csv / bs_real.csv -------------------------------------------
bh = ["K,dtype,N,scen,cs,BS,base,pr,op26,pr_vs_base,op26_vs_pr,pr_exact"]
for key in sorted(syn_bs):
    K, dt, N, scen, BS = key
    d = syn_bs[key]
    b, p, o = d.get("gvr_base"), d.get("gvr_pr"), d.get("op26_r0auto")
    bh.append(f"{K},{dt},{N},{scen},{_cs(d, N)},{BS},{b or ''},{p or ''},{o or ''},"
              f"{_r3(b, p)},{_r3(o, p)},{d.get('pr_exact', '')}")
open(os.path.join(HERE, "bs_synth.csv"), "w").write("\n".join(bh) + "\n")

bh2 = ["model,isl,L,dtype,N,cs,BS,hit,base,pr,op26,pr_vs_base,op26_vs_pr,pr_exact"]
for key in sorted(real_bs, key=lambda k: (k[0], ISL_ORD.get(k[1], 99), k[2], k[4])):
    model, isl, dt, N, BS = key
    d = real_bs[key]
    b, p, o = d.get("gvr_base"), d.get("gvr_pr"), d.get("op26_r0auto")
    bh2.append(f"{model},{isl},{LMAP.get(model, '')},{dt},{N},{_cs(d, N)},{BS},"
               f"{d.get('hit', '')},{b or ''},{p or ''},{o or ''},"
               f"{_r3(b, p)},{_r3(o, p)},{d.get('pr_exact', '')}")
open(os.path.join(HERE, "bs_real.csv"), "w").write("\n".join(bh2) + "\n")

# ---- §8 rival_long.csv: replace GVR-family rows -------------------------------
COLS = ["family", "sweep", "scenario", "model", "op", "K", "dtype", "N", "BS",
        "isl", "cr", "hit", "us", "us_span", "exact"]
GVR_OPS = {"gvr_base", "gvr_pr", "op26_r0auto"}
old = list(csv.DictReader(open(os.path.join(HERE, "rival_long.csv"))))
kept = [r for r in old if r["op"] not in GVR_OPS]
new_rows = []
for r in rows:
    new_rows.append({c: r.get(c, "") for c in COLS})
merged = kept + new_rows
merged.sort(key=lambda r: (r["family"], r["sweep"], r.get("scenario", ""),
                           r.get("model", ""), r["dtype"], int(r["K"]),
                           int(r["N"]), int(r["BS"]), r.get("isl", ""), r["op"]))
with open(os.path.join(HERE, "rival_long.csv"), "w") as f:
    f.write(",".join(COLS) + "\n")
    for r in merged:
        f.write(",".join(str(r[c]) for c in COLS) + "\n")
print(f"rival_long.csv: {len(kept)} rival rows kept + {len(new_rows)} refreshed GVR rows")

# ---- anchor drift: new op26 vs old op26 (rival_long backup keys) --------------
old_op26 = {}
for r in old:
    if r["op"] == "op26_r0auto" and r.get("us"):
        old_op26[(r["family"], r["sweep"], r.get("scenario", ""), r.get("model", ""),
                  r["dtype"], r["K"], r["N"], r["BS"], r.get("isl", ""))] = float(r["us"])
drift = []
for r in rows:
    if r["op"] != "op26_r0auto":
        continue
    k = (r["family"], r["sweep"], r.get("scenario", ""), r.get("model", ""),
         r["dtype"], str(r["K"]), str(r["N"]), str(r["BS"]), str(r.get("isl", "")))
    if k in old_op26:
        drift.append(_fnum(r["us"]) / old_op26[k])
if drift:
    drift.sort()
    med = drift[len(drift) // 2]
    p95 = drift[int(len(drift) * 0.95)]
    print(f"anchor drift op26 new/old: n={len(drift)} median={med:.3f} p95={p95:.3f}"
          f"  (gate: p95 within ~1.15 per house rule)")
else:
    print("anchor drift: no overlapping op26 cells found")

print(f"synth_3arm {len(sh)-1} | real_3arm {len(rh)-1} | "
      f"bs_synth {len(bh)-1} | bs_real {len(bh2)-1}")
# per-cell exactness tally
for nm, dd in (("syn_seq", syn_seq), ("real_seq", real_seq),
               ("syn_bs", syn_bs), ("real_bs", real_bs)):
    tot = sum(1 for d in dd.values() if "pr_exact" in d)
    ok = sum(1 for d in dd.values() if d.get("pr_exact"))
    print(f"  {nm}: pr_exact {ok}/{tot}")
