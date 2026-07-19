# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the real-data 3-arm sweep PER LAYER (no median) into
real_3arm_layers.csv — the §4b per-layer view.

Source = rc_shard_*.out + rcv32_shard_*.out (2026-07-15 sweep, 3 GVR-active
layers per model: flash L10/L22/L34, pro L14/L30/L46, v32 L14/L34/L54).
NOTE provenance: this sweep ran at the pre-refresh frozen launch config
(cs4/T1024/mbpm1/v256) on branch HEAD 018251950f — the same raw data whose
per-(model,ISL) layer-MEDIAN used to be §4 before the 2026-07-16 launch-
contract refresh replaced the headline with single-layer (L22/L30/L34)
launch-shape rows. Inter-layer RELATIVE spread is the point of this file;
absolute large-N V4 cells understate PR vs the refreshed contract (§6).
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
K_OF = {"flash": 512, "pro": 1024, "v32": 2048}
CR_OF = {"flash": 4, "pro": 4, "v32": 1}
ISL_ORDER = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]
MODEL_ORDER = {"flash": 0, "pro": 1, "v32": 2}

line_re = re.compile(
    r"^(flash|pro|v32)\s+(\S+)\s+L(\d+)\s+(base|pr|op26)\s+dur=([0-9.]+|NA)"
    r"(?:.*?N=(\d+))?(?:.*?hit=([0-9.]+))?(?:.*?EXACT=(True|False))?")

cells = {}   # (model, isl, layer) -> {arm: dur, N, hit, pr_exact, base_exact}
files = sorted(glob.glob(os.path.join(HERE, "rc_shard_*.out"))
               + glob.glob(os.path.join(HERE, "rcv32_shard_*.out")))
for f in files:
    for ln in open(f):
        m = line_re.match(ln.strip())
        if not m:
            continue
        model, isl, layer, arm, dur, N, hit, exact = m.groups()
        d = cells.setdefault((model, isl, int(layer)), {})
        if dur != "NA":
            d[arm] = float(dur)
        if N:
            d["N"] = int(N)
        if hit:
            d["hit"] = float(hit)
        if exact is not None:
            d[f"{arm}_exact"] = exact

out = ["model,isl,N,K,layer,hit,base,pr,op26,pr_vs_base,pr_vs_op26,"
       "pr_exact,base_exact"]
order = sorted(cells, key=lambda k: (MODEL_ORDER[k[0]], ISL_ORDER.index(k[1]),
                                     k[2]))
for (model, isl, layer) in order:
    d = cells[(model, isl, layer)]
    b, p, o = d.get("base"), d.get("pr"), d.get("op26")
    N = d.get("N") or int(isl[:-1]) * 1024 // CR_OF[model]
    pvb = round(b / p, 3) if (b and p) else ""
    pvo = round(o / p, 3) if (o and p) else ""
    out.append(f"{model},{isl},{N},{K_OF[model]},{layer},"
               f"{round(d.get('hit', float('nan')), 3)},{b},{p},{o},"
               f"{pvb},{pvo},{d.get('pr_exact', '')},{d.get('base_exact', '')}")

txt = "\n".join(out) + "\n"
open(os.path.join(HERE, "real_3arm_layers.csv"), "w").write(txt)
print(txt)
print(f"{len(out) - 1} per-layer rows "
      f"({len(set((m, i) for m, i, _ in cells))} rungs x 3 layers)")
