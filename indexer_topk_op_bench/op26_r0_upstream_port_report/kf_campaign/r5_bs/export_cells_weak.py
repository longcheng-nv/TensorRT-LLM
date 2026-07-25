# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weak-rung export for campaign gvr-topk-weakrung (direction-iii KF arm).

Targets the 28 PR-fallback rungs of the composite envelope
(COMPOSITE_ENVELOPE_20260725.md): BS 256-1024 x 16k-256k throughput wall
plus pro_512k@128 / pro_1024k@16. One low-hit layer per (model, isl),
BS picks sized under the 500 MiB asset cap (big-BS x big-npad corners
stay local-only, verified externally).

Writes assets_weak/wl_*.safetensors, definition_weak.json,
workload_weak.jsonl, baselines_weak.jsonl (PR head nsys cold medians
from grid_r5pr.csv).
"""
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
KF = HERE.parent
BENCH = KF.parent.parent
sys.path.insert(0, str(BENCH / "harness"))
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

# weak-rung plan: (isl, 'lo' layer, [bs...])
PLAN = {
    # per-asset platform cap ~64 MiB => big-npad cells ride BS<=128 proxies
    # (all proxy (cell,BS) combos have measured PR baselines in grid_r5pr)
    "flash": [("16k", "lo", [256, 1024]), ("32k", "lo", [256]),
              ("128k", "lo", [256]), ("256k", "lo", [32, 128])],
    "pro":   [("16k", "lo", [256, 1024]), ("128k", "lo", [256]),
              ("256k", "lo", [16, 128]), ("512k", "lo", [64]),
              ("1024k", "lo", [16])],
    "v32":   [("64k", "lo", [128])],
}
MAX_TIE = 4096

meta = {(m["model"], m["isl"], int(m["layer"])): m
        for m in csv.DictReader(open(HERE / "cells_meta_bs.csv"))}
by_mi = {}
for (mo, isl, lay), m in meta.items():
    by_mi.setdefault((mo, isl), []).append((float(m["hit"]), lay))
for k in by_mi:
    by_mi[k].sort()

adir = HERE / "assets_weak"
adir.mkdir(exist_ok=True)
wl, total = [], 0
for model, entries in PLAN.items():
    for isl, which, bss in entries:
        hits = by_mi[(model, isl)]
        layer = hits[0][1] if which == "lo" else hits[-1][1]
        mod = v32 if model == "v32" else v4
        b = mod.get_bundle(model, isl, layer, "fp32")
        N, K = b["N"], b["K"]
        v = b["logits"][0, :N].float()
        kth = torch.topk(v, K, largest=True, sorted=True).values[-1]
        mand = (v > kth).nonzero(as_tuple=True)[0].cpu().tolist()
        tie = (v == kth).nonzero(as_tuple=True)[0].cpu().tolist()
        assert len(tie) <= MAX_TIE
        uuid = f"{model}_{isl}_L{layer:02d}"
        for bs in bss:
            fn = f"wl_{uuid}_bs{bs}.safetensors"
            save_file({"logits": b["logits"].float().expand(bs, -1).contiguous().cpu(),
                       "pre_idx": b["preIdx"].expand(bs, -1).contiguous().cpu()},
                      adir / fn)
            total += (adir / fn).stat().st_size
            wl.append(json.dumps({
                "uuid": f"{uuid}_bs{bs}",
                "axes": {"b": bs, "n": N, "k": K},
                "inputs": {
                    "logits": {"type": "safetensors", "path": fn, "tensor_key": "logits"},
                    "pre_idx": {"type": "safetensors", "path": fn, "tensor_key": "pre_idx"},
                    "n_valid": {"type": "scalar", "value": N},
                },
                "custom_correctness_kwargs": {
                    "mandatory": mand, "tie": tie, "k": K, "n": N, "b": bs},
            }))

d = json.load(open(HERE / "definition_bs2.json"))
d["name"] = "indexer_topk_decode_weakrung"
json.dump(d, open(HERE / "definition_weak.json", "w"), indent=1)
(HERE / "workload_weak.jsonl").write_text("\n".join(wl) + "\n")

pr5 = {r["cuuid"]: float(r["pr_cold"])
       for r in csv.DictReader(open(HERE / "grid_r5pr.csv")) if r["pr_cold"]}
bl = []
for line in wl:
    w = json.loads(line)
    us = pr5[w["uuid"]]
    bl.append(json.dumps({"uuid": w["uuid"], "execution_time_ms": round(us / 1000, 6)}))
(HERE / "baselines_weak.jsonl").write_text("\n".join(bl) + "\n")
print(f"{len(wl)} workloads, assets {total/1e6:.0f} MB, baselines {len(bl)}")
