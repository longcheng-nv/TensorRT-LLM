# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R6 export: v2 platform subset + valley/blind/bs1 coverage -> workload_bs3.jsonl.

Adds to the 30 v2 workloads: occupancy-valley points (64k x bs64/128),
formerly-blind exactness shapes (512k x bs8/16, 128k x bs64), extra bs1
guards, and bw-band probes (256k x bs32). Assets materialized into assets_v2/
(reuses existing files). Baselines from grid_r4pr2 (bs1) + grid_r5pr (bs>1).
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

PLAN_V2 = {
    "flash": [("4k", "lo", [1, 4, 32, 1024]), ("64k", "lo", [4, 256]),
              ("64k", "hi", [32]), ("512k", "lo", [1, 4, 32])],
    "pro":   [("4k", "lo", [1, 4, 32, 1024]), ("64k", "lo", [4, 256]),
              ("64k", "hi", [32]), ("1024k", "lo", [1, 4, 32])],
    "v32":   [("4k", "lo", [1, 4, 32, 1024]), ("64k", "lo", [4, 128]),
              ("64k", "hi", [32]), ("256k", "lo", [1, 4, 32])],
}
PLAN_ADD = {
    "flash": [("64k", "lo", [64, 128]), ("128k", "lo", [64]),
              ("256k", "lo", [32]), ("64k", "lo", [1])],
    "pro":   [("64k", "lo", [64, 128]), ("128k", "lo", [64]),
              ("512k", "lo", [8, 16]), ("256k", "lo", [32]), ("4k", "lo", [1])],
    "v32":   [("64k", "lo", [64]), ("128k", "lo", [64]), ("64k", "lo", [1]),
              ("32k", "lo", [128])],
}
MAX_TIE = 4096

meta = {(m["model"], m["isl"], int(m["layer"])): m
        for m in csv.DictReader(open(HERE / "cells_meta_bs.csv"))}
by_mi = {}
for (mo, isl, lay), m in meta.items():
    by_mi.setdefault((mo, isl), []).append((float(m["hit"]), lay))
for k in by_mi:
    by_mi[k].sort()

adir = HERE / "assets_v2"
wl, total, seen = [], 0, set()
for plan in (PLAN_V2, PLAN_ADD):
    for model, entries in plan.items():
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
                key = (uuid, bs)
                if key in seen:
                    continue
                seen.add(key)
                fn = f"wl_{uuid}_bs{bs}.safetensors"
                if not (adir / fn).exists():
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

(HERE / "workload_bs3.jsonl").write_text("\n".join(wl) + "\n")
pr5 = {r["cuuid"]: float(r["pr_cold"])
       for r in csv.DictReader(open(HERE / "grid_r5pr.csv")) if r["pr_cold"]}
pr4 = {r["uuid"]: float(r["pr_cold"])
       for r in csv.DictReader(open(KF / "grid_r4pr2.csv")) if r["pr_cold"]}
bl = []
for line in wl:
    w = json.loads(line)
    u = w["uuid"]
    us = pr5.get(u) or (pr4.get(u[:-4]) if u.endswith("_bs1") else None)
    assert us, u
    bl.append(json.dumps({"uuid": u, "execution_time_ms": round(us / 1000, 6)}))
(HERE / "baselines_bs3.jsonl").write_text("\n".join(bl) + "\n")
print(f"{len(wl)} workloads, assets total {total/1e6:.0f} MB, baselines {len(bl)}")
