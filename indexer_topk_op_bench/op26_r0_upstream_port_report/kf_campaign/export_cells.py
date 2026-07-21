# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export §4 real-data cells -> KernelFactory campaign inputs.

Selects a stratified subset of the 865-cell per-layer BS=1 fp32 grid
(real_3arm_layers_full.csv), and for each cell writes:
  assets/cell_<model>_<isl>_L<layer>.safetensors   {logits [1,Npad] fp32, pre_idx [1,K] int32}
plus campaign files:
  workload.jsonl    (axes n/k, safetensors inputs, scalar n_valid,
                     custom_correctness_kwargs = mandatory/tie index sets)
  baselines.jsonl   (uuid -> execution_time_ms from the REPORT `pr` column,
                     per user directive: baseline == current PR#16457 GVR)
  cells_meta.csv    (bookkeeping: hit, cs, pr/base/op26 timings)

Selection: per model, ISL rungs {4k,32k,128k,512k,1024k} (v32: {4k,32k,128k,256k});
per (model,isl) the min-hit and max-hit layers -> 28 cells.
"""
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BENCH = REPORT.parent
sys.path.insert(0, str(BENCH / "harness"))
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

ISL_SEL = {"flash": ["4k", "32k", "128k", "512k", "1024k"],
           "pro": ["4k", "32k", "128k", "512k", "1024k"],
           "v32": ["4k", "32k", "128k", "256k"]}
MAX_TIE = 4096

rows = list(csv.DictReader(open(REPORT / "real_3arm_layers_full.csv")))
sel = []
for model, isls in ISL_SEL.items():
    for isl in isls:
        cand = [r for r in rows if r["model"] == model and r["isl"] == isl]
        assert cand, (model, isl)
        cand.sort(key=lambda r: float(r["hit"]))
        pick = [cand[0], cand[-1]] if len(cand) > 1 else [cand[0]]
        sel.extend(pick)

assets = HERE / "assets"
assets.mkdir(parents=True, exist_ok=True)
wl_lines, bl_lines, meta = [], [], []
total_bytes = 0
for r in sel:
    model, isl, layer = r["model"], r["isl"], int(r["layer"])
    mod = v32 if model == "v32" else v4
    b = mod.get_bundle(model, isl, layer, "fp32")
    N, K, Npad = b["N"], b["K"], b["Npad"]
    assert Npad == (N + 63) // 64 * 64
    v = b["logits"][0, :N].float()
    kth = torch.topk(v, K, largest=True, sorted=True).values[-1]
    mand = (v > kth).nonzero(as_tuple=True)[0]
    tie = (v == kth).nonzero(as_tuple=True)[0]
    assert mand.numel() < K <= mand.numel() + tie.numel(), (model, isl, layer)
    assert tie.numel() <= MAX_TIE, (model, isl, layer, tie.numel())
    uuid = f"{model}_{isl}_L{layer:02d}"
    fn = f"cell_{uuid}.safetensors"
    save_file({"logits": b["logits"].float().cpu().contiguous(),
               "pre_idx": b["preIdx"].cpu().contiguous()}, assets / fn)
    total_bytes += (assets / fn).stat().st_size
    wl_lines.append(json.dumps({
        "uuid": uuid,
        "axes": {"n": N, "k": K},
        "inputs": {
            "logits": {"type": "safetensors", "path": fn, "tensor_key": "logits"},
            "pre_idx": {"type": "safetensors", "path": fn, "tensor_key": "pre_idx"},
            "n_valid": {"type": "scalar", "value": N},
        },
        "custom_correctness_kwargs": {
            "mandatory": mand.cpu().tolist(),
            "tie": tie.cpu().tolist(),
            "k": K, "n": N,
        },
    }))
    bl_lines.append(json.dumps(
        {"uuid": uuid, "execution_time_ms": float(r["pr"]) / 1000.0}))
    meta.append(dict(uuid=uuid, model=model, isl=isl, layer=layer, N=N, K=K,
                     Npad=Npad, hit=r["hit"], cs=r["cs"], base_us=r["base"],
                     pr_us=r["pr"], op26_us=r["op26"],
                     n_mand=mand.numel(), n_tie=tie.numel()))

(HERE / "workload.jsonl").write_text("\n".join(wl_lines) + "\n")
(HERE / "baselines.jsonl").write_text("\n".join(bl_lines) + "\n")
with open(HERE / "cells_meta.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
    w.writeheader()
    w.writerows(meta)
print(f"{len(sel)} cells, assets {total_bytes/1e6:.1f} MB")
for m in meta:
    print(f"  {m['uuid']:22s} N={m['N']:>7} K={m['K']:>4} hit={m['hit']} "
          f"cs={m['cs']} pr={m['pr_us']}us mand={m['n_mand']} tie={m['n_tie']}")
