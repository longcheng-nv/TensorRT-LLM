# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R5 BS-campaign export: §7b real decode-capture cells -> KF inputs.

Cell universe = REPORT §7b per-layer BS grid (bs_real_layers.csv): 75 cells
(flash L10/22/34 x 9 ISL, pro L14/30/46 x 9 ISL, v32 L14/34/54 x 7 ISL).
Batch construction (canonical §7b): the SAME row materialized BS times.

Writes:
  assets/cell_<uuid>.safetensors      {logits [1,Npad] fp32, pre_idx [1,K] i32}
  definition_bs.json                  axes b/n/k, custom inputs (BS expansion
                                      from staged assets by cell_id) + tie-robust
                                      per-row correctness
  workload_bs.jsonl                   platform subset (stratified cells x BS)
  cells_meta_bs.csv                   bookkeeping incl. cell_id table
"""
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
KF = HERE.parent
REPORT = KF.parent
BENCH = REPORT.parent
sys.path.insert(0, str(BENCH / "harness"))
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

MAX_TIE = 4096
# platform subset: per model, (small, mid, large) ISL; min-hit layer x 4 BS,
# plus max-hit layer at mid ISL x 2 BS, plus BS=1 guards.
SUBSET_ISL = {"flash": ["4k", "64k", "512k"],
              "pro": ["4k", "64k", "1024k"],
              "v32": ["4k", "64k", "256k"]}
SUBSET_BS = [4, 32, 256, 1024]
EXTRA_BS = [32, 1024]
GUARD_BS1 = {"flash": ["4k", "512k"], "pro": ["64k", "1024k"], "v32": ["256k"]}

rows = list(csv.DictReader(open(REPORT / "bs_real_layers.csv")))
cells = {}
for r in rows:
    key = (r["model"], r["isl"], int(r["L"]))
    if key not in cells:
        cells[key] = dict(model=r["model"], isl=r["isl"], layer=int(r["L"]),
                          N=int(r["N"]), hit=float(r["hit"]))
cells = dict(sorted(cells.items()))
print(f"{len(cells)} §7b cells")

wl, meta = [], []
total_bytes = 0
cell_table = []  # cell_id -> filename (embedded into definition reference)
for cid, ((model, isl, layer), c) in enumerate(cells.items()):
    mod = v32 if model == "v32" else v4
    b = mod.get_bundle(model, isl, layer, "fp32")
    N, K, Npad = b["N"], b["K"], b["Npad"]
    assert N == c["N"], (model, isl, layer, N, c["N"])
    v = b["logits"][0, :N].float()
    kth = torch.topk(v, K, largest=True, sorted=True).values[-1]
    mand = (v > kth).nonzero(as_tuple=True)[0]
    tie = (v == kth).nonzero(as_tuple=True)[0]
    assert mand.numel() < K <= mand.numel() + tie.numel()
    assert tie.numel() <= MAX_TIE, (model, isl, layer, tie.numel())
    uuid = f"{model}_{isl}_L{layer:02d}"
    fn = f"cell_{uuid}.safetensors"
    save_file({"logits": b["logits"].float().cpu().contiguous(),
               "pre_idx": b["preIdx"].cpu().contiguous()},
              HERE / "assets" / fn)
    total_bytes += (HERE / "assets" / fn).stat().st_size
    cell_table.append(fn)
    meta.append(dict(cell_id=cid, uuid=uuid, model=model, isl=isl, layer=layer,
                     N=N, K=K, Npad=Npad, hit=c["hit"],
                     n_mand=int(mand.numel()), n_tie=int(tie.numel()),
                     file=fn))
    # platform workloads for this cell
    bs_list = []
    if isl in SUBSET_ISL[model]:
        hits = sorted((cc["hit"], ll) for (mm, ii, ll), cc in cells.items()
                      if mm == model and ii == isl)
        if layer == hits[0][1]:
            bs_list += SUBSET_BS
        elif isl == SUBSET_ISL[model][1] and layer == hits[-1][1]:
            bs_list += EXTRA_BS
    if isl in GUARD_BS1[model] and layer == min(
            ll for (mm, ii, ll) in cells if mm == model and ii == isl):
        bs_list += [1]
    for bs in bs_list:
        wl.append(json.dumps({
            "uuid": f"{uuid}_bs{bs}",
            "axes": {"b": bs, "n": N, "k": K},
            "inputs": {
                "logits": {"type": "custom"},
                "pre_idx": {"type": "custom"},
                "n_valid": {"type": "scalar", "value": N},
                "cell_id": {"type": "scalar", "value": cid},
            },
            "custom_correctness_kwargs": {
                "mandatory": mand.cpu().tolist(),
                "tie": tie.cpu().tolist(),
                "k": K, "n": N, "b": bs,
            },
        }))

REFERENCE = '''import os
import torch

_CELL_FILES = %s


def _load_row(cell_id, device):
    from safetensors.torch import load_file
    fn = _CELL_FILES[int(cell_id)]
    bases = [".", os.environ.get("CUDAGYM_STAGING_DIR", ".")]
    try:
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    for base in bases:
        p = os.path.join(base, fn)
        if os.path.exists(p):
            d = load_file(p)
            return d["logits"].to(device), d["pre_idx"].to(device)
    raise FileNotFoundError(fn)


def gen_inputs(axes_and_scalars, device, seed):
    """SS7b batch construction: the SAME captured row materialized b times."""
    b = int(axes_and_scalars["b"])
    cell_id = int(axes_and_scalars["cell_id"])
    logits, pre_idx = _load_row(cell_id, device)
    return {"logits": logits.expand(b, -1).contiguous().float(),
            "pre_idx": pre_idx.expand(b, -1).contiguous().to(torch.int32)}


def run(logits, pre_idx, n_valid, cell_id):
    b = logits.shape[0]
    k = pre_idx.shape[1]
    vals = logits[:, :n_valid].float()
    return torch.topk(vals, k, dim=1, largest=True,
                      sorted=True).indices.to(torch.int32)


def check_topk(ref_outputs, user_outputs, *, mandatory, tie, k, n, b):
    """Tie-robust top-k index-set check, applied to EVERY batch row
    (all rows are copies of the same captured row)."""
    out = user_outputs["indices"].to(torch.int64).cpu()
    problems = []
    if out.shape != (b, k):
        problems.append(f"shape {tuple(out.shape)} != ({b},{k})")
    else:
        mand = torch.tensor(mandatory, dtype=torch.int64)
        tie_t = (torch.tensor(tie, dtype=torch.int64) if len(tie)
                 else torch.empty(0, dtype=torch.int64))
        allowed = torch.cat([mand, tie_t])
        for i in range(b):
            idx = out[i]
            if int(idx.min()) < 0 or int(idx.max()) >= n:
                problems.append(f"row {i}: index out of range"); break
            if torch.unique(idx).numel() != k:
                problems.append(f"row {i}: duplicate indices"); break
            if not bool(torch.isin(idx, allowed).all()):
                problems.append(f"row {i}: index outside true top-k set"); break
            if not bool(torch.isin(mand, idx).all()):
                problems.append(f"row {i}: missing mandatory index"); break
    passed = len(problems) == 0
    return {"indices": {"passed": passed, "message": "; ".join(problems) or "ok"}}
''' % json.dumps(cell_table)

definition = {
    "name": "indexer_topk_decode_bs_real",
    "description": ("DeepSeek indexer decode top-k, REAL captured logits rows "
                    "(SS7b grid), batched: BS copies of the same row. fp32 in, "
                    "int32 indices out, tie-robust per-row exactness."),
    "op_type": "selection",
    "axes": {
        "b": {"type": "var", "description": "batch size (1..1024), all rows identical copies"},
        "n": {"type": "var", "description": "valid row length (post-compression)"},
        "k": {"type": "var", "description": "top-k: 512/1024/2048"},
        "npad": {"type": "expr", "expression": "(n + 63) // 64 * 64"},
    },
    "inputs": {
        "logits": {"shape": ["b", "npad"], "dtype": "float32"},
        "pre_idx": {"shape": ["b", "k"], "dtype": "int32"},
        "n_valid": {"shape": None, "dtype": "int32"},
        "cell_id": {"shape": None, "dtype": "int32"},
    },
    "outputs": {"indices": {"shape": ["b", "k"], "dtype": "int32"}},
    "custom_inputs_entrypoint": "gen_inputs",
    "custom_correctness_entrypoint": "check_topk",
    "reference": REFERENCE,
}
(HERE / "definition_bs.json").write_text(json.dumps(definition, indent=1))
(HERE / "workload_bs.jsonl").write_text("\n".join(wl) + "\n")
with open(HERE / "cells_meta_bs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
    w.writeheader()
    w.writerows(meta)
print(f"assets {total_bytes/1e6:.1f} MB, {len(wl)} platform workloads")
