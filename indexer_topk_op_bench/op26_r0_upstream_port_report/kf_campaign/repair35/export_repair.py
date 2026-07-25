# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""repair35 campaign export: HETEROGENEOUS-row workloads for the 35 loser
cases of the 2-arm combined dispatch (16k-256k x BS256-1024 band + pockets).

Anti-exploit design (lessons from R5 + fresh champion DQ):
- batch rows CYCLE over ALL GVR-active layers of the (model, isl) group —
  rows genuinely differ, so identity/broadcast shortcuts never pay;
- per-layer tie-robust reference sets ship inside the asset (mand/tie
  tensors), so correctness is checked per row against its own layer.

Writes: assets/grp_<model>_<isl>.safetensors, definition_repair.json,
workload_repair.jsonl, groups_meta.csv."""
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
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

GROUPS = [  # (model, isl, primary loser BS list, guard BS list)
    ("flash", "16k", [256, 512, 1024], [64]),
    ("flash", "32k", [256, 512, 1024], [64]),
    ("flash", "128k", [256], [64]),
    ("flash", "256k", [32, 128, 256, 512, 1024], [16]),
    ("pro", "128k", [256, 512], [64]),
    ("pro", "256k", [128, 256, 512], [64]),
    ("pro", "1024k", [16], [64]),
    ("v32", "64k", [256, 512, 1024], [64]),
]
LAYERS = {"flash": RV4.MODELS["flash"]["layers"],
          "pro": RV4.MODELS["pro"]["layers"],
          "v32": list(RV32.LAYERS_ALL)}


def main():
    (HERE / "assets").mkdir(exist_ok=True)
    meta = []
    files = []
    for gi, (model, isl, prim, guard) in enumerate(GROUPS):
        RD = RV32 if model == "v32" else RV4
        rows = []
        for L in LAYERS[model]:
            try:
                bd = RD.get_bundle(model, isl, L, "fp32")
            except Exception:
                continue
            rows.append((L, bd["logits"][0].clone(), bd["preIdx"][0].clone(),
                         bd["N"], bd["K"]))
            RV4._bundle_cache.clear()
            RV32._bundle_cache.clear()
        npad0 = rows[0][1].shape[0]
        rows = [r for r in rows if r[1].shape[0] == npad0]
        K = rows[0][4]
        N = max(r[3] for r in rows)
        lg = torch.stack([r[1] for r in rows])
        pre = torch.stack([r[2] for r in rows])
        # per-layer tie-robust sets over valid length N
        mand_l, tie_l = [], []
        for _, lrow, _, _, _ in rows:
            v = lrow[:N].float()
            kth = torch.topk(v, K).values[-1]
            mand = torch.nonzero(v > kth, as_tuple=True)[0]
            tie = torch.nonzero(v == kth, as_tuple=True)[0]
            mand_l.append(mand)
            tie_l.append(tie)
        mm = max(x.numel() for x in mand_l)
        tm = max(1, max(x.numel() for x in tie_l))
        mand_t = torch.full((len(rows), mm), -1, dtype=torch.int64)
        tie_t = torch.full((len(rows), tm), -1, dtype=torch.int64)
        for i, (a, b) in enumerate(zip(mand_l, tie_l)):
            mand_t[i, :a.numel()] = a
            tie_t[i, :b.numel()] = b
        fn = f"grp_{model}_{isl}.safetensors"
        save_file({"logits": lg.contiguous(), "pre_idx": pre.contiguous(),
                   "mand": mand_t, "tie": tie_t}, str(HERE / "assets" / fn))
        files.append(fn)
        meta.append(dict(gid=gi, model=model, isl=isl, K=K, N=N, npad=npad0,
                         n_layers=len(rows), prim=prim, guard=guard,
                         asset=fn, mb=round(lg.numel() * 4 / 1e6, 1)))
        print(f"[export] {fn}: {len(rows)} layers K{K} N{N} "
              f"{meta[-1]['mb']}MB", flush=True)

    ref = f'''import os
import torch

_GRP_FILES = {json.dumps(files)}
_CACHE = {{}}


def _load_grp(gid, device):
    key = (int(gid), str(device))
    if key in _CACHE:
        return _CACHE[key]
    from safetensors.torch import load_file
    fn = _GRP_FILES[int(gid)]
    bases = [".", os.environ.get("CUDAGYM_STAGING_DIR", ".")]
    try:
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    for base in bases:
        p = os.path.join(base, fn)
        if os.path.exists(p):
            d = load_file(p)
            out = tuple(d[k].to(device) for k in
                        ("logits", "pre_idx", "mand", "tie"))
            _CACHE[key] = out
            return out
    raise FileNotFoundError(fn)


def gen_inputs(axes_and_scalars, device, seed):
    """HETEROGENEOUS batch: row i is layer (i mod L) of the group capture —
    rows genuinely differ (production decode batches are heterogeneous)."""
    b = int(axes_and_scalars["b"])
    gid = int(axes_and_scalars["cell"])
    n = int(axes_and_scalars["n"])
    lg, pre, _, _ = _load_grp(gid, device)
    L = lg.shape[0]
    idx = torch.arange(b, device=device) % L
    return {{"logits": lg[idx].contiguous().float(),
            "pre_idx": pre[idx].contiguous().to(torch.int32),
            "n_valid": n}}


def run(logits, pre_idx, n_valid):
    k = pre_idx.shape[1]
    vals = logits[:, :n_valid].float()
    return torch.topk(vals, k, dim=1, largest=True,
                      sorted=True).indices.to(torch.int32)


def check_topk(ref_outputs, user_outputs, *, gid, b, k, n):
    """Tie-robust per-row top-k index-set check; row i is checked against
    ITS OWN layer's reference set (mand/tie loaded from the group asset)."""
    out = user_outputs["indices"].to(torch.int64).cpu()
    _, _, mand_t, tie_t = _load_grp(gid, "cpu")
    L = mand_t.shape[0]
    problems = []
    if out.shape != (b, k):
        problems.append(f"shape {{tuple(out.shape)}} != ({{b}},{{k}})")
    else:
        for i in range(b):
            li = i % L
            mand = mand_t[li]
            mand = mand[mand >= 0]
            tie = tie_t[li]
            tie = tie[tie >= 0]
            allowed = torch.cat([mand, tie])
            idx = out[i]
            if int(idx.min()) < 0 or int(idx.max()) >= n:
                problems.append(f"row {{i}}: index out of range")
                break
            if torch.unique(idx).numel() != k:
                problems.append(f"row {{i}}: duplicate indices")
                break
            if not bool(torch.isin(idx, allowed).all()):
                problems.append(f"row {{i}}: index outside true top-k set")
                break
            if not bool(torch.isin(mand, idx).all()):
                problems.append(f"row {{i}}: missing mandatory index")
                break
    passed = len(problems) == 0
    return {{"indices": {{"passed": passed,
                          "message": "; ".join(problems) or "ok"}}}}
'''
    definition = dict(
        name="indexer_topk_repair35_hetero",
        description=("GVR top-K decode, HETEROGENEOUS-row repair band: the 35 "
                     "combined-dispatch loser cases (mid-N x high-BS) on real "
                     "DSv4/V3.2 decode captures, fp32, B200"),
        op_type="topk",
        axes=dict(
            b={"type": "var", "description":
               "batch size; rows CYCLE over the group's layers (heterogeneous)"},
            n={"type": "var", "description": "valid row length"},
            k={"type": "var", "description": "top-k: 512/1024/2048"},
            npad={"type": "expr", "expression": "(n + 63) // 64 * 64"},
            cell={"type": "var", "description":
                  "group id (input-generator key; not a tensor dim)"},
        ),
        inputs=dict(
            logits={"shape": ["b", "npad"], "dtype": "float32"},
            pre_idx={"shape": ["b", "k"], "dtype": "int32"},
            n_valid={"shape": None, "dtype": "int32"},
        ),
        outputs=dict(indices={"shape": ["b", "k"], "dtype": "int32"}),
        custom_inputs_entrypoint="gen_inputs",
        custom_correctness_entrypoint="check_topk",
        reference=ref,
    )
    json.dump(definition, open(HERE / "definition_repair.json", "w"),
              indent=1)

    with open(HERE / "workload_repair.jsonl", "w") as f:
        for m in meta:
            for bs, kind in ([(x, "prim") for x in m["prim"]] +
                             [(x, "guard") for x in m["guard"]]):
                u = f"{m['model']}_{m['isl']}_bs{bs}" + \
                    ("" if kind == "prim" else "_guard")
                line = dict(
                    uuid=u,
                    axes=dict(b=bs, n=m["N"], k=m["K"], cell=m["gid"]),
                    inputs=dict(logits={"type": "custom"},
                                pre_idx={"type": "custom"},
                                n_valid={"type": "custom"}),
                    custom_correctness_kwargs=dict(gid=m["gid"], b=bs,
                                                   k=m["K"], n=m["N"]),
                )
                f.write(json.dumps(line) + "\n")
    with open(HERE / "groups_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(meta[0].keys()))
        w.writeheader()
        for m in meta:
            m["prim"] = " ".join(map(str, m["prim"]))
            m["guard"] = " ".join(map(str, m["guard"]))
            w.writerows([m])
    tot = sum(m["mb"] for m in meta)
    nwl = sum(1 for _ in open(HERE / "workload_repair.jsonl"))
    print(f"[export] {len(meta)} groups, {tot:.0f}MB assets, {nwl} workloads")


if __name__ == "__main__":
    main()
