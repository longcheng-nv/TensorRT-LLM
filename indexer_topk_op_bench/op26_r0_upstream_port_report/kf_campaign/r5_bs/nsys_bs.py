# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R5 nsys BS worker: candidate vs PR-head GVR (native batch), §7b batch
construction (same row materialized BS times). Run UNDER nsys (cudaProfilerApi).

Arms:
  gvr_pr  — gvrpkg_04a0 GvrTopKKernel.launch(logits[BS,npad], pre_idx[BS,k],
            seq_lens[BS], out[BS,k], K, compress_ratio=cr) one batched call
  kf_cand — compiled candidate run(logits, pre_idx, n_valid, cell_id, out)
            (cell_id passed through; candidates may ignore it)

  python3 nsys_bs.py [--cand DIR] [--arms gvr_pr,kf_cand] [--tag t]
                     [--cells all|uuid,..] [--bs 2,4,...] [--shard i/m]
                     [--reps-cold N] [--reps-warm N]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
KF = HERE.parent
REPORT = KF.parent
BENCH = REPORT.parent
import os  # noqa: E402
sys.path.insert(0, str(KF / os.environ.get("GVRPKG_DIR", "gvrpkg_04a0")))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(KF))

from sweep_nsys import measure_cell  # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402
import quick_ab as Q  # noqa: E402

DEV = "cuda"
DEFAULT_BS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def load_cells():
    return list(csv.DictReader(open(HERE / "cells_meta_bs.csv")))


def bundle_for(row, bs):
    mod = v32 if row["model"] == "v32" else v4
    b = mod.get_bundle(row["model"], row["isl"], int(row["layer"]), "fp32")
    lg = b["logits"].float().expand(bs, -1).contiguous()
    pre = b["preIdx"].expand(bs, -1).contiguous()
    return dict(model=row["model"], N=b["N"], K=b["K"], cr=b["cr"],
                Npad=b["Npad"], logits=lg, preIdx=pre, ref=b["ref"],
                cell_id=int(row["cell_id"]), bs=bs)


def pr_call(b):
    K, cr, N, bs = b["K"], b["cr"], b["N"], b["bs"]
    sl = torch.full((bs,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(bs, K, dtype=torch.int32, device=DEV)
    return (lambda: GvrTopKKernel.launch(b["logits"], b["preIdx"], sl, out, K,
                                         compress_ratio=cr)), out


def cand_call(b, mod, entry):
    K, N, bs = b["K"], b["N"], b["bs"]
    out = torch.empty(bs, K, dtype=torch.int32, device=DEV)
    fn = getattr(mod, entry)
    return (lambda: fn(b["logits"], b["preIdx"], N, out)), out


def exact_bs(b, out):
    lg = b["logits"][0, :b["N"]].float()
    ref = lg[b["ref"].to(torch.int64)].sort().values
    for i in range(b["bs"]):
        idx = out[i].to(torch.int64)
        if idx.numel() != b["K"] or int(idx.min()) < 0 or int(idx.max()) >= b["N"]:
            return False, f"row{i}: range/count"
        if torch.unique(idx).numel() != b["K"]:
            return False, f"row{i}: dup"
        sel = lg[idx].sort().values
        if not torch.equal(sel, ref):
            return False, f"row{i}: vdiff={float((sel-ref).abs().max()):.3e}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand")
    ap.add_argument("--entry", default=None)
    ap.add_argument("--arms", default="gvr_pr,kf_cand")
    ap.add_argument("--tag", default="t0")
    ap.add_argument("--cells", default="all")
    ap.add_argument("--bs", default=",".join(map(str, DEFAULT_BS)))
    ap.add_argument("--shard", default=None)
    ap.add_argument("--reps-cold", type=int, default=15)
    ap.add_argument("--reps-warm", type=int, default=10)
    args = ap.parse_args()

    arms = args.arms.split(",")
    cmod, entry = None, None
    if "kf_cand" in arms:
        cmod, entry = Q.build_candidate(args.cand)
        if args.entry:
            entry = args.entry
        print(f"[nsys_bs] candidate entry: {entry}", flush=True)

    rows = load_cells()
    if args.cells != "all":
        want = set(args.cells.split(","))
        rows = [r for r in rows if r["uuid"] in want]
    bs_list = [int(x) for x in args.bs.split(",")]
    cases = [(r, bs) for r in rows for bs in bs_list]
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        cases = cases[i::m]
    print(f"[nsys_bs] {len(cases)} cases", flush=True)

    exact_log = {}
    prof.start()
    for r, bs in cases:
        b = bundle_for(r, bs)
        cuuid = f"{r['uuid']}_bs{bs}"
        for arm in arms:
            call, out = pr_call(b) if arm == "gvr_pr" else cand_call(b, cmod, entry)
            call()
            torch.cuda.synchronize()
            ok, why = exact_bs(b, out)
            exact_log[f"{cuuid}|{arm}"] = (ok, why)
            if not ok:
                print(f"[nsys_bs] INEXACT {cuuid} {arm}: {why}", flush=True)
            measure_cell(call, f"{arm}|{cuuid}", args.reps_cold, args.reps_warm)
        del b
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()
    prof.stop()
    (HERE / f"exact_{args.tag}.json").write_text(json.dumps(exact_log, indent=1))
    n_bad = sum(1 for v in exact_log.values() if not v[0])
    print(f"[nsys_bs] done, inexact {n_bad}/{len(exact_log)}", flush=True)


if __name__ == "__main__":
    main()
