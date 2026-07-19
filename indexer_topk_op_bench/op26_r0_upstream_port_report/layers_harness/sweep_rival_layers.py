#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""§8 per-layer rival backfill: external arms (radix_cutedsl / sglang_v2 /
flashinfer_topk) + op26_r0auto cross-run anchor, real captures, BS=1 fp32,
the 3 GVR-active bench layers per model. GVR per-layer rows come from
sweep_layers.py (launch contract) — not re-measured here.

Protocol identical to rival_harness/sweep_rival.py (NVTX ranges, cold-L2
evict, 20/50 reps, one nsys-rep per batch, cell-resumable jsonl).
Usage: sweep_rival_layers.py --model M --out-root DIR
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
_REPORT = _HERE.parent
_BENCH = _REPORT.parent
sys.path.insert(0, str(_REPORT / "rival_harness"))
sys.path.insert(0, str(_BENCH / "harness"))

from sweep_nsys import measure_cell                                  # noqa: E402
from ops_rival import build_call_rival                               # noqa: E402
import real_data_v4cap as RV4                                        # noqa: E402
import real_data_v32 as RV32                                         # noqa: E402

ARMS = ["op26_r0auto", "radix_cutedsl", "sglang_v2", "flashinfer_topk"]
BS3_LAYERS = {"flash": [10, 22, 34], "pro": [14, 30, 46], "v32": [14, 34, 54]}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}


def bundle(model, isl, L):
    if model == "v32":
        return RV32.get_bundle(model, isl, L, "fp32")
    return RV4.get_bundle(model, isl, L, "fp32")


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["op"], r["N"], r.get("isl", ""), r.get("L")))
            except Exception:
                pass
    return done


def _exact(getter, logits_full, N, K, BS):
    idx_t = getter()
    ref = torch.topk(logits_full[:N].float(), K).values.sort().values
    for r in ((0, BS - 1) if BS > 1 else (0,)):
        idx = idx_t[r].long()
        if idx.numel() != K or idx.min() < 0 or idx.max() >= N or idx.unique().numel() != K:
            return False
        if not torch.equal(logits_full[idx].float().sort().values, ref):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["flash", "pro", "v32"], required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rival_seqlen_{a.model}.jsonl"
    done = _load_done(path)
    cells = [(isl, L) for isl in REAL_ISLS[a.model] for L in BS3_LAYERS[a.model]]
    print(f"# rival-layers {a.model} cells={len(cells)} arms={ARMS}", flush=True)
    f = open(path, "a")
    prof.start()
    try:
        for i, (isl, L) in enumerate(cells):
            try:
                bd = bundle(a.model, isl, L)
            except Exception as e:
                print(f"  SKIP {a.model}/{isl}/L{L}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            K, N, cr = bd["K"], bd["N"], bd["cr"]
            lg_full = bd["logits"][0, :N]
            for op in ARMS:
                if (op, N, isl, L) in done:
                    continue
                base = f"{op}|{a.model}|{isl}|L{L}|fp32|{N}|1"
                rec = dict(family="real", sweep="seqlen", model=a.model, op=op,
                           K=K, dtype="fp32", N=N, BS=1, cr=cr, L=L,
                           hit=bd["hit_rate"], isl=isl,
                           data_src=f"{a.model}/{isl}/L{L}",
                           range_cold=f"c|{base}", range_warm=f"w|{base}",
                           reps_cold=a.reps, reps_warm=a.reps_warm)
                try:
                    call, keep, extra, getter = build_call_rival(
                        op, K, torch.float32, N, 1, cr, bd["logits"], bd["preIdx"])
                    rec.update(extra)
                    if getter is not None:
                        rec["exact"] = bool(_exact(getter, lg_full, N, K, 1))
                    measure_cell(call, base, a.reps, a.reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
                f.write(json.dumps(rec) + "\n"); f.flush()
                gc.collect(); torch.cuda.empty_cache()
            if (i + 1) % 3 == 0 or i + 1 == len(cells):
                print(f"[rival {a.model}] {i+1}/{len(cells)} (isl={isl} L{L})", flush=True)
    finally:
        prof.stop()
    f.close()


if __name__ == "__main__":
    main()
