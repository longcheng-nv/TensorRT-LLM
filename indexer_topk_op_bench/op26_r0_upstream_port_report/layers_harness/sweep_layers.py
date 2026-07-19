#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op26 report per-layer backfill sweep (launch contract, HEAD 018251950f).

Two sweeps, both real-capture only, 3 in-tree arms (gvr_base / gvr_pr /
op26_r0auto), timing protocol identical to refresh_harness/sweep_refresh.py
(NVTX c|/w| ranges, cold-L2 512MB evict, 20 cold + 50 warm reps, one
nsys-rep per batch, cell-resumable jsonl):

  seqlen  --model M --isl I   : BS=1 fp32, ALL captured GVR-active layers
                                (flash 21 even 2..42, pro 30 even 2..60,
                                v32 58 layers 3..60)  -> report §4b
  bs      --model M --layer L : fp32, 11-BS grid x all ISL rungs, the 3
                                GVR-active bench layers per model -> §7b

v32 all-layer support: the slim cache normally materializes only
BENCH_LAYERS; run `python3 sweep_layers.py --prep-v32` ONCE before sharding
to rebuild the slims with all 58 layers (avoids a force-prepare race between
shards).
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
sys.path.insert(0, str(_REPORT / "refresh_harness"))
sys.path.insert(0, str(_BENCH / "harness"))

from sweep_nsys import measure_cell                                  # noqa: E402
from ops_refresh import build_call_rival, ops_for_rival              # noqa: E402
import real_data_v4cap as RV4                                        # noqa: E402
import real_data_v32 as RV32                                         # noqa: E402

ALL_LAYERS = {"flash": list(range(2, 43, 2)),
              "pro": list(range(2, 61, 2)),
              "v32": list(RV32.LAYERS_ALL)}
BS3_LAYERS = {"flash": [10, 22, 34], "pro": [14, 30, 46], "v32": [14, 34, 54]}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def prep_v32():
    """Rebuild v32 slims with ALL 58 layers (idempotent: skips complete slims)."""
    RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
    for isl in REAL_ISLS["v32"]:
        p = RV32._slim_path(isl)
        if p.exists():
            s = torch.load(p, map_location="cpu", weights_only=False)
            if set(s["cur"].keys()) >= set(RV32.LAYERS_ALL):
                print(f"v32 {isl}: slim already all-layer ({len(s['cur'])})")
                continue
        RV32.prepare(isl, force=True)
        print(f"v32 {isl}: re-slimmed with {len(RV32.LAYERS_ALL)} layers")


def bundle(model, isl, L, dt_name="fp32"):
    if model == "v32":
        RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)   # slim key domain
        return RV32.get_bundle(model, isl, L, dt_name)
    return RV4.get_bundle(model, isl, L, dt_name)


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["op"], r["N"], r["BS"], r.get("isl", ""), r.get("L")))
            except Exception:
                pass
    return done


def _exact(out_idx_getter, logits_full, N, K, BS):
    idx_t = out_idx_getter()
    ref = torch.topk(logits_full[:N].float(), K).values.sort().values
    rows = (0, BS - 1) if BS > 1 else (0,)
    for r in rows:
        idx = idx_t[r].long()
        if idx.numel() != K or idx.min() < 0 or idx.max() >= N or idx.unique().numel() != K:
            return False
        got = logits_full[idx].float().sort().values
        if not torch.equal(got, ref):
            return False
    return True


def run(sweep, model, isl_arg, layer_arg, out_path, rc, rw):
    done = _load_done(out_path)
    if sweep == "seqlen":
        cells = [(isl_arg, L, 1) for L in ALL_LAYERS[model]]
    else:
        cells = [(isl, layer_arg, BS) for isl in REAL_ISLS[model] for BS in BS_GRID]
    print(f"# layers {model} {sweep} isl={isl_arg} L={layer_arg} cells={len(cells)}", flush=True)
    f = open(out_path, "a")
    prof.start()
    try:
        for i, (isl, L, BS) in enumerate(cells):
            try:
                bd = bundle(model, isl, L, "fp32")
            except Exception as e:
                print(f"  SKIP {model}/{isl}/L{L}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            K, N, cr = bd["K"], bd["N"], bd["cr"]
            lg_full = bd["logits"][0, :N]
            for op in ops_for_rival("fp32", K):
                if (op, N, BS, isl, L) in done:
                    continue
                base = f"{op}|{model}|{isl}|L{L}|fp32|{N}|{BS}"
                rec = dict(family="real", sweep=sweep, model=model, op=op, K=K,
                           dtype="fp32", N=N, BS=BS, cr=cr, L=L,
                           hit=bd["hit_rate"], isl=isl,
                           data_src=f"{model}/{isl}/L{L}",
                           range_cold=f"c|{base}", range_warm=f"w|{base}",
                           reps_cold=rc, reps_warm=rw)
                try:
                    call, keep, extra, getter = build_call_rival(
                        op, K, torch.float32, N, BS, cr, bd["logits"], bd["preIdx"])
                    rec.update(extra)
                    if getter is not None:
                        rec["exact"] = bool(_exact(getter, lg_full, N, K, BS))
                    measure_cell(call, base, rc, rw)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
                f.write(json.dumps(rec) + "\n"); f.flush()
                gc.collect(); torch.cuda.empty_cache()
            if (i + 1) % 5 == 0 or i + 1 == len(cells):
                print(f"[{model}/{sweep}] {i+1}/{len(cells)} (isl={isl} L={L} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-v32", action="store_true")
    ap.add_argument("--sweep", choices=["seqlen", "bs"])
    ap.add_argument("--model", choices=["flash", "pro", "v32"])
    ap.add_argument("--isl")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--out-root")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    a = ap.parse_args()
    if a.prep_v32:
        prep_v32()
        return
    out = Path(a.out_root)
    out.mkdir(parents=True, exist_ok=True)
    tag = (f"real_seqlen_{a.model}_{a.isl}" if a.sweep == "seqlen"
           else f"real_bs_{a.model}_L{a.layer}")
    run(a.sweep, a.model, a.isl, a.layer, out / f"{tag}.jsonl", a.reps, a.reps_warm)


if __name__ == "__main__":
    main()
