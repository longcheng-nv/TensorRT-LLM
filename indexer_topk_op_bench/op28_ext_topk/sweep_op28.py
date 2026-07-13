#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op28 — nsys pure-kernel sweep of the LATEST external top-K arms on the
op22rr fixed-hit-rate temporal-synth bundles (BYTE-IDENTICAL inputs to the
op22 REPORT §1-2 re-test dataset):

  arms: gvr_cutedsl (ANCHOR — cross-node transfer to the op22rr µs scale)
        radix_cutedsl (2nd reference, hint-blind rival)
        sglang_streaming (OLD top512 vendor — same-node old-vs-new SGLang)
        sglang_v2 / flashinfer_topk / flashinfer_topk_i32 (op28 external arms)

  fp32 ONLY (both external kernels are fp32); K 512/1024/2048
  (sglang_streaming faithful only for K<=1024).

Timing protocol IDENTICAL to sweep_op22rr.py / harness/sweep_nsys.py:
10 warmup, 50 warm-L2 reps in "w|" NVTX ranges, 20 cold-L2 reps with 512MB
evict (outside range) in "c|" ranges, eager + sync inside the range, whole
loop inside the cudaProfilerApi window. Run UNDER nsys via drive_nsys_op28.sh;
kernel us filled by parse_op28.py.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))
sys.path.insert(0, str(HERE.parents[0] / "op22_temporal_fixed_hr_bench"))

from sweep import DTYPES, KS                      # noqa: E402
from sweep_nsys import measure_cell, _load_done   # noqa: E402
from ops_ext import build_call_ext                # noqa: E402

import bundle_data_rr                              # noqa: E402

OPS6 = ["gvr_cutedsl", "radix_cutedsl", "sglang_streaming",
        "sglang_v2", "flashinfer_topk", "flashinfer_topk_i32"]

N_SEQ_MAIN = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
N_HUGE = [524288, 1048576]
N_SEQ_EXT = N_SEQ_MAIN + N_HUGE
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BS_HUGE = [2, 4, 8, 16, 32, 64]

SUBDIR = {"seqlen": "seqlen_sweep", "bs": "bs_scaling", "bs_hugeN": "bs_hugeN"}


def ops_for(dtype_name, K):
    if dtype_name != "fp32":
        raise ValueError("op28 is fp32-only")
    ops = list(OPS6)
    if K > 1024:
        ops.remove("sglang_streaming")   # top512 family not faithful at K=2048
    return ops


def run_batch(sweep, scenario, cells, out_path, reps_cold, reps_warm):
    done = _load_done(out_path)
    f = open(out_path, "a")
    total = len(cells)
    prof.start()
    try:
        for i, (K, dt_name, N, BS) in enumerate(cells):
            dtype = DTYPES[dt_name]
            b = bundle_data_rr.get_bundle(scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            for op in ops_for(dt_name, K):
                key = (sweep, op, K, dt_name, N, BS)
                if key in done:
                    continue
                base = f"{op}|{K}|{dt_name}|{N}|{BS}"
                rec = {"sweep": sweep, "op": op, "K": K, "dtype": dt_name,
                       "N": N, "BS": BS, "cr": cr,
                       "scenario": scenario, "data_src": b["cfg"],
                       "hit_rate": b["kernel_hit_rate"],
                       "layer": b["row_meta"].get("layer"),
                       "seed": b["seed"],
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "reps_cold": reps_cold, "reps_warm": reps_warm}
                try:
                    call, keep, extra = build_call_ext(op, K, dtype, N, BS, cr,
                                                       logits_row, preidx_row)
                    rec.update(extra)
                    measure_cell(call, base, reps_cold, reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            if (i + 1) % 2 == 0 or i + 1 == total:
                print(f"[{scenario}/{sweep} K={cells[0][0]} {cells[0][1]}] "
                      f"{i+1}/{total} (N={N} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()


def cells_for(sweep, K, dt):
    if sweep == "seqlen":
        return [(K, dt, N, 1) for N in N_SEQ_EXT if N > 2 * K]
    if sweep == "bs":
        return [(K, dt, N, BS) for N in N_SEQ_MAIN if N > 2 * K
                for BS in BS_GRID]
    if sweep == "bs_hugeN":
        return [(K, dt, N, BS) for N in N_HUGE for BS in BS_HUGE]
    raise ValueError(sweep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=list(SUBDIR), required=True)
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--K", type=int, required=True, choices=KS)
    ap.add_argument("--dtype", default="fp32", choices=["fp32"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20, help="cold-L2 reps")
    ap.add_argument("--reps-warm", type=int, default=50, help="warm-L2 reps")
    args = ap.parse_args()

    K, dt = args.K, args.dtype
    sub = SUBDIR[args.sweep]
    results = Path(args.out_root)
    (results / sub).mkdir(parents=True, exist_ok=True)
    out_path = results / sub / f"results_K{K}_{dt}.jsonl"
    cells = cells_for(args.sweep, K, dt)
    print(f"# op28 nsys batch: scenario={args.scenario} sweep={args.sweep} "
          f"K={K} dt={dt} cells={len(cells)} ops={ops_for(dt, K)} "
          f"reps_cold={args.reps} reps_warm={args.reps_warm}", flush=True)
    run_batch(args.sweep, args.scenario, cells, out_path,
              args.reps, args.reps_warm)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
