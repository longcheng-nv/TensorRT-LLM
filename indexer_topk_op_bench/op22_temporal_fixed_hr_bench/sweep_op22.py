# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 W3 — nsys pure-kernel sweep over the 5 campaign ops on the fixed-hit-
rate temporal-synth bundles (bundle_data.get_bundle; gen_bundles.py output).

Clone of harness/sweep_nsys.py with:
  - ops   = gvr_ms_auto (op21) | gvr_cutedsl | gvr_multicta_cutedsl |
            radix_cutedsl | sglang_streaming (fp32-only AND K<=1024)
  - data  = per-scenario disk bundles (--scenario best|worst|real)
  - N     = seqlen sweep extended with 524288 and 1048576
  - sweeps: seqlen (BS=1, N 4K..1M) | bs (BS 1..2048, N 4K..256K)
            | bs_hugeN (stretch: N {512K,1M} x BS 2..64)

Timing protocol is IDENTICAL to sweep_nsys.py (same measure_cell: 10 warmup,
warm-L2 reps in "w|" NVTX ranges, 512MB-evict cold-L2 reps in "c|" ranges,
eager + sync inside the range, whole loop inside cudaProfilerApi window).

Run UNDER nsys via drive_nsys_op22.sh; kernel us filled by parse_op22.py.
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

from sweep import DTYPES, KS                      # noqa: E402
from sweep_nsys import build_call, measure_cell, _load_done  # noqa: E402

import bundle_data                                 # noqa: E402

OPS5 = ["gvr_ms_auto", "gvr_cutedsl", "gvr_multicta_cutedsl",
        "radix_cutedsl", "sglang_streaming"]

N_SEQ_MAIN = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
N_HUGE = [524288, 1048576]
N_SEQ_EXT = N_SEQ_MAIN + N_HUGE
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BS_HUGE = [2, 4, 8, 16, 32, 64]   # BS=1 at huge N covered by the seqlen sweep

SUBDIR = {"seqlen": "seqlen_sweep", "bs": "bs_scaling", "bs_hugeN": "bs_hugeN"}


def ops_for(dtype_name, K):
    ops = [o for o in OPS5 if o != "sglang_streaming"]
    if dtype_name == "fp32" and K <= 1024:
        ops.append("sglang_streaming")   # fp32-only kernel, K<=1024
    return ops


def run_batch(sweep, scenario, cells, out_path, reps_cold, reps_warm):
    done = _load_done(out_path)
    f = open(out_path, "a")
    total = len(cells)
    prof.start()
    try:
        for i, (K, dt_name, N, BS) in enumerate(cells):
            dtype = DTYPES[dt_name]
            b = bundle_data.get_bundle(scenario, K, dtype, N)
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
                    call, keep, extra = build_call(op, K, dtype, N, BS, cr,
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
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    ap.add_argument("--out-root", required=True,
                    help="per-scenario root, e.g. ../results_b200_op22/real")
    ap.add_argument("--reps", type=int, default=20, help="cold-L2 reps")
    ap.add_argument("--reps-warm", type=int, default=50, help="warm-L2 reps")
    args = ap.parse_args()

    K, dt = args.K, args.dtype
    sub = SUBDIR[args.sweep]
    results = Path(args.out_root)
    (results / sub).mkdir(parents=True, exist_ok=True)
    out_path = results / sub / f"results_K{K}_{dt}.jsonl"
    cells = cells_for(args.sweep, K, dt)
    print(f"# op22 nsys batch: scenario={args.scenario} sweep={args.sweep} "
          f"K={K} dt={dt} cells={len(cells)} ops={ops_for(dt, K)} "
          f"reps_cold={args.reps} reps_warm={args.reps_warm}", flush=True)
    run_batch(args.sweep, args.scenario, cells, out_path,
              args.reps, args.reps_warm)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
