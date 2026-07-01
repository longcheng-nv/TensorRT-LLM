# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""op16 nsys pure-kernel cold-L2 A/B — REPORT-IDENTICAL protocol.

Measures, in ONE nsys run (so no cross-session drift), the pure-kernel GPU time
of:
  - gvr_cutedsl_rs   : op#7 rank-scatter baseline (ANCHOR — must match report)
  - gvr_dt           : op16 sampled-histogram P2 init (the candidate)
  - radix_cutedsl    : the binding baseline (user target)
  - sglang_streaming : fp32-only baseline (user target)

Uses harness/sweep_nsys.measure_cell (eager + sync-inside-NVTX-range, 512MB L2
evict OUTSIDE for cold) so the numbers are directly comparable to report.html
(nsys nvtx_kern_sum / NVTX Inst; evict kernel filtered). Same synth inputs
(get_bundle beta_moderate) as the report.

Run UNDER nsys via scripts/run_nsys_ab.sh, then parse with the same
report/parse_nsys_full.parse_rep. The ANCHOR (gvr_cutedsl_rs) re-measured here
vs report's gvr_cutedsl_rs_cold_us validates protocol comparability.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))

from sweep import _build_inputs, DTYPES, DEV, get_bundle  # noqa: E402
from sweep_nsys import measure_cell  # noqa: E402
from gvr_dt_op import gvr_dt  # noqa: E402

OP16 = "gvr_dt"


def build_call(op, K, dtype, N, BS, cr, logits_row, preidx_row,
               sample_size, sample_aim_permille):
    if op == OP16:
        logits = logits_row.to(dtype).expand(BS, -1).contiguous()
        seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
        out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
        pre = preidx_row.expand(BS, -1).contiguous()
        keep = [logits, seq_div, out, pre]
        gvr_dt(logits, pre, seq_div, K, compress_ratio=cr, out=out, sampled=True,
               sample_size=sample_size, sample_aim_permille=sample_aim_permille)  # warm compile
        return (lambda: gvr_dt(logits, pre, seq_div, K, compress_ratio=cr, out=out,
                               sampled=True, sample_size=sample_size,
                               sample_aim_permille=sample_aim_permille)), keep
    return _build_inputs(op, K, dtype, N, BS, cr, logits_row, preidx_row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    ap.add_argument("--cr", type=int, default=None)
    ap.add_argument("--N", type=int, nargs="+",
                    default=[4096, 8192, 16384, 32768, 65536, 131072, 262144])
    ap.add_argument("--cfg", default="beta_moderate")
    ap.add_argument("--sample-size", type=int, default=4096)
    ap.add_argument("--aim", type=int, default=1150, help="sample_aim_permille")
    ap.add_argument("--out", required=True, help="jsonl path")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    args = ap.parse_args()

    K, dt = args.K, args.dtype
    dtype = DTYPES[dt]
    cr = args.cr if args.cr is not None else (1 if K == 2048 else 4)
    ops = ["gvr_cutedsl_rs", OP16, "radix_cutedsl"]
    if dt == "fp32":
        ops.append("sglang_streaming")

    Ns = [n for n in args.N if n > 2 * K]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w")
    print(f"# op16 nsys A/B: K={K} dt={dt} cr={cr} Ns={Ns} ops={ops} "
          f"nsample={args.sample_size} aim={args.aim}", flush=True)
    prof.start()
    try:
        for N in Ns:
            b = get_bundle(K, dtype, N, cfg=args.cfg)
            logits_row, preidx_row = b["logits"], b["preIdx"]
            for op in ops:
                base = f"{op}|{K}|{dt}|{N}|1"
                rec = {"op": op, "K": K, "dtype": dt, "N": N, "BS": 1, "cr": cr,
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "sample_size": args.sample_size, "aim": args.aim}
                try:
                    call, keep = build_call(op, K, dtype, N, 1, cr, logits_row,
                                            preidx_row, args.sample_size, args.aim)
                    measure_cell(call, base, args.reps, args.reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
                f.write(json.dumps(rec) + "\n"); f.flush()
                gc.collect(); torch.cuda.empty_cache()
            print(f"  N={N} done", flush=True)
    finally:
        prof.stop()
    f.close()
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
