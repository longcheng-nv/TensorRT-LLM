#!/usr/bin/env python3
# op26 iter7 预研 — single-cell ncu/nsys driver for the core low-BS negative
# band (65K-262K, BS 1-8; fin negative map, analyze_fin_negatives.py).
# Reuses sweep_op22rr.build_call so the arm call convention (BS expand,
# radix_aux prealloc, cr/seq_lens) is byte-identical to the nsys campaign.
#
# Usage (ncu wraps this; --profile-from-start off):
#   ncu --profile-from-start off --set full -o out.ncu-rep \
#     python3 prof_lowbs_cell.py --arm op26_r0mc --scenario real \
#       --K 1024 --dtype fp32 --N 131072 --BS 1 --reps 5
import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, os.path.join(_here, "..", "op22_temporal_fixed_hr_bench"))

import torch  # noqa: E402

import bundle_data_rr  # noqa: E402
from sweep_op22rr import DTYPES, build_call  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="op26_r0mc")
    ap.add_argument("--scenario", default="real",
                    choices=["real", "best", "worst"])
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--N", type=int, default=131072)
    ap.add_argument("--BS", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    b = bundle_data_rr.get_bundle(args.scenario, args.K, dtype, args.N)
    call, keep, extra = build_call(args.arm, args.K, dtype, args.N, args.BS,
                                   b["cr"], b["logits"], b["preIdx"])
    print(f"cell {args.arm}|{args.K}|{args.dtype}|{args.N}|BS{args.BS} "
          f"cr={b['cr']} extra={extra}", flush=True)

    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.reps):
        call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("profiled reps done", flush=True)


if __name__ == "__main__":
    main()
