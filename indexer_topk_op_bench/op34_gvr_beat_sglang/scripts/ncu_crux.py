# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decisive CRUX: NCU-attribute op26_r0auto vs sglang_v2 at the anchor cell
(pro/256k large-N, best regime). Answers WHY sglang wins: single-SM cold-HBM
bandwidth (grid=(1,1,1)) vs 8-CTA cluster, or pass count, or phase overhead.

Runs each kernel exactly ONCE (after warmup) under ncu control. Invoke via:
  ncu --target-processes all --launch-skip <W> --launch-count <C> \
      --metrics <list> --csv python3 ncu_crux.py --arm <op26_r0auto|sglang_v2>
We instead run BOTH in one process and let ncu profile all launches; parse by
kernel name. Simpler: two separate ncu invocations (see drive_ncu_crux.sh).
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OPBENCH = HERE.parents[1]
sys.path.insert(0, str(OPBENCH / "harness"))
sys.path.insert(0, str(OPBENCH / "op26_gvr_logfalsi_rs" / "src"))

import real_data_v4cap as RD4
from sglang_v2_op import topk_v2, plan as sglv2_plan
from gvr_op26_r0_op import gvr_r0_op26

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["op26_r0auto", "sglang_v2"])
    ap.add_argument("--model", default="pro")
    ap.add_argument("--isl", default="256k")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()

    layers = RD4.MODELS[args.model]["layers"]
    L = layers[len(layers) // 2]
    b = RD4.get_bundle(args.model, args.isl, L, "fp32")
    lg, pre, N, K, cr = b["logits"], b["preIdx"], b["N"], b["K"], b["cr"]
    out = torch.empty((1, K), dtype=torch.int32, device=DEV)

    if args.arm == "sglang_v2":
        sl = torch.full((1,), N, dtype=torch.int32, device=DEV)
        md = sglv2_plan(sl)
        torch.cuda.synchronize()
        def run():
            topk_v2(lg, sl, K, out=out, metadata=md, max_seq_len=N)
    else:
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        def run():
            gvr_r0_op26(lg, pre, sl, K, cr, out=out)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()   # ncu profiles from here if --profile-from-start off
    for _ in range(args.reps):
        run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"{args.arm} done N={N} K={K}")


if __name__ == "__main__":
    main()
