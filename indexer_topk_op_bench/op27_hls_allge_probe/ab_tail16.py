# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op27 verify — same-node paired A/B of the K2048 tail ladder on the
16-bit REAL BS=1 large-N cells where the cross-batch CSV showed a dip.

arms: notail = gvr_ms_auto OP27_K2048_TAIL=0 (op25-equivalent)
      tail   = gvr_ms_auto OP27_K2048_TAIL=1 (op27)
Protocol identical to ab_decomp.py (NVTX c|arm|cell, evict, interleaved).

Usage: nsys wrapper, --scenario real --dtype bf16|fp16
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[0]
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))

from sweep import DTYPES  # noqa: E402
from sweep_nsys import build_call, _EVICT  # noqa: E402
import bundle_data_rr  # noqa: E402

CELLS = [(2048, N, 1) for N in (65536, 131072, 262144, 524288, 1048576)]
# --cells "N:BS,N:BS" overrides (K fixed 2048)
ARMS = (("notail", {"OP27_K2048_TAIL": "0"}),
        ("tail", {"OP27_K2048_TAIL": "1"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="real")
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--cells", default=None)
    args = ap.parse_args()
    global CELLS
    if args.cells:
        CELLS = [(2048, int(nb.split(":")[0]), int(nb.split(":")[1]))
                 for nb in args.cells.split(",")]
    dtype = DTYPES[args.dtype]

    f = open(args.out, "a")
    prof.start()
    try:
        for (K, N, BS) in CELLS:
            b = bundle_data_rr.get_bundle(args.scenario, K, dtype, N)
            lg, pi, cr_ = b["logits"], b["preIdx"], b["cr"]
            tag = f"{args.scenario}|{K}|{args.dtype}|{N}|{BS}"
            rec = {"scenario": args.scenario, "K": K, "dtype": args.dtype,
                   "N": N, "BS": BS, "reps": args.reps}
            try:
                arms = {}
                for arm, env in ARMS:
                    for k, v in env.items():
                        os.environ[k] = v
                    call, keep, extra = build_call("gvr_ms_auto", K, dtype,
                                                   N, BS, cr_, lg, pi)

                    def wrapped(_c=call, _e=dict(env)):
                        for k, v in _e.items():
                            os.environ[k] = v
                        _c()
                    arms[arm] = wrapped
                    ref = torch.topk(lg[0, :N].float(), K).values.sort().values
                    wrapped(); torch.cuda.synchronize()
                    got = lg[0, :N].float()[
                        keep[3][0].clamp(min=0).long()].sort().values
                    rec[f"exact_{arm}"] = "ok" if torch.equal(got, ref) \
                        else "FAIL"
                for arm, call in arms.items():
                    for _ in range(args.warmup):
                        call()
                torch.cuda.synchronize()
                for _ in range(args.reps):
                    for arm, call in arms.items():
                        _EVICT.uniform_(0, 1)
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push(f"c|{arm}|{tag}")
                        call()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                del arms
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"done {tag}", flush=True)
    finally:
        prof.stop()
    f.close()
    print("AB BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
