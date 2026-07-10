# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op27 iter3 — 5-arm same-node A/B: fix candidates for the two confirmed
op27 regressions vs op25 at K2048.

arms (env per call; K2048 only):
  notail  = op25 bit-path         (TAIL=0)
  tail    = op27 shipped          (TAIL=1)
  tail_r2 = + R_rounds=2/bAcc4096 (ms path only; msc asserts R==1)
  mid     = midcol ladder (0.75, 0.5, 0.048)
  mid_r2  = midcol + R2

cells = confirmed pockets (worst bf16 65K highBS = ms; real 16-bit 65K BS1
= msc) + zero-regression guards.
"""
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

CELLS = [  # (scenario, dtype, N, BS)
    ("worst", "bf16", 65536, 64),
    ("worst", "bf16", 65536, 256),
    ("worst", "bf16", 65536, 1024),
    ("worst", "bf16", 262144, 1024),
    ("real", "bf16", 65536, 1),
    ("real", "fp16", 65536, 1),
    ("real", "fp32", 131072, 1),
    ("best", "bf16", 262144, 1),
    ("worst", "fp32", 32768, 8),
    ("real", "fp32", 16384, 1),
    ("worst", "bf16", 16384, 1),
]
MID = "0.75,0.5,0.048"
ARMS = (
    ("notail", {"OP27_K2048_TAIL": "0", "OP27_R2": "0", "OP25_QFRACS": None}),
    ("tail", {"OP27_K2048_TAIL": "1", "OP27_R2": "0", "OP25_QFRACS": None}),
    ("tail_r2", {"OP27_K2048_TAIL": "1", "OP27_R2": "1",
                 "OP25_QFRACS": None}),
    ("mid", {"OP27_K2048_TAIL": "1", "OP27_R2": "0", "OP25_QFRACS": MID}),
    ("mid_r2", {"OP27_K2048_TAIL": "1", "OP27_R2": "1", "OP25_QFRACS": MID}),
)


def pin(env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        HERE / "results/nsys/ab_iter3/iter3.jsonl")
    reps, warm = 30, 10
    f = open(out, "a")
    prof.start()
    try:
        for (scen, dt_name, N, BS) in CELLS:
            dtype = DTYPES[dt_name]
            b = bundle_data_rr.get_bundle(scen, 2048, dtype, N)
            lg, pi, cr_ = b["logits"], b["preIdx"], b["cr"]
            tag = f"{scen}|2048|{dt_name}|{N}|{BS}"
            rec = {"scenario": scen, "K": 2048, "dtype": dt_name,
                   "N": N, "BS": BS, "reps": reps}
            try:
                arms = {}
                for arm, env in ARMS:
                    pin(env)
                    call, keep, extra = build_call("gvr_ms_auto", 2048,
                                                   dtype, N, BS, cr_, lg, pi)

                    def wrapped(_c=call, _e=dict(env)):
                        pin(_e)
                        _c()
                    arms[arm] = wrapped
                    ref = torch.topk(lg[0, :N].float(),
                                     2048).values.sort().values
                    wrapped(); torch.cuda.synchronize()
                    got = lg[0, :N].float()[
                        keep[3][0].clamp(min=0).long()].sort().values
                    rec[f"exact_{arm}"] = "ok" if torch.equal(got, ref) \
                        else "FAIL"
                for arm, call in arms.items():
                    for _ in range(warm):
                        call()
                torch.cuda.synchronize()
                for _ in range(reps):
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
