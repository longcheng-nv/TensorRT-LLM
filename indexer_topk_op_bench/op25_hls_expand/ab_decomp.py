# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 — decompose the ship regression: ladder (wide4b M=5) vs slot_scale=2.

4 arms via env knobs, paired cold-L2 (ab_qfracs protocol):
    base   OP25_QFRACS=base OP25_SLOTCAP=1
    ladder OP25_QFRACS=<default table> OP25_SLOTCAP=1
    slots  OP25_QFRACS=base OP25_SLOTCAP=2
    ship   defaults (table + 2)
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

CELLS = [(512, 8192, 1), (512, 16384, 1), (512, 65536, 1), (512, 262144, 1),
         (1024, 8192, 1), (1024, 16384, 1), (1024, 65536, 1),
         (1024, 262144, 1), (1024, 131072, 16),
         (2048, 65536, 1), (2048, 262144, 1)]

ARMS = (
    ("base", {"OP25_QFRACS": "base", "OP25_SLOTCAP": "1"}),
    ("ladder", {"OP25_QFRACS": None, "OP25_SLOTCAP": "1"}),
    ("slots", {"OP25_QFRACS": "base", "OP25_SLOTCAP": "2"}),
    ("ship", {"OP25_QFRACS": None, "OP25_SLOTCAP": None}),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="best")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=25)
    args = ap.parse_args()
    dtype = DTYPES["fp32"]
    f = open(args.out, "a")
    prof.start()
    try:
        for (K, N, BS) in CELLS:
            b = bundle_data_rr.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            tag = f"{args.scenario}|{K}|fp32|{N}|{BS}"
            rec = {"scenario": args.scenario, "K": K, "N": N, "BS": BS,
                   "reps": args.reps}
            try:
                arms = {}
                for arm, env in ARMS:
                    call, keep, _ = build_call("gvr_ms_auto", K, dtype, N,
                                               BS, cr, logits_row,
                                               preidx_row)

                    def wrapped(_c=call, _e=dict(env)):
                        for k, v in _e.items():
                            if v is None:
                                os.environ.pop(k, None)
                            else:
                                os.environ[k] = v
                        _c()
                    arms[arm] = wrapped
                for arm, call in arms.items():
                    for _ in range(8):
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
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
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
