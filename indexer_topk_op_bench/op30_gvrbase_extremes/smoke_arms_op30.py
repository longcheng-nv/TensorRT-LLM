# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op30 pre-flight — build + exactness smoke of all 10 arms on THIS node,
using existing op22rr bundles (bundles_op30 not needed). Catches missing JIT
caches / imports / dispatch regressions before the campaign starts.

Usage: CUDA_VISIBLE_DEVICES=3 python3 smoke_arms_op30.py
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))
sys.path.insert(0, str(HERE.parents[0] / "op28_ext_topk"))
sys.path.insert(0, str(HERE.parents[0] / "op22_temporal_fixed_hr_bench"))

from ops_ext import build_call_ext                # noqa: E402
import bundle_data_rr                              # noqa: E402

from sweep_op30 import ARMS, KNOBS, FP32_ONLY, _exact_idx  # noqa: E402

CASES = [  # (K, dtype, N, BS)
    (512, "fp32", 65536, 1), (512, "bf16", 65536, 1),
    (1024, "fp32", 131072, 1), (1024, "fp16", 16384, 1),
    (2048, "fp32", 262144, 1), (2048, "bf16", 65536, 1),
]
DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def pin(env):
    for var in KNOBS:
        val = env.get(var)
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def main():
    fails = 0
    for K, dt_name, N, BS in CASES:
        b = bundle_data_rr.get_bundle("best", K, DT[dt_name], N)
        logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
        for arm, op, env in ARMS:
            if dt_name != "fp32" and arm in FP32_ONLY:
                continue
            try:
                pin(env)
                call, keep, extra = build_call_ext(op, K, DT[dt_name], N, BS,
                                                   cr, logits_row, preidx_row)
                call()
                torch.cuda.synchronize()
                ref = torch.topk(logits_row[0, :N].float(),
                                 K).values.sort().values
                got = logits_row[0, :N].float()[
                    _exact_idx(arm, keep).long()].sort().values
                ok = torch.equal(got, ref)
                print(f"{arm:22s} K={K:4d} {dt_name} N={N:7d}: "
                      f"{'OK' if ok else '**EXACT FAIL**'} {extra}",
                      flush=True)
                if not ok:
                    fails += 1
                del call, keep
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{arm:22s} K={K:4d} {dt_name} N={N:7d}: "
                      f"**ERROR** {type(e).__name__}: {str(e)[:150]}",
                      flush=True)
                fails += 1
    print(f"SMOKE {'PASS' if fails == 0 else f'FAIL ({fails})'}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
