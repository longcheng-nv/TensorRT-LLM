# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op27 iter2 — silicon exactness smoke for the K2048 tail ladder.

gvr_ms_auto with OP27_K2048_TAIL=1 (default) on op22rr K2048 bundles,
sorted-value-set criterion vs torch.topk. Full authority remains the
81-cell arm sweep; this is the fast pre-gate.

Usage: python3 gate_exact_k2048.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPB = HERE.parent
sys.path.insert(0, str(OPB / "harness"))
sys.path.insert(0, str(OPB / "op22_temporal_fixed_hr_bench"))

import torch  # noqa: E402

from sweep import DTYPES  # noqa: E402
from sweep_nsys import build_call  # noqa: E402
import bundle_data_rr  # noqa: E402

os.environ.pop("OP25_QFRACS", None)
os.environ["OP27_K2048_TAIL"] = "1"

fails = ok = 0
for scen in ("worst", "real", "best"):
    for dt_name in ("fp32", "bf16", "fp16"):
        for N in (16384, 131072, 1048576):
            dtype = DTYPES[dt_name]
            b = bundle_data_rr.get_bundle(scen, 2048, dtype, N)
            lg, pi, cr_ = b["logits"], b["preIdx"], b["cr"]
            call, keep, extra = build_call("gvr_ms_auto", 2048, dtype, N, 1,
                                           cr_, lg, pi)
            call()
            torch.cuda.synchronize()
            ref = torch.topk(lg[0, :N].float(), 2048).values.sort().values
            got = lg[0, :N].float()[
                keep[3][0].clamp(min=0).long()].sort().values
            good = torch.equal(got, ref)
            ok += good
            fails += not good
            print(f"{scen:5} {dt_name} N{N:>8} "
                  f"{extra.get('ms_path', '?'):8} "
                  f"{'ok' if good else '**FAIL**'}", flush=True)
            del call, keep
            torch.cuda.empty_cache()
print(f"\nexact {ok}/{ok + fails}" + (" ALL OK" if not fails else " FAIL"))
sys.exit(1 if fails else 0)
