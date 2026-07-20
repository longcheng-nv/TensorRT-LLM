#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""PR-head (@0d6fc4f1f2) vs rivals, same process: arms = gvr_prhead
(gvrpkgprod2 launch contract) + op26_r0auto + sglang_v2 + flashinfer_topk
(ops_rival builders). BS=1 fp32, bench layer per model, all ISL rungs,
NVTX c|/w| protocol, one nsys rep per model. sglang canonical time =
projected span (parse side)."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
_REPORT = _HERE.parent
_BENCH = _REPORT.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPORT / "rival_harness"))
sys.path.insert(0, str(_BENCH / "harness"))

from sweep_nsys import measure_cell                                  # noqa: E402
from ops_rival import build_call_rival                               # noqa: E402
from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as PrK   # noqa: E402
import real_data_v4cap as RV4                                        # noqa: E402
import real_data_v32 as RV32                                         # noqa: E402
RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)

DEV = "cuda"
BENCH_L = {"flash": 22, "pro": 30, "v32": 34}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
EXT = ["op26_r0auto", "sglang_v2", "flashinfer_topk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    a = ap.parse_args()
    out = Path(a.out_root)
    out.mkdir(parents=True, exist_ok=True)
    RD = RV32 if a.model == "v32" else RV4
    f = open(out / f"pr4_{a.model}.jsonl", "w")
    prof.start()
    try:
        for isl in REAL_ISLS[a.model]:
            L = BENCH_L[a.model]
            bd = RD.get_bundle(a.model, isl, L, "fp32")
            K, N, cr = bd["K"], bd["N"], bd["cr"]
            lg = bd["logits"].contiguous()
            pre = bd["preIdx"].contiguous()
            sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
            outb = torch.empty(1, K, dtype=torch.int32, device=DEV)
            for rnd in range(3):
                for arm in ["gvr_prhead"] + EXT:
                    base = f"{arm}|{a.model}|{isl}|{N}|r{rnd}"
                    rec = dict(model=a.model, isl=isl, L=L, N=N, K=K, arm=arm,
                               rnd=rnd, hit=bd["hit_rate"],
                               range_cold=f"c|{base}", range_warm=f"w|{base}")
                    try:
                        if arm == "gvr_prhead":
                            call = (lambda lg=lg, pre=pre, sl=sl, outb=outb:
                                    PrK.launch(lg, pre, sl, outb, K,
                                               compress_ratio=cr))
                            call()
                            torch.cuda.synchronize()
                            keep = None
                        else:
                            call, keep, extra, getter = build_call_rival(
                                arm, K, torch.float32, N, 1, cr, lg, pre)
                            rec.update(extra)
                        measure_cell(call, base, a.reps, a.reps_warm)
                        del call
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            print(f"[{a.model}] {isl} done", flush=True)
    finally:
        prof.stop()
    f.close()


if __name__ == "__main__":
    main()
