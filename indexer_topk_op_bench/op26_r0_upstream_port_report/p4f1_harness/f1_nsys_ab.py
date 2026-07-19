#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""F1 Gate C: nsys A/B perf gate — p4_finebin_loop OFF vs ON, same process,
paired per cell. Cells: all ISL rungs at the model's bench layer (the 99%
path where ON must be a wash) + the model's fixture layers (the rows where
the extra recursion actually fires). BS=1 fp32, launch contract, NVTX
c|/w| ranges (20 cold w/ 512MB evict + 50 warm), one nsys rep per model.

Usage: under nsys (see drive), --model M --out-root DIR
Gate: per-cell ON/OFF cold ratio; any cell slower than 1.025 blocks.
"""
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
sys.path.insert(0, str(_BENCH / "harness"))

from sweep_nsys import measure_cell                        # noqa: E402
from gvrpkgf1.top_k.gvr_topk_decode import GvrTopKKernel   # noqa: E402
import real_data_v4cap as RV4                              # noqa: E402
import real_data_v32 as RV32                               # noqa: E402
RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)

DEV = "cuda"
BENCH_L = {"flash": 22, "pro": 30, "v32": 34}
FIXTURE_CELLS = {"flash": [],
                 "pro": [("64k", 22), ("128k", 6), ("512k", 48), ("512k", 60)],
                 "v32": [("8k", 8), ("8k", 39), ("16k", 38), ("64k", 16),
                         ("128k", 25)]}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}


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
    cells = [(isl, BENCH_L[a.model]) for isl in REAL_ISLS[a.model]] \
        + FIXTURE_CELLS[a.model]
    f = open(out / f"f1ab_{a.model}.jsonl", "w")
    prof.start()
    try:
        for isl, L in cells:
            bd = RD.get_bundle(a.model, isl, L, "fp32")
            K, N, cr = bd["K"], bd["N"], bd["cr"]
            lg = bd["logits"].contiguous()
            pre = bd["preIdx"].contiguous()
            sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
            outb = torch.empty(1, K, dtype=torch.int32, device=DEV)
            # 3 in-process rounds, arms interleaved (op35 verdict protocol):
            # per-cell verdict = median over rounds of the paired ratio.
            for rnd in range(3):
                for arm, ovr in (("off", {}), ("on", {"p4_finebin_loop": True})):
                    base = f"{arm}|{a.model}|{isl}|L{L}|{N}|r{rnd}"
                    call = (lambda lg=lg, pre=pre, sl=sl, outb=outb, ovr=ovr:
                            GvrTopKKernel.launch(lg, pre, sl, outb, K,
                                                 compress_ratio=cr, **ovr))
                    call()
                    torch.cuda.synchronize()
                    measure_cell(call, base, a.reps, a.reps_warm)
                    f.write(json.dumps(dict(model=a.model, isl=isl, L=L, N=N,
                                            arm=arm, rnd=rnd,
                                            range_cold=f"c|{base}",
                                            range_warm=f"w|{base}")) + "\n")
                    f.flush()
            print(f"[{a.model}] {isl}/L{L} done", flush=True)
    finally:
        prof.stop()
    f.close()


if __name__ == "__main__":
    main()
