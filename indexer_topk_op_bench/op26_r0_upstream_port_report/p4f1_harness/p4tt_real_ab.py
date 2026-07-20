#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Latest-PR-head vs REPORT-baseline GVR A/B.

Arms (same process, paired, 3 interleaved rounds, NVTX c|/w| protocol):
  slow = gvrpkgprod2 p4_tail_fast=False == PR#16457 head @1128c0544f
  fast = gvrpkgprod2 p4_tail_fast=True  (tiny-tie collect+select fast path)
Cells: all ISL rungs x bench layer per model, BS=1 fp32, launch contract.
Usage (under nsys): --model M --out-root DIR
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
sys.path.insert(0, str(_REPORT / "gvrpkg_snapshot"))
sys.path.insert(0, str(_BENCH / "harness"))

from sweep_nsys import measure_cell                                  # noqa: E402
from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as K2   # noqa: E402
import real_data_v4cap as RV4                                        # noqa: E402
import real_data_v32 as RV32                                         # noqa: E402
RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)

DEV = "cuda"
BENCH_L = {"flash": 22, "pro": 30, "v32": 34}
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
    f = open(out / f"ab_{a.model}.jsonl", "w")
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
                for arm, tf in (("slow", False), ("fast", True)):
                    base = f"{arm}|{a.model}|{isl}|L{L}|{N}|r{rnd}"
                    call = (lambda lg=lg, pre=pre, sl=sl, outb=outb, tf=tf:
                            K2.launch(lg, pre, sl, outb, K, compress_ratio=cr,
                                      p4_tail_fast=tf))
                    call()
                    torch.cuda.synchronize()
                    # exactness folded (both arms must be exact)
                    idx = outb[0].long()
                    v = lg[0, :N].float()
                    ex = bool(idx.unique().numel() == K and torch.equal(
                        v[idx].sort().values, torch.topk(v, K).values.sort().values))
                    measure_cell(call, base, a.reps, a.reps_warm)
                    f.write(json.dumps(dict(model=a.model, isl=isl, L=L, N=N,
                                            arm=arm, rnd=rnd, exact=ex,
                                            range_cold=f"c|{base}",
                                            range_warm=f"w|{base}")) + "\n")
                    f.flush()
            print(f"[{a.model}] {isl} done", flush=True)
    finally:
        prof.stop()
    f.close()


if __name__ == "__main__":
    main()
