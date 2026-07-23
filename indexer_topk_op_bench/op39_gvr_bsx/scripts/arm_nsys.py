# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 nsys-axis screen of the PRODUCTION arm (real preIdx thresholds), same
protocol/cells as mb_nsys.py (oracle). NVTX c|arm|<cell>|BS<n>."""
import argparse
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch, exact_rows, timeit  # noqa: E402
from arm2_gate import build_arm2, bufs  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pro:64k:30,pro:1024k:30,flash:256k:22,"
                                       "v32:64k:34,flash:16k:22,pro:256k:46,"
                                       "v32:128k:54,flash:512k:34")
    ap.add_argument("--bs", default="16,32,64,128,256,512,1024")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    arm = build_arm2()
    evict = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    prof.start()
    for cell in args.cells.split(","):
        model, isl, L = cell.split(":")
        L = int(L)
        b = bundle(model, isl, L)
        K, N = b["K"], b["N"]
        for bs in (int(x) for x in args.bs.split(",")):
            lg, pre = make_batch(b, bs)
            bb = bufs(bs, K)
            best = None
            for chunks in sorted({1, 2, 4, max(1, 296 // bs), max(1, 592 // bs)}):
                arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                        bb["done"], bb["ovf"], bb["resc"], bb["out"], chunks)
                torch.cuda.synchronize()
                us = timeit(lambda: arm.run(lg, pre, bb["thr"], bb["cv"],
                                            bb["ci"], bb["cnt"], bb["done"],
                                            bb["ovf"], bb["resc"], bb["out"],
                                            chunks), reps=7)
                if best is None or us < best[0]:
                    best = (us, chunks)
            chunks = best[1]
            assert not exact_rows(b, bb["out"], bs)
            for _ in range(5):
                arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                        bb["done"], bb["ovf"], bb["resc"], bb["out"], chunks)
            torch.cuda.synchronize()
            rname = f"c|arm|{model}_{isl}_L{L:02d}|BS{bs}"
            for _ in range(args.reps):
                evict.zero_()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(rname)
                arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                        bb["done"], bb["ovf"], bb["resc"], bb["out"], chunks)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            del lg, pre, bb
        print(f"[arm_nsys] {model}_{isl}_L{L:02d} done", flush=True)
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()
    prof.stop()


if __name__ == "__main__":
    main()
