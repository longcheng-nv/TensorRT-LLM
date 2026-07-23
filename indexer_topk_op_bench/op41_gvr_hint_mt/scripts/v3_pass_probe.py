# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41: measure stock-v3 P2 secant pass counts per envelope cell (BS=1).
Uses the DBG_PASSES build of src/v3dbg (emit = dbg_pass*10 + fell_to_descent,
early return before P3/P4). Cells with npad <= 12288 take the direct path
(no hints) and are reported as 'direct'."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch  # noqa: E402
from bs38_nsys import all_cells  # noqa: E402


def build_dbg():
    from torch.utils.cpp_extension import load
    kdir = HERE.parent / "src" / "v3dbg"
    (kdir / "build_pt").mkdir(exist_ok=True)
    return load(name="op41_v3dbg",
                sources=[str(kdir / "kernel.cu"), str(kdir / "main.cpp")],
                build_directory=str(kdir / "build_pt"),
                extra_cuda_cflags=["-O3", "-DDBG_PASSES", "-gencode",
                                   "arch=compute_100a,code=sm_100a"],
                verbose=False)


def main():
    mod = build_dbg()
    print("cell,K,npad,passes,descent")
    hist = {}
    for model, isl, L in all_cells():
        b = bundle(model, isl, L)
        K, npad = b["K"], b["logits"].shape[1]
        lg, pre = make_batch(b, 1)
        out = torch.full((1, K), -7, dtype=torch.int32, device="cuda")
        mod.run(lg, pre, b["N"], out)
        torch.cuda.synchronize()
        v = int(out[0, 0].item())
        if npad <= 12288:
            tag, passes, desc = "direct", "", ""
        else:
            passes, desc = v // 10, v % 10
            tag = str(passes)
            hist[(passes, desc)] = hist.get((passes, desc), 0) + 1
        print(f"{model}_{isl}_L{L:02d},{K},{npad},{tag},{desc}", flush=True)
    print("\n== pass-count histogram (npad>12288 cells) ==", file=sys.stderr)
    for k in sorted(hist):
        print(f"  passes={k[0]} descent={k[1]}: {hist[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
