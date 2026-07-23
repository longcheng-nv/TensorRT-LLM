# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 iter13 A/B: cp.async double-buffer collect (ARM39_TA=1) vs iter12
baseline (ARM39_TA=0), paired interleaved event-axis on the big-N band.
Exactness checked per (cell, BS, mode). chunks laddered per mode."""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch, exact_rows, timeit  # noqa: E402
from arm2_gate import build_arm2, bufs  # noqa: E402

CELLS = ["flash:512k:22", "flash:1024k:10", "pro:512k:14", "pro:1024k:30",
         "v32:256k:34", "flash:256k:22"]
BS_LIST = [64, 256, 512, 1024]


def run_mode(arm, lg, pre, bb, chunks):
    arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"], bb["done"],
            bb["ovf"], bb["resc"], bb["out"], chunks)


def main():
    arm = build_arm2()
    print("cell,BS,ta0_us,ta1_us,x_ta,c0,c1")
    for cell in CELLS:
        model, isl, L = cell.split(":")
        b = bundle(model, isl, int(L))
        K = b["K"]
        for bs in BS_LIST:
            lg, pre = make_batch(b, bs)
            bb = bufs(bs, K)
            best = {}
            for mode in (0, 1):
                os.environ["ARM39_TA"] = str(mode)
                for chunks in sorted({1, 2, 4, max(1, 296 // bs),
                                      max(1, 592 // bs)}):
                    run_mode(arm, lg, pre, bb, chunks)
                    torch.cuda.synchronize()
                    bad = exact_rows(b, bb["out"], bs)
                    if bad:
                        print(f"INEXACT {cell} BS{bs} ta{mode} "
                              f"chunks{chunks}: {bad}")
                        sys.exit(1)
                    us = timeit(lambda: run_mode(arm, lg, pre, bb, chunks),
                                reps=9)
                    if mode not in best or us < best[mode][0]:
                        best[mode] = (us, chunks)
            # paired interleaved re-measure at each mode's best chunks
            t = {0: [], 1: []}
            for _ in range(7):
                for mode in (0, 1):
                    os.environ["ARM39_TA"] = str(mode)
                    t[mode].append(timeit(
                        lambda: run_mode(arm, lg, pre, bb, best[mode][1]),
                        reps=5))
            u0 = min(t[0])
            u1 = min(t[1])
            print(f"{model}_{isl}_L{int(L):02d},{bs},{u0:.1f},{u1:.1f},"
                  f"{u0 / u1:.3f},{best[0][1]},{best[1][1]}", flush=True)


if __name__ == "__main__":
    main()
