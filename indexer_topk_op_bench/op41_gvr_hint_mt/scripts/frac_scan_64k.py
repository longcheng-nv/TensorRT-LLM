# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41: focused K2048 AR4 frac scan on the v32_64k hetero regression +
guard groups v32_16k / v32_256k (must keep their wins)."""
import sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))
from probe import timeit, build  # noqa: E402
from ab_v3mt import hetero  # noqa: E402

CANDS = [(55, 88), (60, 90), (50, 85), (25, 65)]


def build_variant(lo, hi):
    from torch.utils.cpp_extension import load
    kdir = HERE.parent / "src" / "v3mt"
    bdir = kdir / f"build_q{lo}_{hi}"
    bdir.mkdir(exist_ok=True)
    return load(name=f"op41_v3mt_q{lo}_{hi}",
                sources=[str(kdir / "kernel.cu"), str(kdir / "main.cpp")],
                build_directory=str(bdir),
                extra_cuda_cflags=["-O3", f"-DQT2048_LO={lo}",
                                   f"-DQT2048_HI={hi}", "-gencode",
                                   "arch=compute_100a,code=sm_100a"],
                verbose=False)


def main():
    v3 = build("kernel_bs")
    mods = {(lo, hi): build_variant(lo, hi) for lo, hi in CANDS}
    print("group,BS,stock_us," +
          ",".join(f"x{lo}_{hi}" for lo, hi in CANDS))
    for model, isl in [("v32", "64k"), ("v32", "16k"), ("v32", "256k")]:
        lg, pre, n_valid, rows = hetero(model, isl)
        n = lg.shape[0]
        for bs in (64, 256, 1024):
            idx = torch.arange(bs) % n
            lgb = lg[idx].contiguous()
            preb = pre[idx].contiguous()
            K = preb.shape[1]
            out = torch.empty((bs, K), dtype=torch.int32, device="cuda")

            def t(mod):
                mod.run(lgb, preb, n_valid, out)
                torch.cuda.synchronize()
                best = None
                for _ in range(5):
                    us = timeit(lambda: mod.run(lgb, preb, n_valid, out),
                                reps=7)
                    best = us if best is None or us < best else best
                return best
            t0 = t(v3)
            xs = [t0 / t(mods[c]) for c in CANDS]
            print(f"{model}_{isl},{bs},{t0:.1f}," +
                  ",".join(f"{x:.3f}" for x in xs), flush=True)


if __name__ == "__main__":
    main()
