# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op42 iter9: single-config ncu probe. Issues ONE arm call inside a
cudaProfiler window (run under `ncu --profile-from-start off`).
Usage: ncu ... python3 scripts/ncu_probe.py --cell flash_128k_L22 --bs 32 --arm bsx
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab import Repl, build_bsx, bsx_call, head_call  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--arm", choices=["bsx", "gvr_pr"], default="bsx")
    ap.add_argument("--src", default=None)
    args = ap.parse_args()

    mod = build_bsx(args.src) if args.arm == "bsx" else None
    stack = Repl(args.cell, args.bs)
    call, out = (bsx_call(stack, mod, args.bs) if args.arm == "bsx"
                 else head_call(stack, args.bs))
    call()  # JIT/warm outside window
    torch.cuda.synchronize()
    bad = stack.exact(out, args.bs)
    print(f"[ncu_probe] {args.cell} BS{args.bs} {args.arm} "
          f"K={stack.K} N={stack.N} Npad={stack.Npad} exact={not bad}",
          flush=True)

    import torch.cuda.profiler as prof
    prof.start()
    call()
    torch.cuda.synchronize()
    prof.stop()


if __name__ == "__main__":
    main()
