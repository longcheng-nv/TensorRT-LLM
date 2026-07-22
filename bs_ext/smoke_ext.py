# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exactness smoke for the compB BS>1 extension: all cells x BS x 3 arms,
no nsys, plus CUDA-event rough timing (diagnostic only, NOT the verdict)."""
import time

import torch

import bs_ext as X

torch.cuda.init()


def evt_ms(call, reps=30):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    seq = X.load_compb_seq()
    ext = X.load_compb_ext()
    fails = 0
    for model, isl, L, kind, bs_grid in X.CELLS:
        b = X.real_bundle(model, isl, L)
        N, K = b["N"], b["K"]
        info = [int(x) for x in ext.ext_info(N, K, max(bs_grid))]
        print(f"== {model}_{isl}_L{L} kind={kind} N={N} K={K} "
              f"ext_info(maxBS)={{path:{info[0]},team:{info[1]},cap:{info[2]},"
              f"rpw:{info[3]},waves:{info[4]}}}", flush=True)
        for BS in bs_grid:
            row = []
            for op, fn in (("seq", lambda: X.compb_call(b, BS, seq)),
                           ("ext", lambda: X.compb_ext_call(b, BS, ext))):
                call, keep, extra, out = fn()
                ok = X.exact_rows(out, b, BS)
                ms = evt_ms(call)
                row.append(f"{op}: exact={ok} {ms*1e3:8.1f}us"
                           + (f" waves={extra.get('waves')}" if op == "ext" else ""))
                if not ok:
                    fails += 1
                del call, keep, out
            print(f"   BS={BS:5d}  " + "   ".join(row), flush=True)
        torch.cuda.empty_cache()
    print("SMOKE", "FAIL" if fails else "PASS", f"failures={fails}")


if __name__ == "__main__":
    main()
