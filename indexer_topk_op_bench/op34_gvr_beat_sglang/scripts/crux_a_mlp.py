# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CRUX-A: does the LATENCY-BOUND scan keep speeding up past 8 CTAs/row at BS=1?

The NCU CRUX showed op26_r0 (1 CTA) and sglang (8 CTA) are both <1% DRAM/SM =
latency-bound; sglang's edge is 8x MLP. If a bare count-ge scan keeps dropping
as C grows past 8, a multi-CTA GVR could EXCEED sglang's fixed-8 MLP. This bare
Triton microbench isolates that scan-latency-vs-C curve (no phases, no reduction
tax) to set the MLP ceiling. Cold-L2 (512MB evict) each launch.

Reports us_cold(C) for C in {1..128} at N in {65539, 262144, 1048576}, plus the
sglang anchor (28.2us NCU / 21.6us wall @N=65539) for reference.
"""
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

DEV = "cuda"


@triton.jit
def _count_ge_chunk(x_ptr, N, C, thr, out_ptr, BLOCK: tl.constexpr):
    """block c scans its contiguous [c*chunk, (c+1)*chunk) slice, counts >=thr,
    atomic-adds to out[0]. grid = (C,). Pure latency-bound strided scan."""
    c = tl.program_id(0)
    chunk = (N + C - 1) // C
    start = c * chunk
    end = tl.minimum(start + chunk, N)
    acc = 0
    off = start
    while off < end:
        idx = off + tl.arange(0, BLOCK)
        m = idx < end
        v = tl.load(x_ptr + idx, mask=m, other=-3.4e38)
        acc += tl.sum((v >= thr).to(tl.int32))
        off += BLOCK
    tl.atomic_add(out_ptr, acc)


def cold_median(fn, evict, iters=40, warmup=8):
    for _ in range(warmup):
        evict(); fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        evict()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def profile_one():
    """NCU entry: one launch each for (N,C) pairs after warmup, so NCU reports
    pure-kernel gpu__time per launch (grid size = C distinguishes them)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--C", type=int, required=True)
    a = ap.parse_args()
    x = torch.randn(a.N, dtype=torch.float32, device=DEV)
    out = torch.zeros(1, dtype=torch.int32, device=DEV)
    def fn():
        out.zero_()
        _count_ge_chunk[(a.C,)](x, a.N, a.C, 0.0, out, BLOCK=1024)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    fn()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"profiled N={a.N} C={a.C}")


def main():
    if "--profile" in sys.argv:
        sys.argv.remove("--profile")
        return profile_one()
    torch.manual_seed(0)
    evbuf = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device=DEV)  # 512MB
    def evict():
        evbuf.uniform_()

    Cs = [1, 2, 4, 8, 16, 32, 64, 128]
    BLOCK = 1024
    out = torch.zeros(1, dtype=torch.int32, device=DEV)
    print(f"{'N':>9} | " + " ".join(f"C{c:<3}" for c in Cs))
    for N in (65539, 262144, 1048576):
        x = torch.randn(N, dtype=torch.float32, device=DEV)
        thr = 0.0
        row = []
        for C in Cs:
            def fn(C=C):
                out.zero_()
                _count_ge_chunk[(C,)](x, N, C, thr, out, BLOCK=BLOCK)
            fn()  # warm/compile
            t = cold_median(fn, evict)
            row.append(t)
        best = min(row); bestC = Cs[row.index(best)]
        cells = " ".join(f"{t:4.1f}" for t in row)
        print(f"{N:>9} | {cells}   best={best:.1f}us @C{bestC}")
    print("\nsglang anchor @N=65539: 28.2us(NCU pure-kernel) / 21.6us(wall) using C=8.")
    print("If bare-scan us keeps dropping past C=8, multi-CTA GVR can exceed sglang MLP.")


if __name__ == "__main__":
    main()
