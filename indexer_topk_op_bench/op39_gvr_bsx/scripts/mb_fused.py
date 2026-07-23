# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 iter2 microbench: fused single-kernel collect+reduce (oracle thr),
with a chunks ladder per (cell, BS). Event-axis L1 screen vs report pr."""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch, timeit  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

CAP = 8192


def build_fused():
    from torch.utils.cpp_extension import load
    kdir = HERE.parent / "src" / "mb_fused"
    (kdir / "build_pt").mkdir(exist_ok=True)
    return load(name="op39_mb_fused",
                sources=[str(kdir / "kernel.cu"), str(kdir / "main.cpp")],
                build_directory=str(kdir / "build_pt"),
                extra_cuda_cflags=["-O3", "-gencode",
                                   "arch=compute_100a,code=sm_100a"],
                verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="pro:64k:30,pro:1024k:30,flash:256k:22,v32:64k:34,flash:16k:22")
    ap.add_argument("--bs", default="16,64,256,1024")
    args = ap.parse_args()
    mb = build_fused()
    import csv
    pr = {}
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" / "bs_real_layers.csv")):
        pr[(r["model"], r["isl"], int(r["L"]), int(r["BS"]))] = float(r["pr"])

    for cell in args.cells.split(","):
        model, isl, L = cell.split(":")
        L = int(L)
        b = bundle(model, isl, L)
        K, N = b["K"], b["N"]
        lg0 = b["logits"][0, :N].float()
        kth = torch.topk(lg0, K).values[-1].item()
        ref_sorted = torch.topk(lg0, K).values.sort().values
        for bs in (int(x) for x in args.bs.split(",")):
            lg, pre = make_batch(b, bs)
            thr = torch.full((bs,), kth, dtype=torch.float32, device="cuda")
            cv = torch.empty(bs, CAP, dtype=torch.float32, device="cuda")
            ci = torch.empty(bs, CAP, dtype=torch.int32, device="cuda")
            cnt = torch.zeros(bs, dtype=torch.int32, device="cuda")
            done = torch.zeros(bs, dtype=torch.int32, device="cuda")
            out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
            best = None
            for chunks in sorted({1, 2, 4, max(1, 296 // bs), max(1, 592 // bs),
                                  max(1, 1184 // bs)}):
                mb.run(lg, thr, cv, ci, cnt, done, out, K, chunks)
                torch.cuda.synchronize()
                ok = True
                for r_ in (0, bs - 1):
                    idx = out[r_].to(torch.int64)
                    if int(idx.min()) < 0 or int(idx.max()) >= N:
                        ok = False
                        break
                    sel = lg0[idx].sort().values
                    if not torch.allclose(sel, ref_sorted):
                        ok = False
                        break
                if not ok:
                    print(f"  {model}_{isl} BS{bs} chunks{chunks} INEXACT", flush=True)
                    continue
                for _ in range(5):
                    mb.run(lg, thr, cv, ci, cnt, done, out, K, chunks)
                torch.cuda.synchronize()
                us = timeit(lambda: mb.run(lg, thr, cv, ci, cnt, done, out, K, chunks),
                            reps=15)
                if best is None or us < best[0]:
                    best = (us, chunks)
            p = pr.get((model, isl, L, bs))
            print(f"{model}_{isl}_L{L:02d} N={N:7d} BS{bs:5d} fused={best[0]:8.2f}us "
                  f"(chunks={best[1]}) pr={p or 0:8.2f} x_vs_pr={(p or 0) / best[0]:6.3f}",
                  flush=True)
            del lg, pre, cv, ci, cnt, done, out
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
