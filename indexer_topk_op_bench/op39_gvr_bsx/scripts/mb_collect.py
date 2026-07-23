# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 rung-2 microbench: oracle-threshold 1-pass collect structure vs
incumbent times, on the two crux cells + the worst-loss battleground cells.
CUDA-event timing (L1 screen; cold-L2 evict per rep). Correctness: value
multiset of out vs reference topk (oracle threshold -> should be exact modulo
threshold-bucket ties).

  python3 mb_collect.py [--cells pro:64k:30,...] [--bs 256,512]
"""
import argparse
import statistics
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


def build_mb():
    from torch.utils.cpp_extension import load
    kdir = HERE.parent / "src" / "mb_collect"
    (kdir / "build_pt").mkdir(exist_ok=True)
    return load(name="op39_mb_collect",
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
    mb = build_mb()
    # incumbent numbers come from the op38 report anchors
    import csv
    pr = {}
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" / "bs_real_layers.csv")):
        pr[(r["model"], r["isl"], int(r["L"]), int(r["BS"]))] = float(r["pr"])
    v3 = {}
    for r in csv.DictReader(open(BENCH / "op38_r3v11_bs" / "v3_data.csv")):
        m, isl = r["cell"].split("_")[0], r["cell"].split("_")[1]
        L = int(r["cell"].rsplit("_L", 1)[1])
        v3[(m, isl, L, int(r["BS"]))] = float(r["cand_us"])

    for cell in args.cells.split(","):
        model, isl, L = cell.split(":")
        L = int(L)
        b = bundle(model, isl, L)
        K, N, Npad = b["K"], b["N"], b["Npad"]
        lg0 = b["logits"][0, :N].float()
        kth = torch.topk(lg0, K).values[-1].item()
        ref_sorted = torch.topk(lg0, K).values.sort().values
        for bs in (int(x) for x in args.bs.split(",")):
            lg, pre = make_batch(b, bs)
            thr = torch.full((bs,), kth, dtype=torch.float32, device="cuda")
            cv = torch.empty(bs, CAP, dtype=torch.float32, device="cuda")
            ci = torch.empty(bs, CAP, dtype=torch.int32, device="cuda")
            cnt = torch.empty(bs, dtype=torch.int32, device="cuda")
            out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
            # chunks: cover machine with ~2 waves at 512 thr: 148*4 CTAs total
            chunks = max(1, (148 * 4) // bs)
            mb.run(lg, thr, cv, ci, cnt, out, K, chunks, 1)
            torch.cuda.synchronize()
            # correctness (multiset, row 0 and last row)
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
            nc = int(cnt.max())
            for _ in range(5):
                mb.run(lg, thr, cv, ci, cnt, out, K, chunks, 1)
            torch.cuda.synchronize()
            us = timeit(lambda: mb.run(lg, thr, cv, ci, cnt, out, K, chunks, 1), reps=15)
            us1 = timeit(lambda: mb.run(lg, thr, cv, ci, cnt, out, K, chunks, 0), reps=15)
            p = pr.get((model, isl, L, bs))
            c3 = v3.get((model, isl, L, bs))
            print(f"{model}_{isl}_L{L:02d} N={N:7d} BS{bs:5d} mb={us:8.2f}us k1={us1:7.2f} "
                  f"pr={p or 0:8.2f} v3={c3 or 0:8.2f} x_vs_pr={(p or 0) / us:6.3f} "
                  f"cand_max={nc:5d} exact={'OK' if ok else 'FAIL'} chunks={chunks}",
                  flush=True)
            del lg, pre, cv, ci, cnt, out
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
