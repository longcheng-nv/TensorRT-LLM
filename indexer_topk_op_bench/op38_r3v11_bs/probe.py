# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op38 quick probe: batched r3_v11 (kernel_bs) vs local PR head, §7b batch
construction (same row expanded to BS materialized copies). CUDA-event coarse
screen (256MB L2 zero before each rep, median of 15); exactness every row.

  python3 probe.py [--cells ...] [--bs ...] [--kdir kernel_bs]
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
KF = BENCH / "op26_r0_upstream_port_report" / "kf_campaign"
sys.path.insert(0, str(KF / "gvrpkg_04a0"))
sys.path.insert(0, str(BENCH / "harness"))

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

DEV = "cuda"

# §7b per-layer envelope: (model, isl, layer). N from loader.
PROBE_CELLS = [
    ("flash", "4k", 22), ("flash", "32k", 22), ("flash", "128k", 22),
    ("flash", "1024k", 22), ("pro", "128k", 30), ("pro", "1024k", 30),
    ("v32", "16k", 34), ("v32", "256k", 34),
]


def build(kdir):
    from torch.utils.cpp_extension import load
    kdir = HERE / kdir
    (kdir / "build_pt").mkdir(exist_ok=True)
    return load(name=f"op38_{kdir.name}",
                sources=[str(kdir / "kernel.cu"), str(kdir / "main.cpp")],
                build_directory=str(kdir / "build_pt"),
                extra_cuda_cflags=["-O3", "-gencode",
                                   "arch=compute_100a,code=sm_100a"],
                verbose=False)


def bundle(model, isl, L):
    mod = v32 if model == "v32" else v4
    return mod.get_bundle(model, isl, L, "fp32")


def make_batch(b, bs):
    lg = b["logits"].expand(bs, -1).contiguous()
    pre = b["preIdx"].expand(bs, -1).contiguous()
    return lg, pre


def exact_rows(b, out, bs):
    lg = b["logits"][0, :b["N"]].float()
    ref = lg[b["ref"].to(torch.int64)].sort().values
    K, N = b["K"], b["N"]
    idx = out.to(torch.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= N:
        return "range"
    sel = lg[idx.reshape(-1)].reshape(bs, K).sort(dim=1).values
    if not torch.equal(sel, ref.unsqueeze(0).expand(bs, -1)):
        return "vdiff"
    for i in range(bs):          # dup check per row
        if torch.unique(idx[i]).numel() != K:
            return f"dup@row{i}"
    return ""


def timeit(call, reps=15):
    l2 = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEV)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        l2.zero_()
        torch.cuda.synchronize()
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kdir", default="kernel_bs")
    ap.add_argument("--bs", default="1,2,8,64,256,1024")
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    bss = [int(x) for x in args.bs.split(",")]
    mod = build(args.kdir)
    print(f"[probe] {args.kdir} built", flush=True)

    for model, isl, L in PROBE_CELLS:
        b = bundle(model, isl, L)
        K, N, cr = b["K"], b["N"], b["cr"]
        for bs in bss:
            lg, pre = make_batch(b, bs)
            # candidate
            out_c = torch.empty(bs, K, dtype=torch.int32, device=DEV)
            mod.run(lg, pre, N, out_c)
            torch.cuda.synchronize()
            bad_c = exact_rows(b, out_c, bs)
            # pr head (local, paired)
            sl = torch.full((bs,), N * cr, dtype=torch.int32, device=DEV)
            out_p = torch.empty(bs, K, dtype=torch.int32, device=DEV)
            GvrTopKKernel.launch(lg, pre, sl, out_p, K, compress_ratio=cr)
            torch.cuda.synchronize()
            bad_p = exact_rows(b, out_p, bs)
            for _ in range(9):
                mod.run(lg, pre, N, out_c)
                GvrTopKKernel.launch(lg, pre, sl, out_p, K, compress_ratio=cr)
            torch.cuda.synchronize()
            t_c = timeit(lambda: mod.run(lg, pre, N, out_c), args.reps)
            t_p = timeit(lambda: GvrTopKKernel.launch(
                lg, pre, sl, out_p, K, compress_ratio=cr), args.reps)
            print(f"{model}_{isl}_L{L:02d} N={N:6d} K={K:4d} BS{bs:5d} "
                  f"cand={t_c:9.2f}us pr={t_p:9.2f}us x{t_p / t_c:6.3f} "
                  f"exact(c/p)={'OK' if not bad_c else bad_c}/"
                  f"{'OK' if not bad_p else bad_p}", flush=True)
            del lg, pre, out_c, out_p
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
