# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 arm v1 exactness gate (Phase 4): full §7b envelope, real preIdx (real
hit rates incl. low-hit pro L46), tie-aware multiset check per row; plus
host-side candidate-count stats (fallback-rate prediction: cnt>CAP or <K).

  python3 arm_gate.py [--bs 2,16,256] [--limit N]
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch, exact_rows  # noqa: E402
from bs38_nsys import all_cells  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

CAP = 8192


def build_arm():
    from torch.utils.cpp_extension import load
    kdir = HERE.parent / "src" / "arm_v1"
    (kdir / "build_pt").mkdir(exist_ok=True)
    return load(name="op39_arm_v1",
                sources=[str(kdir / "kernel.cu"), str(kdir / "main.cpp")],
                build_directory=str(kdir / "build_pt"),
                extra_cuda_cflags=["-O3", "-gencode",
                                   "arch=compute_100a,code=sm_100a"],
                verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="2,16,256")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    arm = build_arm()
    cells = all_cells()
    if args.limit:
        cells = cells[:args.limit]
    n_bad = n_tot = 0
    fb_cells = []
    for model, isl, L in cells:
        b = bundle(model, isl, L)
        K, N, Npad = b["K"], b["N"], b["Npad"]
        # host-side fallback prediction from row 0 (rows identical in batch)
        lgr = b["logits"][0].float()
        pre0 = b["preIdx"][0, :K].to(torch.int64).clamp(0, Npad - 1)
        t_lo = lgr[pre0].min()
        ccount = int((lgr >= t_lo).sum())
        fb = "OVERFLOW" if ccount > CAP else ("UNDER" if ccount < K else "")
        if fb:
            fb_cells.append((f"{model}_{isl}_L{L:02d}", ccount))
        cname = f"{model}_{isl}_L{L:02d}"
        for bs in (int(x) for x in args.bs.split(",")):
            lg, pre = make_batch(b, bs)
            thr = torch.empty(bs, dtype=torch.float32, device="cuda")
            cv = torch.empty(bs, CAP, dtype=torch.float32, device="cuda")
            ci = torch.empty(bs, CAP, dtype=torch.int32, device="cuda")
            cnt = torch.zeros(bs, dtype=torch.int32, device="cuda")
            done = torch.zeros(bs, dtype=torch.int32, device="cuda")
            out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
            chunks = max(1, 592 // bs)
            arm.run(lg, pre, thr, cv, ci, cnt, done, out, chunks)
            torch.cuda.synchronize()
            bad = exact_rows(b, out, bs)
            n_tot += 1
            if bad:
                n_bad += 1
                print(f"INEXACT {cname} BS{bs}: {bad} (cand={ccount} {fb})",
                      flush=True)
            del lg, pre, thr, cv, ci, cnt, done, out
        print(f"[gate] {cname} K={K} Npad={Npad} cand={ccount} {fb}", flush=True)
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()
    print(f"\n[gate] inexact {n_bad}/{n_tot}; fallback-predicted cells: "
          f"{len(fb_cells)}/{len(cells)}")
    for c, n in fb_cells:
        print(f"  {c}: cand={n}")


if __name__ == "__main__":
    main()
