# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41: v3mt (per-K rung fractions) vs stock v3.
1. GATE: exactness on all 75 envelope cells x BS {2,16,256} (replicated) +
   hetero all-layer batches x BS {64, 256} for the 8 study groups.
2. TAX: hetero-batch paired timing stock vs v3mt (straggler recovery).
3. REG: replicated-row paired timing on the probe cells x BS ladder
   (no-regression bar on the envelope-of-record shape).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch, exact_rows, timeit, build  # noqa: E402
from bs38_nsys import all_cells  # noqa: E402

LAYERS = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)),
          "v32": list(range(3, 61))}
GROUPS = [("v32", "16k"), ("v32", "64k"), ("v32", "128k"), ("v32", "256k"),
          ("pro", "256k"), ("pro", "512k"), ("pro", "1024k"),
          ("flash", "512k")]


def build_mt():
    from torch.utils.cpp_extension import load
    kdir = HERE.parent / "src" / "v3mt"
    (kdir / "build_pt").mkdir(exist_ok=True)
    return load(name="op41_v3mt",
                sources=[str(kdir / "kernel.cu"), str(kdir / "main.cpp")],
                build_directory=str(kdir / "build_pt"),
                extra_cuda_cflags=["-O3", "-gencode",
                                   "arch=compute_100a,code=sm_100a"],
                verbose=False)


def hetero(model, isl):
    rows, npad_ref = [], None
    for L in LAYERS[model]:
        try:
            b = bundle(model, isl, L)
        except Exception:
            continue
        npad = b["logits"].shape[1]
        if npad_ref is None:
            npad_ref = npad
        if npad != npad_ref:
            continue
        rows.append((b["logits"][0], b["preIdx"][0], b["N"],
                     b["logits"][0][:b["N"]].float()))
    lg = torch.stack([r[0] for r in rows]).cuda()
    pre = torch.stack([r[1] for r in rows]).cuda()
    return lg, pre, max(r[2] for r in rows), rows


def main():
    mt = build_mt()
    v3 = build("kernel_bs")
    # ---- 1. gate: replicated cells ----
    bad = 0
    for model, isl, L in all_cells():
        b = bundle(model, isl, L)
        for bs in (2, 16, 256):
            lg, pre = make_batch(b, bs)
            out = torch.empty((bs, b["K"]), dtype=torch.int32, device="cuda")
            mt.run(lg, pre, b["N"], out)
            torch.cuda.synchronize()
            e = exact_rows(b, out, bs)
            if e:
                print(f"GATE FAIL {model}_{isl}_L{L} BS{bs}: {e}")
                bad += 1
    # hetero gate: per-row exactness via per-row topk reference
    for model, isl in GROUPS:
        lg, pre, n_valid, rows = hetero(model, isl)
        n = lg.shape[0]
        for bs in (64, 256):
            idx = torch.arange(bs) % n
            lgb = lg[idx].contiguous()
            preb = pre[idx].contiguous()
            K = preb.shape[1]
            out = torch.empty((bs, K), dtype=torch.int32, device="cuda")
            mt.run(lgb, preb, n_valid, out)
            torch.cuda.synchronize()
            for i in range(bs):
                r = rows[int(idx[i])]
                ref = r[3].cuda().topk(K).values.sort().values
                sel = r[3].cuda()[out[i].to(torch.int64)].sort().values
                if not torch.equal(ref, sel):
                    print(f"GATE FAIL hetero {model}_{isl} BS{bs} row{i}")
                    bad += 1
                    break
    print(f"[gate] fails: {bad}", flush=True)
    if bad:
        sys.exit(1)
    # ---- 2. hetero tax A/B ----
    print("\nTAX: model,isl,BS,stock_us,mt_us,x_mt")
    for model, isl in GROUPS:
        lg, pre, n_valid, rows = hetero(model, isl)
        n = lg.shape[0]
        for bs in (16, 64, 256, 1024):
            idx = torch.arange(bs) % n
            lgb = lg[idx].contiguous()
            preb = pre[idx].contiguous()
            K = preb.shape[1]
            out = torch.empty((bs, K), dtype=torch.int32, device="cuda")
            t = {}
            for tag, mod in (("s", v3), ("m", mt)):
                mod.run(lgb, preb, n_valid, out)
                torch.cuda.synchronize()
                best = None
                for _ in range(5):
                    us = timeit(lambda: mod.run(lgb, preb, n_valid, out),
                                reps=7)
                    best = us if best is None or us < best else best
                t[tag] = best
            print(f"{model},{isl},{bs},{t['s']:.1f},{t['m']:.1f},"
                  f"{t['s'] / t['m']:.3f}", flush=True)
    # ---- 3. replicated no-regression ----
    print("\nREG: cell,BS,stock_us,mt_us,x_mt")
    for model, isl, L in [("flash", "512k", 22), ("flash", "1024k", 10),
                          ("pro", "512k", 14), ("pro", "1024k", 30),
                          ("v32", "128k", 14), ("v32", "256k", 34),
                          ("v32", "64k", 54), ("pro", "64k", 30),
                          ("flash", "64k", 22), ("v32", "16k", 14)]:
        b = bundle(model, isl, L)
        for bs in (16, 64, 256, 1024):
            lg, pre = make_batch(b, bs)
            out = torch.empty((bs, b["K"]), dtype=torch.int32, device="cuda")
            t = {}
            for tag, mod in (("s", v3), ("m", mt)):
                mod.run(lg, pre, b["N"], out)
                torch.cuda.synchronize()
                best = None
                for _ in range(5):
                    us = timeit(lambda: mod.run(lg, pre, b["N"], out), reps=7)
                    best = us if best is None or us < best else best
                t[tag] = best
            print(f"{model}_{isl}_L{L:02d},{bs},{t['s']:.1f},{t['m']:.1f},"
                  f"{t['s'] / t['m']:.3f}", flush=True)


if __name__ == "__main__":
    main()
