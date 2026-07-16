#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Radix cuteDSL (vendored SinglePassMultiCTARadixTopK, the §8 `radix_cutedsl`
arm) put through the SAME correctness battery that falsified SGLang v2's
unconditional-exactness claim and validated FlashInfer (fi_topk_correctness.py).

Mechanism context: pure radix top-K — iterative 8-bit-digit refinement over
the FULL ordered-bit pattern of the score (fp32: 4 digit rounds; fp16/bf16: 2),
no histogram-of-truncated-bits threshold bin and no fixed-size tie buffer, so
there is no SGLang-style kMaxNumTie hazard class. The auto heuristic picks
single-CTA or multi-CTA per (N, BS); both get exercised by the battery's
shape spread. No deterministic-mode axis (single algorithm).

Parts (identical to fi_topk_correctness.py):
  1. Adversarial synthetic battery (uniform sglang-killer, all-equal row,
     fp16-collision tie block). fp32 + fp16 + bf16 (all supported dtypes).
  2. The sglang-v2-FAILING real rows + below-cap controls.
  3. Broad real sweep, ALL layers x ALL ISLs (last decode step), batched.
  4. V3.2 128k + 256k: ALL 58 layers x ALL 15 decode steps (870 cells each).

Gate: indices in-range, unique, and the selected-value multiset equals
torch.topk's on the identical valid slice (order-free).

Run:  PYTHONNOUSERSITE=1 \
      PYTHONPATH=/tmp/gvrval1/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrval1/cutlass450 \
      python3 radix_topk_correctness.py
"""
import sys
from pathlib import Path

import torch

BENCH = Path("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench")
sys.path.insert(0, str(BENCH / "harness"))
import real_data_v4cap as RV4        # noqa: E402
import real_data_v32 as RV32         # noqa: E402
from radix_cutedsl_op import radix_cutedsl  # noqa: E402

DEV = "cuda"
TOTAL = {"pass": 0, "fail": 0}
FAILS = []


def _gate(tag, idx, valid, N, K, verbose):
    ref = torch.topk(valid.float(), K).values.sort().values
    uniq = int(idx.unique().numel())
    got = valid.float()[idx.clamp(0, N - 1)].sort().values
    exact = bool(idx.min() >= 0 and idx.max() < N and uniq == K
                 and torch.equal(got, ref))
    TOTAL["pass" if exact else "fail"] += 1
    if not exact:
        FAILS.append(tag)
    if verbose or not exact:
        nbad = int((got != ref).sum())
        maxerr = float((ref - got).abs().max())
        print(f"{tag:58s} N={N:<8d} K={K:<5d} exact={str(exact):5s} uniq={uniq:<5d} "
              f"mismatched={nbad:<5d} max_val_err={maxerr:.6f}")
    return exact


def check(tag, row, N, K, dtype=torch.float32, verbose=True):
    """row: 1-D cpu tensor. Returns exact bool."""
    W = ((N + 63) // 64) * 64
    lg = torch.full((1, W), torch.finfo(dtype).min, dtype=dtype, device=DEV)
    lg[0, :N] = row[:N].to(dtype).to(DEV)
    sl = torch.full((1,), N, dtype=torch.int32, device=DEV)
    out = torch.empty((1, K), dtype=torch.int32, device=DEV)
    torch.cuda.synchronize()
    radix_cutedsl(lg, sl, K, out=out)
    torch.cuda.synchronize()
    return _gate(tag, out[0].long().cpu(), lg[0, :N].cpu(), N, K, verbose)


def check_batch(tag, rows, N, K, dtype=torch.float32):
    """rows: list of 1-D cpu tensors sharing valid length N -> one BS=len call."""
    B = len(rows)
    W = ((N + 63) // 64) * 64
    lg = torch.full((B, W), torch.finfo(dtype).min, dtype=dtype, device=DEV)
    for b, r in enumerate(rows):
        lg[b, :N] = r[:N].to(dtype).to(DEV)
    sl = torch.full((B,), N, dtype=torch.int32, device=DEV)
    out = torch.empty((B, K), dtype=torch.int32, device=DEV)
    torch.cuda.synchronize()
    radix_cutedsl(lg, sl, K, out=out)
    torch.cuda.synchronize()
    nfail = 0
    for b in range(B):
        i = out[b].long().cpu()
        valid = lg[b, :N].cpu().float()
        ref = torch.topk(valid, K).values.sort().values
        got = valid[i.clamp(0, N - 1)].sort().values
        ok = bool(i.min() >= 0 and i.max() < N and i.unique().numel() == K
                  and torch.equal(got, ref))
        TOTAL["pass" if ok else "fail"] += 1
        if not ok:
            nfail += 1
            FAILS.append(f"{tag} row{b}")
            print(f"  !! {tag} row {b}: exact=False mismatched={int((got != ref).sum())} "
                  f"max_val_err={float((ref - got).abs().max()):.6f}")
    print(f"{tag:58s} N={N:<8d} K={K:<5d} rows={B:<3d} fail={nfail}")
    return nfail


# ---------------------------------------------------------------- Part 1
print("=== Part 1: adversarial synthetic battery ===")
g = torch.Generator().manual_seed(20260716)
for dt, m in ((torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")):
    for K in (512, 1024, 2048):
        u = torch.rand(131072, generator=g)
        check(f"uniform[0,1) N=128K K={K} [{m}]", u, 131072, K, dtype=dt)
    u = torch.rand(1048576, generator=g)
    check(f"uniform[0,1) N=1M K=2048 [{m}]", u, 1048576, 2048, dtype=dt)
    # all-equal row: every element ties at the boundary (worst possible tie load)
    check(f"all-equal(0.5) N=128K K=2048 [{m}]",
          torch.full((131072,), 0.5), 131072, 2048, dtype=dt)
    # fp16-collision tie block: 8192 fp32-distinct values inside ONE fp16 ulp,
    # straddling the K boundary. In fp32 a full-bit radix must separate them;
    # in fp16/bf16 the cast makes them true ties (worst boundary-tie load).
    base = torch.tensor(1.0)
    eps = torch.finfo(torch.float32).eps
    blk = base + torch.arange(8192, dtype=torch.float32) * eps * base
    row = torch.rand(131072, generator=g) * 0.5          # background below the block
    row[:8192] = blk
    perm = torch.randperm(131072, generator=g)
    check(f"fp16-collision block(8192@1ulp16) K=2048 [{m}]",
          row[perm], 131072, 2048, dtype=dt)

# ---------------------------------------------------------------- Part 2
print("\n=== Part 2: the sglang-v2-FAILING real rows (V3.2 K=2048) ===")

def load_step_row(isl, L, step):
    d = RV32._layer_dir(isl, L)
    lg = torch.load(d / "decode.logits.in.pt", map_location="cpu", weights_only=False)
    pk = torch.load(d / "decode.topk.out.pt", map_location="cpu", weights_only=False)
    Ns = int(pk[step].max()) + 1
    row = lg[step]
    return (row[0] if row.dim() == 2 else row).float()[:Ns].clone(), Ns

for isl, L, st, note in (("256k", 52, 3, "sglv2 FAIL"), ("256k", 52, 6, "sglv2 FAIL"),
                         ("256k", 52, 12, "sglv2 FAIL"), ("128k", 52, 4, "over-cap")):
    r, N = load_step_row(isl, L, st)
    check(f"real v32 {isl} L{L} step{st} ({note})", r, N, 2048)
for isl, L, tie in (("256k", 52, 1466), ("8k", 52, 1382), ("64k", 39, 1237)):
    s = RV32._slim(isl)
    r, N = load_step_row(isl, L, s["s_last"])
    check(f"real v32 {isl} L{L} last-step control (tie={tie})", r, N, 2048)

# ---------------------------------------------------------------- Part 3
print("\n=== Part 3: broad real sweep — ALL layers x ALL ISLs, last step ===")
for model in ("flash", "pro"):
    K = RV4.MODELS[model]["K"]
    for isl in RV4.ISLS:
        s = RV4._slim(model, isl)
        rows = [s["cur"][L][:s["N"]].float() for L in sorted(s["cur"])]
        check_batch(f"real {model} {isl} all-layers", rows, s["N"], K)
for isl in RV32.ISLS:
    s = RV32._slim(isl)
    rows = [s["cur"][L][:s["N"]].float() for L in sorted(s["cur"])]
    check_batch(f"real v32 {isl} all-layers", rows, s["N"], 2048)

# ---------------------------------------------------------------- Part 4
print("\n=== Part 4: V3.2 128k/256k — ALL layers x ALL decode steps ===")
for isl in ("128k", "256k"):
    ncell = nfail = 0
    for L in RV32.LAYERS_ALL:
        d = RV32._layer_dir(isl, L)
        try:
            lg = torch.load(d / "decode.logits.in.pt", map_location="cpu",
                            weights_only=False)
            pk = torch.load(d / "decode.topk.out.pt", map_location="cpu",
                            weights_only=False)
        except Exception as e:
            print(f"  !! v32 {isl} L{L}: {e}")
            continue
        for st in sorted(lg.keys()):
            Ns = int(pk[st].max()) + 1
            row = lg[st]
            r = (row[0] if row.dim() == 2 else row).float()[:Ns]
            ok = check(f"v32 {isl} L{L} step{st}", r, Ns, 2048, verbose=False)
            ncell += 1
            nfail += (not ok)
        del lg, pk
    print(f"v32 {isl} all-layers x all-steps: {ncell - nfail} / {ncell} exact "
          f"({nfail} FAIL)")

# ---------------------------------------------------------------- summary
print(f"\n==== SUMMARY: {TOTAL['pass']} pass / {TOTAL['fail']} fail ====")
for f in FAILS:
    print("  FAIL:", f)
