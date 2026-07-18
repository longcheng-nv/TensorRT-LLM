#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op36 Track B ship gate] sgl_bx correctness — the SAME 2245-cell battery
that falsified vendored sglang v2 (fi_topk_correctness.py lineage), applied
to the Track-B exactness port (kernel + overflow flag + radix escape).

The decisive rows: V3.2 256k L52 steps 3/6/12 (+128k L52 step 4) where the
UNGUARDED sglang v2 measurably fails (tie count over kMaxNumTie=2048). The
port must flag them and the escape must restore exactness. fp32 only (the
sglang v2 kernel contract; the harness never routes other dtypes to it).

Gate: indices in-range, unique, value multiset == torch.topk on the valid
slice. Reports total escape re-runs (expected: only on over-cap rows).

Run: env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=3 python3 bx_topk_correctness.py
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
BENCH = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(BENCH / "harness"))
import real_data_v4cap as RV4   # noqa: E402
import real_data_v32 as RV32    # noqa: E402
from sgl_bx_op import topk_bx_exact  # noqa: E402

DEV = "cuda"
TOTAL = {"pass": 0, "fail": 0, "rerun": 0}
FAILS = []


def _run_bx(lg, N, K):
    B = lg.shape[0]
    sl = torch.full((B,), N, dtype=torch.int32, device=DEV)
    out, n_rerun = topk_bx_exact(lg, sl, K, max_seq_len=N)
    TOTAL["rerun"] += n_rerun
    return out, n_rerun


def check(tag, row, N, K, verbose=True):
    W = ((N + 63) // 64) * 64
    lg = torch.full((1, W), torch.finfo(torch.float32).min,
                    dtype=torch.float32, device=DEV)
    lg[0, :N] = row[:N].float().to(DEV)
    torch.cuda.synchronize()
    out, n_rerun = _run_bx(lg, N, K)
    torch.cuda.synchronize()
    idx = out[0].long().cpu()
    valid = lg[0, :N].cpu().float()
    ref = torch.topk(valid, K).values.sort().values
    uniq = int(idx.unique().numel())
    got = valid[idx.clamp(0, N - 1)].sort().values
    exact = bool(idx.min() >= 0 and idx.max() < N and uniq == K
                 and torch.equal(got, ref))
    TOTAL["pass" if exact else "fail"] += 1
    if not exact:
        FAILS.append(tag)
    if verbose or not exact:
        nbad = int((got != ref).sum())
        print(f"{tag:58s} N={N:<8d} K={K:<5d} exact={str(exact):5s} "
              f"uniq={uniq:<5d} mismatched={nbad:<5d} rerun={n_rerun}")
    return exact


def check_batch(tag, rows, N, K):
    B = len(rows)
    W = ((N + 63) // 64) * 64
    lg = torch.full((B, W), torch.finfo(torch.float32).min,
                    dtype=torch.float32, device=DEV)
    for b, r in enumerate(rows):
        lg[b, :N] = r[:N].float().to(DEV)
    torch.cuda.synchronize()
    out, n_rerun = _run_bx(lg, N, K)
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
            print(f"  !! {tag} row {b}: exact=False "
                  f"mismatched={int((got != ref).sum())}")
    print(f"{tag:58s} N={N:<8d} K={K:<5d} rows={B:<3d} fail={nfail} "
          f"rerun={n_rerun}")
    return nfail


# ---------------------------------------------------------------- Part 1
print("=== Part 1: adversarial synthetic battery (fp32) ===")
g = torch.Generator().manual_seed(20260716)
for K in (512, 1024, 2048):
    u = torch.rand(131072, generator=g)
    check(f"uniform[0,1) N=128K K={K}", u, 131072, K)
u = torch.rand(1048576, generator=g)
check("uniform[0,1) N=1M K=2048", u, 1048576, 2048)
check("all-equal(0.5) N=128K K=2048",
      torch.full((131072,), 0.5), 131072, 2048)
base = torch.tensor(1.0)
eps = torch.finfo(torch.float32).eps
blk = base + torch.arange(8192, dtype=torch.float32) * eps * base
row = torch.rand(131072, generator=g) * 0.5
row[:8192] = blk
perm = torch.randperm(131072, generator=g)
check("fp16-collision block(8192@1ulp16) K=2048", row[perm], 131072, 2048)

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
print(f"\n==== SUMMARY: {TOTAL['pass']} pass / {TOTAL['fail']} fail / "
      f"{TOTAL['rerun']} escape re-runs ====")
for f in FAILS:
    print("  FAIL:", f)
sys.exit(1 if TOTAL["fail"] else 0)
