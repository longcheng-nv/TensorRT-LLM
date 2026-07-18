# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op36 Track B] Adversarial tie/overflow exactness battery for sgl_bx.

Gates (TRACKB_DESIGN.md): 100% exact including the forced-overflow escape, on
every dispatch path (Register2/4, Streaming, small-batch fused cluster,
persistent cluster + main<3>), plus the single-rank-chunk edge where capped
peer contributions sum to exactly kMaxNumTie at rank 0.

Also proves the battery has teeth: the VENDORED sglang_v2 (no guard) must be
INEXACT on the forced-overflow rows.

Run: python battery_bx.py  (single GPU, ~2 min incl. builds)
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH / "harness"))

from sgl_bx_op import plan as bx_plan, topk_bx_exact, topk_bx  # noqa: E402

DEV = "cuda"
KMAXTIE = 2048


def exact_multiset(x_row, idx_row, K):
    """Value-multiset compare vs torch.topk (ties broken arbitrarily is OK)."""
    got = x_row[idx_row.long()].sort(descending=True).values
    ref = torch.topk(x_row, K).values
    return torch.equal(got, ref)


def make_near_tie_row(N, n_tie, tie_at=0, seed=0):
    """Row with n_tie DISTINCT fp32 values inside ONE fp16 12-bit coarse bin
    (uniform in [1.0, 1.0 + 14*ulp16(1.0))) starting at position `tie_at`;
    everything else is a distinct descending sub-tie background well below.
    Truncating the tie collect at 2048 is provably inexact here for K <
    n_tie: the exact top-K is a specific value subset of the tie bin."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.linspace(-2.0, -1.0, N)                 # distinct background
    ulp16 = 2.0 ** -10                                # fp16 ulp at 1.0
    ties = 1.0 + torch.rand(n_tie, generator=g) * (14 * ulp16)
    x[tie_at:tie_at + n_tie] = ties
    return x.to(DEV)


def run_case(name, x, K, expect_overflow):
    BS, N = x.shape
    sl = torch.full((BS,), N, dtype=torch.int32, device=DEV)
    md, fl = bx_plan(sl)
    out, n_rerun = topk_bx_exact(x, sl, K, metadata=md, flags=fl, max_seq_len=N)
    n_flag = int(fl.sum().item())
    ok = all(exact_multiset(x[r], out[r], K) for r in range(BS))
    flag_ok = (n_flag > 0) == expect_overflow if isinstance(expect_overflow, bool) \
        else n_flag == expect_overflow
    status = "PASS" if (ok and flag_ok) else "FAIL"
    print(f"  [{status}] {name}: BS={BS} N={N} K={K} flags={n_flag} "
          f"rerun={n_rerun} exact={ok}")
    return ok and flag_ok


def main():
    torch.manual_seed(0)
    results = []

    # ---- 1. random data, all dispatch paths (flags must stay 0) -------------
    print("== 1. random rows (no overflow expected) ==")
    for K in (512, 1024, 2048):
        for N in (4096, 8192, 16384, 65536, 131072):
            for BS in (1, 4, 31, 64):
                x = torch.randn(BS, N, dtype=torch.float32, device=DEV)
                results.append(run_case(f"rand", x, K, expect_overflow=0))

    # ---- 2. forced overflow: all-tie rows (equal_count == N) ----------------
    print("== 2. all-tie rows (every element identical) ==")
    for K in (512, 2048):
        for N, BS in ((4096, 1), (16384, 4), (65536, 1), (131072, 2),
                      (262144, 8), (262144, 64)):
            x = torch.ones(BS, N, dtype=torch.float32, device=DEV)
            results.append(run_case("all-tie", x, K, expect_overflow=True))

    # ---- 3. forced overflow: near-tie (distinct fp32, one coarse bin) -------
    #      truncation here is PROVABLY inexact -> escape must restore exactness
    print("== 3. near-tie rows (> kMaxNumTie distinct values in one bin) ==")
    for K in (512, 2048):
        for N, BS in ((4096, 1), (8192, 3), (16384, 1), (16384, 64),
                      (65536, 1), (131072, 4), (262144, 8), (262144, 64)):
            rows = [make_near_tie_row(N, n_tie=3000, seed=17 * r + K)
                    for r in range(BS)]
            x = torch.stack(rows)
            results.append(run_case("near-tie", x, K, expect_overflow=True))

    # ---- 4. cluster single-rank-chunk edge (peer caps sum to kMaxNumTie) ----
    print("== 4. cluster path, ties confined to one rank's chunk ==")
    for BS in (2, 8, 40):   # 2/8 -> fused small-batch cluster; 40 -> persistent+main
        N = 262144
        chunk = -(-N // 8)  # div_ceil over kClusterSize=8 (approx; alignment ok)
        rows = [make_near_tie_row(N, n_tie=3000, tie_at=3 * chunk + 64,
                                  seed=101 + r) for r in range(BS)]
        x = torch.stack(rows)
        results.append(run_case("one-rank-ties", x, 2048, expect_overflow=True))

    # ---- 5. mixed batch: only flagged rows re-run ---------------------------
    print("== 5. mixed batch (2 overflow rows in a 16-row batch) ==")
    N, K = 16384, 2048
    x = torch.randn(16, N, dtype=torch.float32, device=DEV)
    x[3] = make_near_tie_row(N, n_tie=4000, seed=1)
    x[11] = torch.ones(N)
    sl = torch.full((16,), N, dtype=torch.int32, device=DEV)
    md, fl = bx_plan(sl)
    out, n_rerun = topk_bx_exact(x, sl, K, metadata=md, flags=fl, max_seq_len=N)
    flagged = set(fl.nonzero().flatten().tolist())
    ok = all(exact_multiset(x[r], out[r], K) for r in range(16))
    good = flagged == {3, 11} and n_rerun == 2 and ok
    print(f"  [{'PASS' if good else 'FAIL'}] mixed: flagged={sorted(flagged)} "
          f"rerun={n_rerun} exact={ok}")
    results.append(good)

    # ---- 6. TEETH: vendored sglang_v2 (no guard) must FAIL near-tie ---------
    print("== 6. teeth check: vendored sglang_v2 is inexact on these rows ==")
    from sglang_v2_op import topk_v2, plan as sglv2_plan
    N, K = 16384, 2048
    x = make_near_tie_row(N, n_tie=4000, seed=7).unsqueeze(0)
    sl = torch.full((1,), N, dtype=torch.int32, device=DEV)
    md = sglv2_plan(sl)
    idx = topk_v2(x, sl, K, metadata=md, max_seq_len=N)
    vend_exact = exact_multiset(x[0], idx[0], K)
    print(f"  [{'PASS' if not vend_exact else 'FAIL'}] vendored exact={vend_exact} "
          f"(expected False — battery has teeth)")
    results.append(not vend_exact)

    n_pass = sum(results)
    print(f"\n==== BATTERY: {n_pass}/{len(results)} PASS ====")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
