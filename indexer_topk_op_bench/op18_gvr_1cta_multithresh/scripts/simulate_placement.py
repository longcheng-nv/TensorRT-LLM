# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Offline simulator: baseline secant vs M-ary multi-threshold P2 on the real
# synth bundles. Reports full-N passes used and final cand_count (P4 working
# set) per policy — guides (M, R, placement, accept) tuning without kernel runs.
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
import synth_data  # noqa: E402

KC = {512: 5120, 1024: 5120, 2048: 6144}
KFT = {512: 512, 1024: 1024, 2048: 3072}  # cr=4 fp32 (512/1024); cr=1 (2048)


def band(logits, pre, offset, N):
    idx = pre[0].long() + offset
    idx = idx[(idx >= 0) & (idx < N)]
    v = logits[0].float()[idx]
    return v.min().item(), v.max().item(), v.mean().item()


def shrink_passes(sv, K, kC, lo, hi):
    """done=2 path: P3 recount at lo + bisect until count<=kC (extra passes)."""
    p = 1  # recount at current threshold
    c = count_ge(sv, lo)
    rs = 0
    while rs < 10 and c > kC:
        mid = (lo + hi) * 0.5
        if mid == lo:
            mid = hi
        c = count_ge(sv, mid); p += 1
        if c > kC:
            lo = mid
        elif c < K:
            hi = mid
        rs += 1
    return p, c


def count_ge(sv, thr):
    # sv = sorted descending values tensor; count via searchsorted on -sv
    return int(torch.searchsorted(sv, -thr, right=True).item())


def sim_baseline(sv, K, pmin, pmax, pmean, max_iters=10):
    kC, kFT = KC[K], KFT[K]
    passes, lo, hi, clo, chi = 0, pmin, pmax, 0, 0
    thr = pmean
    c = count_ge(sv, thr); passes += 1
    if K <= c <= kC:
        return passes, c
    if c > kC:
        lo, clo = thr, c
    else:
        hi, chi = thr, c
    for it in range(max_iters):
        rng = hi - lo
        if clo > chi and rng > 1e-10:
            f = (clo - kFT) / (clo - chi)
            f = max(0.05, min(0.95, f))
            if it == 0:
                f = min(f, 0.5)
            nv = lo + rng * f
        else:
            nv = (lo + hi) * 0.5
        nv = max(lo + rng * 0.05, min(hi - rng * 0.05, nv))
        if nv == lo or nv == hi:
            nv = (lo + hi) * 0.5
            if nv == lo or nv == hi:
                return passes, count_ge(sv, lo)
        thr = nv
        c = count_ge(sv, thr); passes += 1
        if K <= c <= kC:
            return passes, c
        if c > kC:
            lo, clo = thr, c
        else:
            hi, chi = thr, c
    return passes, count_ge(sv, lo)


def place(mode, lo, hi, pmean, M, first_round):
    d = hi - lo
    if not first_round:
        return [lo + d * (m + 1) / (M + 1) for m in range(M)]
    if mode == 0:
        return [lo + d * m / M for m in range(M)]
    if mode == 1:
        return [lo] + [lo + d / (1 << (M - 1 - m)) for m in range(M - 1)]
    pm = pmean if lo < pmean < hi else (lo + hi) / 2
    half = M // 2
    return ([lo + (pm - lo) * m / half for m in range(half)]
            + [pm + (hi - pm) * m / (M - half) for m in range(M - half)])


def sim_mt(sv, K, pmin, pmax, pmean, M, R, acc_mult, mode):
    lo, hi = pmin, pmax
    best_c, best_t = None, pmin
    cacc = int(K * acc_mult)
    passes = 0
    for r in range(R):
        thrs = place(mode, lo, hi, pmean, M, r == 0)
        cnts = [count_ge(sv, t) for t in thrs]
        passes += 1
        bm = max((m for m in range(M) if cnts[m] >= K), default=-1)
        if bm >= 0:
            if best_c is None or cnts[bm] <= best_c:
                best_c, best_t = cnts[bm], thrs[bm]
            lo = thrs[bm]
            if bm < M - 1:
                hi = thrs[bm + 1]
        else:
            hi = thrs[0]
        if best_c is not None and best_c <= cacc:
            break
        if hi <= lo:
            break
    if best_c is None:
        best_c = count_ge(sv, pmin)
    if best_c > KC[K]:  # done=2 -> P3 retry-shrink extra passes
        ep, best_c = shrink_passes(sv, K, KC[K], best_t, hi)
        passes += ep
    return passes, best_c


if __name__ == "__main__":
    crmap = {512: 4, 1024: 4, 2048: 1}
    policies = [
        ("M4 R2 u a2.0", dict(M=4, R=2, acc_mult=2.0, mode=0)),
        ("M4 R1 u", dict(M=4, R=1, acc_mult=1.0, mode=0)),
        ("M4 R1 dy", dict(M=4, R=1, acc_mult=1.0, mode=1)),
        ("M8 R1 dy", dict(M=8, R=1, acc_mult=1.0, mode=1)),
        ("M4 R2 dy a1.5", dict(M=4, R=2, acc_mult=1.5, mode=1)),
        ("M8 R2 dy a1.25", dict(M=8, R=2, acc_mult=1.25, mode=1)),
        ("M4 R1 pm", dict(M=4, R=1, acc_mult=1.0, mode=2)),
        ("M4 R2 pm a1.5", dict(M=4, R=2, acc_mult=1.5, mode=2)),
        ("M2 R1 dy", dict(M=2, R=1, acc_mult=1.0, mode=1)),
        ("M2 R2 dy a2.0", dict(M=2, R=2, acc_mult=2.0, mode=1)),
    ]
    hdr = f"{'K':>5} {'N':>7} | {'base p/cand':>12} |" + "".join(f" {n:>16}" for n, _ in policies)
    print("passes/cand_count per policy (fp32 bundles)")
    print(hdr)
    for K in (512, 1024, 2048):
        for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            if N <= 2 * K:
                continue
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"], b["preIdx"]
            offset = 1 if crmap[K] == 1 else 0
            pmin, pmax, pmean = band(logits, pre, offset, logits.shape[1])
            neg_sorted = -torch.sort(logits[0].float(), descending=True).values  # ascending -v
            sv = neg_sorted
            bp, bc = sim_baseline(sv, K, pmin, pmax, pmean)
            row = f"{K:>5} {N:>7} | {bp:>4}/{bc:>7} |"
            for _, kw in policies:
                p, c = sim_mt(sv, K, pmin, pmax, pmean, **kw)
                row += f" {p:>4}/{c:>11}"
            print(row, flush=True)
