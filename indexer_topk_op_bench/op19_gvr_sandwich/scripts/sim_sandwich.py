# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""op19 iter0 — offline sandwich simulator on the real synth bundles.

Replays op18's M-ary multi-threshold P2 (incl. CDF-aware place_mode=3 fracs)
and additionally tracks the SANDWICH pair per policy:
  thr1 = tightest evaluated threshold with count >= K   -> M1 (P4 cand today)
  thr0 = evaluated threshold with count < K maximizing count -> M0 (direct-write)
Reports per (K, N, policy): passes, M1, M0, band=M1-M0, k_rem=K-M0, and the
P4 working-set shrink band/M1. This is the exact information the kernel gets
for free from the cached ladder — no extra passes are ever simulated.

Also sims a band-targeted accept variant: keep refining while band > b_acc
(rounds permitting) instead of op18's count<=c_acc.
"""
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
import synth_data  # noqa: E402

KC = {512: 5120, 1024: 5120, 2048: 6144}
_FRACS = json.load(open(_BENCH / "op18_gvr_1cta_multithresh" / "results" / "fracs_table.json"))


def fracs_for(K, n, M):
    cands = []
    for key, v in _FRACS.items():
        k_, n_, m_ = (int(x) for x in key.split("_"))
        if k_ == K and m_ == M:
            cands.append((abs(n_ - n), v["fracs"]))
    if not cands:
        return None
    fr = sorted(cands)[0][1]
    while len(fr) < M:
        fr = fr + [min(0.999, fr[-1] + 0.01)]
    return fr[:M]


def band_stats(logits, pre, offset, N):
    idx = pre[0].long() + offset
    idx = idx[(idx >= 0) & (idx < N)]
    v = logits[0].float()[idx]
    return v.min().item(), v.max().item(), v.mean().item()


def count_ge(sv, thr):
    # sv = ascending tensor of -values; count of v >= thr
    return int(torch.searchsorted(sv, -thr, right=True).item())


def place(mode, lo, hi, pmean, M, first_round, fr=None):
    d = hi - lo
    if not first_round:
        return [lo + d * (m + 1) / (M + 1) for m in range(M)]
    if mode == 3 and fr is not None:
        return [lo + d * f for f in fr]
    if mode == 0:
        return [lo + d * m / M for m in range(M)]
    if mode == 1:
        return [lo] + [lo + d / (1 << (M - 1 - m)) for m in range(M - 1)]
    pm = pmean if lo < pmean < hi else (lo + hi) / 2
    half = M // 2
    return ([lo + (pm - lo) * m / half for m in range(half)]
            + [pm + (hi - pm) * m / (M - half) for m in range(M - half)])


def sim_sandwich(sv, K, pmin, pmax, pmean, M, R, mode, fr=None,
                 acc_mult=None, band_acc=None):
    """Returns dict(passes, M1, M0, band, k_rem, done2)."""
    kC = KC[K]
    lo, hi = pmin, pmax
    best_c = None          # M1: tightest count >= K
    best_t = pmin
    up_c = 0               # M0: max count < K (thr strictly above thr1)
    passes = 0
    for r in range(R):
        thrs = place(mode, lo, hi, pmean, M, r == 0, fr)
        cnts = [count_ge(sv, t) for t in thrs]
        passes += 1
        bm = max((m for m in range(M) if cnts[m] >= K), default=-1)
        # sandwich upper-side tracking: every evaluated count < K is a
        # candidate M0 (monotone: count<K => thr > v_K >= any thr1)
        for c in cnts:
            if c < K and c > up_c:
                up_c = c
        if bm >= 0:
            if best_c is None or cnts[bm] <= best_c:
                best_c, best_t = cnts[bm], thrs[bm]
            lo = thrs[bm]
            if bm < M - 1:
                hi = thrs[bm + 1]
        else:
            hi = thrs[0]
        if best_c is not None:
            stop = False
            if acc_mult is not None and best_c <= int(K * acc_mult):
                stop = True
            if band_acc is not None and (best_c - up_c) <= band_acc:
                stop = True
            if band_acc is None and acc_mult is None:
                stop = False
            if stop:
                break
        if hi <= lo:
            break
    done2 = False
    if best_c is None:
        best_c, best_t = count_ge(sv, pmin), pmin
    if best_c > kC:
        done2 = True  # would fall back to retry-shrink (no sandwich)
        up_c = 0
    return dict(passes=passes, M1=best_c, M0=up_c, band=best_c - up_c,
                k_rem=K - up_c, done2=done2)


if __name__ == "__main__":
    crmap = {512: 4, 1024: 4, 2048: 1}
    out_path = _HERE.parent / "results" / "sim_sandwich_fp32.jsonl"
    fout = open(out_path, "w")
    print(f"{'K':>5} {'N':>7} | {'policy':>14} | {'p':>2} {'M1':>5} {'M0':>5} "
          f"{'band':>5} {'k_rem':>5} | band/M1 M0/K")
    for K in (512, 1024, 2048):
        for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            if N <= 2 * K:
                continue
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"], b["preIdx"]
            offset = 1 if crmap[K] == 1 else 0
            pmin, pmax, pmean = band_stats(logits, pre, offset, logits.shape[1])
            sv = -torch.sort(logits[0].float(), descending=True).values
            policies = []
            for M in (2, 3, 4, 6, 8):
                fr = fracs_for(K, N, M)
                if fr is None:
                    continue
                policies.append((f"M{M}R1f3", dict(M=M, R=1, mode=3, fr=fr)))
                policies.append((f"M{M}R2f3_b64", dict(M=M, R=2, mode=3, fr=fr,
                                                       band_acc=64)))
                policies.append((f"M{M}R3f3_b64", dict(M=M, R=3, mode=3, fr=fr,
                                                       band_acc=64)))
            for name, kw in policies:
                r = sim_sandwich(sv, K, pmin, pmax, pmean, **kw)
                rec = dict(K=K, N=N, policy=name, **r)
                fout.write(json.dumps(rec) + "\n")
                tag = " D2!" if r["done2"] else ""
                print(f"{K:>5} {N:>7} | {name:>14} | {r['passes']:>2} "
                      f"{r['M1']:>5} {r['M0']:>5} {r['band']:>5} "
                      f"{r['k_rem']:>5} | {r['band']/max(r['M1'],1):>6.3f} "
                      f"{r['M0']/K:>5.3f}{tag}", flush=True)
    fout.close()
    print(f"\nwrote {out_path}")
