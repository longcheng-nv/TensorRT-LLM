# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Host-exact replay of GVR P2/P3-fallback on the op22rr bundle rows.

Counts block_count_ge passes (the dominant, N-scan cost at large BS where the
kernel is DRAM-bound) for the anchor (vendored linear secant + one-sided
retry-shrink) vs op26_1cta (log interp / narrowed window / fb_fix), per
scenario row. Semantics transcribed 1:1 from:
  ops/cute_vendored/blackwell/top_k/gvr_topk_decode.py (P1 seeds, P2, P3 retry)
  op26_gvr_logfalsi_rs/src/gvr_op26_op.py             (log interp, fb_fix)
"""
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[0]
sys.path.insert(0, str(_BENCH / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(_BENCH / "harness"))

import bundle_data_rr  # noqa: E402

MAX_REFINE_ITERS = 15
STOCK = {  # cr=4 table from gvr_topk_decode.py: (kFTarget, kC)
    ("fp32", 512): (512, 5120), ("fp32", 1024): (1024, 5120),
    ("fp32", 2048): (3072, 6144),
    ("bf16", 512): (512, 5120), ("bf16", 1024): (1024, 5120),
    ("bf16", 2048): (4096, 5120),
    ("fp16", 512): (512, 5120), ("fp16", 1024): (1024, 5120),
    ("fp16", 2048): (4096, 5120),
}
DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class Row:
    def __init__(self, logits_row_f32, N):
        self.v = logits_row_f32[:N].contiguous()
        self.passes = 0
        self.trace = []

    def count_ge(self, thr):
        self.passes += 1
        c = int((self.v >= thr).sum().item())
        self.trace.append((float(thr), c))
        return c


def p1_seeds(row_f32, pre_idx, N, K):
    idx = pre_idx[(pre_idx >= 0) & (pre_idx < N)].long()
    pv = row_f32[idx]
    pmin, pmax = float(pv.min()), float(pv.max())
    pmean = float(pv.mean())
    cnt_lo_seed = K + (K >> 2)   # 1.25*K at v=pmin, NEVER measured
    return pmean, pmin, pmax, cnt_lo_seed, 1  # chi seed = 1 at v=pmax


def p2(row, thr0, vlo, vhi, clo, chi, kK, kCC, kFT, use_log):
    """Vendored phase2_secant_search. Returns (done, thr, vlo, vhi, clo, chi)."""
    c0 = row.count_ge(thr0)
    if kK <= c0 <= kCC:
        return 1, thr0, vlo, vhi, clo, chi
    if c0 > kCC:
        vlo, clo = thr0, c0
    else:
        vhi, chi = thr0, c0
    done, thr = 0, thr0
    for it in range(MAX_REFINE_ITERS):
        rng = vhi - vlo
        if clo > chi and rng > 1e-10:
            if use_log:
                chi_f = max(float(chi), 1.0)
                den = math.log2(clo / chi_f)
                if den > 0.0:
                    f = math.log2(clo / kFT) / den
                else:
                    f = (clo - kFT) / (clo - chi)
            else:
                f = (clo - kFT) / (clo - chi)
            f = min(max(f, 0.05), 0.95)
            if it == 0:
                f = min(f, 0.5)
            nv = vlo + rng * f
        else:
            nv = (vlo + vhi) * 0.5
        if nv <= vlo:
            nv = vlo + rng * 0.05
        if nv >= vhi:
            nv = vhi - rng * 0.05
        if nv == vlo or nv == vhi:
            nv = (vlo + vhi) * 0.5
            if nv == vlo or nv == vhi:
                return 2, vlo, vlo, vhi, clo, chi   # give up
        c = row.count_ge(nv)
        if kK <= c <= kCC:
            return 1, nv, vlo, vhi, clo, chi
        if c > kCC:
            vlo, clo = nv, c
        else:
            vhi, chi = nv, c
        thr = nv
    # post-loop forced threshold
    thr = vlo if clo <= 2 * kCC else vhi
    return 2, thr, vlo, vhi, clo, chi


def anchor_p3_retry(row, thr, vlo, vhi, kK, kCC):
    """Vendored one-sided retry-shrink (exits on FIRST count<=kCC)."""
    c = row.count_ge(thr)
    if c > kCC:
        vlo = thr
    for _ in range(10):
        if c <= kCC:
            break
        mid = (vlo + vhi) * 0.5
        if mid == vlo:
            mid = vhi
        thr = mid
        c = row.count_ge(thr)
        if c > kCC:
            vlo = thr
        elif c < kK:
            vhi = thr
    return thr, c


def op26_fb_fix(row, thr, vlo, vhi, kK, kCC, log2_mstar):
    """op26 corrected bounded refine (gvr_op26_op.py phase3 prologue)."""
    clo = chi = -1
    thr_c = thr
    for rs in range(30):
        if rs > 0:
            if chi < 0:
                thr_c = vhi
            elif clo < 0:
                thr_c = vlo
            else:
                chic = max(chi, 1)
                l_lo, l_hi = math.log2(clo), math.log2(chic)
                den = l_lo - l_hi
                thr_c = (vlo + vhi) * 0.5
                if den > 0.0:
                    t = (log2_mstar - l_hi) / den
                    cnd = vhi + t * (vlo - vhi)
                    if vlo < cnd < vhi:
                        thr_c = cnd
        c = row.count_ge(thr_c)
        if kK <= c <= kCC:
            return thr_c, c
        if c > kCC:
            vlo, clo = thr_c, c
            if thr_c >= vhi:
                rng = max(vhi - vlo, 1.0)
                vhi, chi = vhi + rng * 8.0, -1
        else:
            vhi, chi = thr_c, c
            if thr_c <= vlo:
                rng = max(vhi - vlo, 1.0)
                vlo, clo = vlo - rng * 8.0, -1
    c = row.count_ge(vhi)   # exhausted: land on measured undershoot side
    return vhi, c


def run_arm(row_f32, pre_idx, N, K, kFT, kCC, use_log, fb):
    row = Row(row_f32, N)
    thr0, vlo, vhi, clo, chi = p1_seeds(row_f32, pre_idx, N, K)
    done, thr, vlo, vhi, clo, chi = p2(row, thr0, vlo, vhi, clo, chi,
                                       K, kCC, kFT, use_log)
    fb_entered = int(done != 1)
    if done != 1:
        if fb == "anchor":
            thr, c = anchor_p3_retry(row, thr, vlo, vhi, K, kCC)
        else:
            mstar = math.log2(K * (kCC / K) ** 0.2)
            thr, c = op26_fb_fix(row, thr, vlo, vhi, K, kCC, mstar)
    else:
        c = row.trace[-1][1]
    return dict(passes=row.passes, cand=c, fb=fb_entered, trace=row.trace)


def ccdf_loglin(row_f32, N, K, kC):
    """Local log-linearity of count(v) between the K-th and kC-th value."""
    v = row_f32[:N]
    top = torch.topk(v, min(kC * 2, N)).values
    vK, vC = float(top[K - 1]), float(top[min(kC, len(top)) - 1])
    ts = torch.linspace(vC, vK, 9)
    cs = torch.tensor([float((v >= t).sum()) for t in ts])
    lc = torch.log2(cs)
    x = ts - ts.mean()
    slope = float((x * (lc - lc.mean())).sum() / (x * x).sum())
    resid = lc - (lc.mean() + slope * x)
    r2 = 1.0 - float((resid ** 2).sum() / ((lc - lc.mean()) ** 2).sum())
    return r2, float(cs[0]), float(cs[-1])


def main():
    cells = [
        # (K, dt, N, label)
        (1024, "fp32", 131072, "REGRESSION: fp32 log-narrow @131K"),
        (1024, "fp32", 8192,   "WIN control: same cfg @8K"),
        (1024, "fp32", 32768,  "WIN control: same cfg @32K"),
        (1024, "bf16", 131072, "REGRESSION: 16b log stock @131K"),
        (2048, "bf16", 32768,  "REGRESSION: 16b log stock K2048"),
        (2048, "fp16", 65536,  "REGRESSION: 16b log stock K2048"),
        (2048, "fp32", 32768,  "control: fp32 log-narrow K2048"),
    ]
    for K, dt, N, label in cells:
        print(f"\n=== K={K} {dt} N={N}  [{label}] ===")
        kFT_s, kC_s = STOCK[(dt, K)]
        # op26 dispatch (mirror dispatch_p2_op26)
        if dt == "fp32":
            if K == 1024:
                o_log, o_kC, o_kFT = True, 2048, 1024
            elif K == 2048:
                o_log, o_kC, o_kFT = True, 4096, 2048
            else:
                o_log, o_kC, o_kFT = False, 1536, 1280
        else:
            o_log, o_kC, o_kFT = True, kC_s, kFT_s
        for scen in ("real", "best", "worst"):
            b = bundle_data_rr.get_bundle(scen, K, DT[dt], N, device="cuda")
            row = b["logits"][0].float()
            pre = b["preIdx"][0]
            a = run_arm(row, pre, N, K, kFT_s, kC_s, False, "anchor")
            o = run_arm(row, pre, N, K, o_kFT, o_kC, o_log, "fb_fix")
            r2, cC, cK = ccdf_loglin(row, N, K, kC_s)
            print(f"  {scen:5s} hr={b['kernel_hit_rate']:.2f} | anchor: "
                  f"passes={a['passes']:2d} cand={a['cand']:5d} fb={a['fb']} | "
                  f"op26: passes={o['passes']:2d} cand={o['cand']:5d} "
                  f"fb={o['fb']} | pass_ratio={o['passes']/a['passes']:.2f} "
                  f"| CCDF loglin R2[{K}..{kC_s}]={r2:.3f}")
            if o["passes"] >= 2 * a["passes"]:
                print(f"        op26 trace: {[(round(t,3),c) for t,c in o['trace'][:14]]}")
                print(f"        anch trace: {[(round(t,3),c) for t,c in a['trace'][:8]]}")


if __name__ == "__main__":
    main()
