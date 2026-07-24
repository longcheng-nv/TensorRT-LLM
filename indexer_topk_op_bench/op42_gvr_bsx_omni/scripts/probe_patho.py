# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iter8 attribution probe: offline exact replay of gvr_topk_tp's P1 rung
ladder + P2a sampled pivot selection on REAL cell rows, predicting the number
of full-row streaming passes per row. Confirms/refutes the re-stream
hypothesis for the 5 M1 pathological cells without touching the kernel.

Replays (float32, bit-faithful where it matters):
  phase1: hmin/hmax -> 64-bin trim hist (qtrim=97%K) -> fine hist ->
          rungs[1..6] at CCDF targets [10,25,45,65,82,94]%K,
          rungs[0]=hmax+(hmax-tlow), rungs[7]=hmin.
  P2a:    sampled count, uniform every-SS-th float4 (SS=32), est=cnt*SS;
          pivot = rung with est in [1.5K, 0.6kC] nearest tgt=min(3K, hi),
          fallback rung 7 (hmin).
  fused:  pivot true count in [K, kC] -> 1 stream (+reuse). Else secant loop
          (AR=8 rung interpolation) counted in full-row passes + final collect.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OP42 = HERE.parent
BENCH = OP42.parent
sys.path.insert(0, str(BENCH / "report"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE))
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402
from ab import parse_cell  # noqa: E402

AR = 8
SS = 32
QT = [10, 25, 45, 65, 82, 94]
MAXPASS = 8


def f32(x):
    return np.float32(x)


def phase1(hints, K):
    hv = hints.astype(np.float32)
    hmin, hmax = f32(hv.min()), f32(hv.max())
    if not (hmax - hmin > 0):
        return np.full(AR, hmin, np.float32), hmin, hmax
    scale = f32(64.0) / (hmax - hmin)
    b1 = np.clip(((hv - hmin) * scale).astype(np.int32), 0, 63)
    h1 = np.bincount(b1, minlength=64)
    suf = np.cumsum(h1[::-1])[::-1]  # suf[b] = count in bins >= b
    qtrim = (K * 97) // 100
    binw = (hmax - hmin) * f32(1.0 / 64.0)
    tlow = hmin
    for b in range(63, -1, -1):
        if suf[b] >= qtrim:
            tlow = f32(hmin + binw * f32(b))
            break
    rungs = np.zeros(AR, np.float32)
    rungs[AR - 1] = hmin
    if not (hmax - tlow > 0):
        rungs[: AR - 1] = hmax
        return rungs, hmin, hmax
    scale2 = f32(64.0) / (hmax - tlow)
    sel = hv >= tlow
    b2 = np.clip(((hv[sel] - tlow) * scale2).astype(np.int32), 0, 63)
    h2 = np.bincount(b2, minlength=64)
    suf2 = np.cumsum(h2[::-1])[::-1]
    tot = int(suf2[0])
    binw2 = (hmax - tlow) * f32(1.0 / 64.0)
    for r, q in enumerate(QT):
        qt = (K * q) // 100
        rung = tlow  # fallback: target below trim range
        for b in range(63, -1, -1):
            if suf2[b] >= qt:
                rung = f32(tlow + binw2 * f32(b))
                break
        if tot < qt:
            rung = tlow
        rungs[r + 1] = rung
    rungs[0] = f32(hmax + (hmax - tlow))
    return rungs, hmin, hmax


def simulate(cell):
    model, isl, L = parse_cell(cell)
    mod = v32 if model == "v32" else v4
    b = mod.get_bundle(model, isl, L, "fp32")
    K, N, Npad = b["K"], b["N"], b["Npad"]
    row = b["logits"][0].float().cpu().numpy()[:Npad]
    pre = b["preIdx"][0].cpu().numpy()[:K]
    v4._bundle_cache.clear()
    v32._bundle_cache.clear()

    kC = 8192 if K >= 2048 else 6144
    hints = row[np.clip(pre, 0, Npad - 1)]
    rungs, hmin, hmax = phase1(hints, K)

    cnt = np.array([(row >= t).sum() for t in rungs], np.int64)
    # sampled: every SS-th float4 = 4 consecutive floats each 4*SS floats
    f4 = row[: (Npad // 4) * 4].reshape(-1, 4)
    samp = f4[::SS].ravel()
    est = np.array([SS * (samp >= t).sum() for t in rungs], np.int64)

    lo, hi = (3 * K) // 2, (6 * kC) // 10
    tgt = min(3 * K, hi)
    tgt = max(tgt, lo)
    best, bestd = AR - 1, 1 << 60
    for j in range(AR):
        if lo <= est[j] <= hi:
            d = abs(int(est[j]) - tgt)
            if d < bestd:
                best, bestd = j, d
    in_band = bestd < (1 << 60)
    pC = int(cnt[best])
    fused_ok = K <= pC <= kC

    # count full-row streaming passes (fused pass = 1)
    passes = 1
    detail = ""
    if fused_ok:
        detail = "fast-path"
    else:
        # secant loop: rungs measured so far = {pivot, hmin}
        t_lo, t_hi = -np.inf, np.inf
        meas = [(float(rungs[best]), pC), (float(rungs[AR - 1]), int(cnt[AR - 1]))]
        chosen = None
        for p in range(MAXPASS + 2):
            meas.sort(key=lambda x: -x[0])
            ok = [(t, c) for t, c in meas if K <= c <= kC]
            if ok:
                chosen = ok[0]
                break
            for t, c in meas:
                if c < K and t <= t_hi:
                    t_hi = t
                if c > kC and t >= t_lo:
                    t_lo = t
            if not np.isfinite(t_hi) or not np.isfinite(t_lo):
                span = max(hmax - hmin, 1e-3)
                step = span * (1 << min(p * 3, 24))
                if not np.isfinite(t_hi):
                    nr = [t_lo + step * (1 << (AR - 1 - r)) for r in range(AR)]
                else:
                    nr = [t_hi - step * (1 << r) for r in range(AR)]
            else:
                dt = (t_hi - t_lo) / (AR + 1)
                nr = [t_hi - dt * (r + 1) for r in range(AR)]
            passes += 1
            meas = [(t, int((row >= t).sum())) for t in nr]
        passes += 1  # thr != tpush -> collect_at re-stream
        detail = f"secant->{passes}p"

    hit = np.intersect1d(pre, np.argsort(row)[-K:]).size / K
    return dict(cell=cell, K=K, Npad=Npad, kC=kC, hr=hit,
                pivot=best, in_band=in_band, est=int(est[best]), trueC=pC,
                cnt_hmin=int(cnt[AR - 1]), passes=passes, detail=detail,
                ests=est.tolist(), cnts=cnt.tolist())


if __name__ == "__main__":
    cells = sys.argv[1:] or [
        # patho
        "pro_1024k_L32", "v32_32k_L03", "v32_32k_L41", "v32_64k_L41",
        "v32_256k_L03",
        # healthy controls
        "pro_1024k_L02", "pro_1024k_L60", "v32_32k_L22", "v32_32k_L60",
        "v32_64k_L22", "v32_256k_L22",
    ]
    print(f"{'cell':<16} {'hr':>5} {'piv':>3} {'band':>5} {'est':>7} "
          f"{'trueC':>7} {'cntHmin':>9} {'passes':>6}  detail")
    for c in cells:
        r = simulate(c)
        print(f"{r['cell']:<16} {r['hr']:>5.2f} {r['pivot']:>3} "
              f"{str(r['in_band']):>5} {r['est']:>7} {r['trueC']:>7} "
              f"{r['cnt_hmin']:>9} {r['passes']:>6}  {r['detail']}")
        print(f"    ests={r['ests']}")
        print(f"    cnts={r['cnts']}")
