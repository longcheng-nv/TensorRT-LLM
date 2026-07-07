# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""proto_hls.py — host prototype validating Theorems 1-3 of the HLS design
(MATH_THRESHOLD_ESTIMATION.html Part II) on the op22 three-scenario bundles,
then grid-searching the HLS parameter family to its global optimum.

Data: op22_temporal_fixed_hr_bench/bundles/{best,worst,real}, fp32 only,
BS=1 rows (78 cells: 3 scenarios x {v4flash K512, v4pro K1024, v32 K2048}
x N 4K..1M). Ground truth per row: exact top-K, true hit-rate h.

What is HOST-EXACT here (algorithm level): counts, order statistics,
straddle/collect outcomes, fallback pass counts. What is MODELED (cost
level, constants from measured campaigns): pass price P_N ~ N*4B/190GB/s
single-CTA (op22 ev-slope), warm extra-pass factor gamma=0.235 (op18 L2 +
op22 22/93.3), width tax tau(M) (op18 microbench), cluster sync s_C and
the leader-recount amplification (op21 iter2 / op22 mech).

Simplifications vs the kernel (documented, checked where possible):
  * P1b column placement uses exact sample order statistics; the kernel's
    256-bin histogram adds <= 1 bin quantization -- simulated separately
    as a sensitivity (HIST mode) rather than baked in.
  * fused-collect slot overflow is modeled as c(collect_col) > CAP_COLLECT
    (default 4096; kernel capacity is per-thread-slot-shaped, op22 showed
    overflow at cand ~4500 with kC 5120).
  * fallback pass counting replays the port's control flow (entry count,
    hi-end check, geometric expansion <=12, bisection <=30) faithfully at
    the decision level, not instruction level.

Outputs: results/proto_hls/records.jsonl (per row x policy),
summary.json (theorem tables + optimizer trajectory + final policy).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent           # op21_gvr_prod/scripts
OPB = HERE.parent.parent                          # indexer_topk_op_bench
sys.path.insert(0, str(OPB / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(OPB / "harness"))
sys.path.insert(0, str(OPB / "ops"))

import bundle_data                                   # noqa: E402
from count_gvr_iters import count_iters              # noqa: E402
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrParams  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = HERE.parent / "results" / "proto_hls"

SCENARIOS = ("best", "worst", "real")
KS = (512, 1024, 2048)

# ---------------- cost-model constants (measured anchors) ----------------
TAU = {1: 1.0, 2: 1.0, 3: 1.20, 4: 1.46, 6: 2.05, 8: 2.70}  # op18 cold; 3/6 interp
GAMMA_WARM = 0.235      # extra-pass factor, BS=1 row L2-resident (op22: 22/93.3)
BW_PASS = 190e9         # B/s effective single-CTA streaming (op22 ev slope)
S_CLUSTER = 0.5e-6      # s, one DSMEM merge + cluster barrier round (iter3/4)
CAP_COLLECT = 4096      # fused-collect capacity model (op22: overflow ~4500)
C_CTA = 4               # ms_auto dispatch: C=4 at N>=65536, BS small
FIXED_FLOOR = 8.5e-6    # s, phase-chain fixed machinery (algorithms.html s7)


def p_pass(N, cta=1, warm=False):
    """Model price of one full-row count pass (s)."""
    t = N * 4.0 / BW_PASS / cta
    if warm:
        t *= GAMMA_WARM
    return t


# ---------------- per-row ground truth ----------------
class Row:
    def __init__(self, scen, K, N, bundle):
        self.scen, self.K, self.N = scen, K, N
        cr = bundle["cr"]
        x = bundle["logits"][0, :N].float().to(DEV)
        pre = bundle["preIdx"][0].to(DEV).long()
        off = 1 if cr == 1 else 0
        idx = pre + off
        ok = (idx >= 0) & (idx < N)
        self.gidx = idx[ok]
        self.S = x[self.gidx]                       # gathered values
        self.K_valid = int(ok.sum().item())
        self.x = x
        tv, ti = torch.topk(x, K)
        self.vK = float(tv[-1].item())
        top_set = torch.zeros(N, dtype=torch.bool, device=DEV)
        top_set[ti] = True
        self.h_true = float(top_set[self.gidx].sum().item()) / K
        self.S_sorted, _ = torch.sort(self.S, descending=True)  # rank 1..Kv
        self.gmin = float(self.S_sorted[self.K_valid - 1].item())
        self.gmax = float(self.S_sorted[0].item())
        gp = GvrParams.get("float32", K, cr)
        self.kC, self.kFTarget = gp.kC, gp.kFTarget
        self.hr_meta = float(bundle.get("kernel_hit_rate", float("nan")))
        self.n_pass_eval = 0                        # full-row eval counter
        # boundary-local retention: the TRUE target sample-rank fraction
        # rho* = G_S(v_K)/K_valid  (Part-I H(K); != mean hit rate h under
        # rank-conditional retention — the worst scenario has h=0.05 but
        # rho*/K ~ 0.2, which is what placement actually needs)
        self.rho_true = float((self.S >= self.vK).sum().item()) \
            / max(self.K_valid, 1)

    def count_ge(self, thr):
        self.n_pass_eval += 1
        return int((self.x >= float(thr)).sum().item())

    def count_ge_free(self, thr):
        # measurement without charging a pass (for ground-truth tables)
        return int((self.x >= float(thr)).sum().item())

    def q(self, f):
        """Sample order statistic at rank fraction f of K_valid (exact)."""
        r = min(max(int(round(f * self.K_valid)), 1), self.K_valid)
        return float(self.S_sorted[r - 1].item())

    def q_hist(self, f, bins=256):
        """P1b-faithful: 256-bin histogram left-edge quantile over
        [gmin, gmax] (suffix-scan crossing), to quantify bin error."""
        rng = self.gmax - self.gmin
        if rng <= 0:
            return self.gmin
        inv = (bins - 1 + 0.99) / rng
        b = torch.clamp(((self.S - self.gmin) * inv).long(), 0, bins - 1)
        hist = torch.bincount(b, minlength=bins).flip(0).cumsum(0).flip(0)
        tgt = max(int(f * self.K_valid), 1)
        sfx = hist.cpu().numpy()
        cand = np.nonzero(sfx >= tgt)[0]
        bi = int(cand[-1]) if len(cand) else 0
        return self.gmin + bi * (rng / bins)

    def sample_ccdf(self, thr):
        """G_S(thr): #sample values >= thr (free surrogate)."""
        return int((self.S >= float(thr)).sum().item())

    def sample_inv(self, m):
        """Value at sample rank m (1-based, descending). Clamped."""
        r = min(max(int(round(m)), 1), self.K_valid)
        return float(self.S_sorted[r - 1].item())


# ---------------- placement policies ----------------
def cols_static(row, qfracs=(0.75, 0.5, 0.25), hist=False):
    """Current MS: anchor g_min + order stats at qfracs (ascending value)."""
    qf = row.q_hist if hist else row.q
    cols = [row.gmin] + [qf(f) for f in qfracs]
    fr = [1.0] + list(qfracs)
    return _mono(cols, fr)


def cols_h_aware(row, h_hat, delta, m_thr=4, hist=False, cap=CAP_COLLECT):
    """HLS Theorem-2 placement: columns at h_hat*{1+delta,1,1-delta,...}
    rank fractions (clipped), + g_min anchor. m_thr counts the anchor.
    Theorem-2c collect constraint: shrink the lowest (collect) column's
    rank fraction until the sample-predicted full-row count G_S(f*Kv)/h_lo
    fits 0.8*cap (h_lo = h_hat*(1-delta))."""
    n_free = m_thr - 1
    if n_free == 1:
        mults = [1.0]
    elif n_free == 2:
        mults = [1.0 + delta, 1.0 - delta]
    elif n_free == 3:
        mults = [1.0 + delta, 1.0, 1.0 - delta]
    else:
        mults = list(np.linspace(1.0 + delta, 1.0 - delta, n_free))
    fr = [min(max(h_hat * m, 0.02), 0.98) for m in mults]
    fr = sorted(set(fr), reverse=True)  # descending rank = ascending value
    # collect-cap constraint on the lowest-value column fr[0]
    h_lo = max(h_hat * (1.0 - delta), 0.03)
    for _ in range(6):
        pred = fr[0] * row.K_valid / h_lo
        if pred <= 0.8 * cap:
            break
        fr[0] = fr[0] * (0.8 * cap) / pred
    fr = sorted(set(fr), reverse=True)
    cols = [row.gmin] + [row.q(f) for f in fr]
    return _mono(cols, [1.0] + fr)


def _mono(cols, fracs):
    """Enforce non-descending columns (kernel P1b epilogue)."""
    out = list(cols)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1])
    return out, fracs


# ---------------- R0 one-shot simulator (MS/msc semantics) ----------------
def row_is_fused(row):
    """Kernel semantics: msc (cluster, N>=65536) always runs fused; the
    single-CTA gvr_ms fuses iff the host gate 4*K <= kC passes (fails for
    K2048 fp32). Non-fused P3 collects the band by a dual-predicate
    rescan, so pair01/overflow are NOT misses there."""
    if row.N >= 65536:
        return True
    return 4 * row.K <= row.kC


def simulate_r0(row, cols, fracs, collect_idx=1, cap=CAP_COLLECT):
    """One fused ladder pass. Returns outcome dict. Charges 1 pass."""
    row.n_pass_eval += 1  # one fused pass evaluates all columns
    cnt = [row.count_ge_free(c) for c in cols]
    K, kC = row.K, row.kC
    fused = row_is_fused(row)
    # best_m: largest ascending-col index with count >= K
    best_m = -1
    for m in range(len(cols)):
        if cnt[m] >= K:
            best_m = m
    up = best_m + 1
    out = {"cnt": cnt, "best_m": best_m, "fused": fused}
    if best_m < 0:
        out["mode"] = "all_lt"          # impossible with g_min anchor
        return out
    if up >= len(cols):
        out["mode"] = "all_ge"          # every column count >= K (h < f_min)
        return out
    m1, m0 = cnt[best_m], cnt[up]
    band = m1 - m0
    out.update({"m1": m1, "m0": m0, "band": band})
    if band > kC:
        out["mode"] = "band_gt_kC"
        return out
    if fused and best_m < collect_idx:
        out["mode"] = "pair01"          # pair includes anchor: collect miss
        return out
    if fused and cnt[collect_idx] > cap:
        out["mode"] = "overflow"        # fused-collect slot overflow
        return out
    out["mode"] = "fast"
    return out


# ---------------- fallback simulators ----------------
def fb_bisect(row, r0, start_thr=None):
    """Port-faithful vendored fallback: entry count, hi-end check,
    geometric expansion (<=12), bisection (<=30) to land in [K, kC].
    Returns #full-row passes consumed (beyond R0)."""
    K, kC = row.K, row.kC
    cols, cnt = r0["_cols"], r0["cnt"]
    best_m = r0["best_m"]
    # entry threshold: kernel picks s_mt_thr[best_m] (count>=K col)
    thr = cols[best_m] if start_thr is None else start_thr
    lo_v, hi_v = row.gmin, row.gmax
    if best_m >= 0:
        lo_v = cols[best_m]
        hi_v = cols[best_m + 1] if best_m + 1 < len(cols) else row.gmax
    passes = 0
    c = row.count_ge_free(thr); passes += 1
    if K <= c <= kC:
        return passes, True
    if c > kC:
        blo, bhi = thr, hi_v
    else:
        blo, bhi = lo_v, thr
    # hi-end guarantee check
    c_hi = row.count_ge_free(bhi); passes += 1
    if K <= c_hi <= kC:
        return passes, True
    ex = 0
    while ex < 12 and c_hi >= K:
        rng = bhi - blo
        if rng <= 0:
            rng = 1e-3
        bhi = bhi + rng
        c_hi = row.count_ge_free(bhi); passes += 1
        ex += 1
    for _ in range(30):
        mid = 0.5 * (blo + bhi)
        if mid <= blo:
            mid = bhi
        c = row.count_ge_free(mid); passes += 1
        if K <= c <= kC:
            return passes, True
        if c > kC:
            blo = mid
        else:
            bhi = mid
    return passes, False


def fb_bridge(row, r0, alpha=0.4, max_steps=12):
    """HLS fallback: bridge regula falsi — interpolate on the sample CCDF
    (rank space) inside the measured value bracket, with an Illinois-style
    bisection safeguard guaranteeing bracket progress. Target
    m* = K*(kC/K)^alpha; accept any count in [K, kC].
    Returns (#full-row passes beyond R0, converged)."""
    K, kC = row.K, row.kC
    mstar = K * (kC / K) ** alpha
    cols, cnt = r0["_cols"], r0["cnt"]
    # value-space bracket: lo (count too HIGH, > kC), hi (count too LOW, < K)
    v_lo, c_lo = None, None      # thr with count > kC
    v_hi, c_hi = None, None      # thr with count < K
    for v, c in zip(cols, cnt):
        if c > kC and (v_lo is None or v > v_lo):
            v_lo, c_lo = v, c
        if c < K and (v_hi is None or v < v_hi):
            v_hi, c_hi = v, c
        if K <= c <= kC:
            # a ladder column already lands: the fallback still pays one
            # recount pass at that thr to populate the collect prefix
            return 1, True
    passes = 0
    last_thr = None
    for _ in range(max_steps):
        if v_lo is not None and v_hi is not None:
            # bridge on sample rank between the bracket endpoints
            ra, rb = row.sample_ccdf(v_lo), row.sample_ccdf(v_hi)
            if c_lo != c_hi:
                t = (mstar - c_hi) / (c_lo - c_hi)     # in (0,1)
                r_next = rb + t * (ra - rb)
            else:
                r_next = 0.5 * (ra + rb)
            thr = row.sample_inv(r_next)
            # safeguard: strict interior progress, else midpoint
            eps = max(abs(v_hi - v_lo) * 1e-6, 1e-12)
            if not (v_lo + eps < thr < v_hi - eps) or thr == last_thr:
                thr = 0.5 * (v_lo + v_hi)
                if not (v_lo < thr < v_hi):
                    return passes, False   # value-tie plateau: exhausted
        elif v_hi is None:
            # no count<K point known (h high / all counts >= K impossible
            # here since that means some col < K... this branch: need
            # HIGHER thr). Ratio estimate through the best >=K point.
            vb, cb = max(((v, c) for v, c in zip(cols, cnt)),
                         key=lambda vc: vc[0])
            r_next = max(row.sample_ccdf(vb) * mstar / max(cb, 1), 1)
            thr = row.sample_inv(r_next)
            if thr <= vb:
                thr = vb + max(abs(vb), 1e-3) * 0.05
        else:
            # no count>kC point known: need LOWER thr than all columns —
            # only g_min anchor region; ratio through lowest point
            va, ca = min(((v, c) for v, c in zip(cols, cnt)),
                         key=lambda vc: vc[0])
            r_next = min(row.sample_ccdf(va) * mstar / max(ca, 1),
                         row.K_valid)
            thr = row.sample_inv(r_next)
            if thr >= va:
                thr = va - max(abs(va), 1e-3) * 0.05
        c = row.count_ge_free(thr); passes += 1
        last_thr = thr
        if K <= c <= kC:
            return passes, True
        if c > kC:
            if v_lo is None or thr > v_lo:
                v_lo, c_lo = thr, c
        else:
            if v_hi is None or thr < v_hi:
                v_hi, c_hi = thr, c
    return passes, False


def fb_logfalsi(row, r0, alpha=0.4, max_steps=12):
    """Log-count regula falsi in VALUE space: CCDF tails are ~exponential,
    so log c(theta) is ~linear in theta — robust exactly where the sample
    is uninformative (h->1 stress: bracket spans counts 393..27k with
    ~zero sample resolution). Illinois-style midpoint safeguard."""
    K, kC = row.K, row.kC
    mstar = K * (kC / K) ** alpha
    cols, cnt = r0["_cols"], r0["cnt"]
    v_lo = c_lo = v_hi = c_hi = None
    for v, c in zip(cols, cnt):
        if c > kC and (v_lo is None or v > v_lo):
            v_lo, c_lo = v, c
        if c < K and (v_hi is None or v < v_hi):
            v_hi, c_hi = v, c
        if K <= c <= kC:
            return 1, True
    passes = 0
    last_thr = None
    for _ in range(max_steps):
        if v_lo is not None and v_hi is not None:
            la, lb = math.log(c_lo), math.log(max(c_hi, 1))
            lt = math.log(mstar)
            t = (lt - lb) / (la - lb) if la != lb else 0.5
            thr = v_hi + t * (v_lo - v_hi)
            eps = max(abs(v_hi - v_lo) * 1e-6, 1e-12)
            if not (v_lo + eps < thr < v_hi - eps) or thr == last_thr:
                thr = 0.5 * (v_lo + v_hi)
                if not (v_lo < thr < v_hi):
                    return passes, False
        elif v_hi is None:
            vb = max(cols)
            thr = vb + max(abs(vb), 1e-3) * 0.1
        else:
            va = min(cols)
            thr = va - max(abs(va), 1e-3) * 0.1
        c = row.count_ge_free(thr); passes += 1
        last_thr = thr
        if K <= c <= kC:
            return passes, True
        if c > kC:
            if v_lo is None or thr > v_lo:
                v_lo, c_lo = thr, c
        else:
            if v_hi is None or thr < v_hi:
                v_hi, c_hi = thr, c
    return passes, False


def fb_hybrid(row, r0, alpha=0.4, max_steps=12):
    """Resolution-gated: bridge when the sample resolves the acceptance
    window (counts-per-sample-rank-step <= (kC-K)/4 in the bracket),
    else log-count falsi. This is the Part-II 'surrogate only where
    informative' rule made concrete."""
    K, kC = row.K, row.kC
    cols, cnt = r0["_cols"], r0["cnt"]
    v_lo = c_lo = v_hi = c_hi = None
    for v, c in zip(cols, cnt):
        if c > kC and (v_lo is None or v > v_lo):
            v_lo, c_lo = v, c
        if c < K and (v_hi is None or v < v_hi):
            v_hi, c_hi = v, c
    if v_lo is not None and v_hi is not None:
        ra, rb = row.sample_ccdf(v_lo), row.sample_ccdf(v_hi)
        res = (c_lo - c_hi) / max(ra - rb, 1)
        if res <= (kC - K) / 4:
            return fb_bridge(row, r0, alpha, max_steps)
    return fb_logfalsi(row, r0, alpha, max_steps)


FB = {"bisect": fb_bisect, "bridge": fb_bridge,
      "logfalsi": fb_logfalsi, "hybrid": fb_hybrid}


# ---------------- policy evaluation ----------------
def eval_policy(rows, placement, m_thr, interp, alpha, delta=0.3,
                h_source="oracle", hist=False, cap=CAP_COLLECT):
    """Evaluate a full policy on all rows. Returns per-row records."""
    recs = []
    for row in rows:
        if placement == "static":
            if m_thr == 4:
                cols, fr = cols_static(row, hist=hist)
            elif m_thr == 2:
                cols, fr = cols_static(row, qfracs=(0.5,), hist=hist)
            elif m_thr == 3:
                cols, fr = cols_static(row, qfracs=(0.6, 0.3), hist=hist)
            else:
                cols, fr = cols_static(
                    row, qfracs=tuple(np.linspace(0.85, 0.15, m_thr - 1)),
                    hist=hist)
        else:  # rho-aware (Theorem-2 placement on the boundary-local rank)
            if h_source == "oracle":
                h_hat = row.rho_true
            elif h_source == "est_r0":
                # tracking proxy: the rho estimate a PREVIOUS step's
                # static-R0 counts would hand to this step (stationarity)
                h_hat = getattr(row, "rho_est_static", row.rho_true)
            else:
                raise ValueError(h_source)
            h_hat = max(h_hat, 0.03)
            cols, fr = cols_h_aware(row, h_hat, delta, m_thr, hist=hist,
                                    cap=cap)
        r0 = simulate_r0(row, cols, fr, cap=cap)
        r0["_cols"] = cols
        rec = {"scen": row.scen, "K": row.K, "N": row.N,
               "h": round(row.h_true, 4), "mode": r0["mode"],
               "band": r0.get("band", -1), "m0": r0.get("m0", -1)}
        if r0["mode"] == "fast":
            rec["fb_passes"] = 0
        else:
            rec["fb_passes"], rec["fb_ok"] = FB[interp](row, r0, alpha) \
                if interp != "bisect" else fb_bisect(row, r0)
        # within-step rho* estimate from the R0 (rank, count) points:
        # log-log interpolate/extrapolate the measured retention mapping
        # to the rank where count == K (Part-I 'ladder counts are a free
        # observation' made concrete; a plain f*K/c ratio at one column
        # is biased under rank-conditional retention)
        rho_est = _rho_from_counts(row, fr, r0["cnt"])
        rec["rho_est_r0"] = round(float(rho_est), 4)
        rec["rho_true"] = round(row.rho_true, 4)
        if placement == "static" and not hasattr(row, "rho_est_static"):
            row.rho_est_static = float(rho_est)
        recs.append(rec)
    return recs


def _rho_from_counts(row, fr, cnt):
    """Estimate rho* (sample-rank fraction where global count crosses K)
    from the measured ladder points (rank_i = fr_i*K_valid, count_i)."""
    K = row.K
    pts = [(fr[i] * row.K_valid, cnt[i])
           for i in range(1, len(cnt)) if cnt[i] > 0]
    pts.sort()                                    # ascending rank
    if not pts:
        return row.rho_true
    below = [(r, c) for r, c in pts if c < K]
    above = [(r, c) for r, c in pts if c >= K]
    if below and above:
        r1, c1 = max(below, key=lambda p: p[1])
        r2, c2 = min(above, key=lambda p: p[1])
        if c2 != c1:
            t = (math.log(K) - math.log(max(c1, 1))) \
                / (math.log(c2) - math.log(max(c1, 1)))
            r = r1 + t * (r2 - r1)
        else:
            r = 0.5 * (r1 + r2)
    elif len(pts) >= 2:
        # extrapolate on the two points nearest K in log-log space
        pts2 = sorted(pts, key=lambda p: abs(p[1] - K))[:2]
        (r1, c1), (r2, c2) = sorted(pts2)
        if c1 > 0 and c2 > 0 and c1 != c2 and r1 != r2:
            b = (math.log(c2) - math.log(c1)) / (math.log(r2) - math.log(r1))
            r = r1 * (K / c1) ** (1.0 / b) if b != 0 else r1
        else:
            r, c = pts2[0]
            r = r * K / max(c, 1)
    else:
        r, c = pts[0]
        r = r * K / max(c, 1)
    return min(max(r / max(row.K_valid, 1), 0.02), 0.98)


def cost_of(recs, rows_by_key, m_thr, regime="W", msc_leader=False):
    """Expected variable cost (s) of a policy under the Part-II model."""
    tot = 0.0
    for r in recs:
        N = r["N"]
        cta = C_CTA if N >= 65536 else 1
        t = TAU[m_thr] * p_pass(N, cta) + (S_CLUSTER if cta > 1 else 0)
        fb = r.get("fb_passes", 0)
        if fb:
            if msc_leader and cta > 1:
                t += fb * p_pass(N, 1)                     # leader recount
            else:
                warm = (regime == "W")
                t += fb * (p_pass(N, cta, warm=warm)
                           + (S_CLUSTER if cta > 1 else 0))
        tot += t + FIXED_FLOOR
    return tot / len(recs)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # -------- load rows --------
    rows = []
    for scen in SCENARIOS:
        for K in KS:
            model = {512: "v4flash", 1024: "v4pro", 2048: "v32"}[K]
            base = OPB / "op22_temporal_fixed_hr_bench" / "bundles" / scen
            for d in sorted(base.glob(f"{model}_fp32_N*")):
                N = int(d.name.split("_N")[-1])
                if args.quick and N not in (65536, 262144, 1048576):
                    continue
                b = bundle_data.get_bundle(scen, K, torch.float32, N,
                                           device=DEV)
                rows.append(Row(scen, K, N, b))
        bundle_data._mem_cache.clear()
    print(f"loaded {len(rows)} fp32 rows "
          f"({time.time()-t0:.0f}s)", flush=True)

    summary = {"n_rows": len(rows), "cost_model": {
        "tau": TAU, "gamma_warm": GAMMA_WARM, "bw_pass": BW_PASS,
        "s_cluster": S_CLUSTER, "cap_collect": CAP_COLLECT,
        "c_cta": C_CTA, "fixed_floor": FIXED_FLOOR}}

    # ================= CYCLE 0: baselines =================
    print("\n=== CYCLE 0: baselines ===", flush=True)
    # classic secant (faithful replay; needs raw bundle preIdx convention)
    classic = []
    for scen in SCENARIOS:
        for K in KS:
            model = {512: "v4flash", 1024: "v4pro", 2048: "v32"}[K]
            base = OPB / "op22_temporal_fixed_hr_bench" / "bundles" / scen
            for d in sorted(base.glob(f"{model}_fp32_N*")):
                N = int(d.name.split("_N")[-1])
                if args.quick and N not in (65536, 262144, 1048576):
                    continue
                b = bundle_data.get_bundle(scen, K, torch.float32, N,
                                           device=DEV)
                st = count_iters(b["logits"][0], b["preIdx"][0],
                                 N, K, b["cr"], torch.float32)
                classic.append({"scen": scen, "K": K, "N": N,
                                "p2_evals": st.p2_evals,
                                "p2_converged": bool(st.p2_converged),
                                "cand": st.cand_count})
        bundle_data._mem_cache.clear()
    ev = [c["p2_evals"] for c in classic]
    summary["classic_secant"] = {
        "p2_evals_mean": float(np.mean(ev)), "p2_evals_max": int(max(ev)),
        "by_scen": {s: float(np.mean([c["p2_evals"] for c in classic
                                      if c["scen"] == s]))
                    for s in SCENARIOS}}
    print("classic secant p2_evals:", summary["classic_secant"], flush=True)

    rows_by_key = {(r.scen, r.K, r.N): r for r in rows}

    # static MS baseline (exact order stats + hist sensitivity)
    base_recs = eval_policy(rows, "static", 4, "bisect", 0.4)
    base_hist = eval_policy(rows, "static", 4, "bisect", 0.4, hist=True)

    def agg(recs):
        n = len(recs)
        modes = {}
        for r in recs:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        fb = [r.get("fb_passes", 0) for r in recs if r["mode"] != "fast"]
        bands = [r["band"] for r in recs if r["band"] >= 0]
        return {"n": n, "modes": modes,
                "fast_rate": modes.get("fast", 0) / n,
                "fb_passes_mean": float(np.mean(fb)) if fb else 0.0,
                "fb_passes_max": int(max(fb)) if fb else 0,
                "band_med": int(np.median(bands)) if bands else -1,
                "by_scen_fast": {s: (sum(1 for r in recs
                                         if r["scen"] == s
                                         and r["mode"] == "fast")
                                     / max(sum(1 for r in recs
                                               if r["scen"] == s), 1))
                                 for s in SCENARIOS}}

    summary["baseline_static"] = agg(base_recs)
    summary["baseline_static_hist"] = agg(base_hist)
    print("static (exact quantiles):", summary["baseline_static"], flush=True)
    print("static (256-bin hist)  :", summary["baseline_static_hist"],
          flush=True)

    # rho-estimator accuracy from R0 counts (feeds Theorem-2 est policies)
    h_err = [abs(r["rho_est_r0"] - r["rho_true"]) for r in base_recs]
    summary["rho_est_r0_abs_err"] = {
        "med": float(np.median(h_err)), "p90": float(np.percentile(h_err, 90)),
        "max": float(max(h_err))}
    summary["rho_vs_h"] = [
        {"scen": r["scen"], "K": r["K"], "N": r["N"], "h": r["h"],
         "rho_true": r["rho_true"], "rho_est": r["rho_est_r0"]}
        for r in base_recs]
    print("R0-count rho-estimator |err|:", summary["rho_est_r0_abs_err"],
          flush=True)

    # ================= CYCLE 1: Theorem 2 (placement) =================
    print("\n=== CYCLE 1: Theorem 2 placement sweep ===", flush=True)
    t2 = {}
    for delta in (0.15, 0.25, 0.35, 0.50):
        recs = eval_policy(rows, "h_aware", 4, "bridge", 0.4, delta=delta,
                           h_source="oracle")
        t2[f"oracle_d{delta}"] = agg(recs)
        print(f"oracle delta={delta}: fast={t2[f'oracle_d{delta}']['fast_rate']:.3f} "
              f"modes={t2[f'oracle_d{delta}']['modes']} "
              f"band_med={t2[f'oracle_d{delta}']['band_med']}", flush=True)
    # est_r0: warm-start estimate from a static R0 (2-round scheme upper cost)
    recs_est = eval_policy(rows, "h_aware", 4, "bridge", 0.4, delta=0.25,
                           h_source="est_r0")
    t2["est_r0_d0.25"] = agg(recs_est)
    print("est_r0 delta=0.25:", t2["est_r0_d0.25"], flush=True)
    summary["theorem2"] = t2

    # ================= CYCLE 2: Theorem 3 (aim point) + bridge =================
    print("\n=== CYCLE 2: Theorem 3 alpha sweep + interpolator ===", flush=True)
    # force-enter fallback from static R0 on ALL rows for statistics
    t3 = {}
    for interp in ("bisect", "bridge", "logfalsi", "hybrid"):
        for alpha in ((0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if interp != "bisect"
                      else (None,)):
            key = f"{interp}" + (f"_a{alpha}" if alpha is not None else "")
            passes, fails = [], 0
            for row in rows:
                cols, fr = cols_static(row)
                r0 = simulate_r0(row, cols, fr)
                r0["_cols"] = cols
                if interp == "bisect":
                    p, ok = fb_bisect(row, r0)
                else:
                    p, ok = FB[interp](row, r0, alpha)
                passes.append(p)
                fails += (not ok)
            t3[key] = {"passes_mean": float(np.mean(passes)),
                       "passes_p90": float(np.percentile(passes, 90)),
                       "passes_max": int(max(passes)), "fails": fails}
            print(f"{key}: {t3[key]}", flush=True)
    summary["theorem3"] = t3
    alpha_best = min(
        [(k, v) for k, v in t3.items() if not k.startswith("bisect")],
        key=lambda kv: (kv[1]["fails"], kv[1]["passes_mean"]))[0]
    summary["alpha_best"] = alpha_best
    print("best fallback:", alpha_best, flush=True)

    # ================= CYCLE 3: Theorem 1 (regime E[T]) + global grid =================
    print("\n=== CYCLE 3: global grid over policy family ===", flush=True)
    a_best = float(alpha_best.split("_a")[1])
    fb_best = alpha_best.split("_a")[0]
    grid = []
    for placement, h_src, delta in (
            [("static", None, None)] +
            [("h_aware", "oracle", d) for d in (0.15, 0.25, 0.35, 0.50)] +
            [("h_aware", "est_r0", d) for d in (0.25, 0.35)]):
        for m_thr in (2, 3, 4, 6):
            if placement == "static" and m_thr not in (2, 3, 4, 6):
                continue
            for interp in ("bisect", fb_best):
                recs = eval_policy(rows, placement, m_thr, interp, a_best,
                                   delta=delta or 0.3,
                                   h_source=h_src or "oracle")
                for regime in ("W", "C"):
                    for msc_leader in ((False, True) if regime == "C"
                                       else (False,)):
                        c = cost_of(recs, rows_by_key, m_thr, regime,
                                    msc_leader)
                        grid.append({
                            "placement": placement, "h_src": h_src,
                            "delta": delta, "M": m_thr, "interp": interp,
                            "regime": regime, "msc_leader": msc_leader,
                            "cost_us": c * 1e6,
                            "fast_rate": agg(recs)["fast_rate"]})
    summary["grid"] = grid
    for regime in ("W", "C"):
        g = [x for x in grid if x["regime"] == regime and not x["msc_leader"]]
        g.sort(key=lambda x: x["cost_us"])
        summary[f"optimum_{regime}"] = g[0]
        print(f"regime {regime} optimum: {g[0]}", flush=True)
        stat = [x for x in g if x["placement"] == "static" and x["M"] == 4
                and x["interp"] == "bisect"][0]
        print(f"  vs static-M4-bisect: {stat['cost_us']:.2f}us "
              f"-> {g[0]['cost_us']:.2f}us "
              f"({stat['cost_us']/g[0]['cost_us']:.3f}x)", flush=True)
    # current msc worst case vs HLS in regime C
    cur = [x for x in grid if x["regime"] == "C" and x["msc_leader"]
           and x["placement"] == "static" and x["M"] == 4
           and x["interp"] == "bisect"][0]
    summary["msc_current_C"] = cur
    print(f"current msc (leader recount, C): {cur['cost_us']:.2f}us",
          flush=True)

    # ================= CYCLE 4: convergence + LOSO generalization =================
    print("\n=== CYCLE 4: leave-one-scenario-out check ===", flush=True)
    loso = {}
    for held in SCENARIOS:
        train = [r for r in rows if r.scen != held]
        test = [r for r in rows if r.scen == held]
        best_c, best_p = None, None
        for delta in (0.15, 0.25, 0.35, 0.50):
            for alpha in (0.2, 0.4, 0.6):
                recs = eval_policy(train, "h_aware", 4, fb_best, alpha,
                                   delta=delta, h_source="oracle")
                c = cost_of(recs, rows_by_key, 4, "C", False)
                if best_c is None or c < best_c:
                    best_c, best_p = c, (delta, alpha)
        recs_t = eval_policy(test, "h_aware", 4, fb_best, best_p[1],
                             delta=best_p[0], h_source="oracle")
        recs_b = eval_policy(test, "static", 4, "bisect", 0.4)
        loso[held] = {
            "picked": {"delta": best_p[0], "alpha": best_p[1]},
            "test_cost_us": cost_of(recs_t, rows_by_key, 4, "C", False) * 1e6,
            "base_cost_us": cost_of(recs_b, rows_by_key, 4, "C", False) * 1e6,
            "test_fast": agg(recs_t)["fast_rate"],
            "base_fast": agg(recs_b)["fast_rate"]}
        print(f"held={held}: {loso[held]}", flush=True)
    summary["loso"] = loso

    # ============ CYCLE 5: deployable policy + per-scenario table ============
    print("\n=== CYCLE 5: deployable (est_r0) policy + final table ===",
          flush=True)
    dep_grid = []
    for delta in (0.35, 0.5, 0.65):
        for m_thr in (3, 4):
            recs = eval_policy(rows, "h_aware", m_thr, fb_best, a_best,
                               delta=delta, h_source="est_r0")
            c = cost_of(recs, rows_by_key, m_thr, "C", False)
            dep_grid.append({"delta": delta, "M": m_thr,
                             "cost_us": c * 1e6,
                             "fast": agg(recs)["fast_rate"]})
    dep_grid.sort(key=lambda x: x["cost_us"])
    summary["deployable_grid"] = dep_grid
    dep = dep_grid[0]
    print("deployable optimum:", dep, flush=True)

    def per_row_cost(recs, m_thr, msc_leader):
        out = []
        for r in recs:
            N = r["N"]
            cta = C_CTA if N >= 65536 else 1
            t = TAU[m_thr] * p_pass(N, cta) + (S_CLUSTER if cta > 1 else 0)
            fb = r.get("fb_passes", 0)
            if fb:
                if msc_leader and cta > 1:
                    t += fb * p_pass(N, 1)
                else:
                    t += fb * (p_pass(N, cta) + (S_CLUSTER if cta > 1 else 0))
            out.append((r["scen"], r["K"], r["N"], (t + FIXED_FLOOR) * 1e6))
        return out

    final_table = {}
    pol_defs = {
        "current_msc": ("static", None, 4, "bisect", True),
        "hls_oracle": ("h_aware", "oracle", 3, fb_best, False),
        "hls_deploy": ("h_aware", "est_r0", dep["M"], fb_best, False),
    }
    pol_recs = {}
    for name, (pl, hs, m, itp, leader) in pol_defs.items():
        recs = eval_policy(rows, pl, m, itp, a_best,
                           delta=dep["delta"] if hs == "est_r0" else 0.35,
                           h_source=hs or "oracle")
        pol_recs[name] = recs
        rc = per_row_cost(recs, m, leader)
        final_table[name] = {}
        for scen in SCENARIOS:
            cs = [c for s, k, n, c in rc if s == scen]
            final_table[name][scen] = {
                "mean_us": float(np.mean(cs)), "max_us": float(np.max(cs)),
                "fast": sum(1 for r in recs if r["scen"] == scen
                            and r["mode"] == "fast")
                / max(sum(1 for r in recs if r["scen"] == scen), 1)}
        print(name, {s: {k: round(v, 2) if isinstance(v, float) else v
                         for k, v in d.items()}
                     for s, d in final_table[name].items()}, flush=True)
    summary["final_table"] = final_table
    summary["final_policy"] = {
        "placement": "rho-aware quantile ladder + g_min anchor",
        "rho_source": "cross-step tracked estimate (validated via est_r0 "
                      "proxy; oracle = tracking upper bound)",
        "delta": dep["delta"], "M": dep["M"],
        "collect_rule": "f_c*Kv/rho_lo <= 0.8*cap",
        "fallback": f"{fb_best} alpha={a_best}",
        "msc_fallback_recount": "cluster-parallel (never leader-only)"}

    # -------- persist --------
    with (OUTDIR / "records.jsonl").open("w") as fh:
        for r in base_recs:
            fh.write(json.dumps({"policy": "static_m4", **r}) + "\n")
        for name, recs in pol_recs.items():
            for r in recs:
                fh.write(json.dumps({"policy": name, **r}) + "\n")
    summary["wall_s"] = time.time() - t0
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUTDIR}/summary.json  ({summary['wall_s']:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
