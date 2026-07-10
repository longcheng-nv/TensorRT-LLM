# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op27 host screen — fb_mprobe (M-ary value-space bracket probe) vs the
shipped fb_logfalsi on the op22rr bundles (op24 scenario definitions).

fb_mprobe: identical to fb_logfalsi EXCEPT when the R0 bracket is missing
an end (all_ge: no count<K point; or the symmetric all-high case). There,
instead of serial +0.1|v| value steps (kernel: geometric doubling, one
full-row scan each), it issues ONE M-column probe scan:
  - slope: log-linear CCDF rate lambda from the two highest distinct
    ladder points (the same ~exponential-tail assumption log-falsi uses),
  - targets: count aims mstar * 4^{-m}, m=0..M-1 (spans a 64x slope error
    at M=4), thresholds thr_m = v1 + (ln c1 - ln t_m)/lambda,
  - guard: non-finite/negative slope -> geometric value steps
    thr_m = v1 + rng*2^m (rng from the top ladder gap).
The probe is charged tau4(N) pass-equivalents (count_ge_multi_bench
measured fp32 column); every serial pass is charged 1.0. All M probe
counts update the bracket; the normal falsi loop then lands.

Usage: python3 screen_mprobe.py [--alpha 0.2] [--M 4] [--json out.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # op27_hls_allge_probe
OPB = HERE.parent                                  # indexer_topk_op_bench
sys.path.insert(0, str(OPB / "op21_gvr_prod" / "scripts"))
sys.path.insert(0, str(OPB / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(OPB / "harness"))
sys.path.insert(0, str(OPB / "ops"))

import torch  # noqa: E402

import bundle_data_rr  # noqa: E402
from proto_hls import (Row, cols_static, simulate_r0, fb_logfalsi,  # noqa: E402
                       p_pass)

SCENS = ("worst", "best", "real")
KS = (512, 1024, 2048)
NS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
# ship ladder (op25): w3a for K512/K1024, stock for K2048
SHIP_QFRACS = {512: (0.92, 0.45, 0.048), 1024: (0.92, 0.45, 0.048),
               2048: (0.75, 0.5, 0.25)}
# count_ge_multi_bench REPORT: fp32 M=4 per-N overhead vs M=1 (measured)
TAU4_N = {4096: 1.01, 8192: 1.20, 16384: 1.15, 32768: 1.23,
          65536: 1.31, 131072: 1.39, 262144: 1.46}


def tau4(N):
    ks = sorted(TAU4_N)
    if N <= ks[0]:
        return TAU4_N[ks[0]]
    if N >= ks[-1]:
        return TAU4_N[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= N <= b:
            t = (math.log(N) - math.log(a)) / (math.log(b) - math.log(a))
            return TAU4_N[a] + t * (TAU4_N[b] - TAU4_N[a])
    return 1.46


def _bracket_from(cols, cnt, K, kC):
    v_lo = c_lo = v_hi = c_hi = None
    landed = None
    for v, c in zip(cols, cnt):
        if c > kC and (v_lo is None or v > v_lo):
            v_lo, c_lo = v, c
        if c < K and (v_hi is None or v < v_hi):
            v_hi, c_hi = v, c
        if K <= c <= kC:
            landed = (v, c)
    return v_lo, c_lo, v_hi, c_hi, landed


def fb_mprobe(row, r0, alpha=0.2, M=4, max_steps=12):
    """Returns (pass_equivalents: float, converged: bool, probe_rounds)."""
    K, kC = row.K, row.kC
    mstar = K * (kC / K) ** alpha
    cols, cnt = list(r0["_cols"]), list(r0["cnt"])
    v_lo, c_lo, v_hi, c_hi, landed = _bracket_from(cols, cnt, K, kC)
    if landed is not None:
        return 1.0, True, 0            # ladder column already in-band
    passes = 0.0
    probe_rounds = 0
    for _pr in range(2):               # at most 2 probe rounds
        if v_lo is not None and v_hi is not None:
            break                       # bracket complete
        probe_rounds += 1
        # distinct (v, c) points sorted by value; slope from the top pair
        pts = sorted({(float(v), int(c)) for v, c in zip(cols, cnt)})
        thrs = []
        if v_hi is None:
            # need HIGHER thresholds (all_ge). slope from top two points.
            (v0, c0), (v1, c1) = pts[-2], pts[-1]
            lam = None
            if v1 > v0 and c0 > c1 > 0:
                lam = (math.log(c0) - math.log(c1)) / (v1 - v0)
            if lam is not None and lam > 0 and math.isfinite(lam):
                for m in range(M):
                    tgt = max(mstar * (0.25 ** m), 1.0)
                    thrs.append(v1 + (math.log(max(c1, 1)) - math.log(tgt))
                                / lam)
            else:
                rng = max(v1 - v0, abs(v1) * 0.05, 1e-3)
                thrs = [v1 + rng * (2 ** m) for m in range(M)]
            thrs = sorted(set(max(t, v1 + 1e-7) for t in thrs))
        else:
            # need LOWER thresholds (no count>kC end) — symmetric
            (v0, c0), (v1, c1) = pts[0], pts[1]
            lam = None
            if v1 > v0 and c0 > c1 > 0:
                lam = (math.log(c0) - math.log(c1)) / (v1 - v0)
            if lam is not None and lam > 0 and math.isfinite(lam):
                for m in range(M):
                    tgt = mstar * (4.0 ** m)
                    thrs.append(v0 - (math.log(max(tgt, 1))
                                      - math.log(max(c0, 1))) / lam)
            else:
                rng = max(v1 - v0, abs(v0) * 0.05, 1e-3)
                thrs = [v0 - rng * (2 ** m) for m in range(M)]
            thrs = sorted(set(min(t, v0 - 1e-7) for t in thrs))
        # ONE fused M-column scan
        passes += tau4(row.N)
        pcnt = [row.count_ge_free(t) for t in thrs]
        cols += thrs
        cnt += pcnt
        v_lo, c_lo, v_hi, c_hi, landed = _bracket_from(cols, cnt, K, kC)
        if landed is not None:
            return passes + 1.0, True, probe_rounds  # +1 recount at landing
    # normal log-falsi landing loop on the (now bracketed) interval
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
        c = row.count_ge_free(thr)
        passes += 1.0
        last_thr = thr
        if K <= c <= kC:
            return passes, True, probe_rounds
        if c > kC:
            if v_lo is None or thr > v_lo:
                v_lo, c_lo = thr, c
        else:
            if v_hi is None or thr < v_hi:
                v_hi, c_hi = thr, c
    return passes, False, probe_rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--json", default=str(HERE / "results_screen.json"))
    args = ap.parse_args()

    recs = []
    for scen in SCENS:
        for K in KS:
            for N in NS:
                try:
                    b = bundle_data_rr.get_bundle(scen, K, torch.float32, N)
                except Exception as e:
                    print(f"skip {scen} K{K} N{N}: {type(e).__name__}")
                    continue
                row = Row(scen, K, N, b)
                cols, fr = cols_static(row, qfracs=SHIP_QFRACS[K])
                r0 = simulate_r0(row, cols, fr)
                r0["_cols"] = cols
                rec = {"scen": scen, "K": K, "N": N, "mode": r0["mode"],
                       "h": round(row.h_true, 3)}
                if r0["mode"] == "fast":
                    rec["ship_p"] = rec["probe_p"] = 0.0
                    rec["ship_ok"] = rec["probe_ok"] = True
                    rec["probe_rounds"] = 0
                else:
                    sp, sok = fb_logfalsi(row, r0, args.alpha)
                    pp, pok, prr = fb_mprobe(row, r0, args.alpha, args.M)
                    rec.update(ship_p=float(sp), ship_ok=bool(sok),
                               probe_p=round(pp, 2), probe_ok=bool(pok),
                               probe_rounds=prr)
                # model us for the fallback portion (cold, single-CTA)
                pp_us = p_pass(N) * 1e6
                rec["ship_us"] = round(rec["ship_p"] * pp_us, 1)
                rec["probe_us"] = round(rec["probe_p"] * pp_us, 1)
                recs.append(rec)
                print(f"{scen:5} K{K:<4} N{N:>6} {rec['mode']:10} "
                      f"ship {rec['ship_p']:>4.1f}p/{rec['ship_us']:>6.1f}us "
                      f"probe {rec['probe_p']:>4.2f}p/{rec['probe_us']:>6.1f}us "
                      f"ok={rec['probe_ok']} rounds={rec['probe_rounds']}")
    Path(args.json).write_text(json.dumps(recs, indent=1))
    # aggregate
    print("\n== aggregate fallback pass-equivalents (mean over cells) ==")
    for scen in SCENS:
        for K in KS:
            sel = [r for r in recs if r["scen"] == scen and r["K"] == K]
            if not sel:
                continue
            fb = [r for r in sel if r["mode"] != "fast"]
            sp = sum(r["ship_p"] for r in sel) / len(sel)
            pp = sum(r["probe_p"] for r in sel) / len(sel)
            ok = all(r["probe_ok"] for r in sel)
            print(f"  {scen:5} K{K:<4}: cells={len(sel)} fb={len(fb)} "
                  f"ship={sp:.2f}p probe={pp:.2f}p all_ok={ok}")


if __name__ == "__main__":
    main()
