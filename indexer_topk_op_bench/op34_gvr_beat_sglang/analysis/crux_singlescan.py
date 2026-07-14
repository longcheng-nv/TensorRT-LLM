# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op34 iter1 rung-0 CRUX — single-scan multi-threshold bracket + fast-write.

Tests the user hypothesis on REAL v4cap bundles, entirely in host Python (no
kernel, no timing): given ONLY the hint (prev-topK gathered values) the kernel
can cheaply compute order-statistic rung thresholds; during ONE scan it measures
the TRUE count over N at each rung. This script emulates that exactly and asks:

  Q1 one-scan happy rate: does SOME hint rung give TRUE count in [K, kC]? (=> a
     single collect scan suffices; fallback fires only on the complement).
  Q2 fast-write share: at t_hi = deepest rung with true-count <= K, the count_hi
     elements are CERTAIN winners (safe to fast-write, skip P4). share=count_hi/K.
  Q3 contested band: count_lo - count_hi = the P4 workload under this scheme;
     compare to the current kC (op26_r0 P3 over-collects up to kC).
  Q4 does the fallback (no happy rung) fire ONLY on near-zero-correlation rows?
     (correlate happy/fail with hit_rate — the user's key claim).

Rungs: hint order-statistic thresholds at h-quantiles q (threshold = the value
with ceil(q*K) hint values >= it). We use a dense ladder so the scheme has good
coverage (the user permits multi-threshold). kC = op26 dispatch (3072 @K512 else 5120).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
import real_data_v4cap as RD4  # noqa: E402

# dense h-quantile ladder (descending h => ascending threshold value)
QLAD = [0.98, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
OUT = _HERE / "crux_singlescan.json"


def kC_of(K):
    return 3072 if K == 512 else 5120


def cell_metrics(model, isl, layer):
    b = RD4.get_bundle(model, isl, layer, "fp32")
    lg = b["logits"][0, :b["N"]].float()
    K, N = b["K"], b["N"]
    kC = kC_of(K)
    pre = b["preIdx"][0].long()
    hint = lg[pre]                       # K prev-topK gathered values
    hint_desc = hint.sort(descending=True).values
    # rung thresholds from hint order stats
    rung_thr = []
    for q in QLAD:
        j = min(K - 1, max(0, int(math.ceil(q * K)) - 1))
        rung_thr.append(hint_desc[j].item())
    rung_thr = sorted(set(rung_thr))     # ascending value
    # TRUE count over N at each rung (what the single scan measures)
    lg_sorted = lg.sort(descending=True).values
    true_cnt = [int((lg >= t).sum().item()) for t in rung_thr]
    # Q1 happy: some rung with count in [K, kC]
    happy = any(K <= c <= kC for c in true_cnt)
    # Q2/Q3: t_hi = deepest rung (largest value) with count <= K
    #        t_lo = shallowest rung (smallest value) with count in [K, kC]
    count_hi = 0
    for t, c in zip(rung_thr, true_cnt):        # ascending value => ascending t
        if c <= K:
            count_hi = c                        # take the largest such count (<=K)
            break
    # actually want the LARGEST count that is still <= K (deepest safe t_hi):
    cand_hi = [c for c in true_cnt if c <= K]
    count_hi = max(cand_hi) if cand_hi else 0
    cand_lo = [c for c in true_cnt if K <= c <= kC]
    count_lo = min(cand_lo) if cand_lo else (min([c for c in true_cnt if c >= K],
                                                 default=N))
    fast_share = count_hi / K
    band = max(0, count_lo - count_hi)
    # oracle ceiling: exact boundary
    thr_true = lg_sorted[K - 1].item()
    strict_above = int((lg > thr_true).sum().item())
    oracle_fast = strict_above / K
    return dict(N=N, K=K, kC=kC, hit_rate=b["hit_rate"], happy=happy,
                fast_share=fast_share, band=band, band_over_K=band / K,
                count_hi=count_hi, count_lo=count_lo, oracle_fast=oracle_fast,
                cur_overcollect=kC / K)


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    models = [a for a in sys.argv[1:] if a in RD4.MODELS] or list(RD4.MODELS)
    res = {}
    ISLN = RD4.ISL_TOKENS
    print(f"{'cell':13s} {'N':>7s} {'hr_med':>6s} {'happy%':>6s} "
          f"{'fast_med':>8s} {'band/K_med':>10s} {'oracle_med':>10s} {'kC/K':>5s}")
    for model in models:
        m = RD4.MODELS[model]
        for isl in RD4.ISLS:
            if not (RD4._layer_dir(model, isl, m["layers"][0]) /
                    "decode.topk.out.pt").exists():
                continue
            cms = [cell_metrics(model, isl, L) for L in m["layers"]]
            hr = sorted(c["hit_rate"] for c in cms)
            fs = sorted(c["fast_share"] for c in cms)
            bk = sorted(c["band_over_K"] for c in cms)
            orc = sorted(c["oracle_fast"] for c in cms)
            happy_pct = 100.0 * sum(c["happy"] for c in cms) / len(cms)
            key = f"{model}/{isl}"
            res[key] = dict(model=model, isl=isl, N=cms[0]["N"], K=cms[0]["K"],
                            kC=cms[0]["kC"], happy_pct=happy_pct,
                            hr_med=hr[len(hr)//2], fast_med=fs[len(fs)//2],
                            band_over_K_med=bk[len(bk)//2],
                            oracle_med=orc[len(orc)//2],
                            per_layer=[dict(layer=L, **{k: c[k] for k in
                                       ("hit_rate", "happy", "fast_share",
                                        "band_over_K", "oracle_fast")})
                                       for L, c in zip(m["layers"], cms)])
            print(f"{key:13s} {cms[0]['N']:>7d} {hr[len(hr)//2]:6.3f} "
                  f"{happy_pct:6.1f} {fs[len(fs)//2]:8.3f} "
                  f"{bk[len(bk)//2]:10.2f} {orc[len(orc)//2]:10.3f} "
                  f"{cms[0]['kC']/cms[0]['K']:5.1f}")
    OUT.write_text(json.dumps(res, indent=1))
    # grand
    allf = [pl["fast_share"] for c in res.values() for pl in c["per_layer"]]
    allb = [pl["band_over_K"] for c in res.values() for pl in c["per_layer"]]
    allh = [pl["happy"] for c in res.values() for pl in c["per_layer"]]
    print("-" * 74)
    print(f"GRAND: happy={100.0*sum(allh)/len(allh):.1f}%  "
          f"fast_share med={sorted(allf)[len(allf)//2]:.3f}  "
          f"band/K med={sorted(allb)[len(allb)//2]:.2f}  "
          f"(current kC/K over-collect: 6.0 @K512, 5.0 @K1024)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
