#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op29 iter1 CRUX (rung 0, host+torch): can the preIdx hint predict the
sglang-v2-style 12-bit coarse threshold bin well enough for a 1-pass fast
path?

Fast-path success condition (HBE design):
    b_hat := coarse_bin(min(hint_values)) - margin
    success iff  b_hat <= b*  (collect-superset at v_lo(b_hat) contains all
                 of top-K; exact resolve happens inside the candidate buffer)
             AND cand_count(bins >= b_hat) <= slot_cap (smem budget)
    miss (b_hat > b*)  -> redo collect pass at true bin (known from the
                 inline histogram)  == rival's 2-pass cost.

Metrics per (scenario, K, N), fp32 op22rr bundles:
    signed gap (b* - b_hat0) at margin 0, per-margin success rate,
    cand_count as multiple of K, expected DRAM passes.
GO if real/best expected passes <= ~1.3 with cand <= ~4xK at some margin.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(HERE.parents[1] / "harness"))
import bundle_data_rr  # noqa: E402

KBITS = 12
NBINS = 1 << KBITS


def coarse_bin(x_f32: torch.Tensor) -> torch.Tensor:
    """Vectorized extract_coarse_bin<12> (fp32 -> fp16 RN -> monotone key)."""
    h = x_f32.to(torch.float16)
    bits = h.view(torch.int16).to(torch.int32) & 0xFFFF
    neg = (bits & 0x8000) != 0
    key = torch.where(neg, (~bits) & 0xFFFF, bits | 0x8000)
    return (key >> (16 - KBITS)).to(torch.int32)


MARGINS = [0, 1, 2, 4, 8, 16]
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


def main():
    dev = "cuda"
    print(f"# crux_hint_bin: 12-bit bins, margins {MARGINS}")
    agg = {}  # (scen, K) -> list of per-N dicts
    for scen in ("real", "best", "worst"):
        for K in (512, 1024, 2048):
            rows = []
            for N in N_GRID:
                if N <= 2 * K:
                    continue
                b = bundle_data_rr.get_bundle(scen, K, torch.float32, N,
                                              device=dev)
                logits = b["logits"][0, :N].float()
                pre = b["preIdx"][0].long().clamp_(0, N - 1)
                hint_vals = logits[pre]
                bins_all = coarse_bin(logits)
                hist = torch.bincount(bins_all, minlength=NBINS)
                # b* : largest bin with above(b) < K <= above(b)+hist[b]
                above = hist.flip(0).cumsum(0).flip(0) - hist  # count bins > b
                ok = (above < K) & (above + hist >= K)
                bstar = int(torch.nonzero(ok).max())
                bhat0 = int(coarse_bin(hint_vals.min().unsqueeze(0)))
                gap = bstar - bhat0
                cum_ge = hist.flip(0).cumsum(0).flip(0)  # count bins >= b
                per_m = {}
                for m in MARGINS:
                    bh = max(bhat0 - m, 0)
                    succ = bh <= bstar
                    cand = int(cum_ge[bh]) if succ else None
                    per_m[m] = (succ, cand)
                rows.append({"N": N, "gap": gap, "bstar": bstar,
                             "bhat0": bhat0, "per_m": per_m,
                             "hr": b["kernel_hit_rate"]})
            agg[(scen, K)] = rows
    # report
    for (scen, K), rows in agg.items():
        print(f"\n== {scen} K={K} (hr={rows[0]['hr']:.3f})")
        print(f"{'N':>8} {'gap b*-b^':>9} " +
              " ".join(f"m={m:<2}succ/cand_xK" for m in MARGINS))
        for r in rows:
            cells = []
            for m in MARGINS:
                s, c = r["per_m"][m]
                cells.append(f"{'Y' if s else 'N'}/"
                             f"{c / K:.1f}" if s else "N/--")
            print(f"{r['N']:>8} {r['gap']:>9} " +
                  " ".join(f"{c:>13}" for c in cells))
    # summary: per scenario, best margin with 100% success, cand cap, E[passes]
    print("\n== summary (pooled over K,N per scenario)")
    for scen in ("real", "best", "worst"):
        rows = [r for (s, k), rs in agg.items() if s == scen for r in rs]
        for m in MARGINS:
            succ = [r["per_m"][m][0] for r in rows]
            cands = [r["per_m"][m][1] / (512 if False else 1)
                     for r in rows if r["per_m"][m][0]]
            # cand as xK needs per-row K; recompute
            cxk = []
            for (s, k), rs in agg.items():
                if s != scen:
                    continue
                for r in rs:
                    ok, c = r["per_m"][m]
                    if ok:
                        cxk.append(c / k)
            rate = sum(succ) / len(succ)
            ep = 1 * rate + 2 * (1 - rate)
            mx = max(cxk) if cxk else float("nan")
            print(f"  {scen} m={m:>2}: success {rate:5.1%}  E[passes] {ep:.2f}"
                  f"  cand max {mx:.1f}xK")


if __name__ == "__main__":
    main()
