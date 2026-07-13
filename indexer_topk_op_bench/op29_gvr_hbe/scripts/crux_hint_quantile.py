#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op29 iter1 CRUX part 2: quantile-of-hint predictor sweep.

t_hat(q) = q-th ranked hint value (descending, rank ceil(q*K)); predictor
bin b_hat = coarse_bin(t_hat) - margin_bins. One-sided success: b_hat <= b*.
Sweep q x margin; report success rate + cand size (xK) per scenario.
Also the oracle: cand size at b_hat = b* exactly (lower bound of buffer need).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "op22_temporal_fixed_hr_bench"))
import bundle_data_rr  # noqa: E402

KBITS = 12
NBINS = 1 << KBITS


def coarse_bin(x):
    h = x.to(torch.float16)
    bits = h.view(torch.int16).to(torch.int32) & 0xFFFF
    neg = (bits & 0x8000) != 0
    key = torch.where(neg, (~bits) & 0xFFFF, bits | 0x8000)
    return (key >> (16 - KBITS)).to(torch.int32)


QS = [0.5, 0.75, 0.9, 0.95, 0.98, 1.0]
MARGINS = [0, 1, 2, 4]
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


def main():
    dev = "cuda"
    stats = {}  # (scen) -> list of (K, N, bstar, cand_at_bstar, {(q,m):(succ,candxK)})
    for scen in ("real", "best", "worst"):
        rows = []
        for K in (512, 1024, 2048):
            for N in N_GRID:
                if N <= 2 * K:
                    continue
                b = bundle_data_rr.get_bundle(scen, K, torch.float32, N,
                                              device=dev)
                logits = b["logits"][0, :N].float()
                pre = b["preIdx"][0].long().clamp_(0, N - 1)
                hv = logits[pre].sort(descending=True).values
                bins_all = coarse_bin(logits)
                hist = torch.bincount(bins_all, minlength=NBINS)
                cum_ge = hist.flip(0).cumsum(0).flip(0)
                above = cum_ge - hist
                ok = (above < K) & (above + hist >= K)
                bstar = int(torch.nonzero(ok).max())
                cand_star = int(cum_ge[bstar])
                per = {}
                for q in QS:
                    r = min(K - 1, max(0, int(round(q * K)) - 1))
                    bq = int(coarse_bin(hv[r].unsqueeze(0)))
                    for m in MARGINS:
                        bh = max(bq - m, 0)
                        succ = bh <= bstar
                        cand = int(cum_ge[bh]) if succ else 0
                        per[(q, m)] = (succ, cand / K)
                rows.append((K, N, bstar, cand_star / K, per))
        stats[scen] = rows

    for scen, rows in stats.items():
        print(f"\n== {scen}: oracle cand(b*)/K: "
              f"med {sorted(r[3] for r in rows)[len(rows)//2]:.2f} "
              f"max {max(r[3] for r in rows):.2f}")
        print(f"{'q':>5} {'m':>2} {'succ%':>6} {'cand/K med':>10} "
              f"{'p90':>6} {'max':>6}")
        for q in QS:
            for m in MARGINS:
                succ = [r[4][(q, m)][0] for r in rows]
                cands = sorted(r[4][(q, m)][1] for r in rows if r[4][(q, m)][0])
                rate = sum(succ) / len(succ)
                if cands:
                    med = cands[len(cands) // 2]
                    p90 = cands[int(len(cands) * 0.9) - 1]
                    mx = cands[-1]
                else:
                    med = p90 = mx = float("nan")
                print(f"{q:>5} {m:>2} {rate:>6.1%} {med:>10.2f} "
                      f"{p90:>6.2f} {mx:>6.2f}")
    # failing cells detail for the best-looking configs
    print("\n== per-cell failures for q=0.9,m=2 / q=0.95,m=2:")
    for scen, rows in stats.items():
        for (q, m) in ((0.9, 2), (0.95, 2)):
            bad = [(r[0], r[1]) for r in rows if not r[4][(q, m)][0]]
            big = [(r[0], r[1], round(r[4][(q, m)][1], 1)) for r in rows
                   if r[4][(q, m)][0] and r[4][(q, m)][1] > 8]
            print(f"  {scen} q={q} m={m}: fail={bad}  cand>8K: {big}")


if __name__ == "__main__":
    main()
