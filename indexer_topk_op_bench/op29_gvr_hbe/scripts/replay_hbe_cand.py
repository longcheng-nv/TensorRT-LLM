#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter11 rung-1: host replay of HBE Phase H0 column placement on real bundles.

Replicates the chunked row-sample estimator (topk_hbe.cuh Phase H0) bit-level:
fp16-ordered 12-bit binning, 64x64 chunk sample, find_threshold at ranks
2*rS_K / 8*rS_K, binB <= binA clamp. Then counts the REAL candidate loads the
fused pass would see (cnt_a, cnt_b vs capA/capB/spill), tier outcome, boundary
bin population (tie machinery load) and a spill-traffic estimate.

Pilot cells expand ONE bundle row to BS identical rows, so a single-row replay
is exact for every CTA in the cell.

Usage: python3 replay_hbe_cand.py [--json out.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "op22_temporal_fixed_hr_bench"))
import bundle_data_rr  # noqa: E402

DEV = "cuda"
HIST_BITS = 12
NBINS = 1 << HIST_BITS
SAMPLE_CHUNKS, CHUNK_ELEMS = 64, 64
NSAMP = SAMPLE_CHUNKS * CHUNK_ELEMS


def coarse_bin(x):
    """fp16-ordered key >> (16-12); bit-identical to extract_coarse_bin<12>."""
    bits = x.half().view(torch.int16).to(torch.int32) & 0xFFFF
    key = torch.where((bits & 0x8000).bool(), (~bits) & 0xFFFF, bits | 0x8000)
    return key >> (16 - HIST_BITS)


def find_threshold(hist, rank):
    """threshold bin b: count(bins>b) < rank <= count(bins>=b)."""
    cum_top = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])
    ok = (cum_top >= rank).nonzero()
    return int(ok.max().item()) if ok.numel() else 0


def cap_a(k):
    return 4 * k if k <= 1024 else 2 * k


def cap_b(k):
    return 2 * k


def replay(logits, N, K):
    row = logits[0, :N].float()
    bins = coarse_bin(row)

    # Phase H0 chunked sample
    stride = max(CHUNK_ELEMS, N // SAMPLE_CHUNKS)
    t = torch.arange(NSAMP, device=row.device)
    idx = torch.clamp((t // CHUNK_ELEMS) * stride + (t % CHUNK_ELEMS), max=N - 1)
    shist = torch.bincount(bins[idx], minlength=NBINS)
    rS_K = max(1, (NSAMP * K) // N)
    rk_a, rk_b = min(NSAMP, 2 * rS_K), min(NSAMP, 8 * rS_K)
    binA = find_threshold(shist, rk_a)
    binB = min(find_threshold(shist, rk_b), binA)

    # full-row loads
    fhist = torch.bincount(bins, minlength=NBINS)
    cnt_a = int((bins >= binA).sum())
    cnt_b = int(((bins >= binB) & (bins < binA)).sum())
    bstar_row = find_threshold(fhist, K)          # true b* over the full row
    capA, capB = cap_a(K), cap_b(K)
    spA = spB = 28 * K
    tierA = cnt_a >= K and cnt_a <= capA + spA
    tierB = (not tierA) and (cnt_a + cnt_b >= K) and cnt_a <= capA + spA \
        and cnt_b <= capB + spB
    ovA, ovB = max(0, cnt_a - capA), max(0, cnt_b - capB)
    # spill DRAM (per row): write 8B + ~2 reads (mini-hist + resolve) per entry
    spill_bytes = ovA * 8 * 3 + (ovB * 8 * 3 if tierB else 0)
    # tierB re-gather: capB-resident idx random reads at resolve + mini-hist
    regather_bytes = (min(cnt_b, capB) * 32 * 2) if tierB else 0
    return {
        "N": N, "K": K, "rk_a": rk_a, "rk_b": rk_b,
        "binA": binA, "binB": binB, "bstar": bstar_row,
        "one_sided_ok": binA <= bstar_row,
        "cnt_a": cnt_a, "cnt_a/K": round(cnt_a / K, 2), "capA": capA,
        "ovA": ovA, "cnt_b": cnt_b, "capB": capB, "ovB": ovB,
        "tier": "A" if tierA else ("B" if tierB else "MISS"),
        "bstar_bin_pop": int(fhist[bstar_row]),          # tie candidates
        "above_bstar": int((bins > bstar_row).sum()),    # count_gt at resolve
        "spill_bytes_row": spill_bytes,
        "regather_bytes_row": regather_bytes,
        "mainpass_bytes_row": N * 4,
        "extra_vs_main_pct": round(
            100 * (spill_bytes + regather_bytes) / (N * 4), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = open(args.json, "a") if args.json else None
    for scen in ("real", "best", "worst"):
        for K in (512, 1024, 2048):
            for N in (131072, 262144):
                b = bundle_data_rr.get_bundle(scen, K, torch.float32, N,
                                              device=DEV)
                r = replay(b["logits"], N, K)
                r["scenario"] = scen
                print(json.dumps(r), flush=True)
                if out:
                    out.write(json.dumps(r) + "\n")
    if out:
        out.close()


if __name__ == "__main__":
    main()
