# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H8a crux (rung-1 host replay): sweep r0_qfracs ladders per K on the real
865 grid; score = R0 hit rate + median cand/K on hits + band-carve rescue
rate (miss cells where crossing-band carve fits: band_size <= kC - c_hi).

  CUDA_VISIBLE_DEVICES=<g> python3 probe_ladder_sweep.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
BENCH = OP40.parent
sys.path.insert(0, str(BENCH / "harness"))

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"

REAL_ISLS = {"flash": RV4.ISLS, "pro": RV4.ISLS,
             "v32": ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
REAL_LAYERS = {"flash": RV4.MODELS["flash"]["layers"],
               "pro": RV4.MODELS["pro"]["layers"],
               "v32": list(RV32.LAYERS_ALL)}

LADDERS = {
    512: [("cur", (0.85,)), ("a", (0.85, 0.55)), ("b", (0.7,)),
          ("c", (0.9, 0.6, 0.35))],
    1024: [("cur", (0.85,)), ("a", (0.85, 0.55)), ("b", (0.7, 0.4)),
           ("c", (0.9, 0.6, 0.35))],
    2048: [("cur", (0.6, 0.35)), ("a", (0.75, 0.5, 0.3)),
           ("b", (0.8, 0.6, 0.4, 0.25)), ("c", (0.5, 0.3)),
           ("d", (0.7, 0.45, 0.25))],
}


def pick_cs(N):
    return 1 if N < 65536 else (8 if N >= 131072 else 4)


def cell_stats(model, isl, L):
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, L, "fp32")
    row = b["logits"][0].float()
    N, K = b["N"], b["K"]
    row = row[:N]
    pre = b["preIdx"][0].long()
    pre = pre[(pre >= 0) & (pre < N)]
    vals = row[pre]
    return row, vals, N, K


def replay_ladder(row, vals, N, K, qf, use_vseed=True):
    pmin, pmax, pmean = float(vals.min()), float(vals.max()), float(vals.mean())
    kc = {512: 3072 if pick_cs(N) == 1 else 5120, 1024: 5120, 2048: 6144}[K]
    nb = 256
    width = (pmax - pmin) / nb
    if width <= 0:
        return None
    bins = ((vals - pmin) / width).long().clamp(0, nb - 1)
    hist = torch.bincount(bins, minlength=nb)
    cum_hi = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])
    thrs = []
    for q in qf:
        need = max(1, int(-(-q * K // 1)))
        ok = (cum_hi >= need).nonzero()
        bidx = int(ok.max()) if ok.numel() else 0
        thrs.append(pmin + bidx * width)
    if use_vseed:
        thrs.append(pmean)
    thrs = sorted(set(thrs))
    counts = [int((row >= t).sum()) for t in thrs]
    adm = [c for c in counts if K <= c <= kc]
    if adm:
        return dict(hit=1, cand=min(adm), carve=0, K=K)
    # band-carve rescue: bracket rungs around K
    lo_cands = [(t, c) for t, c in zip(thrs, counts) if c > kc]
    hi_cands = [(t, c) for t, c in zip(thrs, counts) if c < K]
    c_hi = min((c for _, c in hi_cands), default=0)
    # tightest bracket
    c_lo = min((c for _, c in lo_cands), default=None)
    if c_lo is None:
        return dict(hit=0, cand=None, carve=0, K=K)  # all rungs under K: refine down
    band = c_lo - c_hi
    carve = 1 if band <= kc - 0 and (K - c_hi) > 0 and band <= kc else 0
    return dict(hit=0, cand=None, carve=carve, K=K, band=band, c_hi=c_hi)


def main():
    rows = []
    for m in ("flash", "pro", "v32"):
        for isl in REAL_ISLS[m]:
            for L in REAL_LAYERS[m]:
                try:
                    row, vals, N, K = cell_stats(m, isl, L)
                except Exception:
                    continue
                for name, qf in LADDERS[K]:
                    r = replay_ladder(row, vals, N, K, qf)
                    if r:
                        r.update(ladder=name, K=K)
                        rows.append(r)
            RV4._bundle_cache.clear()
            RV32._bundle_cache.clear()
    import statistics as st
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["K"], r["ladder"])].append(r)
    print(f"{'K':>5s} {'ladder':>7s} {'n':>4s} {'hit%':>6s} {'carve+%':>8s} "
          f"{'eff%':>6s} {'med cand/K':>10s}")
    for (K, lad) in sorted(by):
        v = by[(K, lad)]
        hits = [x for x in v if x["hit"]]
        carv = [x for x in v if x["carve"]]
        ck = sorted(x["cand"] / K for x in hits)
        med = st.median(ck) if ck else float("nan")
        print(f"{K:5d} {lad:>7s} {len(v):4d} {100*len(hits)/len(v):6.1f} "
              f"{100*len(carv)/len(v):8.1f} "
              f"{100*(len(hits)+len(carv))/len(v):6.1f} {med:10.2f}")


if __name__ == "__main__":
    main()
