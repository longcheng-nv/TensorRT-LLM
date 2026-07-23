# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H6 crux (rung-1 host replay): on the real 865 grid, replay P1b rung
placement + R0 admission to get the admitted candidate count distribution
(cand/K) and the R0 miss rate. P4 work scales with cand; if cand/K is fat,
admission tightening is a live lever; if already ~1.2-1.5x, it is dead.

Replays the e612 defaults: K2048 -> qfracs (0.6, 0.35) + vseed pmean;
K512/K1024 -> (0.85,) + vseed. kC: K512 cs1 kc_diet=3072 else 5120; K1024
5120; K2048 6144. 256-bin hist over [pmin, pmax].

  CUDA_VISIBLE_DEVICES=<g> python3 probe_candcount.py > ../results/candcount.csv
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


def pick_cs(N):
    if N < 65536:
        return 1
    if N >= 131072:
        return 8
    return 4  # BS=1: num_rows<=4 handled above; mirrors pick_config for BS=1


def replay(model, isl, L):
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, L, "fp32")
    row = b["logits"][0].float()
    pre = b["preIdx"][0].long()
    N, K = b["N"], b["K"]
    row = row[:N]
    pre = pre[(pre >= 0) & (pre < N)]
    vals = row[pre]
    pmin, pmax, pmean = float(vals.min()), float(vals.max()), float(vals.mean())
    kc = {512: 3072 if pick_cs(N) == 1 else 5120, 1024: 5120, 2048: 6144}[K]
    qf = (0.6, 0.35) if K == 2048 else (0.85,)
    # rung placement: hist of gathered vals, cum from high side
    nb = 256
    width = (pmax - pmin) / nb
    if width <= 0:
        return None
    bins = ((vals - pmin) / width).long().clamp(0, nb - 1)
    hist = torch.bincount(bins, minlength=nb)
    cum_hi = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])  # cum >= bin b
    thrs = []
    for q in qf:
        need = max(1, int(-(-q * K // 1)))
        # rung fires at crossing bin: smallest b (from top) with cum >= need
        ok = (cum_hi >= need).nonzero()
        bidx = int(ok.max()) if ok.numel() else 0
        thrs.append(pmin + bidx * width)
    thrs.append(pmean)  # vseed column
    counts = [int((row >= t).sum()) for t in thrs]
    adm = [(c, i) for i, c in enumerate(counts) if K <= c <= kc]
    hit = bool(adm)
    cand, col = min(adm) if adm else (None, None)
    vseed_col = len(thrs) - 1
    return dict(uuid=f"{model}_{isl}_L{L:02d}", model=model, N=N, K=K, kc=kc,
                hit=int(hit), cand=cand, ratio=(cand / K if cand else None),
                counts=counts, admit_is_vseed=int(col == vseed_col) if hit else None,
                vseed_admissible=int(K <= counts[vseed_col] <= kc))


def main():
    print("uuid,model,N,K,kc,hit,cand,ratio,admit_is_vseed,vseed_admissible")
    stats = []
    for m in ("flash", "pro", "v32"):
        for isl in REAL_ISLS[m]:
            for L in REAL_LAYERS[m]:
                r = replay(m, isl, L)
                if r is None:
                    continue
                stats.append(r)
                print(f"{r['uuid']},{r['model']},{r['N']},{r['K']},{r['kc']},"
                      f"{r['hit']},{r['cand']},{r['ratio']},{r['admit_is_vseed']},{r['vseed_admissible']}")
            RV4._bundle_cache.clear()
            RV32._bundle_cache.clear()
    import statistics as st
    hits = [s for s in stats if s["hit"]]
    rs = sorted(s["ratio"] for s in hits)
    sys.stderr.write(
        f"\ncells {len(stats)} · R0 hit {len(hits)} ({100*len(hits)/len(stats):.1f}%)\n"
        f"cand/K on hits: p10 {rs[int(0.1*len(rs))]:.2f} med {st.median(rs):.2f} "
        f"p90 {rs[int(0.9*len(rs))]:.2f} max {rs[-1]:.2f}\n")
    for K in (512, 1024, 2048):
        rk = sorted(s["ratio"] for s in hits if s["K"] == K)
        hk = [s for s in stats if s["K"] == K]
        if rk:
            va = sum(1 for s in stats if s["K"] == K and s["vseed_admissible"])
            av = sum(1 for s in hits if s["K"] == K and s["admit_is_vseed"])
            sys.stderr.write(f"K={K}: hit {len(rk)}/{len(hk)} med cand/K "
                             f"{st.median(rk):.2f} p90 {rk[int(0.9*len(rk))]:.2f} "
                             f"vseed_admissible {va}/{len(hk)} admit_is_vseed {av}/{len(rk)}\n")


if __name__ == "__main__":
    main()
