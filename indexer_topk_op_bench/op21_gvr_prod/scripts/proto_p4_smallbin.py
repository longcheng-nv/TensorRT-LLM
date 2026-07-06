#!/usr/bin/env python3
"""iter6 host probe: distribution of cnt(b*) — the straddling coarse-bin
population inside phase4_band_rank_scatter — on synth + real bundles.

Replays the kernel's decision chain on host (mode-5 P1b rank-quantile
seeding -> M=4 ladder counts -> sandwich pair (thr1, thr0) -> band ->
coarse kNumBins hist over [thr1, thr0) -> straddling bin b*), then asks:
  A) how often rank_above + cnt(b*) == k_rem  (whole-bin emit, skip fine)
  B) how often cnt(b*) <= 32 / 64             (register-select path)
GO gate: A+B coverage >= 95% of sandwich-path rows, p90 cnt(b*) <= 64.
"""
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
import synth_data, real_data_v2  # noqa: E402

QBINS = 256
QFRACS = (0.75, 0.5, 0.25)
KNB = {512: 1024, 1024: 1024, 2048: 2048}  # fp32 kNumBins per K


def p4_stats(row, pre_idx, K, N):
    """Replay P1b + ladder + sandwich + P4 coarse binning; return dict or
    None if the row falls off the sandwich path (fallback territory)."""
    v = row[:N].float()
    idx = pre_idx[pre_idx >= 0]
    idx = idx[idx < N].long()
    g = v[idx]
    if g.numel() == 0:
        return None
    lo, hi = g.min().item(), g.max().item()
    if hi <= lo:
        return None
    # P1b: QBINS hist of gathered values, suffix scan, quantile columns
    hist = torch.histc(g, bins=QBINS, min=lo, max=hi)
    # kernel binning: b = int((v-lo)*inv), inv=(QBINS-1+0.99)/rng — close
    # enough to histc for a statistics probe
    sfx = torch.flip(torch.cumsum(torch.flip(hist, (0,)), 0), (0,))
    kv = g.numel()
    binw = (hi - lo) / QBINS
    thrs = [lo]
    for f in QFRACS:
        tgt = max(1, int(kv * f))
        cand = (sfx >= tgt).nonzero()
        b = cand.max().item() if cand.numel() else 0
        thrs.append(lo + b * binw)
    thrs = sorted(thrs)  # ascending, col0 = g_min anchor
    cnts = [(v >= t).sum().item() for t in thrs]
    # sandwich pair: thr1 = tightest count>=K; thr0 = next col (count<K)
    best_m = -1
    for m in range(len(thrs)):
        if cnts[m] >= K:
            best_m = m
    if best_m < 0 or best_m == len(thrs) - 1 or cnts[best_m + 1] >= K:
        return None  # all_lt (impossible) or all_ge -> round-2/fallback
    thr1, thr0 = thrs[best_m], thrs[best_m + 1]
    m0 = cnts[best_m + 1]
    band_mask = (v >= thr1) & (v < thr0)
    band = v[band_mask]
    k_rem = K - m0
    if band.numel() < k_rem or k_rem <= 0:
        return None
    # P4 coarse hist over [thr1, thr0)
    kbins = KNB[K]
    inv1 = (kbins - 1 + 0.99) / (thr0 - thr1)
    bins = ((band - thr1) * inv1).long().clamp(0, kbins - 1)
    bh = torch.bincount(bins, minlength=kbins)
    csum = torch.flip(torch.cumsum(torch.flip(bh, (0,)), 0), (0,))
    cand2 = (csum >= k_rem).nonzero()
    b_star = cand2.max().item()
    rank_above = int(csum[b_star + 1].item()) if b_star + 1 < kbins else 0
    cbs = int(bh[b_star].item())
    return {"band": band.numel(), "k_rem": k_rem, "cnt_bstar": cbs,
            "eq_hit": rank_above + cbs == k_rem}


rows = []
for K in (512, 1024, 2048):
    for N in (65536, 262144):
        for seed in (42, 7, 1234):
            b = synth_data.get_bundle(K, torch.float32, N, seed=seed)
            r = p4_stats(b["logits"][0].cuda(), b["preIdx"][0].cuda(), K, N)
            if r:
                r["src"] = f"synth K{K} N{N}"
                rows.append(r)
for model, layers in (("pro", range(2, 61, 2)), ("flash", range(2, 43, 2)),
                      ("v32", (0, 1, 20, 21, 22, 40, 41, 42, 60))):
    for L in layers:
        b = real_data_v2.get_real_bundle_v2(model, L, "fp32")
        r = p4_stats(b["logits"][0], b["preIdx"][0], b["K"], b["N"])
        if r:
            r["src"] = f"real {model} L{L}"
            rows.append(r)

n = len(rows)
cbs = sorted(r["cnt_bstar"] for r in rows)
eq = sum(r["eq_hit"] for r in rows)
le32 = sum(r["cnt_bstar"] <= 32 for r in rows)
le64 = sum(r["cnt_bstar"] <= 64 for r in rows)
cover32 = sum(r["eq_hit"] or r["cnt_bstar"] <= 32 for r in rows)
cover64 = sum(r["eq_hit"] or r["cnt_bstar"] <= 64 for r in rows)
print(f"sandwich-path rows: {n}  (synth 18 + real 60 attempted)")
print(f"band size:   p50={sorted(r['band'] for r in rows)[n//2]}  "
      f"max={max(r['band'] for r in rows)}")
print(f"cnt(b*):     p50={cbs[n//2]}  p90={cbs[int(n*0.9)]}  max={cbs[-1]}")
print(f"path A (eq_hit):        {eq}/{n} = {eq/n:.1%}")
print(f"path B (cnt<=32 / 64):  {le32}/{n} = {le32/n:.1%}  /  "
      f"{le64}/{n} = {le64/n:.1%}")
print(f"A+B coverage (CAP=32):  {cover32}/{n} = {cover32/n:.1%}")
print(f"A+B coverage (CAP=64):  {cover64}/{n} = {cover64/n:.1%}")
big = [r for r in rows if not r["eq_hit"] and r["cnt_bstar"] > 64]
for r in big[:8]:
    print(f"  miss: {r['src']}: band={r['band']} k_rem={r['k_rem']} "
          f"cnt(b*)={r['cnt_bstar']}")
