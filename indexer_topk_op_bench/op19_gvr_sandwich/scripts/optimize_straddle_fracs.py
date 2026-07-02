# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""op19 straddle-fracs optimizer (multi-seed).

For each (K, N, M) pick M round-1 fracs so the ladder STRADDLES v_K robustly:
  fracs[0] = 0.0                     exactness anchor (count(pmin) >= K always)
  fracs[1] = l1: largest frac with count >= K on ALL seeds (tight thr1 seed)
  fracs[M-1] = l0: smallest frac with count < K on ALL seeds (thr0 = direct-
               write threshold; M0 = count(l0) is just below K)
  fracs[2..M-2] = linspace strictly inside (l1, l0) — self-sorting refinement:
               at runtime each lands on whichever side of v_K this seed's CCDF
               puts it, tightening M1 or M0 for free.
Emits results/straddle_fracs.json {"K_N_M": {"fracs": [...], "val": {...}}}
with cross-seed validation (worst band, worst M0, done2 count).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
import synth_data  # noqa: E402

KC = {512: 5120, 1024: 5120, 2048: 6144}
CR = {512: 4, 1024: 4, 2048: 1}
SEEDS = (42, 0, 1, 2, 3)
GRID = 1024


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def row_state(K, N, seed, dtype=torch.float32):
    b = synth_data.get_bundle(K, dtype, N, seed=seed)
    logits, pre = b["logits"][0].float(), b["preIdx"][0].long()
    off = 1 if CR[K] == 1 else 0
    idx = pre + off
    idx = idx[(idx >= 0) & (idx < logits.shape[0])]
    v = logits[idx]
    pmin, pmax = v.min().item(), v.max().item()
    sv_neg = torch.sort(-logits).values
    return pmin, pmax, sv_neg


def counts_on_grid(pmin, pmax, sv_neg):
    lams = torch.linspace(0.0, 1.0, GRID, dtype=torch.float64)
    thrs = (pmin + lams * (pmax - pmin)).to(torch.float32).to(sv_neg.device)
    cnts = torch.searchsorted(sv_neg, -thrs, right=True)
    return lams.tolist(), cnts.tolist()


def sim_round(cnt_fn, fracs, K):
    """One R1 round on one seed: returns (M1, M0, band, done2)."""
    cnts = [cnt_fn(f) for f in fracs]
    geK = [c for c in cnts if c >= K]
    ltK = [c for c in cnts if c < K]
    M1 = min(geK) if geK else None
    M0 = max(ltK) if ltK else 0
    if M1 is None:
        return None, M0, None, True  # can't happen with frac 0 anchor
    band = M1 - M0
    return M1, M0, band, band > KC[K]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32")
    args = ap.parse_args()
    dt = DTYPES[args.dtype]
    out = {}
    print(f"{'K':>5} {'N':>7} {'M':>2} | fracs -> worst-seed (M1, M0, band) d2")
    for K in (512, 1024, 2048):
        for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            if N <= 2 * K:
                continue
            states = [row_state(K, N, s, dt) for s in SEEDS]
            curves = [counts_on_grid(pm, px, sv) for pm, px, sv in states]

            # per-seed crossing: last lambda with count >= K
            l_geK, l_ltK = [], []
            for lams, cnts in curves:
                gi = max(i for i, c in enumerate(cnts) if c >= K)
                l_geK.append(lams[gi])
                li = min((i for i, c in enumerate(cnts) if c < K),
                         default=None)
                l_ltK.append(lams[li] if li is not None else 1.0)
            step = 1.0 / (GRID - 1)
            l1 = max(0.0, min(l_geK) - step)          # count>=K on all seeds
            l0 = min(0.999, max(l_ltK) + step)        # count<K on all seeds

            for M in (2, 3, 4, 6, 8):
                if M == 2:
                    # NO 0-anchor: zero-tax pass; rare all-counts<K seeds fall
                    # back to done=2 retry-shrink (exact, slow) in-kernel.
                    fr = [l1, l0]
                    worst = dict(M1=0, M0=1 << 30, band=0, d2=0)
                    for (pm, px, sv) in states:
                        cnt = lambda f: int(torch.searchsorted(
                            sv, -(pm + f * (px - pm)), right=True).item())
                        M1, M0, band, d2 = sim_round(cnt, fr, K)
                        worst["M1"] = max(worst["M1"], M1 or 0)
                        worst["M0"] = min(worst["M0"], M0)
                        worst["band"] = max(worst["band"], band or 0)
                        worst["d2"] += int(d2 or M1 is None)
                    out[f"{K}_{N}_{M}"] = dict(fracs=[round(f, 6) for f in fr],
                                               val=worst)
                    print(f"{K:>5} {N:>7} {M:>2} | l1={l1:.4f} l0={l0:.4f} -> "
                          f"M1<={worst['M1']:>5} M0>={worst['M0']:>5} "
                          f"band<={worst['band']:>5} d2={worst['d2']}",
                          flush=True)
                    continue
                inner = []
                if M > 3:
                    n_in = M - 3
                    span = l0 - l1
                    inner = [l1 + span * (j + 1) / (n_in + 1)
                             for j in range(n_in)]
                fr = sorted(set([0.0, l1] + inner + [l0]))
                while len(fr) < M:
                    fr.append(min(0.999, fr[-1] + step))
                fr = fr[:M]

                worst = dict(M1=0, M0=1 << 30, band=0, d2=0)
                for (pm, px, sv) in states:
                    cnt = lambda f: int(torch.searchsorted(
                        sv, -(pm + f * (px - pm)), right=True).item())
                    M1, M0, band, d2 = sim_round(cnt, fr, K)
                    worst["M1"] = max(worst["M1"], M1 or 0)
                    worst["M0"] = min(worst["M0"], M0)
                    worst["band"] = max(worst["band"], band or 0)
                    worst["d2"] += int(d2)
                out[f"{K}_{N}_{M}"] = dict(fracs=[round(f, 6) for f in fr],
                                           val=worst)
                print(f"{K:>5} {N:>7} {M:>2} | l1={l1:.4f} l0={l0:.4f} -> "
                      f"M1<={worst['M1']:>5} M0>={worst['M0']:>5} "
                      f"band<={worst['band']:>5} d2={worst['d2']}", flush=True)
    p = _HERE.parent / "results" / ("straddle_fracs.json" if args.dtype == "fp32" else f"straddle_fracs_{args.dtype}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")
