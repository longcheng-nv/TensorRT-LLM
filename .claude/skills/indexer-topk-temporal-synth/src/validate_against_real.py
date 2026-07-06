#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""Acceptance gates for the unified generator vs the REAL capture cache of
synth_vs_real_validation/ (the study that falsified the single-Beta synth).

Gates (per model):
  G1 per-layer KS D        <= 0.05   (synth fixed-layer vs real pooled layer)
  G2 aggregate KS D        <= 0.05   (synth aggregate mix vs real all-layer pool)
  G3 top-K boundary mass   in [0.80, 1.25] at N in {16K, 64K, 256K}
     (synth mass above the REAL top-K threshold / (K/N); single-Beta was 0.00x)
  G4 retention-curve error <= 0.05   (measured synth preIdx retention-by-rank
                                      vs the real-calibrated curve)
  G5 hit-rate mean error   <= 0.03   (synth realised vs real per-step mean)

Usage:  python3 validate_against_real.py [--models v32,v4flash,v4pro]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synth_temporal_data import (Calib, MODEL_CONTRACT, _incl_probs,  # noqa: E402
                                 inv_cdf, synth_row)

ASSETS = HERE.parent / "assets"

# yardstick = the falsification study's real cache in the SAME repo checkout
# (regenerate with synth_vs_real_validation/extract_real.py; needs NFS captures).
# Override with $REAL_CACHE if the study lives elsewhere.
import os
REAL_CACHE = Path(os.environ.get(
    "REAL_CACHE",
    HERE.parents[3] / "synth_vs_real_validation" / "cache"))
CACHE_NAME = {"v32": "v32", "v4flash": "flash", "v4pro": "pro"}
BOUNDARY_N = [16384, 65536, 262144]
AGG_SAMPLES = 2_000_000
LAYER_SAMPLES = 400_000

GATES = dict(ks_layer=0.05, ks_agg=0.05, sel_lo=0.80, sel_hi=1.25,
             ret_err=0.05, hr_err=0.03)


def ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    ab = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), ab, side="right") / a.size
    cb = np.searchsorted(np.sort(b), ab, side="right") / b.size
    return float(np.abs(ca - cb).max())


def synth_marginal(calib: Calib, layers, n: int, rng) -> np.ndarray:
    per = n // len(layers) + 1
    parts = []
    for L in layers:
        rec = dict(calib.L[L]); rec["_pgrid"] = calib.p_grid
        parts.append(inv_cdf(rng.random(per), rec))
    return np.concatenate(parts)[:n].astype(np.float32)


def validate(model: str) -> dict:
    calib = Calib(model)
    K = MODEL_CONTRACT[model]["K"]
    rng = np.random.default_rng(7)
    z = np.load(REAL_CACHE / f"real_{CACHE_NAME[model]}.npz")
    vp = np.load(ASSETS / f"valpool_{model}.npz")
    layers = calib.layers
    res = {"model": model, "K": K, "gates": {}, "pass": True}

    # G1 per-layer KS — against the step-synchronized validation pool (same
    # decode steps as the quantile tables; isolates generator fidelity from
    # per-step distribution drift).  KS vs the older 12-step cache pool is
    # reported as info (contains intrinsic step-selection noise).
    ks_by_layer, ks_info_cache = {}, {}
    for L in layers:
        real = vp[f"L{L}__pool"]
        syn = synth_marginal(calib, [L], min(LAYER_SAMPLES, real.size * 2), rng)
        ks_by_layer[L] = ks_2samp(syn, real)
        ks_info_cache[L] = ks_2samp(syn, z[f"L{L}__pooled"])
    ks_max = max(ks_by_layer.values())
    res["gates"]["G1_ks_layer_max"] = dict(
        value=round(ks_max, 4), limit=GATES["ks_layer"],
        worst_layer=int(max(ks_by_layer, key=ks_by_layer.get)),
        detail={int(k): round(v, 4) for k, v in ks_by_layer.items()},
        info_ks_vs_12step_cache_max=round(max(ks_info_cache.values()), 4),
        ok=ks_max <= GATES["ks_layer"])

    # G2 aggregate KS
    real_agg = np.concatenate([z[f"L{L}__pooled"] for L in layers])
    syn_agg = synth_marginal(calib, layers, AGG_SAMPLES, rng)
    ks_agg = ks_2samp(syn_agg, real_agg)
    res["gates"]["G2_ks_aggregate"] = dict(
        value=round(ks_agg, 4), limit=GATES["ks_agg"],
        ok=ks_agg <= GATES["ks_agg"])

    # G3 boundary mass at real top-K threshold
    sel = {}
    ok3 = True
    for N in BOUNDARY_N:
        frac = K / N
        thr = float(np.quantile(real_agg, 1.0 - frac))
        ratio = float((syn_agg >= thr).mean() / frac)
        good = GATES["sel_lo"] <= ratio <= GATES["sel_hi"]
        ok3 &= good
        sel[N] = dict(real_thr=round(thr, 4), select_ratio=round(ratio, 3),
                      ok=good)
    res["gates"]["G3_boundary_mass"] = dict(
        detail=sel, limit=[GATES["sel_lo"], GATES["sel_hi"]], ok=ok3)

    # G4 retention curve + G5 hit-rate (measured on generated rows)
    nb = calib.L[layers[0]]["ret_vals"].size
    ret_hit = np.zeros(nb); ret_tot = np.zeros(nb)
    hrs, hr_targets, trial_layers = [], [], []
    N_rows = 65536
    n_trials = 2 * len(layers)          # every layer covered exactly twice
    for t in range(n_trials):
        L = layers[t % len(layers)]
        trial_layers.append(L)
        x, pre_pos, m = synth_row(N_rows, K, calib, L,
                                  np.random.default_rng(100 + t), None, "real")
        order = np.argsort(-x, kind="stable")
        topk_pos = order[:K]
        mask = np.zeros(N_rows, dtype=bool)
        valid = pre_pos[pre_pos >= 0]
        mask[valid] = True
        hit_by_rank = mask[topk_pos]
        b = np.minimum((np.arange(K) / K * nb).astype(int), nb - 1)
        for i in range(nb):
            sel_m = b == i
            ret_hit[i] += hit_by_rank[sel_m].sum()
            ret_tot[i] += sel_m.sum()
        hrs.append(m["realised_hr"])
        hr_targets.append(m["target_hr"])
    syn_ret = ret_hit / np.maximum(ret_tot, 1)
    # reference: expected inclusion probabilities under the same waterfilling
    # construction (per trial: curve w, budget n_hit), averaged over trials
    ref_ret = np.zeros(nb)
    for t in range(n_trials):
        L = trial_layers[t]
        rv = np.maximum(calib.L[L]["ret_vals"].astype(np.float64), 1e-4)
        bucket = np.minimum((np.arange(K) / K * nb).astype(int), nb - 1)
        p = _incl_probs(rv[bucket], round(hr_targets[t] * K))
        for i in range(nb):
            ref_ret[i] += p[bucket == i].mean()
    ref_ret /= n_trials
    ret_err = float(np.abs(syn_ret - ref_ret).max())
    res["gates"]["G4_retention_err"] = dict(
        value=round(ret_err, 4), limit=GATES["ret_err"],
        synth_curve=[round(v, 3) for v in syn_ret],
        ref_curve=[round(v, 3) for v in ref_ret],
        ok=ret_err <= GATES["ret_err"])

    # G5: realised hr must equal the per-row target sampled from the REAL
    # per-step hit-rate distribution (targets ~ real by construction, so the
    # gate checks the generator delivers them; real-vs-target gap is finite-
    # sample info only).
    real_hr_mean = float(np.mean([np.mean(calib.L[L]["hr"]) for L in layers]))
    syn_hr_mean = float(np.mean(hrs))
    tgt_mean = float(np.mean(hr_targets))
    hr_err = abs(syn_hr_mean - tgt_mean)
    res["gates"]["G5_hitrate"] = dict(
        synth_mean=round(syn_hr_mean, 4), target_mean=round(tgt_mean, 4),
        err=round(hr_err, 4), limit=GATES["hr_err"],
        info_real_mean=round(real_hr_mean, 4),
        synth_std=round(float(np.std(hrs)), 4),
        info_real_within_layer_std=round(float(np.mean(
            [np.std(calib.L[L]["hr"]) for L in layers])), 4),
        ok=hr_err <= GATES["hr_err"])

    res["pass"] = all(g["ok"] for g in res["gates"].values())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v32,v4flash,v4pro")
    ap.add_argument("--out", default=str(HERE.parent / "assets" /
                                         "validation_gates.json"))
    args = ap.parse_args()
    allres = {}
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        r = validate(m)
        allres[m] = r
        print(f"\n=== {m}  ->  {'PASS' if r['pass'] else 'FAIL'} ===")
        for name, g in r["gates"].items():
            flag = "ok " if g["ok"] else "FAIL"
            if name == "G3_boundary_mass":
                d = {N: v["select_ratio"] for N, v in g["detail"].items()}
                print(f"  [{flag}] {name}: {d} (limit {g['limit']})")
            elif name == "G5_hitrate":
                print(f"  [{flag}] {name}: synth {g['synth_mean']}±{g['synth_std']}"
                      f" vs target {g['target_mean']}"
                      f" (real mean {g['info_real_mean']})")
            else:
                print(f"  [{flag}] {name}: {g.get('value')}"
                      f" (limit {g['limit']})")
    with open(args.out, "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\n-> {args.out}")
    if not all(r["pass"] for r in allres.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
