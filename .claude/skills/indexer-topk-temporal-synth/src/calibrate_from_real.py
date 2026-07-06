#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""Offline calibration: real DSv3.2 / V4-Flash / V4-Pro indexer captures →
compact per-layer assets consumed by the unified generator
(synth_temporal_data.py). Run ONCE per capture refresh; assets are committed
into the skill's assets/ dir so synthesis never needs the multi-GB captures.

Extracted per (model, layer):
  1. Marginal: empirical quantile table q(p) on a tail-densified global p-grid
     + upper-tail GPD (peaks-over-threshold, PWM fit) for extrapolation beyond
     the deepest reliable empirical quantile.  This is what fixes the
     single-Beta tail under-modeling found by synth_vs_real_validation
     (synth mass at real top-K boundary was 0.00x at N>=128K).
  2. Temporal: rank-conditional retention curve  P(pos in preIdx | current
     exact-topK rank bucket),  miss-depth samples  (thr - logit_miss)/sigma,
     per-step hit-rate samples, valid-count fraction samples (V4 undershoot
     sentinels), and lag-1 Gaussian-copula rho (for --steps chain mode).

Real sources (identical to synth_vs_real_validation/extract_real.py):
  V3.2     : Layer_{L}_pd.npy [2025,70690] f32; valid = strip trailing zeros.
             preIdx not captured -> temporal stats from consecutive-row exact
             topK (production preIdx == prev-step topK by GVR closed loop).
  V4 Flash : Q9jCapture 64k (K=512,  cr=4); real preidx.in per cell.
  V4 Pro   : Q9jCapture 64k (K=1024, cr=4); real preidx.in per cell.

Usage:
  python3 calibrate_from_real.py [--models v32,v4flash,v4pro] [--outdir ../assets]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "assets"

Q9J_SRC = ("/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1/"
           "ablation_study/gvr_phase_timing/09_precision_ablation/"
           "11_dsv4_trtllm_indexer_data_capture/src")

V32_DIR = Path("/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/"
               "data_distri/deepseek-v3.2-logging/notebooks/SWE_Bench_64K_decode_logits")
FLASH_CAP = ("/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1/"
             "ablation_study/gvr_phase_timing/09_precision_ablation/"
             "11_dsv4_trtllm_indexer_data_capture/data/"
             "capture_20260520T082958_alllayers_swe64k")
PRO_CAP = ("/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1/"
           "ablation_study/gvr_phase_timing/09_precision_ablation/"
           "12_dsv4_pro_indexer_data_capture/data/"
           "capture_20260520T164146Z_v4pro_K1024_64k")

MODELS = {
    "v32": dict(K=2048, cr=1, layers=[0, 1, 20, 21, 22, 40, 41, 42, 60],
                source=str(V32_DIR)),
    "v4flash": dict(K=512, cr=4, source=FLASH_CAP),
    "v4pro": dict(K=1024, cr=4, source=PRO_CAP),
}

POOL_STEPS = 16      # decode steps pooled for the marginal quantile table
PAIR_STEPS = 24      # consecutive (t-1, t) pairs for temporal stats
N_RET_BUCKETS = 20   # rank buckets over [0, K) for the retention curve
MISS_DEPTH_CAP = 4000
GPD_MIN_EXC = 150    # min exceedances for the tail fit
RHO_SUBSAMPLE = 50_000

# Global p-grid: tail-densified.  Entries beyond the GPD switch are still
# stored (harmless) but the generator uses the GPD there.
P_GRID = np.unique(np.concatenate([
    [0.0],
    np.geomspace(1e-6, 0.01, 40),
    np.linspace(0.01, 0.99, 197),
    1.0 - np.geomspace(0.01, 1e-7, 70),
    [1.0],
])).astype(np.float64)

RET_EDGES = np.linspace(0.0, 1.0, N_RET_BUCKETS + 1)  # fractions of K


def fit_gpd_pwm(exc: np.ndarray):
    """Hosking-Wallis PWM fit of GPD to exceedances (x - u > 0).
    Returns (xi, beta) in the  F(x) = 1 - (1 + xi*x/beta)^(-1/xi)  convention.
    """
    x = np.sort(exc.astype(np.float64))
    n = x.size
    a0 = x.mean()
    pp = (np.arange(1, n + 1) - 0.35) / n          # plotting positions
    a1 = np.mean(x * (1.0 - pp))
    denom = a0 - 2.0 * a1
    if denom <= 0:
        return 0.0, float(a0)                      # exponential-tail fallback
    xi = 2.0 - a0 / denom
    beta = 2.0 * a0 * a1 / denom
    xi = float(np.clip(xi, -0.5, 0.5))
    beta = float(max(beta, 1e-6))
    return xi, beta


VALPOOL_CAP = 200_000  # per-layer downsample kept for validate_against_real.py


def marginal_assets(pooled: np.ndarray):
    """Quantile table + upper-tail GPD from pooled per-layer samples."""
    pooled = pooled.astype(np.float64)
    q = np.quantile(pooled, P_GRID)
    m = pooled.size
    # threshold: at least GPD_MIN_EXC exceedances, at most 2% of mass
    p_u = min(1.0 - GPD_MIN_EXC / m, 0.998)
    u_thr = float(np.quantile(pooled, p_u))
    exc = pooled[pooled > u_thr] - u_thr
    if exc.size < 30:                               # degenerate; disable GPD
        xi, beta = 0.0, float(pooled.std() * 0.1)
    else:
        xi, beta = fit_gpd_pwm(exc)
    rng = np.random.default_rng(1234)
    vp = (pooled if m <= VALPOOL_CAP
          else pooled[rng.choice(m, VALPOOL_CAP, replace=False)])
    return dict(
        q=q.astype(np.float32),
        gpd=np.array([p_u, u_thr, xi, beta], dtype=np.float64),
        mean=float(pooled.mean()), std=float(pooled.std()),
        min=float(pooled.min()), max=float(pooled.max()),
        n_pooled=int(m),
        valpool=vp.astype(np.float32),
    )


def temporal_accumulate(cur: np.ndarray, pre_positions: np.ndarray, K: int, acc: dict):
    """Update temporal accumulators from one (current row, preIdx positions) cell.

    cur           : current-step logits (float, len N)
    pre_positions : valid preIdx positions (int, current-row coordinates)
    """
    N = cur.size
    if N <= K + 8:
        return
    order = np.argsort(-cur, kind="stable")
    topk_pos = order[:K]
    thr = float(cur[order[K - 1]])
    sigma = float(cur.std()) or 1.0

    pre_set_mask = np.zeros(N, dtype=bool)
    valid = pre_positions[(pre_positions >= 0) & (pre_positions < N)]
    pre_set_mask[valid] = True

    hit_by_rank = pre_set_mask[topk_pos]            # [K] bool, rank-ordered
    bucket_idx = np.minimum((np.arange(K) / K * N_RET_BUCKETS).astype(int),
                            N_RET_BUCKETS - 1)
    for b in range(N_RET_BUCKETS):
        m = bucket_idx == b
        acc["ret_hit"][b] += int(hit_by_rank[m].sum())
        acc["ret_tot"][b] += int(m.sum())
    acc["rank0_hit"] += int(hit_by_rank[0])
    acc["rank0_tot"] += 1

    # hit rate over K (matches kernel-side definition)
    acc["hr"].append(float(hit_by_rank.sum()) / K)
    acc["nvalid_frac"].append(float(valid.size) / K)

    # miss depth: valid preIdx positions NOT in current exact topK
    topk_mask = np.zeros(N, dtype=bool)
    topk_mask[topk_pos] = True
    miss_pos = valid[~topk_mask[valid]]
    if miss_pos.size:
        depth = (thr - cur[miss_pos]) / sigma
        acc["miss_depth"].append(depth[depth >= 0].astype(np.float32))


def spearman_rho(a: np.ndarray, b: np.ndarray, rng) -> float:
    n = min(a.size, b.size)
    if n > RHO_SUBSAMPLE:
        idx = rng.choice(n, RHO_SUBSAMPLE, replace=False)
    else:
        idx = np.arange(n)
    ra = np.argsort(np.argsort(a[idx])).astype(np.float64)
    rb = np.argsort(np.argsort(b[idx])).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if d == 0:
        return 0.0
    rho_s = float((ra * rb).sum() / d)
    # Spearman -> Pearson rho of the Gaussian copula
    return float(np.clip(2.0 * np.sin(np.pi / 6.0 * rho_s), -0.999, 0.999))


def new_acc():
    return dict(ret_hit=np.zeros(N_RET_BUCKETS, dtype=np.int64),
                ret_tot=np.zeros(N_RET_BUCKETS, dtype=np.int64),
                rank0_hit=0, rank0_tot=0,
                hr=[], nvalid_frac=[], miss_depth=[], rho=[])


def finalize_layer(rec_marg: dict, acc: dict, rng) -> dict:
    ret = np.where(acc["ret_tot"] > 0,
                   acc["ret_hit"] / np.maximum(acc["ret_tot"], 1), 0.0)
    md = (np.concatenate(acc["miss_depth"]) if acc["miss_depth"]
          else np.array([0.05], dtype=np.float32))
    if md.size > MISS_DEPTH_CAP:
        md = md[rng.choice(md.size, MISS_DEPTH_CAP, replace=False)]
    out = dict(rec_marg)
    out.update(
        ret_vals=ret.astype(np.float32),
        ret_rank0=float(acc["rank0_hit"] / max(acc["rank0_tot"], 1)),
        miss_depth=np.sort(md).astype(np.float32),
        hr=np.array(acc["hr"], dtype=np.float32),
        nvalid_frac=np.array(acc["nvalid_frac"], dtype=np.float32),
        rho=float(np.mean(acc["rho"])) if acc["rho"] else 0.95,
    )
    return out


# ---------------- V3.2 ----------------

def calibrate_v32():
    cfg = MODELS["v32"]
    K = cfg["K"]
    rng = np.random.default_rng(0)
    layers = {}
    for L in cfg["layers"]:
        t0 = time.time()
        a = np.load(V32_DIR / f"Layer_{L}_pd.npy", mmap_mode="r")   # [2025, 70690]
        R = a.shape[0]
        lo_row = int(R * 0.72)

        def valid_row(r):
            row = np.ascontiguousarray(a[r]).astype(np.float32)
            nz = np.nonzero(row)[0]
            Lv = int(nz[-1] + 1) if nz.size else row.size
            return row[:Lv]

        pool_rows = np.unique(np.linspace(lo_row, R - 1, POOL_STEPS).astype(int))
        pooled = np.concatenate([valid_row(r) for r in pool_rows])
        marg = marginal_assets(pooled)

        acc = new_acc()
        pair_rows = np.unique(np.linspace(lo_row, R - 2, PAIR_STEPS).astype(int))
        for r in pair_rows:
            prev, cur = valid_row(r), valid_row(r + 1)
            n = min(prev.size, cur.size)
            prev_topk = np.argpartition(-prev, K)[:K]      # positions, prev coords
            # production preIdx == prev-step topK (GVR closed loop)
            temporal_accumulate(cur, prev_topk[prev_topk < cur.size], K, acc)
            acc["rho"].append(spearman_rho(prev[:n], cur[:n], rng))
        layers[L] = finalize_layer(marg, acc, rng)
        print(f"  v32 L{L:2d}: pool={marg['n_pooled']} "
              f"hr={np.mean(layers[L]['hr']):.3f} rho={layers[L]['rho']:.3f} "
              f"gpd(xi={marg['gpd'][2]:.3f}) [{time.time()-t0:.1f}s]")
    return layers


# ---------------- V4 (Flash / Pro via Q9j) ----------------

def calibrate_q9j(cap_dir: str, K: int):
    sys.path.insert(0, Q9J_SRC)
    from q9j_load import Q9jCapture
    cap = Q9jCapture(cap_dir)
    rng = np.random.default_rng(0)
    steps = list(cap.steady_state_steps)
    pool_sel = np.unique(np.linspace(steps[2], steps[-2], POOL_STEPS).astype(int))
    pair_sel = np.unique(np.linspace(steps[2], steps[-3], PAIR_STEPS).astype(int))
    layers = {}
    for L in cap.layers:
        t0 = time.time()
        pooled_parts = []
        for s in pool_sel:
            try:
                pooled_parts.append(cap[(int(L), int(s))].logits.float().numpy())
            except (KeyError, RuntimeError):
                continue
        marg = marginal_assets(np.concatenate(pooled_parts))

        acc = new_acc()
        for s in pair_sel:
            try:
                cell = cap[(int(L), int(s))]
                prev_cell = cap[(int(L), int(s - 1))]
            except (KeyError, RuntimeError):
                continue
            cur = cell.logits.float().numpy()
            # REAL production seed: captured preidx.in of this step
            temporal_accumulate(cur, cell.preidx.numpy().astype(np.int64), K, acc)
            prev = prev_cell.logits.float().numpy()
            n = min(prev.size, cur.size)
            acc["rho"].append(spearman_rho(prev[:n], cur[:n], rng))
        layers[int(L)] = finalize_layer(marg, acc, rng)
        print(f"  L{L:2d}: pool={marg['n_pooled']} "
              f"hr={np.mean(layers[int(L)]['hr']):.3f} "
              f"nvalid={np.mean(layers[int(L)]['nvalid_frac']):.3f} "
              f"rho={layers[int(L)]['rho']:.3f} "
              f"gpd(xi={marg['gpd'][2]:.3f}) [{time.time()-t0:.1f}s]")
    return layers


# ---------------- assemble & save ----------------

def bucket_split(layers: dict) -> dict:
    """Tercile by pooled mean (desc): shallow = highest-mean third, then
    moderate, deep — same semantic as the legacy skills' beta buckets."""
    order = sorted(layers, key=lambda L: -layers[L]["mean"])
    n = len(order)
    a, b = (n + 2) // 3, (2 * n + 1) // 3
    return {"beta_shallow": order[:a],
            "beta_moderate": order[a:b],
            "beta_deep": order[b:]}


def save_assets(model: str, layers: dict, outdir: Path):
    cfg = MODELS[model]
    buckets = bucket_split(layers)
    # separate validation pool (not needed at synth time; not shipped w/ synth)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / f"valpool_{model}.npz",
        **{f"L{L}__pool": rec.pop("valpool") for L, rec in layers.items()})
    meta = dict(
        model=model, K=cfg["K"], compress_ratio=cfg["cr"],
        layers=sorted(layers), buckets=buckets,
        source=cfg["source"],
        pool_steps=POOL_STEPS, pair_steps=PAIR_STEPS,
        n_ret_buckets=N_RET_BUCKETS,
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        note=("Empirical inverse-CDF + GPD tail + rank-conditional temporal "
              "calibration from real 64K production captures. Supersedes the "
              "moment-matched single-Beta cfgs of swebench-temporal-synth*."),
    )
    flat = {"meta_json": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
            "p_grid": P_GRID, "ret_edges": RET_EDGES}
    for L, rec in layers.items():
        pre = f"L{L}__"
        flat[pre + "q"] = rec["q"]
        flat[pre + "gpd"] = rec["gpd"]
        flat[pre + "stats"] = np.array(
            [rec["mean"], rec["std"], rec["min"], rec["max"]], dtype=np.float64)
        flat[pre + "ret_vals"] = rec["ret_vals"]
        flat[pre + "ret_rank0"] = np.float64(rec["ret_rank0"])
        flat[pre + "miss_depth"] = rec["miss_depth"]
        flat[pre + "hr"] = rec["hr"]
        flat[pre + "nvalid_frac"] = rec["nvalid_frac"]
        flat[pre + "rho"] = np.float64(rec["rho"])
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"calib_{model}.npz"
    np.savez_compressed(path, **flat)
    print(f"  -> {path} ({path.stat().st_size/1e6:.2f} MB)")


def main():
    global V32_DIR, FLASH_CAP, PRO_CAP, Q9J_SRC
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="v32,v4flash,v4pro")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--v32_dir", default=str(V32_DIR),
                    help="V3.2 Layer_{L}_pd.npy dir (default: canonical NFS)")
    ap.add_argument("--flash_cap", default=FLASH_CAP,
                    help="V4-Flash Q9j capture dir (default: canonical NFS)")
    ap.add_argument("--pro_cap", default=PRO_CAP,
                    help="V4-Pro Q9j capture dir (default: canonical NFS)")
    ap.add_argument("--q9j_src", default=Q9J_SRC,
                    help="dir containing q9j_load.py (default: canonical NFS)")
    args = ap.parse_args()
    V32_DIR = Path(args.v32_dir)
    FLASH_CAP, PRO_CAP, Q9J_SRC = args.flash_cap, args.pro_cap, args.q9j_src
    MODELS["v32"]["source"] = str(V32_DIR)
    MODELS["v4flash"]["source"] = FLASH_CAP
    MODELS["v4pro"]["source"] = PRO_CAP
    outdir = Path(args.outdir)
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        print(f"=== {m} ===")
        if m == "v32":
            layers = calibrate_v32()
        elif m == "v4flash":
            layers = calibrate_q9j(FLASH_CAP, MODELS["v4flash"]["K"])
        elif m == "v4pro":
            layers = calibrate_q9j(PRO_CAP, MODELS["v4pro"]["K"])
        else:
            raise SystemExit(f"unknown model {m}")
        save_assets(m, layers, outdir)
    print("DONE")


if __name__ == "__main__":
    main()
