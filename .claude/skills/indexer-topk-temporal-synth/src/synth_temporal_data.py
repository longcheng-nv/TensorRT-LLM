#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
r"""Unified DSv3.2 / DSv4-Flash / DSv4-Pro temporally-coherent synthetic
indexer decode logits + preIdx generator.

Supersedes swebench-temporal-synth, swebench-temporal-synth-v4flash and
swebench-temporal-synth-v4pro.  Two structural upgrades over those skills:

MARGINAL — empirical inverse-CDF + GPD tail (replaces moment-matched Beta):
  Each logit is drawn by inverting a per-layer empirical quantile table
  extracted from the real 64K production captures, with a peaks-over-threshold
  GPD extension for quantiles deeper than the empirical resolution.  This
  reproduces the real heavy positive tail that the single Beta flattened
  (synth_vs_real_validation: synth mass at the real top-K boundary was
  0.00x at N>=128K; empirical resampling makes it ~1.0x by construction).
  Rows can additionally mix over the real per-layer family ("aggregate"
  mode) — the marginal the kernel actually sees across the whole model —
  fixing the single-beta_moderate unrepresentativeness (KS 0.19-0.20 for
  Flash/V3.2).

TEMPORAL — rank-conditional construction (replaces iid-Gaussian-noise + c
  binary search):
  preIdx is built directly from the real-calibrated statistics:
    * retention-by-rank curve  P(pos in preIdx | current topK rank bucket)
      (head ranks are near-certain, boundary ranks churn — as measured);
    * per-step hit-rate DISTRIBUTION (sampled per row, not a fixed scalar);
    * miss positions placed at real-measured depths below the selection
      threshold ((thr - logit)/sigma samples -> boundary-band with the real
      deep-tail component);
    * V4 undershoot sentinels: n_valid < K sampled from the real
      valid-count distribution (GVR non-convergence; -1 tail slots).
  Optional --steps T emits a Gaussian-copula AR(1) chain (lag-1 rho fit from
  real consecutive decode steps) with the exact closed-loop invariant
  preIdx_t = topK(row_{t-1}).

Kernel contracts preserved per model (verbatim from the legacy skills):
  v32     : K=2048 cr=1  preIdx = pos-1 (kernel +1), pad align 4, seq_lens=N
  v4flash : K=512  cr=4  preIdx = pos   (offset 0),  pad align 8,
            seq_lens = N*cr + next_n - 1;  radix_aux_* pre-alloc (post-#14297)
  v4pro   : K=1024 cr=4  same as v4flash

Outputs per (cfg, N, BS) — same layout as the legacy skills:
  {cfg}_N{N}_bs{BS}/
    logits.pt    [BS, N_padded] (--dtype fp32|bf16|fp16), -inf padded
    preIdx.pt    [BS, K] int32
    seq_lens.pt  [BS] int32
    meta.json    model, per-row layer/hit-rate/n_valid, invariants
  (--steps T adds step{t}_logits.pt / step{t}_preIdx.pt for t = 1..T-1)

Usage:
  python3 synth_temporal_data.py --model v4flash --N 64K --cfg aggregate \
      --bs 8 --outdir /tmp/synth_out
  python3 synth_temporal_data.py --model v32 --N 128K --cfg beta_moderate \
      --bs 1 --outdir /tmp/synth_v32   # legacy bucket names still work
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"

MODEL_CONTRACT = {
    "v32": dict(K=2048, compress_ratio=1, preidx_caller_offset=-1,
                pad_align=4, seqlens="N", radix_aux=False),
    "v4flash": dict(K=512, compress_ratio=4, preidx_caller_offset=0,
                    pad_align=8, seqlens="N*cr+next_n-1", radix_aux=True),
    "v4pro": dict(K=1024, compress_ratio=4, preidx_caller_offset=0,
                  pad_align=8, seqlens="N*cr+next_n-1", radix_aux=True),
}

NEXT_N = 1


# ---------------- calibration assets ----------------

class Calib:
    def __init__(self, model: str, assets_dir: Path = ASSETS):
        path = assets_dir / f"calib_{model}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run calibrate_from_real.py first")
        z = np.load(path)
        self.meta = json.loads(bytes(z["meta_json"]).decode())
        self.p_grid = z["p_grid"]
        self.ret_edges = z["ret_edges"]
        self.layers = list(self.meta["layers"])
        self.buckets = {k: list(v) for k, v in self.meta["buckets"].items()}
        self.L = {}
        for L in self.layers:
            pre = f"L{L}__"
            self.L[L] = dict(
                q=z[pre + "q"].astype(np.float64),
                gpd=z[pre + "gpd"],
                stats=z[pre + "stats"],
                ret_vals=z[pre + "ret_vals"].astype(np.float64),
                ret_rank0=float(z[pre + "ret_rank0"]),
                miss_depth=z[pre + "miss_depth"].astype(np.float64),
                hr=z[pre + "hr"].astype(np.float64),
                nvalid_frac=z[pre + "nvalid_frac"].astype(np.float64),
                rho=float(z[pre + "rho"]),
            )
        self.pos = _PosCalib(model, assets_dir)       # positional / gather model

    def layer_pool(self, cfg: str):
        """cfg -> list of layers a row may sample its marginal from."""
        if cfg == "aggregate":
            return self.layers
        if cfg in self.buckets:
            return self.buckets[cfg]
        if cfg.upper().startswith("L") and cfg[1:].isdigit():
            L = int(cfg[1:])
            if L not in self.L:
                raise ValueError(f"layer {L} not calibrated; have {self.layers}")
            return [L]
        raise ValueError(
            f"unknown cfg '{cfg}': use aggregate | beta_shallow | "
            f"beta_moderate | beta_deep | L<layer> | all")


def inv_cdf(u: np.ndarray, rec: dict) -> np.ndarray:
    """Empirical inverse CDF with GPD upper-tail extrapolation."""
    p_grid, q = rec["_pgrid"], rec["q"]
    x = np.interp(u, p_grid, q)
    p_u, u_thr, xi, beta = rec["gpd"]
    tail = u > p_u
    if tail.any():
        qt = (1.0 - u[tail]) / max(1.0 - p_u, 1e-12)   # exceedance prob
        qt = np.maximum(qt, 1e-12)
        if abs(xi) < 1e-6:
            xv = u_thr - beta * np.log(qt)
        else:
            xv = u_thr + beta / xi * (qt ** (-xi) - 1.0)
        # sanity cap: don't extrapolate absurdly past the observed max
        cap = rec["stats"][3] + 5.0 * max(rec["stats"][3] - u_thr, beta)
        x[tail] = np.minimum(xv, cap)
    return x


# ---------------- positional / gather model (Part-3) ----------------
# Assigns logit VALUES to POSITIONS so the top-K (hence preIdx) is spatially
# clustered + recency/sink-shaped as measured in the real captures, instead of
# uniform-random. Marginal-preserving (a permutation) so every value-distribution
# gate is untouched. Calibration: assets/posz_<model>.npz (calib_positional.py).
# Disable with SYNTH_POSITIONAL=0 (legacy IID placement).
_POS_ENABLED = os.environ.get("SYNTH_POSITIONAL", "1") == "1"
_RHO_CACHE = {}


class _PosCalib:
    """Per-(model,layer) positional record; absent file -> {} (IID fallback)."""

    def __init__(self, model: str, assets_dir: Path):
        self.L = {}
        p = assets_dir / f"posz_{model}.npz"
        if not p.exists():
            return
        z = np.load(p, allow_pickle=True)
        for L in list(z["layers"]):
            t = z[f"L{L}__targets"]
            self.L[int(L)] = dict(
                mu_norm=z[f"L{L}__mu_norm"].astype(np.float64),
                frac_adj=float(t[0]))


def _ar1_field(N: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    if rho <= 0:
        return rng.standard_normal(N)
    a = math.sqrt(1.0 - rho * rho)
    eps = rng.standard_normal(N)
    eps[0] = eps[0] / max(a, 1e-9)
    try:
        from scipy.signal import lfilter
        return lfilter([a], [1.0, -rho], eps)
    except ImportError:
        c = np.empty(N)
        c[0] = eps[0] * a
        for i in range(1, N):
            c[i] = rho * c[i - 1] + a * eps[i]
        return c


def _logmu(N: int, mu: np.ndarray) -> np.ndarray:
    idx = np.minimum((np.arange(N) / N * mu.size).astype(int), mu.size - 1)
    return np.log(np.maximum(mu[idx], 1e-6))


def _tune_rho(N, K, mu, frac_adj_target, rng,
              grid=(0.0, 0.9, 0.97, 0.99, 0.995, 0.998, 0.999)) -> float:
    key = (N, K, round(frac_adj_target, 3))
    if key in _RHO_CACHE:
        return _RHO_CACHE[key]
    lm = _logmu(N, mu)
    best, be = 0.0, 1e9
    for rho in grid:
        s = _ar1_field(N, rho, rng) + 0.15 * lm
        fa = float((np.diff(np.sort(np.argpartition(-s, K)[:K])) <= 2).mean())
        if abs(fa - frac_adj_target) < be:
            best, be = rho, abs(fa - frac_adj_target)
    _RHO_CACHE[key] = best
    return best


def _positional_order(N, K, pos_rec, rng) -> np.ndarray:
    """Position ranking (high->low score) from real mu_L + AR(1) clustering."""
    mu = pos_rec["mu_norm"]
    rho = _tune_rho(N, K, mu, pos_rec["frac_adj"], rng)
    s = _ar1_field(N, rho, rng) + 1.0 * _logmu(N, mu)
    return np.argsort(-s)


def _assign_by_order(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    x = np.empty(order.size, dtype=np.float32)
    x[order] = np.sort(values)[::-1]                  # largest value -> top score
    return x


# ---------------- per-row synthesis ----------------

def _incl_probs(w: np.ndarray, n: float) -> np.ndarray:
    """Waterfilling: p = min(s*w, 1) with sum(p) = n.  Keeps per-rank
    inclusion probabilities PROPORTIONAL to the retention curve (up to
    clipping at 1) — weighted choice-without-replacement does not."""
    n = min(max(n, 0.0), float(w.size))
    if n <= 0:
        return np.zeros_like(w)
    lo, hi = 0.0, n / max(w.sum(), 1e-12)
    while np.minimum(hi * w, 1.0).sum() < n and hi < 1e12:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.minimum(mid * w, 1.0).sum() < n:
            lo = mid
        else:
            hi = mid
    return np.minimum(hi * w, 1.0)


def _select_hits(K: int, n_hit: int, ret_vals: np.ndarray, ret_edges: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Choose n_hit current-topK ranks whose inclusion probabilities follow
    the real retention-by-rank curve; rank 0 (argmax) always retained."""
    if n_hit <= 0:
        return np.array([0], dtype=np.int64)          # argmax invariant
    nb = ret_vals.size
    bucket = np.minimum((np.arange(K) / K * nb).astype(int), nb - 1)
    w = np.maximum(ret_vals[bucket], 1e-4)
    p = _incl_probs(w, float(n_hit))
    keep = rng.random(K) < p
    keep[0] = True
    # exact-count correction (uniform among flippable slots -> curve shape
    # is scaled, not distorted)
    excess = int(keep.sum()) - n_hit
    if excess > 0:
        cand = np.nonzero(keep)[0][1:]                # never drop rank 0
        drop = rng.choice(cand, size=min(excess, cand.size), replace=False)
        keep[drop] = False
    elif excess < 0:
        cand = np.nonzero(~keep)[0]
        pc = p[cand]
        pc = pc / pc.sum() if pc.sum() > 0 else None
        add = rng.choice(cand, size=min(-excess, cand.size), replace=False,
                         p=pc)
        keep[add] = True
    return np.nonzero(keep)[0]                        # kept ranks


def _place_misses(x: np.ndarray, order: np.ndarray, K: int, n_miss: int,
                  thr: float, sigma: float, depth_samples: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Positions for preIdx entries outside current topK, at real-measured
    value depths below the selection threshold."""
    if n_miss <= 0:
        return np.empty(0, dtype=np.int64)
    N = x.size
    depths = rng.choice(depth_samples, size=n_miss)
    target = thr - depths * sigma
    x_sorted_desc = x[order]
    ranks = np.searchsorted(-x_sorted_desc, -target)  # descending search
    ranks = np.clip(ranks, K, N - 1)
    # resolve duplicate ranks -> next unused
    used = set()
    out = np.empty(n_miss, dtype=np.int64)
    for i, r in enumerate(np.sort(ranks)):
        r = int(r)
        while r in used and r < N - 1:
            r += 1
        used.add(r)
        out[i] = r
    return order[out]


def synth_row(N: int, K: int, calib: Calib, layer: int,
              rng: np.random.Generator,
              target_hr: float | None, sentinel_mode: str):
    """One (logits row, preIdx positions[, prev pseudo-order]) draw.
    Returns (x fp32[N], pre_pos int64[K] with -1 sentinels, row_meta)."""
    rec = dict(calib.L[layer])
    rec["_pgrid"] = calib.p_grid

    values = inv_cdf(rng.random(N), rec).astype(np.float32)
    if _POS_ENABLED and layer in calib.pos.L:          # cluster values -> real gather
        x = _assign_by_order(values, _positional_order(N, K, calib.pos.L[layer], rng))
    else:
        x = values                                     # legacy IID placement

    order = np.argsort(-x, kind="stable").astype(np.int64)
    topk_pos = order[:K]
    thr = float(x[order[K - 1]])
    sigma = float(x.std()) or 1.0

    hr = float(target_hr if target_hr is not None else rng.choice(rec["hr"]))
    hr = min(max(hr, 0.0), 1.0)
    if sentinel_mode == "real" and rec["nvalid_frac"].size:
        n_valid = int(round(float(rng.choice(rec["nvalid_frac"])) * K))
    else:
        n_valid = K
    n_valid = max(min(n_valid, K), 1)
    n_hit = min(int(round(hr * K)), n_valid)
    n_miss = n_valid - n_hit

    kept_ranks = _select_hits(K, n_hit, rec["ret_vals"], calib.ret_edges, rng)
    hit_pos = topk_pos[kept_ranks]
    miss_pos = _place_misses(x, order, K, n_miss, thr, sigma,
                             rec["miss_depth"], rng)

    # pseudo prev-step ordering: hits keep their value; misses were above the
    # *previous* threshold, so rank them just above thr with small jitter.
    pos = np.concatenate([hit_pos, miss_pos])
    pv = np.concatenate([
        x[hit_pos].astype(np.float64),
        thr + np.abs(rng.normal(0.0, 0.05 * sigma, size=miss_pos.size)),
    ])
    pos = pos[np.argsort(-pv, kind="stable")]

    pre_pos = np.full(K, -1, dtype=np.int64)
    pre_pos[:pos.size] = pos

    realised_hr = float(np.isin(pos, topk_pos).sum()) / K
    row_meta = dict(layer=int(layer), target_hr=hr, realised_hr=realised_hr,
                    n_valid=int(pos.size), thr=thr,
                    row_mean=float(x.mean()), row_std=sigma,
                    row_min=float(x.min()), row_max=float(x.max()))
    return x, pre_pos, row_meta


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _norm_cdf_fast(z: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf as _erf          # optional fast path
        return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    except ImportError:
        try:
            import torch
            return (0.5 * (1.0 + torch.erf(
                torch.from_numpy(z / math.sqrt(2.0))))).numpy()
        except ImportError:
            return _norm_cdf(z)


def synth_chain(N: int, K: int, calib: Calib, layer: int, steps: int,
                rng: np.random.Generator):
    """Gaussian-copula AR(1) chain of `steps` rows with the exact GVR
    closed-loop invariant preIdx_t = topK(row_{t-1})."""
    rec = dict(calib.L[layer])
    rec["_pgrid"] = calib.p_grid
    rho = rec["rho"]
    z = rng.standard_normal(N)
    rows, pres = [], []
    prev_topk = None
    # shared positional order across steps: consistent spatial clustering while
    # the copula z preserves temporal value-correlation (marginal-preserving).
    pos_order = (_positional_order(N, K, calib.pos.L[layer], rng)
                 if _POS_ENABLED and layer in calib.pos.L else None)
    for _ in range(steps):
        values = inv_cdf(np.clip(_norm_cdf_fast(z), 1e-12, 1 - 1e-12),
                         rec).astype(np.float32)
        x = _assign_by_order(values, pos_order) if pos_order is not None else values
        order = np.argsort(-x, kind="stable").astype(np.int64)
        rows.append(x)
        pres.append(prev_topk)
        prev_topk = order[:K].copy()
        z = rho * z + math.sqrt(max(1.0 - rho * rho, 0.0)) * \
            rng.standard_normal(N)
    return rows, pres, rho


# ---------------- bundle assembly ----------------

def _pad_align_inf_np(x: np.ndarray, align: int) -> np.ndarray:
    r = x.size % align
    if r == 0:
        return x
    return np.concatenate([x, np.full(align - r, -np.inf, dtype=x.dtype)])


def synthesize(model: str, N: int, BS: int, cfg: str,
               K: int | None = None, target_hr: float | None = None,
               seed: int = 42, row_mode: str = "independent",
               sentinel_mode: str = "real", steps: int = 1,
               dtype: str = "fp32", assets_dir: Path = ASSETS):
    import torch
    contract = MODEL_CONTRACT[model]
    K = K or contract["K"]
    cr = contract["compress_ratio"]
    if N <= 2 * K:
        raise ValueError(f"N={N} too small: requires N > 2*K = {2 * K}")
    if model == "v32":
        sentinel_mode = "full"          # V3.2 capture has no preidx stream

    calib = Calib(model, assets_dir)
    pool = calib.layer_pool(cfg)
    rng = np.random.default_rng(seed)
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16,
                   "fp16": torch.float16}[dtype]

    n_rows = 1 if row_mode == "replicate" else BS
    rows, pre_rows, row_metas = [], [], []
    chain_extra = None

    for b in range(n_rows):
        layer = int(pool[rng.integers(len(pool))])
        if steps > 1:
            xs, ps, rho = synth_chain(N, K, calib, layer, steps + 1,
                                      np.random.default_rng(seed + 7919 * b))
            # step 0 of the chain has no prev -> use single-step construction
            x0, pre0, m0 = synth_row(N, K, calib, layer,
                                     np.random.default_rng(seed + 7919 * b),
                                     target_hr, sentinel_mode)
            m0["chain_rho"] = rho
            rows.append(xs[1])
            pre_rows.append(ps[1])      # = topK(xs[0]) exact closed loop
            row_metas.append(m0)
            if b == 0:
                chain_extra = (xs, ps)
        else:
            x, pre, m = synth_row(N, K, calib, layer,
                                  np.random.default_rng(seed + 7919 * b),
                                  target_hr, sentinel_mode)
            rows.append(x)
            pre_rows.append(pre)
            row_metas.append(m)

    align = contract["pad_align"]
    off = contract["preidx_caller_offset"]

    def to_preidx(pre_pos: np.ndarray) -> np.ndarray:
        out = pre_pos.copy()
        valid = out >= 0
        out[valid] = out[valid] + off      # v32: kernel adds +1 back
        return out.astype(np.int32)

    logits_np = np.stack([_pad_align_inf_np(r, align) for r in rows])
    pre_np = np.stack([to_preidx(p) for p in pre_rows])
    if row_mode == "replicate":
        logits_np = np.repeat(logits_np, BS, axis=0)
        pre_np = np.repeat(pre_np, BS, axis=0)

    logits = torch.from_numpy(logits_np).to(torch_dtype).contiguous()
    preIdx = torch.from_numpy(pre_np).contiguous()
    seq_val = N if contract["seqlens"] == "N" else N * cr + NEXT_N - 1
    seq_lens = torch.full((BS,), seq_val, dtype=torch.int32)

    hrs = [m["realised_hr"] for m in row_metas]
    meta = {
        "skill": "indexer-topk-temporal-synth",
        "model": model,
        "cfg": cfg,
        "layer_pool": pool,
        "N": N, "Npad": int(logits.shape[1]), "BS": BS, "K": K,
        "compress_ratio": cr, "next_n": NEXT_N, "seq_lens_val": int(seq_val),
        "seed": seed, "dtype": dtype,
        "row_mode": row_mode, "sentinel_mode": sentinel_mode, "steps": steps,
        "preidx_caller_offset": off,
        "logit_alignment": align,
        "radix_aux_required": contract["radix_aux"],
        "marginal": "empirical inverse-CDF + GPD tail (real 64K captures)",
        "temporal": ("rank-conditional retention + real miss-depth + "
                     "sampled hit-rate" + (" + copula AR(1) chain"
                                           if steps > 1 else "")),
        "realised_hr_mean": float(np.mean(hrs)),
        "realised_hr_std": float(np.std(hrs)),
        "rows": row_metas,
        "calib_meta": calib.meta,
    }
    bundle = {"logits": logits, "preIdx": preIdx, "seq_lens": seq_lens,
              "meta": meta}
    if steps > 1 and chain_extra is not None:
        xs, ps = chain_extra
        bundle["chain"] = [
            {"logits": torch.from_numpy(
                _pad_align_inf_np(xs[t], align)).to(torch_dtype),
             "preIdx": torch.from_numpy(to_preidx(
                 ps[t] if ps[t] is not None else np.full(K, -1, np.int64)))}
            for t in range(1, len(xs))
        ]
    return bundle


def save(out_subdir: str, bundle: dict):
    import torch
    os.makedirs(out_subdir, exist_ok=True)
    torch.save(bundle["logits"].cpu(), os.path.join(out_subdir, "logits.pt"))
    torch.save(bundle["preIdx"].cpu(), os.path.join(out_subdir, "preIdx.pt"))
    torch.save(bundle["seq_lens"].cpu(), os.path.join(out_subdir, "seq_lens.pt"))
    if "chain" in bundle:
        for t, st in enumerate(bundle["chain"], start=1):
            torch.save(st["logits"].cpu(),
                       os.path.join(out_subdir, f"step{t}_logits.pt"))
            torch.save(st["preIdx"].cpu(),
                       os.path.join(out_subdir, f"step{t}_preIdx.pt"))
    with open(os.path.join(out_subdir, "meta.json"), "w") as f:
        meta = dict(bundle["meta"])
        meta["rows"] = meta["rows"][:32]      # cap row detail for huge BS
        json.dump(meta, f, indent=2)


def parse_n(arg: str) -> int:
    s = arg.upper().strip()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODEL_CONTRACT))
    p.add_argument("--N", type=parse_n, default=65536,
                   help="post-compress seq len; accepts '64K' etc.")
    p.add_argument("--cfg", default="aggregate",
                   help="aggregate (default; real per-layer mixture) | "
                        "beta_shallow|beta_moderate|beta_deep (legacy bucket "
                        "names, now real-layer terciles) | L<layer> | all")
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--K", type=int, default=None,
                   help="override model-native K (ablation only)")
    p.add_argument("--target_hr", type=float, default=None,
                   help="fixed hit-rate; default samples the real per-step "
                        "hit-rate distribution")
    p.add_argument("--row_mode", choices=["independent", "replicate"],
                   default="independent",
                   help="independent (default): each row draws its own layer "
                        "+ sample; replicate: legacy single-row broadcast")
    p.add_argument("--sentinel_mode", choices=["real", "full"], default="real",
                   help="real (default): V4 n_valid<K undershoot sentinels "
                        "sampled from capture; full: always K valid entries")
    p.add_argument("--steps", type=int, default=1,
                   help=">1: copula AR(1) chain; emits step{t}_* tensors")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    cfgs = (["aggregate", "beta_shallow", "beta_moderate", "beta_deep"]
            if args.cfg == "all" else [args.cfg])
    os.makedirs(args.outdir, exist_ok=True)
    for cfg in cfgs:
        t0 = time.time()
        bundle = synthesize(args.model, args.N, args.bs, cfg, K=args.K,
                            target_hr=args.target_hr, seed=args.seed,
                            row_mode=args.row_mode,
                            sentinel_mode=args.sentinel_mode,
                            steps=args.steps, dtype=args.dtype)
        sub = os.path.join(args.outdir,
                           f"{cfg}_N{args.N}_bs{args.bs}")
        save(sub, bundle)
        m = bundle["meta"]
        r0 = m["rows"][0]
        print(f"=== {args.model} {cfg} N={args.N} BS={args.bs} K={m['K']} ===")
        print(f"  rows: {len(m['rows'])} layer(s) e.g. L{r0['layer']}; "
              f"row0 mean={r0['row_mean']:.3f} std={r0['row_std']:.3f} "
              f"[{r0['row_min']:.2f}, {r0['row_max']:.2f}]")
        print(f"  hit-rate realised mean={m['realised_hr_mean']:.3f} "
              f"std={m['realised_hr_std']:.3f}  "
              f"n_valid(row0)={r0['n_valid']}/{m['K']}")
        print(f"  [{time.time()-t0:.2f}s] -> {sub}/")


if __name__ == "__main__":
    main()
