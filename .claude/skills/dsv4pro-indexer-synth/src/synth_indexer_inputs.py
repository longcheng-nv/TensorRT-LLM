"""
synth_indexer_inputs.py  —  DSV4 indexer input data synthesizer

Generates (Q, K_cache, weights, logits, topK, preIdx) tensors that match
the statistical distribution of real DSV4 Pro/Flash indexer inputs captured
from SWE-bench workloads, with controllable temporal hit-rate (GVR behavior).

Algorithm:
  1. Logits-First: sample per-step logits from the fitted per-layer distribution
     (Gumbel_r for shallow layers, Normal for deep layers).
  2. Rank-Transform: map random Q/K inner-product onto logits target distribution.
  3. Temporal Bias: apply binary-searched coefficient c to prev_topK positions
     AFTER rank-transform to achieve the target GVR hit-rate exactly.
  4. FP4 Quantize: pack Q and K into FP4 e2m1 int8 format (2 values per byte).

Usage:
    python3 synth_indexer_inputs.py \\
        --model   pro|flash                 # required
        --bs      1                         # batch size (default 1)
        --isl     65536                     # input sequence length (tokens)
        --osl     500                       # decode steps to synthesize
        --target-hr 0.69                    # GVR hit-rate; 0 = no temporal bias
        --layers  all|even|2,12,20          # default: even (GVR-active)
        --format  pt|npz                    # default pt
        --out-dir /path/to/output           # required
        --seed    42
        --params  /path/to/custom_params.json  # override built-in params
        --no-k-cache                        # skip K-cache (saves memory)
        --no-logits                         # skip logits (saves memory)

Output files (per-layer layout):
    <out-dir>/
      manifest.json
      layer_XX/
        q_fp4.pt        {step: Tensor [BS,1,n_heads,head_dim//2] int8}
        k_cache.pt      {step: Tensor [n_blocks,tpb,1,bytes_per_token] int8}
        weights.pt      {step: Tensor [BS,n_heads] float32}
        logits.pt       {step: Tensor [BS,kv_len] float32}
        topk.pt         {step: Tensor [BS,K] int32}
        preidx.pt       {step: Tensor [BS,K] int32}   ← prev step's topK (GVR input)

Measured Pro distribution (SWE-bench 64K, B300, 2026-06-05):
    Q:       N(0.00, 1.93) FP4 packed   [all 30 layers identical]
    K:       N(0.08, 2.32) FP4 packed   [all 30 layers identical]
    Weights: N(0.02, 0.05) fp32         [per-layer params available]
    Logits:  Gumbel_r(mu≈-1.4, s≈0.59) shallow L02-L10
             Normal(mu≈-1.1, s≈0.95)   deep    L12-L60
    Hit-rate: 0.686 mean (L02=0.757, L12=0.597)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── FP4 helpers ──────────────────────────────────────────────────────────────

FP4_LUT = np.array(
    [0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.],
    dtype=np.float32,
)


def fp4_quantize(arr: np.ndarray) -> np.ndarray:
    """Quantize float32 → FP4 e2m1 packed int8 (2 values per byte).
    Input shape (..., N) where N is even → output (..., N//2) int8.
    """
    x = arr.reshape(-1).astype(np.float32)
    diffs = np.abs(x[:, None] - FP4_LUT[None, :])
    idx = diffs.argmin(axis=1).astype(np.uint8)
    lo, hi = idx[0::2], idx[1::2]
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return packed.view(np.int8).reshape(arr.shape[:-1] + (arr.shape[-1] // 2,))


def fp4_dequantize(packed: np.ndarray) -> np.ndarray:
    """Unpack FP4 int8 → float32. Input (..., N//2) → output (..., N)."""
    b = packed.astype(np.uint8)
    lo, hi = b & 0x0F, (b >> 4) & 0x0F
    return np.stack([FP4_LUT[lo], FP4_LUT[hi]], axis=-1).reshape(
        packed.shape[:-1] + (packed.shape[-1] * 2,)
    ).astype(np.float32)


def fp8_quantize(arr: np.ndarray) -> np.ndarray:
    """Quantize float32 → FP8 e4m3fn (1 byte per value). Stored as int8."""
    t = torch.from_numpy(arr)
    t8 = t.to(torch.float8_e4m3fn)
    return t8.view(torch.int8).numpy()


# ── Distribution sampling ─────────────────────────────────────────────────────

def sample_logits(dist: str, mu: float, sigma: float,
                  n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n logit values from the given distribution parameterised by (mu, sigma)."""
    if dist == "norm" or not HAS_SCIPY:
        return rng.normal(mu, sigma, n).astype(np.float32)
    if dist == "gumbel_r":
        euler_gamma = 0.5772156649
        scale = sigma * math.sqrt(6) / math.pi
        loc = mu - scale * euler_gamma
        return sp_stats.gumbel_r.rvs(
            loc=loc, scale=scale, size=n,
            random_state=int(rng.integers(2**31))
        ).astype(np.float32)
    if dist == "laplace":
        scale = sigma / math.sqrt(2)
        return sp_stats.laplace.rvs(
            loc=mu, scale=scale, size=n,
            random_state=int(rng.integers(2**31))
        ).astype(np.float32)
    return rng.normal(mu, sigma, n).astype(np.float32)


def rank_transform(raw: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map raw onto target distribution by quantile matching (rank transform).
    Preserves ranking of raw; values are drawn from sorted target.
    """
    return np.sort(target)[np.argsort(np.argsort(raw))].astype(np.float32)


def binary_search_coeff(logits: np.ndarray, prev_topk: np.ndarray,
                        target_hr: float, K: int,
                        max_c: float = 20.0) -> float:
    """Binary-search coefficient c such that
    hit_rate(topK(logits + c·mask(prev_topk)), prev_topk) == target_hr.
    35 iterations → precision < 1e-9. Must be called AFTER rank_transform.
    """
    if target_hr <= 0.0 or prev_topk is None or K == 0:
        return 0.0
    lo, hi = 0.0, max_c
    mask = np.zeros_like(logits)
    mask[prev_topk] = 1.0
    for _ in range(35):
        c = (lo + hi) / 2.0
        topk_c = np.argpartition(logits + c * mask, -K)[-K:]
        hr = float(len(np.intersect1d(topk_c, prev_topk))) / K
        if hr < target_hr:
            lo = c
        else:
            hi = c
    return (lo + hi) / 2.0


# ── Layer parameter lookup ───────────────────────────────────────────────────

def get_logits_params(layer: int, params: dict) -> tuple[str, float, float]:
    """Return (dist_name, mu, sigma) for logits at this layer.
    Adds small step-level jitter to mu/sigma (sampled once per call).
    """
    lp = params["logits"]
    if layer in lp.get("shallow_layers", []):
        bucket = lp["shallow"]
    else:
        bucket = lp["deep"]
    dist = bucket["dist"]
    mu   = bucket["mu_mean"]
    sig  = bucket["sig_mean"]
    return dist, mu, sig


def get_weights_params(layer_idx: int, params: dict) -> tuple[float, float]:
    """Return (mu, sigma) for weights at this layer (0-indexed into GVR layer list)."""
    wp = params["weights"]
    per_mu  = wp.get("per_layer_mu")
    per_sig = wp.get("per_layer_sigma")
    if per_mu and per_sig and layer_idx < len(per_mu):
        return per_mu[layer_idx], per_sig[layer_idx]
    return wp["mu"], wp["sigma"]


# ── Single step synthesis ─────────────────────────────────────────────────────

def synth_one_step(
    layer: int, layer_idx: int, params: dict,
    bs: int, n_kv: int, n_heads: int, head_dim: int, K: int,
    step_mu: float, step_sig: float,
    rng: np.random.Generator,
    prev_topk: np.ndarray = None,
    target_hr: float = 0.0,
    gen_k_cache: bool = True,
    gen_logits: bool = True,
) -> dict:
    """Synthesize one decode step for one layer.

    Returns dict with: q_fp4, k_cache (opt), weights, logits (opt), topk, actual_hr.
    """
    qp = params["q"]
    kp = params["k"]

    # ── Q  [BS, 1, n_heads, head_dim] FP4 packed ─────────────────────────────
    q_f32 = np.clip(
        rng.normal(qp["mu"], qp["sigma"], (bs, 1, n_heads, head_dim)),
        qp["clip_low"], qp["clip_high"]
    ).astype(np.float32)
    q_fp4 = fp4_quantize(q_f32)                 # [BS, 1, n_heads, head_dim//2]

    # ── Weights  [BS, n_heads] float32 ───────────────────────────────────────
    w_mu, w_sig = get_weights_params(layer_idx, params)
    weights = rng.normal(w_mu, w_sig, (bs, n_heads)).astype(np.float32)

    # ── K cache  [n_blocks, tpb, 1, bytes_per_token] int8 ───────────────────
    k_cache = None
    if gen_k_cache:
        tpb = kp.get("tokens_per_block", 32)
        n_blocks = math.ceil(n_kv / tpb)
        k_f32 = np.clip(
            rng.normal(kp["mu"], kp["sigma"], (n_blocks * tpb, head_dim)),
            kp.get("clip_low", -6.0), kp.get("clip_high", 6.0)
        ).astype(np.float32)
        if kp.get("dtype", "fp4_e2m1_packed_int8").startswith("fp4"):
            k_packed = fp4_quantize(k_f32)       # [n_blocks*tpb, head_dim//2]
            # Scale bytes: 4 bytes per token (one float32 scale per FP4 block)
            scale_bytes = np.ones((n_blocks * tpb, 4), dtype=np.int8)
            k_row = np.concatenate([k_packed, scale_bytes], axis=1)  # [N, head_dim//2+4]
        else:
            k_row = fp8_quantize(k_f32)          # FP8 fallback
            scale_bytes = np.ones((n_blocks * tpb, 4), dtype=np.int8)
            k_row = np.concatenate([k_row, scale_bytes], axis=1)
        k_cache = k_row.reshape(n_blocks, tpb, 1, k_row.shape[-1])

    # ── Logits via rank-transform ─────────────────────────────────────────────
    logits = None
    topk = None
    actual_hr = 0.0

    if gen_logits or (prev_topk is not None and target_hr > 0.0):
        # Target distribution for this step
        logits_target = sample_logits(
            params["logits"]["shallow" if layer in params["logits"].get("shallow_layers",[]) else "deep"]["dist"],
            step_mu, step_sig, n_kv, rng
        )
        # Raw inner-product logits (sum_h w_h * Q_h @ K_flat.T / sqrt(d))
        # For efficiency: use random unit vectors
        scale = 1.0 / math.sqrt(head_dim)
        q_flat = q_f32[0, 0]                              # [n_heads, head_dim]
        k_flat = np.clip(
            rng.normal(kp["mu"], kp["sigma"], (n_kv, head_dim)),
            kp.get("clip_low", -6.0), kp.get("clip_high", 6.0)
        ).astype(np.float32)
        per_head = (q_flat @ k_flat.T) * scale            # [n_heads, n_kv]
        raw_logits = (weights[0] @ per_head).astype(np.float32)   # [n_kv]

        # Rank-transform raw onto target
        logits_synth = rank_transform(raw_logits, logits_target)   # [n_kv]

        # Temporal bias (MUST be after rank-transform)
        if prev_topk is not None and target_hr > 0.0:
            c = binary_search_coeff(logits_synth, prev_topk, target_hr, K)
            if c > 0.0:
                logits_synth = logits_synth.copy()
                logits_synth[prev_topk] += c

        # topK
        topk_flat = np.argpartition(logits_synth, -K)[-K:]
        topk_sorted = topk_flat[np.argsort(logits_synth[topk_flat])[::-1]]
        topk = topk_sorted.astype(np.int32)

        if prev_topk is not None:
            actual_hr = float(len(np.intersect1d(topk, prev_topk))) / K

        # Replicate to [BS, n_kv]
        if gen_logits:
            logits = np.tile(logits_synth[None, :], (bs, 1))

        # Replicate topK to [BS, K]
        topk = np.tile(topk[None, :], (bs, 1))

    return {
        "q_fp4":   torch.from_numpy(q_fp4).to(torch.int8),
        "k_cache": torch.from_numpy(k_cache) if k_cache is not None else None,
        "weights": torch.from_numpy(weights),
        "logits":  torch.from_numpy(logits) if logits is not None else None,
        "topk":    torch.from_numpy(topk) if topk is not None else None,
        "actual_hr": actual_hr,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def resolve_params(model: str, custom_path: str = None) -> dict:
    """Load distribution parameters for the given model."""
    if custom_path:
        with open(custom_path) as f:
            return json.load(f)
    skill_dir = Path(__file__).parent
    name = "pro_params.json" if "pro" in model.lower() else "flash_params.json"
    default = skill_dir / "params" / name
    if not default.is_file():
        raise FileNotFoundError(f"Built-in params not found: {default}")
    with open(default) as f:
        return json.load(f)


def resolve_layers(spec: str, params: dict) -> list[int]:
    all_gvr = params.get("gvr_active_layers", list(range(2, 62, 2)))
    if spec == "even" or spec == "all":
        return all_gvr
    return [int(x) for x in spec.split(",") if x.strip()]


def save_dict(dct: dict, path_no_ext: str, fmt: str) -> str:
    if fmt == "npz":
        path = path_no_ext + ".npz"
        np.savez(path, **{f"s{k:04d}": v.numpy() if isinstance(v, torch.Tensor) else v
                          for k, v in dct.items()})
    else:
        path = path_no_ext + ".pt"
        torch.save(dct, path)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",     required=True, help="pro | flash | <path/to/custom_params.json>")
    p.add_argument("--bs",        type=int, default=1, help="batch size (default 1)")
    p.add_argument("--isl",       type=int, default=65536, help="input sequence length in tokens (default 65536)")
    p.add_argument("--osl",       type=int, default=500,   help="decode steps to synthesize (default 500)")
    p.add_argument("--target-hr", type=float, default=-1.0,
                   help="GVR hit-rate per step. -1 = use model default (Pro=0.69, Flash=0.40). 0 = disable.")
    p.add_argument("--layers",    default="even", help="even | all | comma-list (default even)")
    p.add_argument("--format",    choices=("pt", "npz"), default="pt")
    p.add_argument("--out-dir",   required=True)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--params",    default="", help="path to custom params JSON (overrides built-in)")
    p.add_argument("--no-k-cache",  action="store_true", help="skip K-cache synthesis")
    p.add_argument("--no-logits",   action="store_true", help="skip logits (forces no topK/preIdx)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load params
    if os.path.isfile(args.model):
        params = resolve_params("custom", args.model)
    else:
        params = resolve_params(args.model, args.params or None)

    K        = params["K"]
    n_heads  = params["n_heads"]
    head_dim = params["head_dim_fp4"]
    cr       = params["compress_ratio"]
    tpb      = params.get("k", {}).get("tokens_per_block", 32)

    # n_kv: compressed KV slots, aligned to 64
    n_kv = math.ceil(args.isl / cr)
    n_kv = ((n_kv + 63) // 64) * 64

    # target hit-rate
    if args.target_hr < 0.0:
        target_hr = params["temporal"]["default_target_hr"]
    else:
        target_hr = args.target_hr

    layers = resolve_layers(args.layers, params)

    print(f"[synth] model={params['model']}  BS={args.bs}  ISL={args.isl}  "
          f"n_kv={n_kv}  K={K}  osl={args.osl}  "
          f"target_hr={target_hr:.3f}  layers={len(layers)}  fmt={args.format}",
          flush=True)

    rng = np.random.default_rng(args.seed)

    # Build per-step logits (mu, sigma) from the step distribution
    # mu drifts slightly as kv grows; we model as a small linear trend
    # derived from the measured step_means distribution
    lp = params["logits"]

    gen_k    = not args.no_k_cache
    gen_log  = not args.no_logits

    for li, layer in enumerate(layers):
        ldir = os.path.join(args.out_dir, f"layer_{layer:02d}")
        os.makedirs(ldir, exist_ok=True)

        # Determine logits bucket for this layer
        if layer in lp.get("shallow_layers", []):
            bucket = lp["shallow"]
        else:
            bucket = lp["deep"]

        dist   = bucket["dist"]
        mu_m   = bucket["mu_mean"]
        sig_m  = bucket["sig_mean"]
        mu_s   = bucket.get("mu_std",  0.0)
        sig_s  = bucket.get("sig_std", 0.0)

        q_buf, k_buf, w_buf, log_buf, topk_buf, pre_buf = {}, {}, {}, {}, {}, {}
        prev_topk = None
        hit_rates = []

        for step in range(args.osl):
            # Per-step logits params: add step-level noise around the bucket mean
            step_mu  = float(rng.normal(mu_m, mu_s))  if mu_s  > 0 else mu_m
            step_sig = abs(float(rng.normal(sig_m, sig_s))) if sig_s > 0 else sig_m
            step_sig = max(step_sig, 0.1)  # guard against near-zero

            out = synth_one_step(
                layer, li, params,
                args.bs, n_kv, n_heads, head_dim, K,
                step_mu, step_sig,
                rng,
                prev_topk=prev_topk,
                target_hr=target_hr,
                gen_k_cache=gen_k,
                gen_logits=gen_log,
            )

            q_buf[step]    = out["q_fp4"]
            w_buf[step]    = out["weights"]
            if gen_k and out["k_cache"] is not None:
                k_buf[step] = out["k_cache"]
            if gen_log:
                if out["logits"] is not None:
                    log_buf[step]  = out["logits"]
                if out["topk"] is not None:
                    topk_buf[step] = out["topk"]
                    pre_buf[step]  = (torch.from_numpy(prev_topk[None, :])
                                      if prev_topk is not None
                                      else torch.zeros(args.bs, K, dtype=torch.int32))

            # Advance temporal state
            if out["topk"] is not None:
                prev_topk = out["topk"][0].numpy()
                if step > 0:
                    hit_rates.append(out["actual_hr"])

        # Save
        save_dict(q_buf, os.path.join(ldir, "q_fp4"), args.format)
        save_dict(w_buf, os.path.join(ldir, "weights"), args.format)
        if k_buf:
            save_dict(k_buf, os.path.join(ldir, "k_cache"), args.format)
        if log_buf:
            save_dict(log_buf,  os.path.join(ldir, "logits"), args.format)
            save_dict(topk_buf, os.path.join(ldir, "topk"), args.format)
            save_dict(pre_buf,  os.path.join(ldir, "preidx"), args.format)

        hr_str = (f"actual_hr={np.mean(hit_rates):.3f}±{np.std(hit_rates):.3f}"
                  if hit_rates else "step0-only")
        print(f"  L{layer:02d} ({dist}): {args.osl} steps  n_kv={n_kv}  {hr_str}",
              flush=True)

    # Manifest
    manifest = {
        "model":     params["model"],
        "K":         K,
        "n_heads":   n_heads,
        "head_dim":  head_dim,
        "bs":        args.bs,
        "isl":       args.isl,
        "osl":       args.osl,
        "n_kv":      n_kv,
        "compress_ratio": cr,
        "target_hr": target_hr,
        "layers":    layers,
        "format":    args.format,
        "seed":      args.seed,
        "algorithm": "logits-first + rank-transform + temporal-bias",
        "q_dtype":   params["q"]["dtype"],
        "k_dtype":   params["k"]["dtype"],
        "has_k_cache": gen_k,
        "has_logits":  gen_log,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[synth] manifest -> {os.path.join(args.out_dir, 'manifest.json')}", flush=True)
    print(f"[synth] DONE — {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
