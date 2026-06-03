#!/usr/bin/env python3
r"""
Generate V4-Pro-aligned synthetic decode logits + temporal-coherence preIdx.

Target hardware: NVIDIA **Blackwell** (B200 sm_100 / B300 sm_103).
Indexer top-K: K = 1024 (V4 Pro native; Blackwell-tuned).

Self-contained adaptation of the in-tree `swebench-temporal-synth-v4pro`
methodology (no skill dependency). Tuned for the two cases of
`gvr_perfsim_UT`:

  * case-1: BS=1,   N=65536, K=1024, beta_moderate, target hit-rate=0.6
  * case-2: BS=256, N=65536, K=1024, beta_moderate, target hit-rate=0.6
    (BS rows literally replicate the BS=1 row — the user-requested
     "不同 BS 下直接复制相同的数据")

V4 Pro invariants (verified against
  tensorrt_llm/.../heuristicTopKDecode.cu  +  PR #14297 contract):

  K               = 1024                       (Pro native; Flash = 512)
  compress_ratio  = 4                          (V4 DSA path)
  preIdx offset   = 0  (caller side)           (kernel reads preIdx[i] directly
                                                when compress_ratio != 1)
  seq_lens_val    = N * compress_ratio + (next_n - 1),  next_n = 1
  radix_aux_*     = caller-allocated (BS * RADIX_AUX_BLOCKS_MAX * K) buffers
                    required when blocksPerRow > 1 (post-PR #14297)

Typical V4 Pro logit distribution (`beta_moderate` bucket, mid of the 30
GVR-active layers L02..L60 fitted from real swe-bench captures, 32K / 64K
ISL pool):

  mean=-1.885, std=1.025, clip_low=-6.15, clip_high=8.45
  (asymmetric tail — positive outliers larger than negative; matches V4 Pro
   indexer logit envelope across layer depth.)

Output layout (per call → one cell directory):

  <outdir>/
    logits.pt    [BS, N_pad] fp32, -inf right-padded to multiple of 8
    preIdx.pt    [BS, K]     int32
    seq_lens.pt  [BS]        int32 = N * compress_ratio + 0
    meta.json    cfg, calibrated noise c, realised hit-rate, V4 invariants
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

K_DEFAULT = 1024
COMPRESS_RATIO_DEFAULT = 4
RADIX_AUX_BLOCKS_MAX = 32  # contract from indexerTopK.cu kMaxBlocksPerRowDecode * 2 buffer

# Mid-bucket Pro distribution from real swe-bench V4 Pro captures.
BETA_MODERATE = dict(
    mean=-1.885, std=1.025, clip_low=-6.15, clip_high=8.45,
)


def fit_beta_params(mean: float, std: float, low: float, high: float):
    """Solve Beta(α, β) on [low, high] matching target (mean, std)."""
    r = high - low
    mu = (mean - low) / r
    var = min((std / r) ** 2, mu * (1 - mu) * 0.99)
    conc = mu * (1 - mu) / var - 1
    return conc * mu, conc * (1 - mu)


def sample_beta_row(N: int, cfg: dict, seed: int, device: str) -> torch.Tensor:
    alpha, beta_p = fit_beta_params(
        cfg["mean"], cfg["std"], cfg["clip_low"], cfg["clip_high"])
    rng = np.random.default_rng(seed)
    samples = (rng.beta(alpha, beta_p, size=N) * (cfg["clip_high"] - cfg["clip_low"])
               + cfg["clip_low"]).astype(np.float32)
    samples = np.clip(samples, cfg["clip_low"], cfg["clip_high"]).astype(np.float32)
    return torch.from_numpy(samples).to(device=device)


def pad_align_inf(t1d: torch.Tensor, align: int = 8) -> torch.Tensor:
    n = t1d.numel()
    r = n % align
    if r == 0:
        return t1d.contiguous()
    pad = torch.full((align - r,), -float("inf"), dtype=t1d.dtype, device=t1d.device)
    return torch.cat([t1d, pad]).contiguous()


def synthesize(
    N: int,
    BS: int,
    K: int = K_DEFAULT,
    compress_ratio: int = COMPRESS_RATIO_DEFAULT,
    target_hr: float = 0.6,
    seed: int = 42,
    max_c: float = 5.0,
    calib_iters: int = 30,
    calib_tol: float = 0.005,
    cfg: dict | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    """Sample / calibrate / build preIdx in fp32, then optionally cast logits
    to `dtype` (bf16 / fp16) via RNE truncation — matches V4 Pro production
    semantics (prev fp32 preIdx + current half-prec kernel logits)."""
    if N <= 2 * K:
        raise ValueError(f"N={N} too small: requires N > 2·K = {2 * K}")
    cfg = cfg or BETA_MODERATE
    torch.manual_seed(seed)

    row = sample_beta_row(N, cfg, seed, device)  # always fp32 here
    row_pad = pad_align_inf(row, align=8)
    row_std = float(row.std().item())

    current_topk = torch.topk(row, K, largest=True, sorted=True).indices.to(torch.long)
    current_argmax = int(current_topk[0].item())
    current_topk_set = set(current_topk.cpu().tolist())

    noise_seed = seed + 1000

    def hit_rate_at(c: float) -> float:
        torch.manual_seed(noise_seed)
        noise = torch.randn(N, dtype=torch.float32, device=device) * (c * row_std)
        prev_set = set(torch.topk(row + noise, K).indices.cpu().tolist())
        return len(current_topk_set & prev_set) / K

    # Binary-search noise coefficient c: hit-rate is monotonically decreasing in c.
    lo, hi = 0.0, max_c
    c_best, hr_best = max_c / 2, hit_rate_at(max_c / 2)
    saturated = False
    for _ in range(calib_iters):
        mid = (lo + hi) / 2
        hr = hit_rate_at(mid)
        if abs(hr - target_hr) < calib_tol:
            c_best, hr_best = mid, hr
            break
        if hr > target_hr:
            lo = mid
        else:
            hi = mid
        c_best, hr_best = mid, hr
    if abs(c_best - max_c) < 1e-3 and hr_best > target_hr + 0.02:
        saturated = True

    torch.manual_seed(noise_seed)
    noise = torch.randn(N, dtype=torch.float32, device=device) * (c_best * row_std)
    prev_topk = torch.topk(row + noise, K, sorted=True).indices.to(torch.long)
    if current_argmax not in prev_topk.tolist():
        prev_topk[-1] = current_argmax  # kernel argmax invariant

    pre_idx = prev_topk.to(torch.int32)  # V4 caller offset = 0
    assert pre_idx.numel() == K

    in_topk_mask = torch.zeros(N, dtype=torch.bool, device=device)
    in_topk_mask[current_topk] = True
    kernel_hit_rate = float(
        in_topk_mask[pre_idx.to(torch.long).clamp(min=0, max=N - 1)].float().mean().item()
    )

    # Cast fp32 logits → target dtype (RNE truncation). preIdx stays int32.
    if dtype != torch.float32:
        row_pad_cast = row_pad.to(dtype)
    else:
        row_pad_cast = row_pad

    # Replicate to BS rows (the user's spec: "不同 BS 下直接复制相同的数据").
    Npad = row_pad_cast.numel()
    logits = row_pad_cast.unsqueeze(0).expand(BS, -1).contiguous()
    preIdx = pre_idx.unsqueeze(0).expand(BS, -1).contiguous()
    next_n = 1
    seq_lens_val = N * compress_ratio + next_n - 1
    seq_lens = torch.full((BS,), seq_lens_val, dtype=torch.int32, device=device)

    meta = {
        "schema": "gvr_perfsim_UT v1",
        "cfg_name": "beta_moderate_v4pro_typical",
        "cfg_params": cfg,
        "N": N,
        "Npad": Npad,
        "BS": BS,
        "K": K,
        "compress_ratio": compress_ratio,
        "next_n": next_n,
        "seq_lens_val": seq_lens_val,
        "seed": seed,
        "noise_seed": noise_seed,
        "target_hr": target_hr,
        "calibrated_noise_c": c_best,
        "calibration_realised_hr": hr_best,
        "kernel_side_hit_rate": kernel_hit_rate,
        "calibration_saturated": saturated,
        "row_stats": {
            "mean": float(row.mean().item()),
            "std": row_std,
            "min": float(row.min().item()),
            "max": float(row.max().item()),
        },
        "preidx_caller_offset": 0,
        "preidx_construction": "V4 Pro temporal-coherence (caller offset=0; kernel uses preIdx[i] directly for cr=4)",
        "logit_alignment": 8,
        "radix_aux_blocks_max": RADIX_AUX_BLOCKS_MAX,
        "logits_dtype": str(dtype),
        "halfprec_method": ("RNE-cast from fp32 sample/calibrate (V4 Pro Option A)"
                            if dtype != torch.float32 else "native fp32"),
    }
    return {"logits": logits, "preIdx": preIdx, "seq_lens": seq_lens, "meta": meta}


def save(outdir: str, bundle: dict):
    os.makedirs(outdir, exist_ok=True)
    torch.save(bundle["logits"].cpu(), os.path.join(outdir, "logits.pt"))
    torch.save(bundle["preIdx"].cpu(), os.path.join(outdir, "preIdx.pt"))
    torch.save(bundle["seq_lens"].cpu(), os.path.join(outdir, "seq_lens.pt"))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(bundle["meta"], f, indent=2)


def parse_n(arg: str) -> int:
    s = arg.upper().strip()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def parse_dtype(arg: str) -> torch.dtype:
    a = arg.lower().strip()
    if a in ("fp32", "float32", "float"):
        return torch.float32
    if a in ("bf16", "bfloat16"):
        return torch.bfloat16
    if a in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"unknown dtype: {arg}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=parse_n, default=65536,
                   help="post-compress seq len (default 65536 = 64K)")
    p.add_argument("--bs", type=int, default=1,
                   help="batch size (BS rows literally replicate the BS=1 row)")
    p.add_argument("--K", type=int, default=K_DEFAULT,
                   help="top-K (default 1024 = V4 Pro native indexer top-K; "
                        "Blackwell-tuned in heuristic_topk.cuh)")
    p.add_argument("--compress_ratio", type=int, default=COMPRESS_RATIO_DEFAULT)
    p.add_argument("--target_hr", type=float, default=0.6,
                   help="target preIdx ∩ true-topK hit rate (default 0.6)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_c", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", type=parse_dtype, default=torch.float32,
                   help="logits dtype: fp32 (default), bf16, or fp16 "
                        "(V4 Pro Option A — sample/calibrate in fp32, then "
                        "RNE-cast the saved logits to half-prec)")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    t0 = time.time()
    bundle = synthesize(
        N=args.N, BS=args.bs, K=args.K,
        compress_ratio=args.compress_ratio,
        target_hr=args.target_hr, seed=args.seed,
        max_c=args.max_c, device=args.device, dtype=args.dtype)
    save(args.outdir, bundle)

    m = bundle["meta"]
    print(f"=== beta_moderate (V4 Pro typical) N={args.N} BS={args.bs} K={args.K} cr={args.compress_ratio} dtype={args.dtype} ===")
    print(f"  row mean={m['row_stats']['mean']:.3f} std={m['row_stats']['std']:.3f} "
          f"[{m['row_stats']['min']:.2f}, {m['row_stats']['max']:.2f}]")
    print(f"  target_hr={m['target_hr']:.3f}  c_calibrated={m['calibrated_noise_c']:.4f}")
    print(f"  realised hit_rate={m['calibration_realised_hr']:.4f}"
          f"{'  (SATURATED)' if m['calibration_saturated'] else ''}")
    print(f"  kernel-side hit_rate (caller offset=0) = {m['kernel_side_hit_rate']:.4f}")
    print(f"  synth_time = {time.time() - t0:.2f}s")
    print(f"  → {args.outdir}/{{logits,preIdx,seq_lens}}.pt + meta.json")


if __name__ == "__main__":
    main()
