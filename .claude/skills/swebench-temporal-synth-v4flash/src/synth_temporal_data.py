#!/usr/bin/env python3
r"""Generate V4-Flash-aligned synthetic decode logits + temporal-coherence preIdx.

V4 Flash specifics (vs V3.2 sibling skill swebench-temporal-synth):
  - K = 512  (Flash native; V3.2 was 2048)
  - compress_ratio = 4  (DSv4 indexer in compressed-token-index space)
  - preIdx caller offset = 0
      (V4 kernel uses preIdx[i] directly when cr != 1 — see
       cpp/tensorrt_llm/kernels/heuristicTopKDecode.cu:
         preIdxOffset = (compressRatio == 1) ? ((rowIdx % next_n) + 1) : 0
       The V3.2 sibling applies a caller-side -1 offset; V4 does NOT.)
  - Per-cfg target_hr derived from real V4 Flash captures
    (~0.36 / 0.46 / 0.44 for shallow / moderate / deep).
  - Caller-allocated radix_aux_indices + radix_aux_logits required when
    blocksPerRow > 1 (post-PR #14297 contract).

Beta distribution cfgs were fitted from real V4 Flash swe-bench captures
(see /tmp/dsv4/v4_dist_flash_K512.json), three-bucketed by mean across the
21 GVR-active layers.

Workflow (matches V3.2 swebench-temporal-synth):
  1. Sample row [N] from one of 3 V4-Flash beta cfgs.
  2. current_topk = topK(row, K=512, sorted=True).
  3. Binary-search noise c so |topK(row+c·σ·noise) ∩ current_topk|/K ≈ target_hr.
  4. prev_topk = topK(row + c·σ·noise) at calibrated c.
  5. If current_argmax ∉ prev_topk, replace prev_topk[-1] with current_argmax.
  6. preIdx = prev_topk.int32  (V4: NO -1 offset).
  7. radix_aux_{indices,logits} pre-allocated for kernel call safety.

Outputs per (cfg, N, BS):
  {cfg}_N{N}_bs{BS}/
    logits.pt           [BS, N_padded] fp32, -inf-padded
    preIdx.pt           [BS, K=512] int32
    seq_lens.pt         [BS] int32 = N * compress_ratio + next_n - 1
                        (so kernel internally computes row_end[b]=N)
    radix_aux_*.pt      pre-allocated scratch (optional, generated only with --bench)
    meta.json           noise_c, realised_hr, row stats, V4 invariants
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

K_DEFAULT = 512
COMPRESS_RATIO_DEFAULT = 4
RADIX_AUX_BLOCKS_MAX = 32  # safe upper bound on blocksPerRow for ISL ≤ 100K

# Fitted from real V4 Flash swe-bench captures, 21 layers (even 2..42),
# three-bucketed by mean across (32K, 64K, 100K) ISL captures.
# clip_low / clip_high are the actual observed (min, max) bounds (NOT a
# symmetric envelope around mean) — V4 logits have asymmetric tails with
# stronger positive outliers. See /tmp/dsv4/v4_dist_flash_K512_v2.json.
BETA_CFGS = {
    "beta_shallow": dict(mean=-1.315, std=0.605, clip_low=-4.36, clip_high=7.83, target_hr=0.36),
    "beta_moderate": dict(mean=-2.088, std=0.728, clip_low=-5.08, clip_high=7.17, target_hr=0.46),
    "beta_deep": dict(mean=-2.595, std=0.785, clip_low=-6.28, clip_high=7.55, target_hr=0.44),
}


def _fit_beta_params(mean: float, std: float, low: float, high: float):
    """Solve Beta(α, β) on [low, high] matching target (mean, std)."""
    r = high - low
    mu = (mean - low) / r
    var = min((std / r) ** 2, mu * (1 - mu) * 0.99)
    conc = mu * (1 - mu) / var - 1
    return conc * mu, conc * (1 - mu)


def sample_beta_row(N: int, cfg_name: str, seed: int) -> torch.Tensor:
    """Sample N values from a Beta on [clip_low, clip_high] matching cfg's
    (mean, std). Bounds reflect real V4 logit envelope — asymmetric, with
    positive-tail extending further than negative.
    """
    cfg = BETA_CFGS[cfg_name]
    mean, std = cfg["mean"], cfg["std"]
    low, high = cfg["clip_low"], cfg["clip_high"]
    alpha, beta_p = _fit_beta_params(mean, std, low, high)
    rng = np.random.default_rng(seed)
    samples = (rng.beta(alpha, beta_p, size=N) * (high - low) + low).astype(np.float32)
    samples = np.clip(samples, low, high).astype(np.float32)
    return torch.from_numpy(samples).to(device="cuda")


def _pad_align_inf(t1d: torch.Tensor, align: int = 8) -> torch.Tensor:
    """V4 kernel padding requirement: numColumns multiple of 8."""
    n = t1d.numel()
    r = n % align
    if r == 0:
        return t1d.contiguous()
    pad = torch.full((align - r,), -float("inf"), dtype=t1d.dtype, device=t1d.device)
    return torch.cat([t1d, pad]).contiguous()


def synthesize(
    N: int,
    BS: int,
    cfg_name: str,
    K: int = K_DEFAULT,
    compress_ratio: int = COMPRESS_RATIO_DEFAULT,
    target_hr: float | None = None,
    seed: int = 42,
    max_c: float = 5.0,
    calib_iters: int = 20,
    calib_tol: float = 0.005,
    dtype: torch.dtype = torch.float32,
):
    if N <= 2 * K:
        raise ValueError(f"N={N} too small: requires N > 2·K = {2 * K}")
    if cfg_name not in BETA_CFGS:
        raise ValueError(f"unknown cfg: {cfg_name}; choices = {list(BETA_CFGS)}")

    cfg = BETA_CFGS[cfg_name]
    if target_hr is None:
        target_hr = cfg["target_hr"]

    torch.manual_seed(seed)

    row = sample_beta_row(N, cfg_name, seed)
    row_pad = _pad_align_inf(row, align=8)
    row_std = float(row.std().item())
    row_mean = float(row.mean().item())
    row_min = float(row.min().item())
    row_max = float(row.max().item())

    current_topk = torch.topk(row, K, largest=True, sorted=True).indices.to(torch.long)
    current_argmax = int(current_topk[0].item())
    current_topk_set = set(current_topk.cpu().tolist())

    noise_seed = seed + 1000

    def hit_rate_at(c: float) -> float:
        torch.manual_seed(noise_seed)
        noise = torch.randn(N, dtype=torch.float32, device="cuda") * (c * row_std)
        prev_row = row + noise
        prev_set = set(torch.topk(prev_row, K).indices.cpu().tolist())
        return len(current_topk_set & prev_set) / K

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
    noise = torch.randn(N, dtype=torch.float32, device="cuda") * (c_best * row_std)
    prev_row = row + noise
    prev_topk = torch.topk(prev_row, K, sorted=True).indices.to(torch.long)

    if current_argmax not in prev_topk.tolist():
        prev_topk[-1] = current_argmax

    # V4: NO caller-side offset. preIdx[i] is consumed by kernel as-is.
    pre_idx = prev_topk.to(torch.int32)
    assert pre_idx.numel() == K

    # Verify realised hit rate (kernel-side at offset=0)
    in_topk_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
    in_topk_mask[current_topk] = True
    kernel_hit_rate = float(in_topk_mask[pre_idx.to(torch.long).clamp(min=0, max=N-1)].float().mean().item())

    Npad = row_pad.numel()
    row_pad_cast = row_pad.to(dtype) if dtype != row_pad.dtype else row_pad
    logits = row_pad_cast.unsqueeze(0).expand(BS, -1).contiguous()
    preIdx = pre_idx.unsqueeze(0).expand(BS, -1).contiguous()
    next_n = 1
    seq_lens_val = N * compress_ratio + next_n - 1
    seq_lens = torch.full((BS,), seq_lens_val, dtype=torch.int32, device="cuda")

    meta = {
        "skill": "swebench-temporal-synth-v4flash",
        "cfg": cfg_name,
        "cfg_params": cfg.copy(),
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
        "row_stats": {"mean": row_mean, "std": row_std, "min": row_min, "max": row_max},
        "preidx_caller_offset": 0,
        "preidx_construction": "V4 temporal-coherence (caller offset=0; kernel uses preIdx[i] directly for cr=4)",
        "dtype": str(dtype),
        "logit_alignment": 8,
    }

    return {
        "logits": logits,
        "preIdx": preIdx,
        "seq_lens": seq_lens,
        "meta": meta,
    }


def save(out_subdir: str, bundle: dict):
    os.makedirs(out_subdir, exist_ok=True)
    torch.save(bundle["logits"].cpu(), os.path.join(out_subdir, "logits.pt"))
    torch.save(bundle["preIdx"].cpu(), os.path.join(out_subdir, "preIdx.pt"))
    torch.save(bundle["seq_lens"].cpu(), os.path.join(out_subdir, "seq_lens.pt"))
    with open(os.path.join(out_subdir, "meta.json"), "w") as f:
        json.dump(bundle["meta"], f, indent=2)


# ---------- Optional GVR vs Radix benchmark ----------

LIBTH_COMMON_DEFAULT = (
    "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM"
    "/cpp/build/tensorrt_llm/thop/libth_common.so"
)
L2_FLUSH_BYTES = 128 * 1024 * 1024


def benchmark(bundle: dict, warmup: int = 3, reps: int = 5, libth_common: str | None = None):
    libth_common = libth_common or os.environ.get("LIBTH_COMMON", LIBTH_COMMON_DEFAULT)
    if not os.path.exists(libth_common):
        raise FileNotFoundError(f"libth_common.so not found at {libth_common}. Set $LIBTH_COMMON.")
    torch.ops.load_library(libth_common)
    op = torch.ops.trtllm.indexer_topk_decode

    logits = bundle["logits"].cuda()
    preIdx = bundle["preIdx"].cuda()
    seq_lens = bundle["seq_lens"].cuda()
    BS, _ = logits.shape
    K = preIdx.shape[1]
    compress_ratio = bundle["meta"]["compress_ratio"]
    indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
    scratch = torch.empty((BS * K,), dtype=logits.dtype, device="cuda")
    # radix_aux required by post-#14297 kernel contract when blocksPerRow > 1.
    radix_aux_indices = torch.empty(
        (BS * RADIX_AUX_BLOCKS_MAX * K,), dtype=torch.int32, device="cuda")
    radix_aux_logits = torch.empty(
        (BS * RADIX_AUX_BLOCKS_MAX * K,), dtype=torch.float32, device="cuda")
    flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")

    def time_one(fn):
        for _ in range(warmup):
            flush.zero_(); torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
        evts_us = []
        for _ in range(reps):
            flush.zero_(); torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            evts_us.append(s.elapsed_time(e) * 1000.0)
        return float(np.median(evts_us)), evts_us

    def gvr_fn():
        op(logits, seq_lens, indices, 1, K,
           pre_idx=preIdx,
           heuristic_scratch=scratch,
           compress_ratio=compress_ratio,
           radix_aux_indices=radix_aux_indices,
           radix_aux_logits=radix_aux_logits)

    def radix_fn():
        op(logits, seq_lens, indices, 1, K,
           pre_idx=None,
           heuristic_scratch=None,
           compress_ratio=compress_ratio,
           radix_aux_indices=radix_aux_indices,
           radix_aux_logits=radix_aux_logits)

    gvr_us, _ = time_one(gvr_fn)
    radix_us, _ = time_one(radix_fn)
    return {
        "gvr_us": gvr_us,
        "radix_us": radix_us,
        "speedup_radix_over_gvr": radix_us / gvr_us,
        "warmup": warmup, "reps": reps,
    }


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=parse_n, default=14478,
                   help="post-compress seq len (V4 cr=4: 32K ISL → ~7530, 64K → ~14470, 100K → ~25110). Accepts '14K', '14478', etc.")
    p.add_argument("--cfg", default="beta_moderate", help=f"{list(BETA_CFGS)} or 'all'")
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--K", type=int, default=K_DEFAULT)
    p.add_argument("--compress_ratio", type=int, default=COMPRESS_RATIO_DEFAULT)
    p.add_argument("--target_hr", type=float, default=None,
                   help="override per-cfg default target hit-rate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_c", type=float, default=5.0)
    p.add_argument("--dtype", type=parse_dtype, default=torch.float32,
                   help="logits dtype: fp32 | bf16 | fp16")
    p.add_argument("--outdir", required=True)
    p.add_argument("--bench", action="store_true")
    args = p.parse_args()

    cfgs = list(BETA_CFGS) if args.cfg == "all" else [args.cfg]
    os.makedirs(args.outdir, exist_ok=True)

    for cfg_name in cfgs:
        t0 = time.time()
        bundle = synthesize(
            N=args.N, BS=args.bs, cfg_name=cfg_name, K=args.K,
            compress_ratio=args.compress_ratio,
            target_hr=args.target_hr, seed=args.seed,
            max_c=args.max_c, dtype=args.dtype)
        sub = os.path.join(args.outdir, f"{cfg_name}_N{args.N}_bs{args.bs}")
        save(sub, bundle)

        m = bundle["meta"]
        print(f"\n=== {cfg_name} N={args.N} BS={args.bs} K={args.K} cr={args.compress_ratio} ===")
        print(f"  row mean={m['row_stats']['mean']:.3f} std={m['row_stats']['std']:.3f} "
              f"[{m['row_stats']['min']:.2f}, {m['row_stats']['max']:.2f}]")
        print(f"  target_hr={m['target_hr']:.3f}  c_calibrated={m['calibrated_noise_c']:.4f}")
        print(f"  realised hit_rate={m['calibration_realised_hr']:.4f}"
              f"{'  SATURATED' if m['calibration_saturated'] else ''}")
        print(f"  kernel-side hit_rate (offset=0) = {m['kernel_side_hit_rate']:.4f}")
        print(f"  synth_time = {time.time() - t0:.2f}s")
        print(f"  → {sub}/{{logits,preIdx,seq_lens}}.pt + meta.json")

        if args.bench:
            try:
                bench = benchmark(bundle)
                print(f"  GVR   wall = {bench['gvr_us']:.2f} µs")
                print(f"  Radix wall = {bench['radix_us']:.2f} µs")
                print(f"  Speedup    = {bench['speedup_radix_over_gvr']:.3f}×")
                with open(os.path.join(sub, "speedup.txt"), "w") as f:
                    json.dump(bench, f, indent=2)
            except FileNotFoundError as e:
                print(f"  [bench skipped: {e}]", file=sys.stderr)


if __name__ == "__main__":
    main()
