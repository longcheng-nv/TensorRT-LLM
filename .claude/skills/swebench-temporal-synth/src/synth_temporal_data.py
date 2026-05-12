#!/usr/bin/env python3
r"""Generate SWE-Bench-aligned synthetic decode logits + temporal-coherence preIdx.

Workflow (verified against Q19 Design C, REPORT_temporal.md):
  1. Sample row [N] from one of 3 beta cfgs (≈ SWE-Bench L20 / L22 / L42 fits).
  2. Compute current_topk = topK(row, K=2048, sorted=True).
  3. Binary-search noise coefficient c such that
       |topK(row + c·σ·N(0,1)) ∩ current_topk| / K  ≈  target_hr (default 0.50)
  4. prev_topk = topK(row + c·σ·noise) at calibrated c (same noise seed).
  5. If current_argmax ∉ prev_topk, replace prev_topk[-1] with current_argmax
     (kernel argmax invariant).
  6. preIdx = (prev_topk - 1).int32                ← caller-side -1 offset
     Kernel will add +1 per heuristicTopKDecode.cu:89,145.

Outputs per (cfg, N, BS):
  {cfg}_N{N}_bs{BS}/
    logits.pt    [BS, N_padded] fp32, with -inf padding past N
    preIdx.pt    [BS, K=2048] int32
    seq_lens.pt  [BS] int32 = N
    meta.json    {cfg moments, calibrated c, realised hit rates, ...}
    [speedup.txt]  if --bench: GVR vs Radix wall + speedup ratio

Usage:
  python3 synth_temporal_data.py --N 65536 --cfg beta_moderate --bs 1 \\
      --outdir /tmp/synth_out
  python3 synth_temporal_data.py --N 131072 --cfg all --bs 1 --bench \\
      --outdir /tmp/synth_out_128k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

K_DEFAULT = 2048

# Inlined verbatim from tests/unittest/_torch/thop/parallel/test_indexer_topk.py
# (3 beta entries of _DECODE_DIST_CONFIGS — temporal mode is beta-only per
# Q19-tempC spec 2026-05-12).
BETA_CFGS = {
    "beta_shallow": dict(mean=-0.75, std=1.90, full_range=13.60),  # ≈ SWE-Bench L20
    "beta_moderate": dict(mean=-2.96, std=1.68, full_range=12.85),  # ≈ SWE-Bench L22 region
    "beta_deep": dict(mean=-4.51, std=1.75, full_range=11.24),  # ≈ SWE-Bench L42
}


def _fit_beta_params(mean: float, std: float, low: float, high: float):
    """α, β for Beta on [low, high] matching (mean, std).

    Verbatim from test_indexer_topk.py.
    """
    r = high - low
    mu = (mean - low) / r
    var = min((std / r) ** 2, mu * (1 - mu) * 0.99)
    conc = mu * (1 - mu) / var - 1
    return conc * mu, conc * (1 - mu)


def sample_beta_row(N: int, cfg_name: str, seed: int) -> torch.Tensor:
    """Return [N] fp32 CUDA tensor; clipped to [mean − fr/2, mean + fr/2]."""
    cfg = BETA_CFGS[cfg_name]
    mean, std, fr = cfg["mean"], cfg["std"], cfg["full_range"]
    low, high = mean - fr / 2, mean + fr / 2
    alpha, beta_p = _fit_beta_params(mean, std, low, high)
    rng = np.random.default_rng(seed)
    samples = (rng.beta(alpha, beta_p, size=N) * (high - low) + low).astype(np.float32)
    samples = np.clip(samples, low, high).astype(np.float32)
    return torch.from_numpy(samples).to(device="cuda")


def _pad_align_inf(t1d: torch.Tensor, align: int = 4) -> torch.Tensor:
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
    target_hr: float = 0.50,
    seed: int = 42,
    max_c: float = 5.0,
    calib_iters: int = 20,
    calib_tol: float = 0.005,
):
    """Returns dict with logits, preIdx, seq_lens tensors + meta."""
    if N <= 2 * K:
        raise ValueError(f"N={N} too small: requires N > 2·K = {2 * K} (miss pool empty otherwise)")
    if cfg_name not in BETA_CFGS:
        raise ValueError(f"unknown cfg: {cfg_name}; choices = {list(BETA_CFGS)} or 'all'")

    torch.manual_seed(seed)

    # Step 1: current row
    row = sample_beta_row(N, cfg_name, seed)
    row_pad = _pad_align_inf(row, align=4)
    row_std = float(row.std().item())
    row_mean = float(row.mean().item())
    row_min = float(row.min().item())
    row_max = float(row.max().item())

    # Step 2: current topK
    current_topk = torch.topk(row, K, largest=True, sorted=True).indices.to(torch.long)
    current_argmax = int(current_topk[0].item())
    current_topk_set = set(current_topk.cpu().tolist())

    noise_seed = seed + 1000  # fixed across calibration + bench

    def hit_rate_at(c: float) -> float:
        torch.manual_seed(noise_seed)
        noise = torch.randn(N, dtype=torch.float32, device="cuda") * (c * row_std)
        prev_row = row + noise
        prev_set = set(torch.topk(prev_row, K).indices.cpu().tolist())
        return len(current_topk_set & prev_set) / K

    # Step 3: binary search c (monotonically decreasing hit_rate in c)
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
    # Detect floor saturation (e.g. K/N >= 0.5 keeps hr stuck above target)
    if abs(c_best - max_c) < 1e-3 and hr_best > target_hr + 0.02:
        saturated = True

    # Step 4: build prev_topk from calibrated noise
    torch.manual_seed(noise_seed)
    noise = torch.randn(N, dtype=torch.float32, device="cuda") * (c_best * row_std)
    prev_row = row + noise
    prev_topk = torch.topk(prev_row, K, sorted=True).indices.to(torch.long)

    # Step 5: argmax invariant
    if current_argmax not in prev_topk.tolist():
        prev_topk[-1] = current_argmax

    # Step 6: -1 offset
    pre_idx_one = (prev_topk - 1).to(torch.int32)
    assert pre_idx_one.numel() == K

    # Verify realised hit rate (kernel-side: read logits[preIdx[i] + 1])
    in_topk_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
    in_topk_mask[current_topk] = True
    plus_one = (pre_idx_one.to(torch.long) + 1).clamp(min=0, max=N - 1)
    kernel_hit_rate = float(in_topk_mask[plus_one].float().mean().item())

    # Replicate
    Npad = row_pad.numel()
    logits = row_pad.unsqueeze(0).expand(BS, -1).contiguous()
    preIdx = pre_idx_one.unsqueeze(0).expand(BS, -1).contiguous()
    seq_lens = torch.full((BS,), N, dtype=torch.int32, device="cuda")

    cfg = BETA_CFGS[cfg_name]
    meta = {
        "cfg": cfg_name,
        "cfg_params": {
            "mean": cfg["mean"],
            "std": cfg["std"],
            "full_range": cfg["full_range"],
            "clip_low": cfg["mean"] - cfg["full_range"] / 2,
            "clip_high": cfg["mean"] + cfg["full_range"] / 2,
        },
        "N": N,
        "Npad": Npad,
        "BS": BS,
        "K": K,
        "seed": seed,
        "noise_seed": noise_seed,
        "target_hr": target_hr,
        "calibrated_noise_c": c_best,
        "calibration_realised_hr": hr_best,
        "kernel_side_hit_rate": kernel_hit_rate,
        "calibration_saturated": saturated,
        "row_stats": {
            "mean": row_mean,
            "std": row_std,
            "min": row_min,
            "max": row_max,
        },
        "preidx_offset_applied": -1,
        "kernel_preidx_offset_expected": 1,
        "preidx_construction": "temporal-coherence (Q19 Design C)",
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

LIBTH_COMMON = os.environ.get(
    "LIBTH_COMMON",
    "/home/scratch.loncheng_gpu/workspace/perf/TensorRT-LLM"
    "/cpp/build/tensorrt_llm/thop/libth_common.so",
)
L2_FLUSH_BYTES = 128 * 1024 * 1024


def benchmark(bundle: dict, warmup: int = 3, reps: int = 5):
    """Optional: measure GVR / Radix walls on the generated tensors.

    Requires libth_common.so to provide torch.ops.trtllm.indexer_topk_decode.
    """
    if not os.path.exists(LIBTH_COMMON):
        raise FileNotFoundError(f"libth_common.so not found at {LIBTH_COMMON}. Set $LIBTH_COMMON.")
    torch.ops.load_library(LIBTH_COMMON)
    op = torch.ops.trtllm.indexer_topk_decode

    logits = bundle["logits"].cuda()
    preIdx = bundle["preIdx"].cuda()
    seq_lens = bundle["seq_lens"].cuda()
    BS = logits.shape[0]
    K = preIdx.shape[1]
    indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
    scratch = torch.empty((BS * K,), dtype=logits.dtype, device="cuda")
    flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")

    def time_one(fn):
        # warmup
        for _ in range(warmup):
            flush.zero_()
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
        # timed via cuda events
        evts_us = []
        for _ in range(reps):
            flush.zero_()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            evts_us.append(s.elapsed_time(e) * 1000.0)  # ms→µs
        return float(np.median(evts_us)), evts_us

    def gvr_fn():
        op(logits, seq_lens, indices, 1, K, preIdx, scratch)

    def radix_fn():
        op(logits, seq_lens, indices, 1, K)

    gvr_us, _ = time_one(gvr_fn)
    radix_us, _ = time_one(radix_fn)
    return {
        "gvr_us": gvr_us,
        "radix_us": radix_us,
        "speedup_radix_over_gvr": radix_us / gvr_us,
        "warmup": warmup,
        "reps": reps,
    }


# ---------- CLI ----------


def parse_n(arg: str) -> int:
    """Accept '64K', '128K', '65536', etc."""
    s = arg.upper().strip()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--N", type=parse_n, default=65536, help="seq length (accepts '64K', '131072', etc.)"
    )
    p.add_argument("--cfg", default="beta_moderate", help=f"one of {list(BETA_CFGS)} or 'all'")
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--K", type=int, default=K_DEFAULT)
    p.add_argument("--target_hr", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_c", type=float, default=5.0)
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--bench", action="store_true", help="run GVR vs Radix speedup measurement after synth"
    )
    args = p.parse_args()

    cfgs = list(BETA_CFGS) if args.cfg == "all" else [args.cfg]
    os.makedirs(args.outdir, exist_ok=True)

    for cfg_name in cfgs:
        t0 = time.time()
        bundle = synthesize(
            N=args.N,
            BS=args.bs,
            cfg_name=cfg_name,
            K=args.K,
            target_hr=args.target_hr,
            seed=args.seed,
            max_c=args.max_c,
        )
        sub = os.path.join(args.outdir, f"{cfg_name}_N{args.N}_bs{args.bs}")
        save(sub, bundle)

        m = bundle["meta"]
        print(f"\n=== {cfg_name} N={args.N} BS={args.bs} ===")
        print(
            f"  row mean={m['row_stats']['mean']:.3f} "
            f"std={m['row_stats']['std']:.3f} "
            f"[{m['row_stats']['min']:.2f}, {m['row_stats']['max']:.2f}]"
        )
        print(f"  calibrated noise_c = {m['calibrated_noise_c']:.4f}")
        print(
            f"  realised hit_rate  = {m['calibration_realised_hr']:.4f} "
            f"(target {m['target_hr']:.2f}"
            f"{', SATURATED' if m['calibration_saturated'] else ''})"
        )
        print(f"  kernel-side hit_rate (preIdx+1 ∈ true_topk) = {m['kernel_side_hit_rate']:.4f}")
        print(f"  synth_time = {time.time() - t0:.2f}s")
        print(f"  saved → {sub}/{{logits,preIdx,seq_lens}}.pt + meta.json")

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
