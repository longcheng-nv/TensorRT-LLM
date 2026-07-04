# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""op13 GVR cuteDSL P2-candidate-count op (N-dispatched kCC/kFTarget tweak).

Identical to harness/gvr_cutedsl_op.py (same vendored GvrTopKKernel, same compile
flags / fake tensors / launch path so local single-op perf == tensorrt_llm
integration perf) EXCEPT it overrides (kCC, kFTarget) per (dtype, K, N) using the
N-dispatch table validated in op13 iters 3-6:

  fp32 only (production indexer logits are fp32 — dsa.py:97 / fp8_paged_mqa_logits
  fp32 output):
    K=512  & N<=65536  -> kCC=1536, kFTarget=1280   (kc3x; ~7-15% pure-kernel win)
    K=1024 & N<=65536  -> kCC=3072, kFTarget=2560   (kc3x)
  everything else (large N, bf16, fp16, K=2048) -> baseline GvrParams default.

The override is a post-ctor attr set on a GvrTopKKernel subclass (kC/kFTarget are
read as const_expr at compile, so a subclass override before cute.compile is the
production-equivalent path; NO edit to the vendored kernel). Large N keeps the
baseline because the narrowed acceptance window costs extra P2 full-N count_ge
scans that dominate once each eval is ~8-12us (iter 3-5: N>=131072 always loses).
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "ops"))  # make cute_vendored importable
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_compiled = {}

# N-dispatch crossover: narrow kCC only at small/mid N. Large N (>=131072) loses
# to P2-eval explosion (iter 3-5); 65536 wins, 131072 loses, so the boundary sits
# in (65536, 131072) — narrow for N <= NARROW_N_MAX.
NARROW_N_MAX = 65536

# (dtype, K) -> (kCC, kFTarget) applied when N <= NARROW_N_MAX.
# SHIP = fp32 K=512 (V4 Flash) ONLY. iter-7 nsys ×3-median A/B (p2c vs baseline op):
#   K=512  fp32: robust WIN — N=4K -14.7%, 8K/16K -11%, 65K -5.5%, 32K ~tie, large N ~tie.
#   K=1024 fp32: NOT shipped — ×3-median is noisy with real REGRESSIONS (N=4096 +15.8%,
#                N=65536 +12.4%) despite wins at 8K/16K/32K. Mixed/unreliable (matches
#                iter 5/6 "K1024 small/noisy") => keep baseline, no narrow dispatch.
#   bf16/fp16: marginal (~3-6%) and not the production dtype; K=2048 baseline already lean.
_NARROW_TABLE = {
    (torch.float32, 512): (1536, 1280),
}


def dispatch_params(dtype, K, n):
    """Return (kCC, kFTarget) override for this cell, or (None, None) for baseline."""
    if n <= NARROW_N_MAX:
        kcc_kft = _NARROW_TABLE.get((dtype, K))
        if kcc_kft is not None:
            return kcc_kft
    return None, None


class GvrP2C(GvrTopKKernel):
    """GvrTopKKernel with kC (=kCC) / kFTarget overridable post-ctor."""

    def __init__(self, *a, kC_override=None, kFTarget_override=None, **kw):
        super().__init__(*a, **kw)
        if kC_override is not None:
            self.kC = int(kC_override)
        if kFTarget_override is not None:
            self.kFTarget = int(kFTarget_override)


def _config(bs, n):
    """Replicate gvr_cutedsl_op.py launch-config heuristic for GvrTopKKernel."""
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def compile_gvr_p2c(dtype, bs, n, K, cr_val):
    kcc, kft = dispatch_params(dtype, K, n)
    key = (dtype, bs, n, K, cr_val, kcc, kft)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrP2C(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
        kC_override=kcc, kFTarget_override=kft,
    )
    n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
    in_align = 32 if use256 else 16
    input_fake = cr.make_fake_compact_tensor(_DT[dtype], (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align)
    pre_idx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_batch, K), stride_order=(1, 0), assumed_align=16)
    seq_lens_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
    out_idx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_rows, K), stride_order=(1, 0), assumed_align=16)
    fake_stream = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kobj, input_fake, pre_idx_fake, seq_lens_fake, None, out_idx_fake,
                            stream=fake_stream, options="--enable-tvm-ffi")
    _compiled[key] = compiled
    return compiled


def gvr_cutedsl_p2c(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None):
    bs, n = logits.shape
    compiled = compile_gvr_p2c(logits.dtype, bs, n, index_topk, compress_ratio)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"NARROW_N_MAX={NARROW_N_MAX}, table={_NARROW_TABLE}")
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for K, crv, N in ((512, 4, 16384), (512, 4, 262144), (1024, 4, 32768), (2048, 1, 32768)):
            kcc, kft = dispatch_params(dt, K, N)
            logits = torch.randn(1, N, dtype=dt, device="cuda")
            pre_idx = torch.topk(logits[0].float(), K).indices.int().view(1, K).contiguous()
            seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
            out = gvr_cutedsl_p2c(logits, pre_idx, seq_lens, K, crv)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[0].tolist()))
            mode = f"narrow(kCC={kcc},kFT={kft})" if kcc else "baseline"
            print(f"  {str(dt):14s} K={K:4d} cr={crv} N={N:6d} {mode:24s}: uniq={nuniq}/{K} valdiff={d:.2e}")
    print("GVR p2c smoke OK")
