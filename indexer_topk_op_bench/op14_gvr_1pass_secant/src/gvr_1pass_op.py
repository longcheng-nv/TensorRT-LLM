# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""op14 GVR ≈1-HBM-pass secant op driver.

Two entry points, both compiled from the LOCAL copied kernel
(src/gvr_topk_decode_1pass.py) and mirroring harness/gvr_cutedsl_rs_op.py EXACTLY
(same launch-config, fake tensors, compile flags) so local single-op perf ==
tensorrt_llm integration perf:

  gvr_rs_base(...)  -> rank-scatter baseline (enable_1pass_compaction=False) ==
                       harness/gvr_cutedsl_rs_op.gvr_cutedsl_rs (op#7). A/B anchor.
  gvr_1pass(...)    -> enable_1pass_compaction=True (the new algorithm; behavior
                       implemented incrementally by /omni-kernel behind the flag).

See ALGORITHM_SPEC.md for the algorithm + analysis.
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "ops"))   # cute_vendored importable
sys.path.insert(0, str(_HERE))                       # local kernel
from gvr_topk_decode_1pass import GvrTopKKernel  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_compiled = {}


def _config(bs, n):
    """Identical launch-config heuristic to gvr_cutedsl_rs_op.py."""
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def _compile(dtype, bs, n, K, cr_val, enable_1pass, C):
    key = (dtype, bs, n, K, cr_val, enable_1pass, C)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrTopKKernel(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
        enable_p4_rank_scatter=True, enable_p4_rank_scatter_exact=True,
        enable_1pass_compaction=enable_1pass, compaction_C=C,
    )
    n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
    in_align = 32 if use256 else 16
    input_fake = cr.make_fake_compact_tensor(_DT[dtype], (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align)
    pre_idx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_batch, K), stride_order=(1, 0), assumed_align=16)
    seq_lens_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
    out_idx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_rows, K), stride_order=(1, 0), assumed_align=16)
    fake_stream = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    if enable_1pass:
        n_cap = cute.sym_int()
        cval_fake = cr.make_fake_compact_tensor(cutlass.Float32, (n_rows, n_cap), stride_order=(1, 0), assumed_align=16)
        cidx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_rows, n_cap), stride_order=(1, 0), assumed_align=16)
        compiled = cute.compile(kobj, input_fake, pre_idx_fake, seq_lens_fake, None, out_idx_fake,
                                stream=fake_stream, cand_val=cval_fake, cand_idx=cidx_fake,
                                options="--enable-tvm-ffi")
    else:
        compiled = cute.compile(kobj, input_fake, pre_idx_fake, seq_lens_fake, None, out_idx_fake,
                                stream=fake_stream, options="--enable-tvm-ffi")
    _compiled[key] = compiled
    return compiled


def gvr_rs_base(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None):
    bs, n = logits.shape
    compiled = _compile(logits.dtype, bs, n, index_topk, compress_ratio, False, 2)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


# Scratch cap = 32*K. ALGORITHM_SPEC §7 suggested 16*K but the measured
# survivor count c0 = #{v >= pmin} grows with N (≈11k for K512 at N=262144),
# so 16*K=8192 overflows → fallback. 32*K keeps the fast path firing across the
# large-N grid. Allocated once per (bs, K) and reused (no per-call alloc).
_scratch = {}


def _get_scratch(bs, K):
    key = (bs, K)
    s = _scratch.get(key)
    if s is None:
        cap = 32 * K
        cval = torch.empty(bs, cap, dtype=torch.float32, device="cuda")
        cidx = torch.empty(bs, cap, dtype=torch.int32, device="cuda")
        _scratch[key] = (cval, cidx)
        s = _scratch[key]
    return s


def gvr_1pass(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None, C=2):
    bs, n = logits.shape
    compiled = _compile(logits.dtype, bs, n, index_topk, compress_ratio, True, C)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    cval, cidx = _get_scratch(bs, index_topk)
    compiled(logits, pre_idx, seq_lens, None, out, cand_val=cval, cand_idx=cidx)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(_HERE.parents[1] / "harness"))
    from synth_data import get_bundle  # noqa: E402
    bad = 0
    for dt in (torch.float32, torch.bfloat16):
        for K, crv, N in ((512, 4, 131072), (1024, 4, 131072)):
            b = get_bundle(K, dt, N, cfg="beta_moderate", seed=42)
            logits = b["logits"].to(dt).contiguous(); pre = b["preIdx"].contiguous()
            sl = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
            for name, fn in (("rs_base", gvr_rs_base), ("1pass", gvr_1pass)):
                out = fn(logits, pre, sl, K, crv)
                torch.cuda.synchronize()
                idx = out[0].clamp(min=0).long()
                v = logits[0].float().gather(0, idx).sort(descending=True).values
                ref = torch.topk(logits[0].float(), K).values
                d = (v - ref).abs().max().item(); nuniq = len(set(out[0].tolist()))
                if d > 1e-3 or nuniq < K:
                    bad += 1
                print(f"  {name:8s} {str(dt):14s} K={K:4d} N={N:6d}: uniq={nuniq}/{K} vdiff={d:.2e}")
    print("op14 smoke " + ("OK" if bad == 0 else f"FAIL ({bad})"))
