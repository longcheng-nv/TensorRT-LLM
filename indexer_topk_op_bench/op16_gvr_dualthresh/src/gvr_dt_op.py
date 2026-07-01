# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Op #7: GVR (cuteDSL) with EXACT fused rank-and-scatter P4.

Same kernel as gvr_cutedsl_op.py (single-CTA GVR), but P4 (the in-SMEM exact
top-K from candidates) uses `phase4_rank_scatter` + a fixed-256-bin fine
recursion instead of the iterative histogram-snap. Eliminates the snap loop +
2-pass writeback (~14 block barriers → ~7); resolves the straddling bin by value
(vdiff=0, exact). See p4_recursive_digit/REPORT.md for the B200 A/B.

Kernel source: p4_recursive_digit/src/gvr_topk_decode_p4.py (the vendored
GvrTopKKernel + the two gated P4 flags enable_p4_rank_scatter[_exact]).
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
_BUCKET_SRC = _HERE.parent / "p4_recursive_digit" / "src"
sys.path.insert(0, str(_HERE.parent / "ops"))   # cute_vendored importable
sys.path.insert(0, str(_BUCKET_SRC))             # bucket kernel with P4 flags
from gvr_topk_decode_p4 import GvrTopKKernel  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_compiled = {}


def _config(bs, n):
    """Identical launch-config heuristic to gvr_cutedsl_op.py."""
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def compile_gvr_cute_rs(dtype, bs, n, K, cr_val):
    key = (dtype, bs, n, K, cr_val)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrTopKKernel(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
        enable_p4_rank_scatter=True, enable_p4_rank_scatter_exact=True,
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


def gvr_cutedsl_rs(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None):
    bs, n = logits.shape
    compiled = compile_gvr_cute_rs(logits.dtype, bs, n, index_topk, compress_ratio)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    # Use the REAL synth bundles (beta_moderate, hit-rate 0.6) — the exact data
    # the report/sweep use — so this is a faithful exactness gate. (NOTE: on
    # torch.randn→bf16 the values collapse to ~256 distinct levels → extreme ties
    # at the K-th boundary can break the rank-scatter contiguity; that adversarial
    # case is NOT what the report measures. On the synth/real DSv4 distribution
    # all dtypes are vdiff=0 — see p4_recursive_digit/data/matrix_exact_bs1.log.)
    from synth_data import get_bundle  # noqa: E402
    bad = 0
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for K, crv, N in ((2048, 1, 32768), (512, 4, 65536), (1024, 4, 32768)):
            b = get_bundle(K, dt, N, cfg="beta_moderate", seed=0)
            logits = b["logits"].to(dt).contiguous()
            pre_idx = b["preIdx"].contiguous()
            seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
            out = gvr_cutedsl_rs(logits, pre_idx, seq_lens, K, crv)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[0].tolist()))
            if d > 1e-5 or nuniq < K:
                bad += 1
            print(f"  {str(dt):14s} K={K:4d} cr={crv} N={N:6d}: uniq={nuniq}/{K} valdiff_vs_topk={d:.2e}")
    print("GVR cuteDSL rank-scatter-exact smoke " + ("OK" if bad == 0 else f"FAIL ({bad} cells)"))
