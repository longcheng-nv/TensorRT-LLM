# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""op16 GVR (cuteDSL, rank-scatter P4) + sampled-histogram P2 init (cheaper-P2).

Mirrors ``harness/gvr_cutedsl_rs_op.py`` EXACTLY (same launch config, same
compile path → local perf == integration perf), but the kernel is the op16
``gvr_topk_decode_dt.GvrTopKKernel`` with the ``enable_sampled_init`` flag.

- ``gvr_dt(..., sampled=False)`` → byte-identical to op#7 rank-scatter (baseline).
- ``gvr_dt(..., sampled=True)``  → sampled-histogram P2 init (op16).

The sampled path replaces P2's iterative full-N secant with a strided-subsample
SMEM histogram → t0 at the aim*K quantile → 1 full-N confirm (secant corrects on
the rare miss). Host-validated 1 pass (vs 2-3.67) + cand~1.1×K + exact.
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "ops"))   # cute_vendored importable
sys.path.insert(0, str(_HERE))                          # op16 kernel
from gvr_topk_decode_dt import GvrTopKKernel  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_compiled = {}


def _config(bs, n):
    """Identical launch-config heuristic to gvr_cutedsl_rs_op.py."""
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def compile_gvr_dt(dtype, bs, n, K, cr_val, sampled=True,
                   sample_size=4096, sample_aim_permille=1150, dual=False):
    key = (dtype, bs, n, K, cr_val, sampled, sample_size, sample_aim_permille, dual)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrTopKKernel(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
        enable_p4_rank_scatter=True, enable_p4_rank_scatter_exact=True,
        enable_sampled_init=sampled, sample_size=sample_size,
        sample_aim_permille=sample_aim_permille, enable_dual_thresh=dual,
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


def gvr_dt(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
           sampled=True, sample_size=4096, sample_aim_permille=1150, dual=False):
    bs, n = logits.shape
    compiled = compile_gvr_dt(logits.dtype, bs, n, index_topk, compress_ratio,
                              sampled, sample_size, sample_aim_permille, dual)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    # Exactness gate on the REAL synth bundles (the report's exact inputs).
    sys.path.insert(0, str(_HERE.parent.parent / "harness"))
    from synth_data import get_bundle  # noqa: E402
    bad = 0
    ncell = 0
    for sampled in (False, True):
        tag = "sampled" if sampled else "baseline"
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            for K, crv in ((512, 4), (1024, 4), (2048, 1)):
                for N in (8192, 16384, 65536, 262144):
                    for cfg in ("beta_shallow", "beta_moderate", "beta_deep"):
                        for seed in (0, 1):
                            b = get_bundle(K, dt, N, cfg=cfg, seed=seed)
                            logits = b["logits"].to(dt).contiguous()
                            pre_idx = b["preIdx"].contiguous()
                            seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
                            out = gvr_dt(logits, pre_idx, seq_lens, K, crv, sampled=sampled)
                            torch.cuda.synchronize()
                            idx = out[0].clamp(min=0).long()
                            v = logits[0].float().gather(0, idx).sort(descending=True).values
                            ref = torch.topk(logits[0].float(), K).values
                            d = (v - ref).abs().max().item()
                            nuniq = len(set(out[0].tolist()))
                            ok = (d <= 1e-5 and nuniq >= K)
                            ncell += 1
                            if not ok:
                                bad += 1
                                print(f"  FAIL[{tag}] {str(dt):14s} K={K} cr={crv} N={N:6d} {cfg} s{seed}: uniq={nuniq}/{K} vdiff={d:.2e}")
    print(f"op16 exactness: {ncell-bad}/{ncell} OK" + ("" if bad == 0 else f"  ({bad} FAIL)"))
