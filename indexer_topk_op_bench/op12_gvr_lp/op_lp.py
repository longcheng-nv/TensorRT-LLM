# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Op #12: GVR (cuteDSL) low-precision + P4-skip experimental op.

Wrapper around the COPIED kernel `op12_gvr_lp/src/gvr_topk_decode_lp.py`
(never touches the in-tree originals). Exposes all config knobs so the
omni-kernel loop can sweep them:

  num_threads     : 512 | 1024
  p4_mode         : "snap" | "rs" | "rs_exact" | "fine_hist" | "interp_seed"
                    | "skip" | "skip_snap"  (skip = new opt-2 path)
  min_bpm         : __launch_bounds__ min_blocks_per_mp
  use256_override : force 256-bit loads on/off (else N>=16384 heuristic)

The input is fp32 (SGLang-comparison dtype). The kernel itself supports
running storage dtype bf16/fp16; the low-precision experiment is driven via
the `storage_dtype` arg (a separate pre-cast lever, opt-1).
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
sys.path.insert(0, str(_HERE.parent / "ops"))   # cute_vendored importable
sys.path.insert(0, str(_SRC))
from gvr_topk_decode_lp import GvrTopKKernel  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_compiled = {}

# P4 mode -> kernel flag kwargs
_P4_FLAGS = {
    "snap":        dict(),
    "rs":          dict(enable_p4_rank_scatter=True),
    "rs_exact":    dict(enable_p4_rank_scatter=True, enable_p4_rank_scatter_exact=True),
    "fine_hist":   dict(enable_p4_fine_hist=True),
    "interp_seed": dict(enable_p4_interp_seed=True),
    # opt-2 skip paths (new flags added to the kernel)
    "skip":        dict(enable_p4_skip=True, enable_p4_rank_scatter=True, enable_p4_rank_scatter_exact=True),
    "skip_snap":   dict(enable_p4_skip=True),
    # DEBUG: time P1+P2+P3 floor (inexact) to size the P4 budget.
    "nop4":        dict(enable_skip_p4_debug=True),
}

# opt-1 mixed-precision (exact): P1-P3 scan bf16/fp16, P4 reloads fp32. Maps
# mode -> (scan_dtype, base_p4_flags). Caller passes fp32 logits; the op casts.
_LP_MODES = {
    "lp_bf16":      (torch.bfloat16, dict(enable_p4_rank_scatter=True, enable_p4_rank_scatter_exact=True)),
    "lp_fp16":      (torch.float16,  dict(enable_p4_rank_scatter=True, enable_p4_rank_scatter_exact=True)),
    "lp_bf16_snap": (torch.bfloat16, dict()),
    "lp_fp16_snap": (torch.float16,  dict()),
}


def _default_config(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def compile_lp(dtype, bs, n, K, cr_val, *, num_threads=None, p4_mode="rs_exact",
               min_bpm=None, use256_override=None, kc_accept=None):
    key = (dtype, bs, n, K, cr_val, num_threads, p4_mode, min_bpm, use256_override, kc_accept)
    if key in _compiled:
        return _compiled[key]
    t_def, use256_def, mbpm_def = _default_config(bs, n)
    t = num_threads if num_threads is not None else t_def
    use256 = use256_override if use256_override is not None else use256_def
    mbpm = min_bpm if min_bpm is not None else mbpm_def
    if p4_mode in _LP_MODES:
        scan_dt, flags = _LP_MODES[p4_mode]
        flags = dict(flags, enable_lp_scan=True)
        scan_cute_dt = _DT[scan_dt]
    else:
        flags = _P4_FLAGS[p4_mode]
        scan_cute_dt = _DT[dtype]
    kobj = GvrTopKKernel(
        dtype=scan_cute_dt, top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=mbpm, return_output_values=False, kc_accept=kc_accept, **flags,
    )
    n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
    in_align = 32 if use256 else 16
    input_fake = cr.make_fake_compact_tensor(scan_cute_dt, (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align)
    pre_idx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_batch, K), stride_order=(1, 0), assumed_align=16)
    seq_lens_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
    out_idx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_rows, K), stride_order=(1, 0), assumed_align=16)
    # orig fp32 tensor for the lp-scan P4 reload (== input for non-lp modes).
    input_fp32_fake = cr.make_fake_compact_tensor(cutlass.Float32, (n_rows, n_cols), stride_order=(1, 0), assumed_align=16)
    fake_stream = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kobj, input_fake, pre_idx_fake, seq_lens_fake, None, out_idx_fake,
                            input_fp32_fake, stream=fake_stream, options="--enable-tvm-ffi")
    _compiled[key] = compiled
    return compiled


def _dispatch_config(bs, n):
    """Best-achievable regime dispatch (B200, fp32, from the op12 A/B sweeps).

    - Large N (>=131072) at low BS: snap P4 with 1024 threads (best long-context).
    - Else: rank-scatter-exact P4 with 512 threads (best all-rounder).
    P4 is barrier-floor-bound, so the choice is dominated by P1-P3 parallelism +
    the P4 variant's fixed barrier count, both of which favor this split.
    Returns (p4_mode, num_threads).
    """
    if bs <= NUM_SMS and n >= 131072:
        return "snap", 1024
    return "rs_exact", 512


def gvr_lp(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
           *, num_threads=None, p4_mode="rs_exact", min_bpm=None, use256_override=None,
           kc_accept=None):
    bs, n = logits.shape
    if p4_mode == "dispatch":
        p4_mode, num_threads = _dispatch_config(bs, n)
    compiled = compile_lp(logits.dtype, bs, n, index_topk, compress_ratio,
                          num_threads=num_threads, p4_mode=p4_mode, min_bpm=min_bpm,
                          use256_override=use256_override, kc_accept=kc_accept)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    if p4_mode in _LP_MODES:
        # opt-1 mixed precision: input_data = bf16/fp16 scan copy, input_fp32 = orig.
        assert logits.dtype == torch.float32, "lp modes take fp32 logits (cast internally)"
        scan_dt = _LP_MODES[p4_mode][0]
        scan = logits.to(scan_dt).contiguous()
        compiled(scan, pre_idx, seq_lens, None, out, logits)
    else:
        compiled(logits, pre_idx, seq_lens, None, out, logits)
    return out


if __name__ == "__main__":
    # exactness gate on the REAL synth bundles (the data the report uses)
    sys.path.insert(0, str(_HERE.parent / "harness"))
    from synth_data import get_bundle  # noqa: E402
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4", default="rs_exact")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    bad = 0
    for dt in (torch.float32,):
        for K, crv, N in ((512, 4, 65536), (1024, 4, 32768), (512, 4, 4096), (1024, 4, 262144)):
            b = get_bundle(K, dt, N, cfg="beta_moderate", seed=0)
            logits = b["logits"].to(dt).contiguous()
            pre_idx = b["preIdx"].contiguous()
            seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
            out = gvr_lp(logits, pre_idx, seq_lens, K, crv, num_threads=args.threads, p4_mode=args.p4)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[0].tolist()))
            ok = (d <= 1e-5 and nuniq == K)
            if not ok:
                bad += 1
            print(f"  p4={args.p4:10s} thr={args.threads} {str(dt):12s} K={K:4d} cr={crv} N={N:6d}: "
                  f"uniq={nuniq}/{K} valdiff={d:.2e} {'OK' if ok else 'FAIL'}")
    print("smoke " + ("OK" if bad == 0 else f"FAIL ({bad})"))
