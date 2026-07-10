# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""iter5 silicon diagnosis: why did K2048 fp16 65536 get WORSE under V3?

Times four single-CTA configs on the exact op22rr bundle rows (BS=1 and
BS=256 expand), CUDA events + L2 flush (cold-L2 canonical):
  iter4 : p2_log=True,  kFT stock(4096), sec2=off   (campaign baseline)
  V1    : p2_log=True,  kFT center(3238), sec2=off  (center aim only)
  V3    : p2_log=True,  kFT center(3238), sec2=on   (iter5 as shipped)
  anchor: vendored linear P2 via p2_log=False stock  (sanity scale)

If V3 >> V1 the secant path itself is the problem on silicon (loop-carried
registers or proposal quality); if V3 ~= V1 ~= iter4 the dispatch isn't
engaging. Companion: diag_p2_variants.py (host replay predicts V3 < iter4).
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str((_HERE / "../op22_temporal_fixed_hr_bench").resolve()))
sys.path.insert(0, str((_HERE / "../harness").resolve()))

import bundle_data_rr  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as cr  # noqa: E402
from gvr_op26_op import GvrOp26Kernel, _DT, _config_1cta  # noqa: E402

_L2 = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")


def compile_variant(dt, K, n, bs, p2_log, kcc, kft, sec2, rs_on):
    t, use256, min_bpm = _config_1cta(bs, n)
    kobj = GvrOp26Kernel(
        dtype=_DT[dt], top_k=K, next_n=1, num_threads=t,
        compress_ratio=4 if K != 2048 else 1, use_256bit_load=use256,
        enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
        enable_p4_rank_scatter=rs_on, enable_p4_rank_scatter_exact=rs_on,
        p2_log=p2_log, kC_override=kcc, kFTarget_override=kft,
        p2_secant2=sec2, fb_fix=True,
    )
    n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
    in_align = 32 if use256 else 16
    input_fake = cr.make_fake_compact_tensor(
        _DT[dt], (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align)
    pre_idx_fake = cr.make_fake_compact_tensor(
        cutlass.Int32, (n_batch, K), stride_order=(1, 0), assumed_align=16)
    seq_lens_fake = cr.make_fake_compact_tensor(
        cutlass.Int32, (n_batch,), stride_order=(0,))
    out_idx_fake = cr.make_fake_compact_tensor(
        cutlass.Int32, (n_rows, K), stride_order=(1, 0), assumed_align=16)
    fake_stream = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kobj, input_fake, pre_idx_fake, seq_lens_fake,
                        None, out_idx_fake, stream=fake_stream,
                        options="--enable-tvm-ffi")


def time_compiled(compiled, logits, pre, seq, K, reps=100):
    out = torch.empty(logits.shape[0], K, dtype=torch.int32, device="cuda")
    for _ in range(10):
        compiled(logits, pre, seq, None, out)
    torch.cuda.synchronize()
    times = []
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        _L2.zero_()  # L2 flush (cold-L2 canonical)
        s.record()
        compiled(logits, pre, seq, None, out)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[len(times) // 2], out


def check_exact(out, logits, K):
    idx = out[0].clamp(min=0).long()
    v = logits[0].float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits[0].float(), K).values
    return (v - ref).abs().max().item() == 0.0 and len(set(out[0].tolist())) == K


def main():
    cells = [(2048, "fp16", torch.float16, 65536, 1),
             (2048, "fp16", torch.float16, 65536, 256),
             (1024, "fp32", torch.float32, 131072, 1),
             (1024, "fp32", torch.float32, 131072, 256)]
    for K, dts, dt, N, BS in cells:
        b = bundle_data_rr.get_bundle("real", K, dt, N, device="cuda")
        logits = b["logits"][:1].repeat(BS, 1).contiguous()
        pre = b["preIdx"][:1].repeat(BS, 1).contiguous()
        cr_v = 4 if K != 2048 else 1
        seq = torch.full((BS,), N * cr_v, dtype=torch.int32, device="cuda")
        stock_kft = {1024: None, 2048: None}[K]  # None = vendored default
        center = {(2048, "fp16"): 3238, (1024, "fp32"): 1448}[(K, dts)]
        narrow_kcc = {(2048, "fp16"): None, (1024, "fp32"): 2048}[(K, dts)]
        variants = [
            ("anchor-lin", dict(p2_log=False, kcc=None, kft=None, sec2=False)),
            ("iter4     ", dict(p2_log=True, kcc=narrow_kcc, kft=stock_kft,
                                sec2=False)),
            ("V1-center ", dict(p2_log=True, kcc=narrow_kcc, kft=center,
                                sec2=False)),
            ("V3-secant ", dict(p2_log=True, kcc=narrow_kcc, kft=center,
                                sec2=True)),
        ]
        rs_on = (dt == torch.float32) or BS >= 256
        print(f"== K{K} {dts} N={N} BS={BS} (rs={rs_on}) ==")
        for name, cfg in variants:
            comp = compile_variant(dt, K, N, BS, cfg["p2_log"], cfg["kcc"],
                                   cfg["kft"], cfg["sec2"], rs_on)
            us, out = time_compiled(comp, logits, pre, seq, K)
            ok = check_exact(out, logits, K)
            print(f"  {name}: {us:8.2f} us  exact={ok}")


if __name__ == "__main__":
    main()
