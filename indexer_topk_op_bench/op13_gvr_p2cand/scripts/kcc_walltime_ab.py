# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""CORRECT wall-time A/B: time the ACTUAL modified-kCC kernel cold-L2, not the
production op's wall x modified-kernel fractions (the iter-2 phase_ab flaw).

Builds a non-instrumented GvrTopKKernel with kC/kFTarget overridden (production
code path, mirrors harness/gvr_cutedsl_op compile), times it cold-L2 with the
same CUDA-graph + cudaEvent + 512MB-evict protocol as report/sweep.py, and
compares median wall-us vs the default-kCC baseline. Also verifies EXACTNESS
(value-equiv to torch.topk) of every variant launched.
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parents[1] / "ops"))
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from synth_data import get_bundle  # noqa: E402
import measure_cute_phases as mp  # noqa: E402  (reuse _EVICT, _config)

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_EVICT = mp._EVICT
_compiled = {}


class GvrOverride(GvrTopKKernel):
    def __init__(self, *a, kC_override=None, kFTarget_override=None, **kw):
        super().__init__(*a, **kw)
        if kC_override is not None:
            self.kC = int(kC_override)
        if kFTarget_override is not None:
            self.kFTarget = int(kFTarget_override)


def _compile(dtype, bs, n, K, cr_val, kCC, kFT):
    key = (dtype, bs, n, K, cr_val, kCC, kFT)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = mp._config(bs, n)
    kobj = GvrOverride(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
        kC_override=kCC, kFTarget_override=kFT)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    in_align = 32 if use256 else 16
    in_f = cr.make_fake_compact_tensor(_DT[dtype], (nr, nc), stride_order=(1, 0), assumed_align=in_align)
    pi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, stream=fs, options="--enable-tvm-ffi")
    _compiled[key] = compiled
    return compiled


def _cold_us(call, reps=60, warmup=5):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    g.replay(); torch.cuda.synchronize()
    cold = []
    for _ in range(reps):
        _EVICT.uniform_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    cold.sort()
    return cold[len(cold) // 2]


def _exact(out, logits_row, K):
    idx = out[0].clamp(min=0).long()
    v = logits_row.float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits_row.float(), K).values
    return (v - ref).abs().max().item() < 1e-3


def time_variant(K, dtype, N, cr_val, kCC, kFT, reps=60):
    b = get_bundle(K, dtype, N, cfg="beta_moderate", seed=0)
    logits = b["logits"].to(dtype).contiguous(); pre = b["preIdx"].contiguous()
    seq_lens = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
    out = torch.empty(1, K, dtype=torch.int32, device="cuda")
    comp = _compile(dtype, 1, N, K, cr_val, kCC, kFT)
    comp(logits, pre, seq_lens, None, out); torch.cuda.synchronize()
    ok = _exact(out, logits[0], K)
    us = _cold_us(lambda: comp(logits, pre, seq_lens, None, out), reps=reps)
    return us, ok


CONFIGS = {
    512:  [("base", None, None), ("kc2x", 1024, 1024), ("kc3x", 1536, 1280)],
    1024: [("base", None, None), ("kc2x", 2048, 2048), ("kc3x", 3072, 2560)],
    2048: [("base", None, None), ("kc2x", 4096, 3686), ("kc3x", 6144, 3686)],
}
N_BY_K = {
    512: [4096, 8192, 16384, 32768, 65536, 131072, 262144],
    1024: [4096, 8192, 16384, 32768, 65536, 131072, 262144],
    2048: [8192, 16384, 32768, 65536, 131072, 262144],
}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--dt", default="fp32")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--ns", default="")  # comma list to subset N
    args = ap.parse_args()
    K = args.K; cr_val = 4 if K in (512, 1024) else 1
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dt]
    ns = [int(x) for x in args.ns.split(",")] if args.ns else N_BY_K[K]
    print(f"=== K={K} {args.dt} cr={cr_val} : WALL-TIME cold-L2 A/B (real modified kernel) ===")
    print(f"{'N':>8} | {'variant':6s} {'wall_us':>8s} {'exact':>5s} {'vs_base':>8s}")
    for N in ns:
        base_us = None
        for label, kcc, kft in CONFIGS[K]:
            us, ok = time_variant(K, dtype, N, cr_val, kcc, kft, reps=args.reps)
            if label == "base":
                base_us = us; rel = ""
            else:
                d = us - base_us
                rel = f"{d:+.2f} ({'WIN' if d < -0.3 else 'loss' if d > 0.3 else '~'})"
            print(f"{N:>8} | {label:6s} {us:8.2f} {str(ok):>5s} {rel:>8s}")
