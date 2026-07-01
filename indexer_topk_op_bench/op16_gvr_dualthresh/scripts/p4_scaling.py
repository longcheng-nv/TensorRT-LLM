# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""DECISIVE B300 experiment: does rank-scatter P4 cost scale with candidate
count, or is it floor-bound (op12's B200 claim)?

Method: reuse the clock64-instrumented rank-scatter kernel
(harness/measure_cute_phases_rs.GvrTopKKernelRsTimed) but subclass it to
override kC (the P2 accept-window upper bound AND candidate cap). Smaller kC
=> higher converged threshold => fewer candidates fed to P4. We read P4 CYCLES
directly from phase_ts (ts[4]-ts[3]) — no wall-us conversion needed to answer
the floor-vs-count question, since cycles are directly comparable across kC.

We also emit the actual cand_count by writing s_iscalars[0] into phase_ts slot
via a tiny kernel-body-free trick: we run op13's host count at the converged
threshold is unavailable here, so instead we sweep kC and report P4 cycles vs
kC; the cand monotonically decreases with kC (op14: K512 default cand ~2.1-2.7k
at kC=5120). Flat P4 cycles across kC => floor-bound.
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
_HARNESS = _HERE.parent.parent / "harness"
sys.path.insert(0, str(_HARNESS))

from measure_cute_phases_rs import (  # noqa: E402
    GvrTopKKernelRsTimed, _DT, _config, _cold_us, _EVICT, NUM_SMS)


class GvrRsTimedOverride(GvrTopKKernelRsTimed):
    """Rank-scatter timed kernel with kC / kFTarget override (no body edit)."""

    def __init__(self, *a, kC_override=None, kFTarget_override=None, **kw):
        super().__init__(*a, **kw)
        if kC_override is not None:
            self.kC = int(kC_override)
        if kFTarget_override is not None:
            self.kFTarget = int(kFTarget_override)


def _compile_ovr(dtype, n, K, cr_val, kC_ovr, kFT_ovr):
    t, use256, min_bpm = _config(1, n)
    kobj = GvrRsTimedOverride(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
        enable_p4_rank_scatter=True, enable_p4_rank_scatter_exact=True,
        kC_override=kC_ovr, kFTarget_override=kFT_ovr)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    in_align = 32 if use256 else 16
    in_f = cr.make_fake_compact_tensor(_DT[dtype], (nr, nc), stride_order=(1, 0), assumed_align=in_align)
    pi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    ts_f = cr.make_fake_compact_tensor(cutlass.Int64, (nr, 6), stride_order=(1, 0), assumed_align=16)
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, ts_f, stream=fs, options="--enable-tvm-ffi")


def run(K, dtype, N, cr_val, kC_kFT_list):
    from synth_data import get_bundle
    b = get_bundle(K, dtype, N, cfg="beta_moderate", seed=0)
    logits = b["logits"].to(dtype).contiguous()
    pre = b["preIdx"].contiguous()
    seq_lens = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
    out = torch.empty(1, K, dtype=torch.int32, device="cuda")
    ts = torch.zeros(1, 6, dtype=torch.int64, device="cuda")
    print(f"\n=== K={K} {dtype} N={N} cr={cr_val} : P4 cycles vs (kC,kFT) [eval-optimal kFT] ===")
    print(f"{'kC':>6} {'kFT':>6} | {'P1_cyc':>8} {'P2_cyc':>9} {'P3_cyc':>9} {'P4_cyc':>9} {'tot_cyc':>9} | {'exact':>6}")
    for kC, kFT in kC_kFT_list:
        compiled = _compile_ovr(dtype, N, K, cr_val, kC, kFT)
        def call():
            compiled(logits, pre, seq_lens, None, out, ts)
        # warm run to fill ts + check exactness
        call(); torch.cuda.synchronize()
        idx = out[0].clamp(min=0).long()
        v = logits[0].float().gather(0, idx).sort(descending=True).values
        ref = torch.topk(logits[0].float(), K).values
        d = (v - ref).abs().max().item()
        nuniq = len(set(out[0].tolist()))
        exact = (d < 1e-5 and nuniq >= K)
        # median cycle counts over a few runs
        p1s, p2s, p3s, p4s, tts = [], [], [], [], []
        for _ in range(9):
            _EVICT.uniform_()
            call(); torch.cuda.synchronize()
            t = ts[0].tolist()
            p1s.append(t[1]-t[0]); p2s.append(t[2]-t[1]); p3s.append(t[3]-t[2])
            p4s.append(t[4]-t[3]); tts.append(t[5]-t[0])
        med = lambda x: sorted(x)[len(x)//2]
        print(f"{kC:>6} {kFT:>6} | {med(p1s):>8d} {med(p2s):>9d} {med(p3s):>9d} {med(p4s):>9d} {med(tts):>9d} | {str(exact):>6}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--dt", default="fp32")
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--N", type=int, nargs="+", default=[4096, 16384, 65536])
    args = ap.parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dt]
    # (kC, kFT) pairs. Baseline (5120,512) + op13 eval-optimal winners for K=512:
    # kc3x=(1536,1280), kc2x=(1024,1024), plus a tight (768,640) to probe the floor.
    # eval-optimal kFT (near kCC) minimizes secant evals per kC (op13 host pre-pass).
    K = args.K
    pairs = [(5120, 512), (2048, 2048), (1536, 1280), (1024, 1024), (768, 640)]
    for N in args.N:
        run(K, dtype, N, args.cr, pairs)
