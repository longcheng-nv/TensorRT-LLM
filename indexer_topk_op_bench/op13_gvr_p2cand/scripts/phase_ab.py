# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""DECISIVE A/B: does narrowing kCC (fewer P4 candidates) actually cut P3+P4
on the REAL plain-snap gvr_cutedsl kernel — and by more than the P2-eval cost?

Reuses harness/measure_cute_phases (clock64 P1-P4 split on the production
cold-L2 wall-us). Builds a kCC/kFTarget-overridable timed kernel so we can
compare BASELINE vs NARROWED candidate windows on identical cold input data.

This is the experiment whose premise the whole task rests on. op12 iter3 ran it
on the op#7 rank-scatter P4 kernel (P4 found barrier-bound). Here we run it on
the user's named base op (plain snap P4) to confirm/refute independently.
"""
import sys
from pathlib import Path

import torch
import cutlass

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parents[1] / "ops"))
import measure_cute_phases as mp  # noqa: E402
from measure_cute_phases import GvrTopKKernelTimed, _DT, _config  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as cr  # noqa: E402
from synth_data import get_bundle  # noqa: E402

_EVICT = mp._EVICT


class GvrTimedOverride(GvrTopKKernelTimed):
    """Timed kernel with kC (candidate cap / P2 accept upper bound) + kFTarget
    overridable post-ctor (both are plain attrs read as const_expr at compile)."""
    def __init__(self, *a, kC_override=None, kFTarget_override=None, **kw):
        super().__init__(*a, **kw)
        if kC_override is not None:
            self.kC = int(kC_override)
        if kFTarget_override is not None:
            self.kFTarget = int(kFTarget_override)


def build(dtype, N, K, cr_val, kC_ovr=None, kFT_ovr=None):
    t, use256, min_bpm = _config(1, N)
    kobj = GvrTimedOverride(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
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


def phase_split(compiled, logits, pre, seq_lens, K, prod_us, reps=40):
    out = torch.empty(1, K, dtype=torch.int32, device="cuda")
    ts = torch.zeros(1, 6, dtype=torch.int64, device="cuda")
    for _ in range(5):
        compiled(logits, pre, seq_lens, None, out, ts)
    torch.cuda.synchronize()
    rows = []
    for _ in range(reps):
        _EVICT.uniform_()
        compiled(logits, pre, seq_lens, None, out, ts)
        torch.cuda.synchronize()
        rows.append(ts[0].cpu().tolist())
    rows.sort(key=lambda r: r[5] - r[0])
    t = rows[len(rows) // 2]
    cyc = dict(P1=t[1]-t[0], P2=t[2]-t[1], P3=t[3]-t[2], P4=t[4]-t[3], end=t[5]-t[4])
    tot = sum(cyc.values())
    return {k: cyc[k]/tot*prod_us for k in cyc}, out


def run(K, dtype, N, cr_val, kC_ovr, kFT_ovr, reps=40):
    from gvr_cutedsl_op import gvr_cutedsl
    b = get_bundle(K, dtype, N, cfg="beta_moderate", seed=0)
    logits = b["logits"].to(dtype).contiguous(); pre = b["preIdx"].contiguous()
    seq_lens = torch.full((1,), N*cr_val, dtype=torch.int32, device="cuda")
    out = torch.empty(1, K, dtype=torch.int32, device="cuda")
    # trusted absolute cold-L2 wall-us of the PRODUCTION (uninstrumented) op
    prod_us = mp._cold_us(lambda: gvr_cutedsl(logits, pre, seq_lens, K, cr_val, out=out), reps=reps)
    comp = build(dtype, N, K, cr_val, kC_ovr=kC_ovr, kFT_ovr=kFT_ovr)
    ph, _ = phase_split(comp, logits, pre, seq_lens, K, prod_us, reps=reps)
    return prod_us, ph


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--dt", default="fp32")
    args = ap.parse_args()
    K = args.K
    cr_val = 4 if K in (512, 1024) else 1
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dt]

    # baseline kC, plus narrowed variants matching the host-sweep cand targets
    NARROW = {512: int(1.25*512), 1024: int(1.25*1024), 2048: int(1.5*2048)}
    kc_n = NARROW[K]
    NS = [4096, 8192, 16384, 65536, 262144]
    NS = [n for n in NS if n > 2*K]
    print(f"=== K={K} {args.dt} cr={cr_val} : BASELINE vs NARROWED kCC={kc_n} (~{kc_n/K:.2f}xK) ===")
    print(f"{'N':>8} | {'variant':9s} | {'tot':>6s} {'P1':>5s} {'P2':>6s} {'P3':>6s} {'P4':>5s} | dP3+P4  dP2  dTOT")
    for N in NS:
        _, base = run(K, dtype, N, cr_val, kC_ovr=None, kFT_ovr=None)
        _, narr = run(K, dtype, N, cr_val, kC_ovr=kc_n, kFT_ovr=min(kc_n, K))
        bt = sum(base.values()); nt = sum(narr.values())
        d_p34 = (narr['P3']+narr['P4']) - (base['P3']+base['P4'])
        d_p2 = narr['P2'] - base['P2']
        d_tot = nt - bt
        print(f"{N:>8} | {'baseline':9s} | {bt:6.2f} {base['P1']:5.2f} {base['P2']:6.2f} {base['P3']:6.2f} {base['P4']:5.2f} |")
        print(f"{'':>8} | {'narrow':9s} | {nt:6.2f} {narr['P1']:5.2f} {narr['P2']:6.2f} {narr['P3']:6.2f} {narr['P4']:5.2f} | "
              f"{d_p34:+5.2f}  {d_p2:+5.2f}  {d_tot:+5.2f}")
