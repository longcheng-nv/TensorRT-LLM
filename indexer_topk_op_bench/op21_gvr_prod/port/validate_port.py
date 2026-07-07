#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""PR-1 port validation: exactness-gate the ASSEMBLED gvr_topk_decode_ms.py
(imported through portshim/, i.e. exactly the file that ships) against the
same gates the bench kernel passed at iter11, plus a bitwise cross-check
against the bench ops (byte-identical composition => identical outputs).

Gates:
  1. synth fp32       K{512,1024,2048} x N{8192,65536,262144} x BS{1,16} x 3 seeds
  2. synth 16-bit     spot grid (bf16/fp16, native ladder ON)
  3. adversarial band 72 cases (iter11 gate; ms/C4/C8, production preIdx dialect)
  4. real captures    60 layers (pro 30 + flash 21 + v32 9) x {ms, C4, C8}
  5. selection-identity vs bench gvr_ms / gvr_msc on a spot grid: SORTED
     index sets must match bitwise. (Positional order is NOT compared:
     the collect cursor is atomic, so the within-row permutation is
     run-to-run nondeterministic — the bench kernel itself permutes
     across identical back-to-back calls. Inherited GVR behavior, the
     selected SET is bit-stable.)

Usage: python3 validate_port.py [--quick]   (--quick = gates 1+3+5, fp32 only)
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BUCKET = _HERE.parent
_BENCH = _BUCKET.parent
sys.path.insert(0, str(_BENCH / "ops"))       # cute_vendored (shim targets)
sys.path.insert(0, str(_BENCH / "harness"))   # synth_data / real_data_v2
sys.path.insert(0, str(_BUCKET / "src"))      # bench ops (cross-check)
sys.path.insert(0, str(_BUCKET / "scripts"))  # adversarial make_row
sys.path.insert(0, str(_HERE))                # portshim

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

from portshim.top_k.gvr_topk_decode_ms import (  # noqa: E402
    GvrMsKernel, GvrMsClusterKernel)

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
       torch.float16: cutlass.Float16}
_compiled = {}


def _cute_compile(kobj, use256):
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if use256 else 16
    in_f = crt.make_fake_compact_tensor(kobj.dtype, (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, kobj.top_k), stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, kobj.top_k), stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, stream=fs,
                        options="--enable-tvm-ffi")


def ms_port(logits, pre_idx, seq_lens, K, cr, next_n=1, out=None):
    """Host wrapper mirroring the bench gvr_ms production entry (mode-5,
    M=4 R=1, fuse gate bs<=SMs && 4K<=kC) on the PORTED single-CTA class."""
    bs, n = logits.shape
    if out is None:
        out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = n >= 16384
    min_bpm = 1 if bs <= NUM_SMS else 3
    fuse = bool(bs <= NUM_SMS and 4 * K <= 5120)
    key = ("ms", logits.dtype, t, use256, min_bpm, K, cr, fuse, next_n)
    if key not in _compiled:
        kobj = GvrMsKernel(
            dtype=_DT[logits.dtype], top_k=K, next_n=next_n, num_threads=t,
            compress_ratio=cr, use_256bit_load=use256,
            min_blocks_per_mp=min_bpm, fuse_collect=fuse)
        _compiled[key] = _cute_compile(kobj, use256)
    _compiled[key](logits, pre_idx, seq_lens, None, out)
    return out


def msc_port(logits, pre_idx, seq_lens, K, cr, C=4, next_n=1, out=None):
    """Host wrapper mirroring the bench gvr_msc (threads=1024, fuse=True)."""
    bs, n = logits.shape
    if out is None:
        out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
    use256 = n >= 16384
    key = ("msc", logits.dtype, use256, K, cr, C, next_n)
    if key not in _compiled:
        kobj = GvrMsClusterKernel(
            dtype=_DT[logits.dtype], top_k=K, next_n=next_n, num_threads=1024,
            compress_ratio=cr, use_256bit_load=use256, min_blocks_per_mp=1,
            cluster_size=C)
        _compiled[key] = _cute_compile(kobj, use256)
    _compiled[key](logits, pre_idx, seq_lens, None, out)
    return out


def _check(out, lg, K, value_metrics):
    ref = torch.topk(lg[0].float(), K).indices
    vd, _rc, nn = value_metrics(out[:1], lg[:1].float(), ref, K)
    u = torch.unique(out[0][out[0] >= 0]).numel()
    return (vd == 0 and nn == 0 and u == K), vd, u


def main():
    quick = "--quick" in sys.argv
    from real_data_v2 import value_metrics  # noqa: E402
    import synth_data  # noqa: E402
    total_bad = 0

    # ---- gate 1: synth fp32 (mirrors scripts/smoke_exact.py) ----
    ok = bad = 0
    for K in (512, 1024, 2048):
        for N in (8192, 65536, 262144):
            if N <= 2 * K:
                continue
            for BS in (1, 16):
                for seed in (42, 7, 1234):
                    b = synth_data.get_bundle(K, torch.float32, N, seed=seed)
                    lg = b["logits"][:1].repeat(BS, 1).contiguous()
                    pre = b["preIdx"][:1].repeat(BS, 1).contiguous()
                    cr = b["cr"]
                    sl = torch.full((BS,), N * cr if cr > 1 else N,
                                    dtype=torch.int32, device="cuda")
                    out = ms_port(lg, pre, sl, K, cr)
                    torch.cuda.synchronize()
                    good, vd, u = _check(out[:, :], lg[:, :N], K, value_metrics)
                    ok += good
                    bad += not good
                    if not good:
                        print(f"FAIL synth K{K} N{N} BS{BS} s{seed}: vd={vd} u={u}")
    print(f"[gate1] synth fp32 ms: {ok} ok / {bad} fail")
    total_bad += bad

    # ---- gate 3: adversarial band (iter11 gate, ms/C4/C8) ----
    from smoke_adversarial_band import make_row  # noqa: E402
    ok = bad = 0
    CASES = []
    for K, N, cr in ((1024, 262144, 4), (1024, 131072, 4), (512, 262144, 4),
                     (2048, 262144, 1)):
        for seed in (11, 22, 33):
            for af in (0.4, 0.55):
                CASES.append((K, N, cr, seed, af))
    for K, N, cr, seed, af in CASES:
        row, pre = make_row(K, N, seed, above_frac=af)
        if cr == 1:
            pre = (pre - 1).clamp(min=0)
        lg = row.unsqueeze(0).contiguous()
        pr = pre.unsqueeze(0).contiguous()
        sl = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
        for tag, fn in (("ms", lambda: ms_port(lg, pr, sl, K, cr)),
                        ("C4", lambda: msc_port(lg, pr, sl, K, cr, C=4)),
                        ("C8", lambda: msc_port(lg, pr, sl, K, cr, C=8))):
            o = fn()
            torch.cuda.synchronize()
            good, vd, u = _check(o, lg, K, value_metrics)
            ok += good
            bad += not good
            if not good:
                print(f"FAIL adv {tag} K{K} N{N} s{seed} af{af}: vd={vd:.3e} u={u}")
    print(f"[gate3] adversarial band: {ok} ok / {bad} fail")
    total_bad += bad

    # ---- gate 5: selection identity vs bench ops (spot grid) ----
    from gvr_ms_op import gvr_ms as bench_ms  # noqa: E402
    from gvr_msc_op import gvr_msc as bench_msc  # noqa: E402
    ok = bad = 0
    for K, N in ((512, 262144), (1024, 131072), (2048, 262144), (1024, 8192)):
        cr = 1 if K == 2048 else 4
        b = synth_data.get_bundle(K, torch.float32, N, seed=42)
        lg, pre = b["logits"][:1].contiguous(), b["preIdx"][:1].contiguous()
        sl = torch.full((1,), N * cr if cr > 1 else N, dtype=torch.int32,
                        device="cuda")
        pairs = [("ms", ms_port(lg, pre, sl, K, cr),
                  bench_ms(lg, pre, sl, K, compress_ratio=cr))]
        if N >= 65536:
            pairs.append(("C4", msc_port(lg, pre, sl, K, cr, C=4),
                          bench_msc(lg, pre, sl, K, cr, C=4)))
        torch.cuda.synchronize()
        for tag, o_p, o_b in pairs:
            same = bool(torch.equal(o_p.sort(dim=-1).values,
                                    o_b.sort(dim=-1).values))
            ok += same
            bad += not same
            if not same:
                print(f"SETDIFF {tag} K{K} N{N}: selected index sets differ")
    print(f"[gate5] selection identity vs bench: {ok} ok / {bad} fail")
    total_bad += bad

    if not quick:
        # ---- gate 2: synth 16-bit spot (native ladder ON by default) ----
        ok = bad = 0
        for dt in (torch.bfloat16, torch.float16):
            for K, N in ((512, 262144), (1024, 131072), (2048, 65536)):
                cr = 1 if K == 2048 else 4
                for BS in (1, 8):
                    b = synth_data.get_bundle(K, dt, N, seed=7)
                    lg = b["logits"][:1].repeat(BS, 1).contiguous()
                    pre = b["preIdx"][:1].repeat(BS, 1).contiguous()
                    sl = torch.full((BS,), N * cr if cr > 1 else N,
                                    dtype=torch.int32, device="cuda")
                    for tag, fn in (("ms", lambda: ms_port(lg, pre, sl, K, cr)),
                                    ("C8", lambda: msc_port(lg, pre, sl, K, cr, C=8))):
                        o = fn()
                        torch.cuda.synchronize()
                        good, vd, u = _check(o[:, :], lg[:, :N], K, value_metrics)
                        ok += good
                        bad += not good
                        if not good:
                            print(f"FAIL 16b {tag} {dt} K{K} N{N} BS{BS}: vd={vd} u={u}")
        print(f"[gate2] synth 16-bit: {ok} ok / {bad} fail")
        total_bad += bad

        # ---- gate 4: real captures x {ms, C4, C8} ----
        import real_data_v2  # noqa: E402
        ok = bad = 0
        for model, layers in (("pro", range(2, 61, 2)),
                              ("flash", range(2, 43, 2)),
                              ("v32", (0, 1, 20, 21, 22, 40, 41, 42, 60))):
            for L in layers:
                b = real_data_v2.get_real_bundle_v2(model, L, "fp32")
                K, cr, N = b["K"], b["cr"], b["N"]
                lg = b["logits"][:, :].contiguous()
                sl = torch.tensor([N * cr if cr > 1 else N], dtype=torch.int32,
                                  device="cuda")
                for tag, fn in (("ms", lambda: ms_port(lg, b["preIdx"], sl, K, cr)),
                                ("C4", lambda: msc_port(lg, b["preIdx"], sl, K, cr, C=4)),
                                ("C8", lambda: msc_port(lg, b["preIdx"], sl, K, cr, C=8))):
                    o = fn()
                    torch.cuda.synchronize()
                    vd, _rc, nn = value_metrics(o, lg[:, :N].float(), b["ref"], K)
                    u = torch.unique(o[0][o[0] >= 0]).numel()
                    good = (vd == 0 and nn == 0 and u == K)
                    ok += good
                    bad += not good
                    if not good:
                        print(f"FAIL real {tag} {model} L{L}: vd={vd:.2e} u={u}")
        print(f"[gate4] real x C: {ok} ok / {bad} fail")
        total_bad += bad

    # gate 6 (next_n + varlen) is standalone: run_gate6_nextn.py
    print(f"validate_port: {'ALL GREEN' if total_bad == 0 else f'{total_bad} FAILURES'}")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
