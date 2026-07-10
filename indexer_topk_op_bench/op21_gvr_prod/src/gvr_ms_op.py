# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op21: rank-quantile sandwich GVR top-K (single-CTA, production-shaped).

Base = op20 gvr_x (op19 sandwich + fused P2+P3 slot-collect). op21 delta:
place_mode=5 "rank quantile" round-0 placement — the M ladder thresholds are
ORDER STATISTICS of the K gathered prev-step values (256-bin in-smem
histogram quantiles at qfracs of the valid count), NOT value-space fractions
from an offline per-(K,N,dtype) straddle table. Column 0 = v_lo = g_min (a
guaranteed count>=K_valid anchor: all K gathered positions hold values >=
g_min). Host-prototype validation (iter0.5, 55 real+synth rows): straddle
94.5% overall / 100% on V4 Pro; only miss mode = all_ge (round-2/fallback).
Distribution-free => no offline table, N drops out of placement, fail-soft
under hit-rate/distribution shift. Original op19 header follows.

op19: sandwich two-threshold GVR top-K (single-CTA, Strategy-A).

Extends op18's M-ary multi-threshold P2 (GvrMultiThreshKernel). While the
ladder rounds run, additionally snapshot the SANDWICH upper threshold:
  thr1 = tightest evaluated threshold with count >= K      (op18's best)
  thr0 = evaluated threshold with count < K, max count M0  (NEW)
Both per-thread count columns are cached from the same scans (zero extra
passes). Every element >= thr0 is a GUARANTEED top-K member (only M0 < K
elements are >= thr0, and they dominate everything below thr0).

Phase-3 (sandwich): ONE scan with two predicates and two prefix-summed
cursors — v >= thr0 direct-writes its index to the output row (positions
0..M0-1); thr1 <= v < thr0 goes to smem as a band candidate. Phase-4 then
selects only k_rem = K - M0 winners from band = M1 - M0 candidates with a
runtime-k histogram snap whose value range is seeded [thr1, thr0) (no
min/max pass). Accept unlock vs op18: smem only needs the BAND, so done=1
requires band <= kC instead of M1 <= kC.

Fallback: no sandwich pair (M0 == 0) -> exact op18 path (P3 collect-all +
const-K P4). done=2 (band > kC) -> baseline retry-shrink path. Exact.
"""
import math
import os
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "ops"))
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_BENCH / "op18_gvr_1cta_multithresh" / "src"))
from gvr_mt_op import GvrMultiThreshKernel, _load_fracs  # noqa: E402
from cute_vendored.blackwell.top_k.gvr_topk_decode import (  # noqa: E402
    _fmin_f32_inline, atomicAdd, float_as_uint32,
)
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
from cutlass._mlir.dialects import llvm  # noqa: E402
from cutlass.cutlass_dsl import T, dsl_user_op  # noqa: E402
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
       torch.float16: cutlass.Float16}
_INT_MAX = 0x7FFFFFFF


# ---------------------------------------------------------------------------
# iter9 native 16-bit compare primitives (inline PTX; `kind` is a trace-time
# python str "bf16"|"f16"). Microbench (probe/count16_native.cu, B200):
# set.ge.{bf16x2,f16x2} + add.rn 16x2 accumulate = 1.73x over cvt->fp32 at
# N262K single-CTA, 1.21x at the C8 slice; counts bit-match the fp32 path
# when thresholds are pre-quantized to the dtype grid.
# ---------------------------------------------------------------------------
def _k16(kind):
    return "bf16" if kind == "bf16" else "f16"


@dsl_user_op
def _quant_f32_16(f, kind, *, loc=None, ip=None):
    k = _k16(kind)
    asm = ("{.reg .b16 h; cvt.rn." + k + ".f32 h, $1; cvt.f32." + k
           + " $0, h;}")
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [f.ir_value(loc=loc, ip=ip)], asm, "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def quant_f32_16(f, kind):
    return _quant_f32_16(f, kind)


@dsl_user_op
def _pack2_16_from_f32(f, kind, *, loc=None, ip=None):
    k = _k16(kind)
    asm = "{.reg .b16 h; cvt.rn." + k + ".f32 h, $1; mov.b32 $0, {h, h};}"
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [f.ir_value(loc=loc, ip=ip)], asm, "=r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def pack2_16_from_f32(f, kind):
    return _pack2_16_from_f32(f, kind)


@dsl_user_op
def _setge_add2_16(acc, v2, t2, kind, *, loc=None, ip=None):
    k = _k16(kind) + "x2"
    asm = ("{.reg .b32 o; set.ge." + k + "." + k + " o, $2, $3; add.rn."
           + k + " $0, $1, o;}")
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [acc.ir_value(loc=loc, ip=ip), v2.ir_value(loc=loc, ip=ip),
                  t2.ir_value(loc=loc, ip=ip)], asm, "=r,r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def setge_add2_16(acc, v2, t2, kind):
    return _setge_add2_16(acc, v2, t2, kind)


@dsl_user_op
def _setge_mask2_16(v2, t2, kind, *, loc=None, ip=None):
    asm = "set.ge.u32." + _k16(kind) + "x2 $0, $1, $2;"
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [v2.ir_value(loc=loc, ip=ip), t2.ir_value(loc=loc, ip=ip)],
        asm, "=r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def setge_mask2_16(v2, t2, kind):
    return _setge_mask2_16(v2, t2, kind)


@dsl_user_op
def _pair_half_f32_16(v2, hi, kind, *, loc=None, ip=None):
    k = _k16(kind)
    which = "hi" if hi else "lo"
    asm = ("{.reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32." + k
           + " $0, " + which + ";}")
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [v2.ir_value(loc=loc, ip=ip)], asm, "=f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def pair_half_f32_16(v2, hi, kind):
    return _pair_half_f32_16(v2, hi, kind)


@dsl_user_op
def _pair_sum_i32_16(acc2, kind, *, loc=None, ip=None):
    k = _k16(kind)
    asm = ("{.reg .b16 lo, hi; .reg .f32 a, b; mov.b32 {lo, hi}, $1; "
           "cvt.f32." + k + " a, lo; cvt.f32." + k + " b, hi; "
           "add.f32 a, a, b; cvt.rzi.s32.f32 $0, a;}")
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [acc2.ir_value(loc=loc, ip=ip)], asm, "=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def pair_sum_i32_16(acc2, kind):
    return _pair_sum_i32_16(acc2, kind)


@dsl_user_op
def _lg2_f32(f, *, loc=None, ip=None):
    # iter13 (HLS log-falsi): approx log2 for the fallback interpolant.
    # Base cancels in the falsi ratio; approx precision only shapes the
    # aim point — the bracket safeguard guarantees convergence regardless.
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [f.ir_value(loc=loc, ip=ip)], "lg2.approx.f32 $0, $1;",
        "=f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def lg2_f32(f):
    return _lg2_f32(f)


class GvrSandwichKernel(GvrMultiThreshKernel):
    """Sandwich two-threshold single-CTA kernel. New tunable: band_accept
    (stop refining once band <= band_accept; replaces op18 c_accept)."""

    def __init__(self, *a, band_accept=64, fuse_collect=False, smem_row_elems=0,
                 qfracs=(0.75, 0.5, 0.25), p4_rank_scatter=True, qbins=256,
                 p4_smallbin=True, p2_native=True, fb_logfalsi=True,
                 fb_alpha=0.2, slot_scale=1, **kw):
        super().__init__(*a, **kw)
        self.band_accept = int(band_accept)
        # op25 S1a: fused-collect slot-capacity scale. The deep collect
        # column (qfracs[0] ~0.92) that fixes the h>0.75 pair01 cliff pays a
        # mid-h overflow at the stock per-thread cap (host replay: Pro-real
        # h<0.5 bucket fast 0.07 at cap-model 4096 vs 0.94 at 8192); x2
        # doubles the slot smem (num_threads*slot_cap*8B, +40-64KB) which is
        # only allocated on the fused path (bs <= NUM_SMS) so high-BS
        # residency is untouched.
        self.slot_scale = max(1, int(slot_scale))
        # op21 iter13 (HLS): log-count regula-falsi fallback refine. The
        # fallback bracket [s_thr[1], s_thr[2]] arrives with ladder-KNOWN
        # counts (s_iscalars[5]/[6]); CCDF tails are ~exponential so log
        # count is ~linear in threshold — the host prototype (proto_hls.py,
        # 78 op22 rows, forced fallback) measured 1.00 mean / 1 max passes
        # vs bisect 1.77 mean / 6 max. Aim at the interior target
        # m* = K*(kC/K)^alpha, alpha=0.2 (HLS Theorem 3 grid optimum);
        # accepting anywhere in [K, kC] is unchanged. False = legacy blind
        # bisection (A/B knob OP21_FB_LOGFALSI=0).
        self.fb_logfalsi = bool(fb_logfalsi)
        # fb_alpha: interior aim exponent. 0.2 = the HLS grid optimum on
        # PASS COUNT; silicon (iter13 A/B) shows the accepted-count side
        # effect (P4 is cand-linear) — smaller alpha lands tighter cand at
        # the cost of more undershoot retries. OP21_FB_ALPHA probes it.
        self.fb_alpha = float(fb_alpha)
        self.log2_mstar = math.log2(
            self.top_k * (self.kC / self.top_k) ** self.fb_alpha)
        # iter9: native 16-bit ladder compares. Thresholds are quantized to
        # the dtype grid at P1b emit (thr_q = f32(dtype(thr))), which makes
        # 16-bit-domain compares bit-equivalent to the fp32 compares every
        # other phase performs on the exactly-embedded values (microbench
        # counts matched on all configs). The M-column counts accumulate in
        # packed 16x2 lanes (set.ge + add.rn), flushed to int32 every 16 vec
        # iters (per-half growth <= 8/iter => <= 128 << the 256 bf16 integer
        # grid). The collect column uses a packed mask (set.ge.u32) so the
        # slot cursor stays exact per element. fp32 path untouched.
        self.p2_native = bool(p2_native)
        # op21 iter5 (b): P1b quantile-histogram bin count. 256 default; 64 at
        # bs > NUM_SMS (production-legal BS rule) cuts the per-row fixed cost
        # (suffix-scan 8 -> 6 double-barrier steps, 4x less zero/crossing
        # work) where rows-per-SM > 1 makes it the dominant term. Seeds only
        # get coarser (wider band); exactness authority unchanged.
        self.qbins = int(qbins)
        assert self.qbins in (64, 128, 256) and self.num_threads >= self.qbins
        # op21 iter5: exact rank-scatter band refine (op8 P4 port) instead of
        # the runtime-k snap-convergence loop. iter4 ablation pinned the snap
        # at 3.9-7us (== the whole remaining rival gap at the P0 holes); the
        # rank-scatter replaces the data-dependent snap iterations with one
        # fine 256-bin recursion + a single scatter pass. False = old snap
        # (kept as the A/B reference).
        self.p4_rank_scatter = bool(p4_rank_scatter)
        # op21 iter6: small-bin P4 fast paths. Host probe (68 synth+real
        # rows): cnt(b*) p50=2 p90=3 max=4 with band~1K in >=1024 coarse
        # bins, so (B) cnt(b*)<=32 -> warp0 register ranking (31-step
        # shuffle ring, exact positions, no fine hist) covers 100%; (A)
        # rank_above+cnt(b*)==k_rem -> whole-bin emit covers big-bin
        # equality; (C) distribution-shift fallback = the EXACT value-edge
        # band snap (iter11: the fixed-depth fine recursion was falsified
        # inexact — smoke_adversarial_band.py). False = fast paths off,
        # always snap.
        self.p4_smallbin = bool(p4_smallbin)
        # op21: rank fractions for place_mode=5 quantile placement, mapped to
        # ladder columns 1..M-1 (descending rank => ascending value). Column 1
        # (qfracs[0]) doubles as the fused-collect column (pred_col=1).
        self.qfracs = tuple(float(f) for f in qfracs)
        self.QBINS = self.qbins
        if self.place_mode == 5:
            assert len(self.qfracs) == self.M_thr - 1, "qfracs must be M-1 long"
            assert int(smem_row_elems) == 0, "mode5 v1: no smem-row path"
            # phase1 stashes the K gathered prev-step values into the (still
            # unused) P3 slot buffer so phase1b histograms from smem instead
            # of re-gathering K L2 loads per row (25%+ extra L2 traffic at
            # small N — the whole p5-vs-p4 overhead at high BS).
            assert self.kC >= self.top_k, "mode5 stash needs kC >= K"
        # op20 iter6: smem-resident row for small N — the whole row is bulk-
        # loaded into smem CONCURRENTLY with P1's gmem gather (both issue in
        # one cold-DRAM latency window), and the ladder pass then reads smem,
        # collapsing the cold critical chain from 2 serial trips to 1.
        # 0 = disabled; >0 = compile-time smem row capacity (elements).
        self.smem_row_elems = int(smem_row_elems)
        assert self.smem_row_elems == 0 or fuse_collect, \
            "smem_row requires the fused P2+P3 path"
        # op20 iter4: fused P2+P3 — during the (single) ladder pass, also
        # append every v >= thr[pred_col] into per-thread smem slots, where
        # pred_col = the l1 straddle column (tightest frac with count>=K by
        # offline design; ascending fracs => index 1 when the 0-anchor is
        # present, else 0). Valid ONLY for R==1 + straddle placement: all
        # thresholds are known before the scan (unlike falsified Opt-L where
        # secant had to converge first). If usable, P3's full-N rescan is
        # replaced by a per-thread walk of <= slot_cap collected entries.
        self.fuse_collect = bool(fuse_collect)
        self.pred_col = 1 if self.M_thr >= 3 else 0
        # slot_cap needs headroom over the per-thread mean lambda=cand/threads
        # (lambda + ~4*sqrt(lambda)); at t=1024 kC/nt=5 overflows constantly
        # (measured N65536 BS64 -7%), so floor at 8. op25: x slot_scale.
        self.slot_cap = max(8, self.kC // self.num_threads) * self.slot_scale
        # deferred direct-write stores values nowhere; indices-only op
        assert not self.return_output_values, "sandwich is indices-only"

    # trace-time helpers for the iter9 native 16-bit ladder path
    def _p2n(self):
        return self.p2_native and self.dtype != cutlass.Float32

    def _kind16(self):
        return "bf16" if self.dtype == cutlass.BFloat16 else "f16"

    def _make_load_copy_atom_u32(self):
        # same vec width/caching as _make_load_copy_atom, u32-typed view
        # (16-bit pairs land packed in 32-bit registers, no repack cost)
        if self.use_constant_hint:
            return cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), cutlass.Int32,
                num_bits_per_copy=self.vec_bits, invariant=True)
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32,
            num_bits_per_copy=self.vec_bits)

    # ------------------------------------------------------------------
    # op20 iter4: fused ladder pass — block_count_ge_multi + slot-append of
    # every v >= thr[pred_col] into per-thread smem slot regions. Slot cursor
    # IS cnt_frag[pred_col] (pre-increment), so overflow == cnt > slot_cap,
    # detectable from the cached per-thread column after the pass.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_collect_multi(
        self, input_row, N, s_mt_thr, smem_ptcnt_multi, smem_wcnt_multi,
        s_mt_cnt, smem_slotk, smem_slotv, tidx, warp_id, lane,
    ):
        M = cutlass.const_expr(self.M_thr)
        PC = cutlass.const_expr(self.pred_col)
        S = cutlass.const_expr(self.slot_cap)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        thr_frag = cute.make_fragment((M,), cutlass.Float32)
        cnt_frag = cute.make_fragment((M,), cutlass.Int32)
        for m in cutlass.range_constexpr(M):
            thr_frag[m] = s_mt_thr[m]
            cnt_frag[m] = cutlass.Int32(0)
        slot_base = tidx * cutlass.Int32(S)

        row_addr = input_row.iterator.toint()
        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_unroll_4:
            big_iters = cutlass.Int32(0)
            if N > i + cutlass.Int32(vec_w - 1):
                big_iters = (N - i - cutlass.Int32(vec_w)) // cutlass.Int32(step_elem) + cutlass.Int32(1)
            if cutlass.const_expr(self._p2n()):
                # iter9: native 16-bit paired path — u32-typed loads (pairs
                # already packed), set.ge/add.rn 16x2 counts for m != PC,
                # packed mask for the collect column (exact slot cursor).
                kind = self._kind16()
                vec_w2 = cutlass.const_expr(vec_w // 2)
                copy_atom_u32 = self._make_load_copy_atom_u32()
                rng2_frag = cute.make_fragment((vec_w2,), cutlass.Int32)
                thr2_frag = cute.make_fragment((M,), cutlass.Int32)
                acc2_frag = cute.make_fragment((M,), cutlass.Int32)
                for m in cutlass.range_constexpr(M):
                    thr2_frag[m] = pack2_16_from_f32(thr_frag[m], kind)
                    acc2_frag[m] = cutlass.Int32(0)
                ccol = cutlass.Int32(0)
                for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                    i_local = i + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        cutlass.Int32, row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem, assumed_align=vec_align)
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w2,)))
                    cute.copy(copy_atom_u32, src_k, rng2_frag)
                    for p in cutlass.range_constexpr(vec_w2):
                        v2 = rng2_frag[p]
                        mpc = setge_mask2_16(v2, thr2_frag[PC], kind)
                        if mpc != cutlass.Int32(0):
                            if (mpc & cutlass.Int32(0xFFFF)) != cutlass.Int32(0):
                                if ccol < cutlass.Int32(S):
                                    smem_slotk[slot_base + ccol] = pair_half_f32_16(v2, 0, kind)
                                    smem_slotv[slot_base + ccol] = i_local + cutlass.Int32(2 * p)
                                ccol = ccol + cutlass.Int32(1)
                            if (mpc >> cutlass.Int32(16)) != cutlass.Int32(0):
                                if ccol < cutlass.Int32(S):
                                    smem_slotk[slot_base + ccol] = pair_half_f32_16(v2, 1, kind)
                                    smem_slotv[slot_base + ccol] = i_local + cutlass.Int32(2 * p + 1)
                                ccol = ccol + cutlass.Int32(1)
                        for m in cutlass.range_constexpr(M):
                            if cutlass.const_expr(m != self.pred_col):
                                acc2_frag[m] = setge_add2_16(acc2_frag[m], v2, thr2_frag[m], kind)
                    if (k & cutlass.Int32(15)) == cutlass.Int32(15):
                        for m in cutlass.range_constexpr(M):
                            if cutlass.const_expr(m != self.pred_col):
                                cnt_frag[m] = cnt_frag[m] + pair_sum_i32_16(acc2_frag[m], kind)
                                acc2_frag[m] = cutlass.Int32(0)
                for m in cutlass.range_constexpr(M):
                    if cutlass.const_expr(m != self.pred_col):
                        cnt_frag[m] = cnt_frag[m] + pair_sum_i32_16(acc2_frag[m], kind)
                cnt_frag[PC] = ccol
            else:
                rng_frag = cute.make_fragment((vec_w,), self.dtype)
                for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                    i_local = i + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        self.dtype, row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem, assumed_align=vec_align)
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                    cute.copy(copy_atom, src_k, rng_frag)
                    for j in cutlass.range_constexpr(vec_w):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj = rng_frag[j]
                        else:
                            vj = cutlass.Float32(rng_frag[j])
                        if vj >= thr_frag[PC]:
                            cpos = cnt_frag[PC]
                            if cpos < cutlass.Int32(S):
                                smem_slotk[slot_base + cpos] = vj
                                smem_slotv[slot_base + cpos] = i_local + cutlass.Int32(j)
                        for m in cutlass.range_constexpr(M):
                            cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < N:
            src_ptr = cute.make_ptr(
                self.dtype, row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem, assumed_align=vec_align)
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= thr_frag[PC]:
                    cpos = cnt_frag[PC]
                    if cpos < cutlass.Int32(S):
                        smem_slotk[slot_base + cpos] = vj
                        smem_slotv[slot_base + cpos] = i + cutlass.Int32(j)
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + step

        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= thr_frag[PC]:
                cpos = cnt_frag[PC]
                if cpos < cutlass.Int32(S):
                    smem_slotk[slot_base + cpos] = v
                    smem_slotv[slot_base + cpos] = it
            for m in cutlass.range_constexpr(M):
                cnt_frag[m] = cnt_frag[m] + cutlass.Int32(v >= thr_frag[m])
            it = it + cutlass.Int32(num_threads)

        for m in cutlass.range_constexpr(M):
            smem_ptcnt_multi[m * num_threads + tidx] = cnt_frag[m]
        for m in cutlass.range_constexpr(M):
            wc = self.warp_reduce_sum_i32(cnt_frag[m])
            if lane == 0:
                smem_wcnt_multi[m * num_warps + warp_id] = wc
        cute.arch.barrier()
        if warp_id == cutlass.Int32(0):
            for m in cutlass.range_constexpr(M):
                v = cutlass.Int32(0)
                if lane < cutlass.Int32(num_warps):
                    v = smem_wcnt_multi[m * num_warps + lane]
                total = self.warp_reduce_sum_i32(v)
                if lane == 0:
                    s_mt_cnt[m] = total

    # ------------------------------------------------------------------
    # op20 iter6: ladder pass reading the smem-resident row (small N). Same
    # contract as block_count_collect_multi; scalar smem reads (row is fp32
    # in smem; smem BW is not the bottleneck at N<=8K).
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_collect_multi_smem(
        self, smem_rowbuf, N, s_mt_thr, smem_ptcnt_multi, smem_wcnt_multi,
        s_mt_cnt, smem_slotk, smem_slotv, tidx, warp_id, lane,
    ):
        M = cutlass.const_expr(self.M_thr)
        PC = cutlass.const_expr(self.pred_col)
        S = cutlass.const_expr(self.slot_cap)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)

        thr_frag = cute.make_fragment((M,), cutlass.Float32)
        cnt_frag = cute.make_fragment((M,), cutlass.Int32)
        for m in cutlass.range_constexpr(M):
            thr_frag[m] = s_mt_thr[m]
            cnt_frag[m] = cutlass.Int32(0)
        slot_base = tidx * cutlass.Int32(S)

        i = cutlass.Int32(tidx)
        while i < N:
            v = smem_rowbuf[i]
            if v >= thr_frag[PC]:
                cpos = cnt_frag[PC]
                if cpos < cutlass.Int32(S):
                    smem_slotk[slot_base + cpos] = v
                    smem_slotv[slot_base + cpos] = i
            for m in cutlass.range_constexpr(M):
                cnt_frag[m] = cnt_frag[m] + cutlass.Int32(v >= thr_frag[m])
            i = i + cutlass.Int32(num_threads)

        for m in cutlass.range_constexpr(M):
            smem_ptcnt_multi[m * num_threads + tidx] = cnt_frag[m]
        for m in cutlass.range_constexpr(M):
            wc = self.warp_reduce_sum_i32(cnt_frag[m])
            if lane == 0:
                smem_wcnt_multi[m * num_warps + warp_id] = wc
        cute.arch.barrier()
        if warp_id == cutlass.Int32(0):
            for m in cutlass.range_constexpr(M):
                vv = cutlass.Int32(0)
                if lane < cutlass.Int32(num_warps):
                    vv = smem_wcnt_multi[m * num_warps + lane]
                total = self.warp_reduce_sum_i32(vv)
                if lane == 0:
                    s_mt_cnt[m] = total

    # ------------------------------------------------------------------
    # op20 iter4: P3 from slots — identical output contract to
    # phase3_sandwich (didx flush to output[0:M0), band -> smem_keys/vals,
    # s_iscalars[4]=M0, s_iscalars[0]=band) but the source is the per-thread
    # slot regions collected during the ladder pass, NOT a full-N rescan.
    # Precondition: thr_best >= thr[pred_col] (slots are a superset) and no
    # per-thread slot overflow.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_from_slots(
        self, smem_slotk, smem_slotv, smem_keys, smem_vals, smem_ptcnt,
        smem_ptcnt_up, smem_ptcnt_multi, smem_wcnt, smem_didx, s_thr, s_swf,
        s_iscalars, output_indices_row, tidx, warp_id, lane,
    ):
        kCC = cutlass.const_expr(self.kC)
        PC = cutlass.const_expr(self.pred_col)
        S = cutlass.const_expr(self.slot_cap)
        num_threads = cutlass.const_expr(self.num_threads)

        my_up = smem_ptcnt_up[tidx]
        my_band = smem_ptcnt[tidx] - my_up
        my_pk = (my_up << cutlass.Int32(16)) | my_band
        tp0 = my_pk
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp0, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp0 = tp0 + other
        my_excl_pk = tp0 - my_pk
        warp_tot_pk = cute.arch.shuffle_sync(tp0, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_tot_pk
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                c = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + c
            s_iscalars[4] = tot >> cutlass.Int32(16)
            s_iscalars[0] = tot & cutlass.Int32(0xFFFF)
        cute.arch.barrier()
        base_pk = smem_wcnt[warp_id] + my_excl_pk
        wc0 = base_pk >> cutlass.Int32(16)
        wcb = base_pk & cutlass.Int32(0xFFFF)

        thr1 = s_thr[0]
        thr0 = s_swf[0]
        my_lc = smem_ptcnt_multi[cutlass.Int32(PC) * cutlass.Int32(num_threads) + tidx]
        if my_lc > cutlass.Int32(S):
            my_lc = cutlass.Int32(S)
        slot_base = tidx * cutlass.Int32(S)
        sw = cutlass.Int32(0)
        while sw < my_lc:
            v = smem_slotk[slot_base + sw]
            if v >= thr1:
                if v >= thr0:
                    if wc0 < cutlass.Int32(self.top_k):
                        smem_didx[wc0] = smem_slotv[slot_base + sw]
                        wc0 = wc0 + cutlass.Int32(1)
                elif wcb < cutlass.Int32(kCC):
                    smem_keys[wcb] = v
                    smem_vals[wcb] = smem_slotv[slot_base + sw]
                    wcb = wcb + cutlass.Int32(1)
            sw = sw + cutlass.Int32(1)
        cute.arch.barrier()

        m0t = s_iscalars[4]
        iF = tidx
        while iF < m0t:
            output_indices_row[iF] = smem_didx[iF]
            iF = iF + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Phase-3 sandwich: dual-predicate scan, direct-write + band-collect.
    # smem_ptcnt      = per-thread counts at thr1 (winning column, cached)
    # smem_ptcnt_up   = per-thread counts at thr0 (sandwich column, cached)
    # After: output_indices_row[0:M0] = direct top-K members;
    #        smem_keys/vals[0:band]   = band candidates;
    #        s_iscalars[0]            = band count.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_sandwich(
        self, input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_ptcnt_up,
        smem_wcnt, smem_didx, s_thr, s_swf, s_iscalars,
        output_values_row, output_indices_row, tidx, warp_id, lane,
    ):
        kCC = cutlass.const_expr(self.kC)
        num_threads = cutlass.const_expr(self.num_threads)

        # ---- ONE packed prefix sum: (direct << 16) | band positions ----
        # Safe: per-thread counts < 2^16, block totals M0 <= K <= 2048 and
        # band <= kC <= 6144 (same bound as block_fused_snap_iter packing).
        my_up = smem_ptcnt_up[tidx]
        my_band = smem_ptcnt[tidx] - my_up
        my_pk = (my_up << cutlass.Int32(16)) | my_band
        tp0 = my_pk
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp0, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp0 = tp0 + other
        my_excl_pk = tp0 - my_pk
        warp_tot_pk = cute.arch.shuffle_sync(tp0, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_tot_pk
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                c = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + c
            s_iscalars[4] = tot >> cutlass.Int32(16)          # M0 total
            s_iscalars[0] = tot & cutlass.Int32(0xFFFF)       # band count
        cute.arch.barrier()
        base_pk = smem_wcnt[warp_id] + my_excl_pk
        my_pos0 = base_pk >> cutlass.Int32(16)
        my_posb = base_pk & cutlass.Int32(0xFFFF)

        # ---- fused stream-write scan ----
        thr1 = s_thr[0]
        thr0 = s_swf[0]
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        row_addr = input_row.iterator.toint()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        wc0 = my_pos0
        wcb = my_posb
        ic = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_phase3_unroll:
            if self.enable_unroll_4:
                rng_frag = cute.make_fragment((vec_w,), self.dtype)
                big_iters = cutlass.Int32(0)
                if N > ic + cutlass.Int32(vec_w - 1):
                    big_iters = (N - ic - cutlass.Int32(vec_w)) // cutlass.Int32(
                        step_elem) + cutlass.Int32(1)
                for k in cutlass.range(big_iters, unroll=4):
                    ic_local = ic + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem, assumed_align=vec_align)
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                    cute.copy(copy_atom, src_k, rng_frag)
                    for j in cutlass.range_constexpr(vec_w):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj = rng_frag[j]
                        else:
                            vj = cutlass.Float32(rng_frag[j])
                        # single rare outer branch (op18-P3 shape): keeps the
                        # 4-way LSU pipeline intact on the common miss path
                        if vj >= thr1:
                            if vj >= thr0:
                                if wc0 < cutlass.Int32(self.top_k):
                                    smem_didx[wc0] = ic_local + cutlass.Int32(j)
                                    wc0 = wc0 + cutlass.Int32(1)
                            elif wcb < cutlass.Int32(kCC):
                                smem_keys[wcb] = vj
                                smem_vals[wcb] = ic_local + cutlass.Int32(j)
                                wcb = wcb + cutlass.Int32(1)
                ic = ic + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while ic + cutlass.Int32(vec_w - 1) < N:
            src_ptr = cute.make_ptr(
                self.dtype, row_addr + cutlass.Int64(ic) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem, assumed_align=vec_align)
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= thr1:
                    if vj >= thr0:
                        if wc0 < cutlass.Int32(self.top_k):
                            smem_didx[wc0] = ic + cutlass.Int32(j)
                            wc0 = wc0 + cutlass.Int32(1)
                    elif wcb < cutlass.Int32(kCC):
                        smem_keys[wcb] = vj
                        smem_vals[wcb] = ic + cutlass.Int32(j)
                        wcb = wcb + cutlass.Int32(1)
            ic = ic + step

        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= thr1:
                if v >= thr0:
                    if wc0 < cutlass.Int32(self.top_k):
                        smem_didx[wc0] = it
                        wc0 = wc0 + cutlass.Int32(1)
                elif wcb < cutlass.Int32(kCC):
                    smem_keys[wcb] = v
                    smem_vals[wcb] = it
                    wcb = wcb + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)
        cute.arch.barrier()

        # coalesced flush of the deferred direct-write indices -> output[0:M0)
        m0t = s_iscalars[4]
        iF = tidx
        while iF < m0t:
            output_indices_row[iF] = smem_didx[iF]
            iF = iF + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Runtime-k snap iteration (block_fused_snap_iter with k_target arg).
    # ------------------------------------------------------------------
    @cute.jit
    def block_band_snap_iter(
        self, smem_keys, smem_wcnt, smem_hist, s_thr, s_iscalars, count,
        k_target, tidx, warp_id, lane,
    ):
        num_threads = cutlass.const_expr(self.num_threads)
        thr = s_thr[0]

        lge = cutlass.Int32(0)
        lgt = cutlass.Int32(0)
        s_up = cutlass.Float32(self.FLT_MAX)
        s_down = cutlass.Float32(self.NEG_FLT_MAX)
        isi = tidx
        while isi < count:
            v = smem_keys[isi]
            if v >= thr:
                lge = lge + cutlass.Int32(1)
            if v > thr:
                lgt = lgt + cutlass.Int32(1)
                s_up = _fmin_f32_inline(s_up, v)
            if v < thr:
                s_down = cute.arch.fmax(s_down, v)
            isi = isi + cutlass.Int32(num_threads)

        packed = (lge << cutlass.Int32(16)) | lgt
        packed = self.warp_reduce_sum_i32(packed)
        s_up = self.warp_reduce_min_f32(s_up)
        s_down = self.warp_reduce_max_f32(s_down)
        if lane == 0:
            smem_wcnt[warp_id] = packed
            smem_hist[warp_id] = float_as_uint32(s_up)
            smem_hist[self.num_warps + warp_id] = float_as_uint32(s_down)
        cute.arch.barrier()

        if tidx == 0:
            tp = cutlass.Int32(0)
            total_up = cutlass.Float32(self.FLT_MAX)
            total_down = cutlass.Float32(self.NEG_FLT_MAX)
            for w in cutlass.range_constexpr(self.num_warps):
                tp = tp + smem_wcnt[w]
                vu = llvm.bitcast(cutlass.Float32.mlir_type, smem_hist[w].ir_value())
                vd = llvm.bitcast(cutlass.Float32.mlir_type,
                                  smem_hist[self.num_warps + w].ir_value())
                total_up = _fmin_f32_inline(total_up, cutlass.Float32(vu))
                total_down = cute.arch.fmax(total_down, cutlass.Float32(vd))
            cge = tp >> cutlass.Int32(16)
            cgt = tp & cutlass.Int32(0xFFFF)
            s_iscalars[2] = cge
            s_iscalars[3] = cgt
            if cgt >= k_target:
                if total_up < cutlass.Float32(self.FLT_MAX):
                    s_thr[0] = total_up
            elif cge < k_target:
                if total_down > cutlass.Float32(self.NEG_FLT_MAX):
                    s_thr[0] = total_down
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Phase-4 band refine dispatcher: pick k_rem winners from band candidates
    # in smem_keys/vals[0:band] (all in [thr1, thr0)), writing output
    # positions m0..K-1. Compile-time switch between the exact rank-scatter
    # (default) and the legacy runtime-k histogram snap (A/B reference).
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_band_snap(
        self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
        s_iscalars, output_values_row, output_indices_row, band, k_rem, m0,
        tidx, warp_id, lane,
    ):
        if cutlass.const_expr(self.p4_rank_scatter):
            self.phase4_band_rank_scatter(
                smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
                s_iscalars, output_values_row, output_indices_row, band,
                k_rem, m0, tidx, warp_id, lane)
        else:
            self.phase4_band_snap_hist(
                smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
                s_iscalars, output_values_row, output_indices_row, band,
                k_rem, m0, tidx, warp_id, lane)

    # ------------------------------------------------------------------
    # op21 iter5/6/11: rank-scatter band refine (op8 phase4_rank_scatter
    # lineage). Differences vs op8: histogram range is the KNOWN
    # [thr1, thr0) band bracket (no min/max pass), rank target is the
    # runtime k_rem (not const K), all output positions offset by m0.
    # Chain: coarse kBins hist -> straddling bin b* + rank_above -> iter6
    # fast paths on b*: (A) whole-bin equality emit, (B) <=32 members ->
    # warp0 exact register ranking; (C) fallback = the EXACT value-edge
    # band snap (iter11). NOTE (iter11 falsification): a fixed-depth
    # sub-histogram of b* is NOT a valid path C — a fine bin is a value
    # INTERVAL, not a tie group, so cutting it at k_rem in stash order can
    # emit below the true K-th rank (upstream revert ec04147502; local
    # proof smoke_adversarial_band.py 72/72). Only paths that terminate on
    # a DATA value (register ranking, value-edge snap) are exact.
    # Scratch: s_iscalars[0..4] (all dead at P4 entry — P4 is terminal at
    # every call site); path B stashes b* members in smem_hist[8..39]
    # (coarse hist dead there).
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_band_rank_scatter(
        self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
        s_iscalars, output_values_row, output_indices_row, band, k_rem, m0,
        tidx, warp_id, lane,
    ):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        if band == k_rem:
            # every band candidate is a winner: emit at offset m0
            i4 = tidx
            while i4 < band:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[m0 + i4] = self.dtype(smem_keys[i4])
                output_indices_row[m0 + i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        else:
            # ---- coarse histogram over the known band range ----
            bmin_r = s_thr[0]
            bmax_r = s_swf[0]
            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            i6 = tidx
            while i6 < cutlass.Int32(kBins):
                smem_hist[i6] = cutlass.Int32(0)
                i6 = i6 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            range1 = bmax_r - bmin_r
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            i7 = tidx
            while i7 < band:
                vk = smem_keys[i7]
                bin_f = (vk - bmin_r) * inv1
                bin_i = cutlass.Int32(bin_f)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # ---- 3-step high->low search: straddling bin b* + rank_above ----
            warp_bin_sum = cutlass.Int32(0)
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (cutlass.Int32(kBins - 1)
                          - warp_id * cutlass.Int32(bins_per_warp)
                          - cutlass.Int32(jb))
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            if lane == 0:
                smem_wcnt[warp_id] = warp_bin_sum
            cute.arch.barrier()

            if tidx == 0:
                cum = cutlass.Int32(0)
                tw = cutlass.Int32(num_warps - 1)
                found = cutlass.Int32(0)
                for w2 in cutlass.range_constexpr(self.num_warps):
                    cum = cum + smem_wcnt[w2]
                    if cum >= k_rem and found == cutlass.Int32(0):
                        tw = cutlass.Int32(w2)
                        found = cutlass.Int32(1)
                cum2 = cutlass.Int32(0)
                for w3 in cutlass.range_constexpr(self.num_warps):
                    if cutlass.Int32(w3) < tw:
                        cum2 = cum2 + smem_wcnt[w3]
                s_iscalars[2] = cum2
                s_iscalars[3] = tw
            cute.arch.barrier()

            target_warp = s_iscalars[3]
            if warp_id == target_warp and lane == cutlass.Int32(0):
                base_cum = s_iscalars[2]
                b_star_l = cutlass.Int32(kBins - 1)
                rank_above_l = base_cum
                set_d = cutlass.Int32(0)
                for jb2 in cutlass.range_constexpr(bins_per_warp):
                    bidx2 = (cutlass.Int32(kBins - 1)
                             - target_warp * cutlass.Int32(bins_per_warp)
                             - cutlass.Int32(jb2))
                    ra_b = base_cum
                    base_cum = base_cum + smem_hist[bidx2]
                    if base_cum >= k_rem and set_d == cutlass.Int32(0):
                        b_star_l = bidx2
                        rank_above_l = ra_b  # count in bins strictly above b*
                        set_d = cutlass.Int32(1)
                s_iscalars[2] = rank_above_l
                s_iscalars[3] = b_star_l
            cute.arch.barrier()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]

            # ---- iter6 small-bin fast paths (host probe: cnt(b*) p50=2
            # p90=3 max=4 on 68 synth+real rows — the snap fallback is
            # cold in practice) ----
            if cutlass.const_expr(self.p4_smallbin):
                cnt_bstar = smem_hist[b_star]
                r_need = k_rem - rank_above
                if cnt_bstar <= cutlass.Int32(32):
                    # path B: stash the <=32 b* members, warp0 exact
                    # register ranking, direct positional emit (no atomics
                    # on the b* group, no fine hist, 2 barriers total)
                    if tidx == 0:
                        s_iscalars[4] = cutlass.Int32(0)  # cnt_above cursor
                        s_iscalars[1] = cutlass.Int32(0)  # b* stash cursor
                    cute.arch.barrier()
                    isc = tidx
                    while isc < band:
                        v = smem_keys[isc]
                        bin_i = cutlass.Int32((v - bmin_r) * inv1)
                        if bin_i < cutlass.Int32(0):
                            bin_i = cutlass.Int32(0)
                        if bin_i > cutlass.Int32(kBins - 1):
                            bin_i = cutlass.Int32(kBins - 1)
                        if bin_i > b_star:
                            o_a = atomicAdd(s_iscalars.iterator + cutlass.Int32(4),
                                            cutlass.Int32(1))
                            pos = m0 + o_a
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif bin_i == b_star:
                            js = atomicAdd(s_iscalars.iterator + cutlass.Int32(1),
                                           cutlass.Int32(1))
                            if js < cutlass.Int32(32):
                                # stash band index; safe even if b_star lands
                                # in [8,40): cnt_bstar is already in a
                                # register and the coarse hist is dead here
                                smem_hist[cutlass.Int32(8) + js] = isc
                        isc = isc + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    if warp_id == cutlass.Int32(0):
                        v_i = cutlass.Float32(self.NEG_FLT_MAX)
                        bidx_i = cutlass.Int32(0)
                        if lane < cnt_bstar:
                            bidx_i = smem_hist[cutlass.Int32(8) + lane]
                            v_i = smem_keys[bidx_i]
                        rankp = cutlass.Int32(0)
                        for src_c in cutlass.range_constexpr(32):
                            v_o = cute.arch.shuffle_sync(
                                v_i, cutlass.Int32(src_c))
                            if cutlass.Int32(src_c) < cnt_bstar:
                                if v_o > v_i:
                                    rankp = rankp + cutlass.Int32(1)
                                elif v_o == v_i and cutlass.Int32(src_c) < lane:
                                    rankp = rankp + cutlass.Int32(1)
                        if lane < cnt_bstar and rankp < r_need:
                            pos = m0 + rank_above + rankp
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v_i)
                                output_indices_row[pos] = smem_vals[bidx_i]
                    cute.arch.barrier()
                elif cnt_bstar == r_need:
                    # path A: every b* member is a winner — whole-bin emit,
                    # skip the fallback entirely
                    if tidx == 0:
                        s_iscalars[4] = cutlass.Int32(0)  # cnt_above cursor
                        s_iscalars[1] = cutlass.Int32(0)  # b* cursor
                    cute.arch.barrier()
                    isc = tidx
                    while isc < band:
                        v = smem_keys[isc]
                        bin_i = cutlass.Int32((v - bmin_r) * inv1)
                        if bin_i < cutlass.Int32(0):
                            bin_i = cutlass.Int32(0)
                        if bin_i > cutlass.Int32(kBins - 1):
                            bin_i = cutlass.Int32(kBins - 1)
                        if bin_i > b_star:
                            o_a = atomicAdd(s_iscalars.iterator + cutlass.Int32(4),
                                            cutlass.Int32(1))
                            pos = m0 + o_a
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif bin_i == b_star:
                            o_s = atomicAdd(s_iscalars.iterator + cutlass.Int32(1),
                                            cutlass.Int32(1))
                            pos = m0 + rank_above + o_s
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        isc = isc + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                else:
                    # path C (distribution shift, never fires on real/synth
                    # probes): fall back to the EXACT value-edge snap on the
                    # whole band. iter11: the previous fixed-depth fine
                    # scatter here was NOT exact on arbitrary continuous
                    # logits (same failure mode upstream reverted in
                    # ec04147502; falsified by smoke_adversarial_band.py,
                    # 72/72) — a fixed-depth histogram cannot separate two
                    # distinct values in one (sub-)bin, so the straddling
                    # bin's stash-order emit can pick a value below the true
                    # K-th rank. The snap converges sel_thr onto an actual
                    # DATA value (block_band_snap_iter steps to value
                    # edges), so its ==sel_thr group is a true tie group and
                    # any k_rem-cut of it is exact.
                    self.phase4_band_snap_hist(
                        smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr,
                        s_swf, s_iscalars, output_values_row,
                        output_indices_row, band, k_rem, m0, tidx, warp_id,
                        lane)
            else:
                # p4_smallbin=False (OP21_P4_FAST=0): fast paths disabled —
                # everything falls to the exact snap (iter11; the old
                # always-fine reference was retired with the inexact fine
                # scatter).
                self.phase4_band_snap_hist(
                    smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
                    s_iscalars, output_values_row, output_indices_row, band,
                    k_rem, m0, tidx, warp_id, lane)

    # ------------------------------------------------------------------
    # Legacy phase-4 band snap (A/B reference): runtime-k histogram snap.
    # Histogram range seeded [thr1, thr0). out_count starts at M0 so the
    # writeback fills output positions M0..K-1.
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_band_snap_hist(
        self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
        s_iscalars, output_values_row, output_indices_row, band, k_rem, m0,
        tidx, warp_id, lane,
    ):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        if band == k_rem:
            # every band candidate is a winner: emit at offset m0
            i4 = tidx
            while i4 < band:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[m0 + i4] = self.dtype(smem_keys[i4])
                output_indices_row[m0 + i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        else:
            # band > k_rem: histogram + runtime-k snap. Range = [thr1, thr0)
            # by construction — no min/max pass needed.
            bmin_r = s_thr[0]
            bmax_r = s_swf[0]
            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            i6 = tidx
            while i6 < cutlass.Int32(kBins):
                smem_hist[i6] = cutlass.Int32(0)
                i6 = i6 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            range1 = bmax_r - bmin_r
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            i7 = tidx
            while i7 < band:
                vk = smem_keys[i7]
                bin_f = (vk - bmin_r) * inv1
                bin_i = cutlass.Int32(bin_f)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # k_rem-th bin search (runtime k)
            warp_bin_sum = cutlass.Int32(0)
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (cutlass.Int32(kBins - 1)
                          - warp_id * cutlass.Int32(bins_per_warp)
                          - cutlass.Int32(jb))
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            if lane == 0:
                smem_wcnt[warp_id] = warp_bin_sum
            cute.arch.barrier()

            if tidx == 0:
                cum = cutlass.Int32(0)
                tw = cutlass.Int32(num_warps - 1)
                found = cutlass.Int32(0)
                for w2 in cutlass.range_constexpr(self.num_warps):
                    cum = cum + smem_wcnt[w2]
                    if cum >= k_rem and found == cutlass.Int32(0):
                        tw = cutlass.Int32(w2)
                        found = cutlass.Int32(1)
                cum2 = cutlass.Int32(0)
                for w3 in cutlass.range_constexpr(self.num_warps):
                    if cutlass.Int32(w3) < tw:
                        cum2 = cum2 + smem_wcnt[w3]
                s_iscalars[2] = cum2
                s_iscalars[3] = tw
            cute.arch.barrier()

            target_warp = s_iscalars[3]
            if warp_id == target_warp and lane == cutlass.Int32(0):
                base_cum = s_iscalars[2]
                thr_local = bmin_r
                set_done = cutlass.Int32(0)
                for jb2 in cutlass.range_constexpr(bins_per_warp):
                    bidx2 = (cutlass.Int32(kBins - 1)
                             - target_warp * cutlass.Int32(bins_per_warp)
                             - cutlass.Int32(jb2))
                    base_cum = base_cum + smem_hist[bidx2]
                    if base_cum >= k_rem and set_done == cutlass.Int32(0):
                        thr_local = bmin_r + cutlass.Float32(bidx2) * range1 / cutlass.Float32(kBins)
                        set_done = cutlass.Int32(1)
                s_thr[0] = thr_local
            cute.arch.barrier()

            # snap convergence: cgt < k_rem <= cge
            si = cutlass.Int32(0)
            done_snap = cutlass.Int32(0)
            while si < band and done_snap == cutlass.Int32(0):
                self.block_band_snap_iter(smem_keys, smem_wcnt, smem_hist,
                                          s_thr, s_iscalars, band, k_rem,
                                          tidx, warp_id, lane)
                if s_iscalars[3] < k_rem and s_iscalars[2] >= k_rem:
                    done_snap = cutlass.Int32(1)
                si = si + cutlass.Int32(1)

            # two-pass writeback at offset m0 (out_count starts at m0)
            sel_thr = s_thr[0]
            if tidx == 0:
                s_iscalars[4] = m0
            cute.arch.barrier()

            base_w = warp_id * cutlass.Int32(self.WARP_SIZE)
            while base_w < band:
                ix1 = base_w + lane
                emit_gt = cutlass.Int32(0)
                v_p1 = cutlass.Float32(self.NEG_FLT_MAX)
                if ix1 < band:
                    v_p1 = smem_keys[ix1]
                    if v_p1 > sel_thr:
                        emit_gt = cutlass.Int32(1)
                mask_gt = cute.arch.vote_ballot_sync(emit_gt != cutlass.Int32(0))
                if mask_gt != cutlass.Uint32(0):
                    cnt_gt = cutlass.Int32(cute.arch.popc(mask_gt))
                    lane_mask_gt = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
                    moff_gt = cutlass.Int32(cute.arch.popc(mask_gt & lane_mask_gt))
                    bp_gt = cutlass.Int32(0)
                    if lane == cutlass.Int32(0):
                        bp_gt = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cnt_gt)
                    bp_gt = cute.arch.shuffle_sync(bp_gt, cutlass.Int32(0))
                    wpos_p1 = bp_gt + moff_gt
                    if emit_gt != cutlass.Int32(0) and wpos_p1 < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[wpos_p1] = self.dtype(v_p1)
                        output_indices_row[wpos_p1] = smem_vals[ix1]
                base_w = base_w + cutlass.Int32(num_threads)
            cute.arch.barrier()

            base_w2 = warp_id * cutlass.Int32(self.WARP_SIZE)
            while base_w2 < band:
                ix2 = base_w2 + lane
                emit_eq = cutlass.Int32(0)
                v_p2 = cutlass.Float32(self.NEG_FLT_MAX)
                if ix2 < band:
                    v_p2 = smem_keys[ix2]
                    if v_p2 == sel_thr:
                        emit_eq = cutlass.Int32(1)
                mask_eq = cute.arch.vote_ballot_sync(emit_eq != cutlass.Int32(0))
                if mask_eq != cutlass.Uint32(0):
                    cnt_eq = cutlass.Int32(cute.arch.popc(mask_eq))
                    lane_mask_eq = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
                    moff_eq = cutlass.Int32(cute.arch.popc(mask_eq & lane_mask_eq))
                    bp_eq = cutlass.Int32(0)
                    if lane == cutlass.Int32(0):
                        bp_eq = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cnt_eq)
                    bp_eq = cute.arch.shuffle_sync(bp_eq, cutlass.Int32(0))
                    wpos_p2 = bp_eq + moff_eq
                    if emit_eq != cutlass.Int32(0) and wpos_p2 < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[wpos_p2] = self.dtype(v_p2)
                        output_indices_row[wpos_p2] = smem_vals[ix2]
                base_w2 = base_w2 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            filled_par = s_iscalars[4]
            if filled_par > cutlass.Int32(kK):
                filled_par = cutlass.Int32(kK)
            ipad = filled_par + tidx
            while ipad < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[ipad] = cutlass.Int32(-1)
                ipad = ipad + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # op21 FIX (root cause of the op18/19 real-data red-card): the vendored
    # done!=1 retry-shrink exits on the FIRST count <= kCC — INCLUDING
    # count < K (undershoot: -1 slots) — and its bracket hi end carries no
    # count < K guarantee (hit~0 rows can have >= K elements above v_hi).
    # This override runs a CORRECT bounded bisection to land count in
    # [kK, kCC] (expanding hi to +LARGE first if count(hi) >= kK), marks
    # done=1, then delegates to the vendored prefix-sum + stream-write
    # (contract: smem_ptcnt caches per-thread counts of the LAST
    # block_count_ge at the final s_thr[0]). If 30 iters exhaust (tie-block,
    # ~16-bit only), it lands on the undershoot side = production fail-soft
    # semantics rather than an arbitrary truncated subset.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_collect_candidates(self, input_row, N, smem_keys, smem_vals,
                                  smem_ptcnt, smem_wcnt, s_thr, s_iscalars,
                                  tidx, warp_id, lane):
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        if s_iscalars[1] != cutlass.Int32(1):
            # iter13 (HLS): s_iscalars[5]/[6] carry the ladder-KNOWN bracket
            # counts (invariants: [5] > kCC == count at s_thr[1]; [6] < kK
            # == count at s_thr[2]; -1 = unknown). Known ends let the entry
            # and hi-end full-row passes be skipped outright, and the refine
            # step aims by log-count regula falsi instead of blind bisection.
            kn_lo = cutlass.Int32(0)
            if cutlass.const_expr(self.fb_logfalsi):
                if s_iscalars[5] > cutlass.Int32(0):
                    kn_lo = cutlass.Int32(1)
            if kn_lo == cutlass.Int32(0):
                # entry count at the current threshold (also warms smem_ptcnt)
                self.block_count_ge(input_row, N, s_thr[0], smem_ptcnt, smem_wcnt,
                                    s_iscalars, tidx, warp_id, lane)
                cute.arch.barrier()  # block_count_ge has NO trailing barrier
            need = cutlass.Int32(0)
            if kn_lo == cutlass.Int32(1):
                need = cutlass.Int32(1)  # count(s_thr[1]) > kCC by invariant
            elif s_iscalars[0] > cutlass.Int32(kCC) or s_iscalars[0] < cutlass.Int32(kK):
                need = cutlass.Int32(1)
            if need == cutlass.Int32(1):
                if kn_lo == cutlass.Int32(0):
                    if tidx == 0:
                        if s_iscalars[0] > cutlass.Int32(kCC):
                            s_thr[1] = s_thr[0]
                            if cutlass.const_expr(self.fb_logfalsi):
                                s_iscalars[5] = s_iscalars[0]
                        else:
                            s_thr[2] = s_thr[0]
                            if cutlass.const_expr(self.fb_logfalsi):
                                s_iscalars[6] = s_iscalars[0]
                    cute.arch.barrier()
                kn_hi = cutlass.Int32(0)
                if cutlass.const_expr(self.fb_logfalsi):
                    if s_iscalars[6] >= cutlass.Int32(0):
                        kn_hi = cutlass.Int32(1)  # count(s_thr[2]) < kK known
                run_refine = cutlass.Int32(1)
                if kn_hi == cutlass.Int32(0):
                    # hi-end guarantee: count(hi) must be < kK, else expand
                    self.block_count_ge(input_row, N, s_thr[2], smem_ptcnt,
                                        smem_wcnt, s_iscalars, tidx, warp_id, lane)
                    cute.arch.barrier()
                    if s_iscalars[0] >= cutlass.Int32(kK) and s_iscalars[0] <= cutlass.Int32(kCC):
                        # hi itself already valid: adopt it
                        if tidx == 0:
                            s_thr[0] = s_thr[2]
                        cute.arch.barrier()
                        run_refine = cutlass.Int32(0)
                    else:
                        # geometric expansion: double the bracket upward until
                        # count(hi) < kK (hit~0 rows can have >= K elements above
                        # the max gathered value; a one-shot huge hi would need
                        # ~96 bisection steps — doubling keeps the bracket tight)
                        ex = cutlass.Int32(0)
                        while ex < cutlass.Int32(12) and s_iscalars[0] >= cutlass.Int32(kK):
                            if tidx == 0:
                                rngx = s_thr[2] - s_thr[1]
                                if rngx <= cutlass.Float32(0.0):
                                    rngx = cutlass.Float32(1e-3)
                                s_thr[2] = s_thr[2] + rngx
                            cute.arch.barrier()
                            self.block_count_ge(input_row, N, s_thr[2], smem_ptcnt,
                                                smem_wcnt, s_iscalars, tidx,
                                                warp_id, lane)
                            cute.arch.barrier()
                            ex = ex + cutlass.Int32(1)
                        if tidx == 0:
                            s_thr[0] = s_thr[2]  # provisional; loop re-places
                            if cutlass.const_expr(self.fb_logfalsi):
                                # learn the hi-end count (hi-end check or last
                                # expansion step; tidx0-only falsi reads it)
                                if s_iscalars[0] < cutlass.Int32(kK):
                                    s_iscalars[6] = s_iscalars[0]
                        cute.arch.barrier()
                if run_refine == cutlass.Int32(1):
                    rs = cutlass.Int32(0)
                    ok3 = cutlass.Int32(0)
                    while rs < cutlass.Int32(30) and ok3 == cutlass.Int32(0):
                        if tidx == 0:
                            lo3 = s_thr[1]
                            hi3 = s_thr[2]
                            mid3 = (lo3 + hi3) * cutlass.Float32(0.5)
                            if cutlass.const_expr(self.fb_logfalsi):
                                # log-count regula falsi: aim where the
                                # ~exponential CCDF predicts count == m*;
                                # strict-interior guard falls back to the
                                # midpoint (Illinois-style safeguard)
                                clo3 = s_iscalars[5]
                                chi3 = s_iscalars[6]
                                if clo3 > cutlass.Int32(0) and chi3 >= cutlass.Int32(0):
                                    chic = chi3
                                    if chic < cutlass.Int32(1):
                                        chic = cutlass.Int32(1)
                                    l_lo = lg2_f32(cutlass.Float32(clo3))
                                    l_hi = lg2_f32(cutlass.Float32(chic))
                                    den3 = l_lo - l_hi
                                    if den3 > cutlass.Float32(0.0):
                                        t3 = (cutlass.Float32(self.log2_mstar)
                                              - l_hi) / den3
                                        cnd3 = hi3 + t3 * (lo3 - hi3)
                                        if cnd3 > lo3 and cnd3 < hi3:
                                            mid3 = cnd3
                            if mid3 <= lo3:
                                mid3 = hi3
                            s_thr[0] = mid3
                        cute.arch.barrier()
                        self.block_count_ge(input_row, N, s_thr[0], smem_ptcnt,
                                            smem_wcnt, s_iscalars, tidx,
                                            warp_id, lane)
                        cute.arch.barrier()
                        c3 = s_iscalars[0]
                        if c3 >= cutlass.Int32(kK) and c3 <= cutlass.Int32(kCC):
                            ok3 = cutlass.Int32(1)
                        if tidx == 0:
                            if c3 > cutlass.Int32(kCC):
                                s_thr[1] = s_thr[0]
                                if cutlass.const_expr(self.fb_logfalsi):
                                    s_iscalars[5] = c3
                            elif c3 < cutlass.Int32(kK):
                                s_thr[2] = s_thr[0]
                                if cutlass.const_expr(self.fb_logfalsi):
                                    s_iscalars[6] = c3
                        cute.arch.barrier()
                        rs = rs + cutlass.Int32(1)
                    if ok3 == cutlass.Int32(0):
                        # tie-block: land on the undershoot (fail-soft) side
                        self.block_count_ge(input_row, N, s_thr[2], smem_ptcnt,
                                            smem_wcnt, s_iscalars, tidx,
                                            warp_id, lane)
                        cute.arch.barrier()
                        if tidx == 0:
                            s_thr[0] = s_thr[2]
                        cute.arch.barrier()
            if tidx == 0:
                s_iscalars[1] = cutlass.Int32(1)  # converged: skip vendored retry
            cute.arch.barrier()
        GvrMultiThreshKernel.phase3_collect_candidates(
            self, input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_wcnt,
            s_thr, s_iscalars, tidx, warp_id, lane)

    # ------------------------------------------------------------------
    # op21: phase1 variant that gathers the prev-K values ONCE — computing
    # the min/max/mean stats exactly like the base phase1_preidx_stats AND
    # stashing every gathered value into smem_stash (the P3 slot buffer,
    # free until the ladder pass). phase1b then histograms from smem, so
    # the K-load L2 re-gather disappears from the critical path.
    # Invalid preIdx entries stash NEG_FLT_MAX (always < v_lo == min of the
    # valid values), which phase1b skips with a single compare.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1_stats_stash(
        self, input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
        smem_stash, smem_wmin_f32, smem_wmax_f32, smem_wsum_f32,
        smem_wcnt_i32, s_thr, s_iscalars, tidx, warp_id, lane,
    ):
        local_min = cutlass.Float32(self.FLT_MAX)
        local_max = cutlass.Float32(self.NEG_FLT_MAX)
        local_sum = cutlass.Float32(0.0)
        local_cnt = cutlass.Int32(0)
        if cutlass.const_expr(pre_idx_count >= self.num_threads):
            n_iters = cutlass.const_expr(pre_idx_count // self.num_threads)
            for u in cutlass.range_constexpr(n_iters):
                i = tidx + cutlass.Int32(u * self.num_threads)
                raw = pre_idx_row[i]
                idx = raw + pre_idx_offset
                v = cutlass.Float32(self.NEG_FLT_MAX)  # stash sentinel
                if idx >= 0 and idx < N:
                    v = self._load_fp32(input_row, idx)
                    local_max = cute.arch.fmax(local_max, v)
                    local_min = _fmin_f32_inline(local_min, v)
                    local_sum = local_sum + v
                    local_cnt = local_cnt + 1
                smem_stash[i] = v
        else:
            idx = cutlass.Int32(-1)
            if tidx < cutlass.Int32(pre_idx_count):
                idx = pre_idx_row[tidx] + pre_idx_offset
            v = cutlass.Float32(self.NEG_FLT_MAX)
            if idx >= 0 and idx < N:
                v = self._load_fp32(input_row, idx)
                local_max = cute.arch.fmax(local_max, v)
                local_min = _fmin_f32_inline(local_min, v)
                local_sum = local_sum + v
                local_cnt = local_cnt + 1
            if tidx < cutlass.Int32(pre_idx_count):
                smem_stash[tidx] = v
        active_preidx_warps = cutlass.const_expr(
            min(pre_idx_count // self.WARP_SIZE, self.num_warps)
        )
        if cutlass.const_expr(active_preidx_warps < self.num_warps):
            if warp_id < cutlass.Int32(active_preidx_warps):
                wmin = self.warp_reduce_min_f32(local_min)
                wmax = self.warp_reduce_max_f32(local_max)
                wsum = self.warp_reduce_sum_f32(local_sum)
                wcnt = self.warp_reduce_sum_i32(local_cnt)
                if lane == 0:
                    smem_wmin_f32[warp_id] = wmin
                    smem_wmax_f32[warp_id] = wmax
                    smem_wsum_f32[warp_id] = wsum
                    smem_wcnt_i32[warp_id] = wcnt
        else:
            wmin = self.warp_reduce_min_f32(local_min)
            wmax = self.warp_reduce_max_f32(local_max)
            wsum = self.warp_reduce_sum_f32(local_sum)
            wcnt = self.warp_reduce_sum_i32(local_cnt)
            if lane == 0:
                smem_wmin_f32[warp_id] = wmin
                smem_wmax_f32[warp_id] = wmax
                smem_wsum_f32[warp_id] = wsum
                smem_wcnt_i32[warp_id] = wcnt
        cute.arch.barrier()
        if tidx == 0:
            pmin = cutlass.Float32(self.FLT_MAX)
            pmax = cutlass.Float32(self.NEG_FLT_MAX)
            psum = cutlass.Float32(0.0)
            pcnt = cutlass.Int32(0)
            for w in cutlass.range_constexpr(active_preidx_warps):
                v_min = smem_wmin_f32[w]
                v_max = smem_wmax_f32[w]
                v_sum = smem_wsum_f32[w]
                v_cnt = smem_wcnt_i32[w]
                pmax = cute.arch.fmax(pmax, v_max)
                pmin = _fmin_f32_inline(pmin, v_min)
                psum = psum + v_sum
                pcnt = pcnt + v_cnt
            pmean = cutlass.Float32(0.0)
            if pcnt > 0:
                pmean = psum / cutlass.Float32(pcnt)
            else:
                pmean = (pmin + pmax) * cutlass.Float32(0.5)
            cnt_lo_seed = pre_idx_count + (pre_idx_count >> 2)
            s_thr[0] = pmean
            s_thr[1] = pmin
            s_thr[2] = pmax
            s_iscalars[0] = cutlass.Int32(0)  # cand_count
            s_iscalars[1] = cutlass.Int32(0)  # done
            s_iscalars[2] = cutlass.Int32(cnt_lo_seed)  # cnt_lo
            s_iscalars[3] = cutlass.Int32(1)  # cnt_hi
            s_iscalars[4] = cutlass.Int32(0)  # out_count
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # op21 place_mode=5: rank-quantile round-0 placement. Histogram the K
    # prev-step values (stashed in smem by phase1_stats_stash) over
    # [v_lo, v_hi] with QBINS fixed bins (smem_hist is free until P4, which
    # re-zeroes it); a parallel suffix-scan then drops ladder column m+1 at
    # the bin left-edge where the ge-count crosses qfracs[m] * K_valid.
    # Thresholds are SEEDS only; exactness is enforced by the unchanged
    # downstream count/snap machinery.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_rank_quantile(
        self, smem_stash, pre_idx_count,
        smem_hist, s_thr, s_mt_thr, s_mt_cnt, tidx,
    ):
        QBINS = cutlass.const_expr(self.QBINS)
        M = cutlass.const_expr(self.M_thr)
        num_threads = cutlass.const_expr(self.num_threads)
        lo = s_thr[1]
        hi = s_thr[2]
        iz = cutlass.Int32(tidx)
        while iz < cutlass.Int32(QBINS):
            smem_hist[iz] = cutlass.Int32(0)
            iz = iz + cutlass.Int32(num_threads)
        cute.arch.barrier()
        rng = hi - lo
        inv = (cutlass.Float32(QBINS - 1) + cutlass.Float32(0.99)) / rng
        # histogram the phase1-stashed values from smem (no L2 re-gather);
        # NEG_FLT_MAX sentinel entries (invalid preIdx) are < lo and skipped
        if cutlass.const_expr(pre_idx_count >= self.num_threads):
            n_iters = cutlass.const_expr(pre_idx_count // self.num_threads)
            for u in cutlass.range_constexpr(n_iters):
                i = tidx + cutlass.Int32(u * self.num_threads)
                v = smem_stash[i]
                if v >= lo:
                    b = cutlass.Int32((v - lo) * inv)
                    if b > cutlass.Int32(QBINS - 1):
                        b = cutlass.Int32(QBINS - 1)
                    atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
        else:
            v = cutlass.Float32(self.NEG_FLT_MAX)
            if tidx < cutlass.Int32(pre_idx_count):
                v = smem_stash[tidx]
            if v >= lo:
                b = cutlass.Int32((v - lo) * inv)
                if b > cutlass.Int32(QBINS - 1):
                    b = cutlass.Int32(QBINS - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
        cute.arch.barrier()
        # Parallel suffix-scan + crossing detect (v1 ran this cum scan serially
        # on tid0: ~1800 dependent smem ops ~= +30us fixed, N-independent).
        binw = rng / cutlass.Float32(QBINS)
        if tidx == 0:
            s_mt_thr[0] = lo  # g_min anchor: count >= K_valid guaranteed
            for md in cutlass.range_constexpr(M - 1):
                s_mt_thr[md + 1] = lo + rng * (cutlass.Float32(md + 1) / cutlass.Float32(M))
        # in-place Hillis-Steele suffix sum over the QBINS bins (thread b owns
        # bin b; num_threads >= QBINS guaranteed by _config). The tid0 default
        # writes above overlap step 0's reads (disjoint smem).
        for e in cutlass.range_constexpr(self.QBINS.bit_length() - 1):
            step = cutlass.const_expr(1 << e)
            v = cutlass.Int32(0)
            if tidx < cutlass.Int32(QBINS):
                v = smem_hist[tidx]
                if tidx + cutlass.Int32(step) < cutlass.Int32(QBINS):
                    v = v + smem_hist[tidx + cutlass.Int32(step)]
            cute.arch.barrier()
            if tidx < cutlass.Int32(QBINS):
                smem_hist[tidx] = v
            cute.arch.barrier()
        # crossing: the unique largest bin b with suffix(b) >= tgt (== first
        # crossing of the old top-down cum scan) writes column m+1's seed
        if tidx < cutlass.Int32(QBINS):
            sfx = smem_hist[tidx]
            nxt = cutlass.Int32(0)
            if tidx < cutlass.Int32(QBINS - 1):
                nxt = smem_hist[tidx + 1]
            total = smem_hist[0]
            for m in cutlass.range_constexpr(M - 1):
                tgt = cutlass.Int32(cutlass.Float32(total) * cutlass.Float32(self.qfracs[m]))
                if tgt < cutlass.Int32(1):
                    tgt = cutlass.Int32(1)
                if sfx >= tgt and (tidx == cutlass.Int32(QBINS - 1) or nxt < tgt):
                    s_mt_thr[m + 1] = lo + cutlass.Float32(tidx) * binw
        cute.arch.barrier()
        if tidx == 0:
            # enforce non-descending columns (equal columns tolerated: counts
            # equal => straddle logic still picks a valid pair)
            for mm in cutlass.range_constexpr(M - 1):
                pv = s_mt_thr[mm]
                if s_mt_thr[mm + 1] < pv:
                    s_mt_thr[mm + 1] = pv
            if cutlass.const_expr(self._p2n()):
                # iter9: quantize every column to the dtype grid so the
                # native 16-bit ladder compares agree bit-for-bit with the
                # fp32 compares in P3/P4/fallback (cvt.rn is monotonic =>
                # the non-descending property survives; column 0 = g_min is
                # a data value, already on the grid).
                for mq in cutlass.range_constexpr(M):
                    s_mt_thr[mq] = quant_f32_16(s_mt_thr[mq],
                                                self._kind16())
        # visibility: callers ladder-init barrier follows before any read

    @cute.kernel
    def gvr_topk_kernel(self, input_data, pre_idx, seq_lens, output_values, output_indices):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        next_n = cutlass.const_expr(self.next_n)
        top_k = cutlass.const_expr(self.top_k)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        kC = cutlass.const_expr(self.kC)
        kNumBins = cutlass.const_expr(self.kNumBins)
        M = cutlass.const_expr(self.M_thr)
        R = cutlass.const_expr(self.R_rounds)
        bAcc = cutlass.const_expr(self.band_accept)
        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        row_idx = bidx
        pre_idx_row_idx = row_idx // next_n
        if cutlass.const_expr(self.compress_ratio == 1):
            pre_idx_offset = cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)
        else:
            pre_idx_offset = cutlass.Int32(0)
        seq_len = seq_lens[pre_idx_row_idx]
        actual_kv_len = (seq_len - cutlass.Int32(next_n) + cutlass.Int32(row_idx % next_n) + cutlass.Int32(1))
        if cutlass.const_expr(self.compress_ratio == 1):
            N = actual_kv_len
        else:
            N = actual_kv_len // cutlass.Int32(self.compress_ratio)
        input_row = input_data[row_idx, None]
        pre_idx_row = pre_idx[pre_idx_row_idx, None]
        if cutlass.const_expr(self.return_output_values):
            output_values_row = output_values[row_idx, None]
        else:
            output_values_row = None
        output_indices_row = output_indices[row_idx, None]
        pre_idx_count = pre_idx.shape[1]
        griddepcontrol_wait()

        smem = SmemAllocator()
        smem_keys = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((kC,), order=(0,)), byte_alignment=128)
        smem_vals = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((kC,), order=(0,)), byte_alignment=128)
        # NOTE: keep hist and didx SEPARATE. A union buffer (8KB less smem at
        # K2048) raises residency 2->3 CTAs/SM at high BS and measures 0.745x
        # vs separate's 0.893x at K2048/16K/BS2048 — lower occupancy wins in
        # that regime. (measured 2026-07-02)
        smem_hist = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((kNumBins,), order=(0,)), byte_alignment=128)
        smem_ptcnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=128)
        smem_wmin = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wmax = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wsum = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wcnt_p1 = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        s_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        # iter13: +2 slots ([5]/[6] = log-falsi bracket counts)
        s_iscalars = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((7,), order=(0,)), byte_alignment=16)
        smem_ptcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_warps,), order=(0,)), byte_alignment=64)
        s_mt_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        s_mt_cnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        # [0]=br_lo, [1]=br_hi, [2]=best_thr
        s_mstf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        # [0]=best_cnt, [1]=best_col_this_round(-1 none), [2]=continue_flag
        s_msti = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        # op19 additions: sandwich upper threshold snapshot
        smem_ptcnt_up = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_didx = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((top_k,), order=(0,)), byte_alignment=128)
        # [0]=thr0
        s_swf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((1,), order=(0,)), byte_alignment=16)
        # [0]=M0, [1]=up_col_this_round (-1 none)
        s_swi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((2,), order=(0,)), byte_alignment=16)
        # op20 iter4: per-thread collect slots for the fused ladder pass
        smem_slotk = None
        smem_slotv = None
        if cutlass.const_expr(self.fuse_collect):
            slot_elems = cutlass.const_expr(self.num_threads * self.slot_cap)
            smem_slotk = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((slot_elems,), order=(0,)), byte_alignment=128)
            smem_slotv = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((slot_elems,), order=(0,)), byte_alignment=128)
        # op20 iter6: smem-resident row buffer (small N)
        smem_rowbuf = None
        if cutlass.const_expr(self.smem_row_elems > 0):
            smem_rowbuf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((self.smem_row_elems,), order=(0,)), byte_alignment=128)

        if N <= cutlass.Int32(top_k):
            jd = tidx
            while jd < N:
                output_indices_row[jd] = cutlass.Int32(jd)
                jd = jd + cutlass.Int32(num_threads)
            jp = N + cutlass.Int32(tidx)
            while jp < cutlass.Int32(top_k):
                output_indices_row[jp] = cutlass.Int32(-1)
                jp = jp + cutlass.Int32(num_threads)
        else:
            row_in_smem = cutlass.Int32(0)
            if cutlass.const_expr(self.smem_row_elems > 0):
                if N <= cutlass.Int32(self.smem_row_elems):
                    row_in_smem = cutlass.Int32(1)
                    ib = cutlass.Int32(tidx)
                    while ib < N:
                        smem_rowbuf[ib] = self._load_fp32(input_row, ib)
                        ib = ib + cutlass.Int32(num_threads)
            # phase1's internal barrier also publishes the bulk-load stores
            if cutlass.const_expr(self.place_mode == 5):
                # gathers once: stats + smem stash (smem_keys free until P3)
                self.phase1_stats_stash(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                        smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)
            else:
                self.phase1_preidx_stats(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                         smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if tidx == 0:
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        je = je + cutlass.Int32(1)
            else:
                # ---- P2: adaptive M-ary ladder + sandwich pair tracking ----
                pmean = s_thr[0]
                if cutlass.const_expr(self.place_mode == 5):
                    self.phase1b_rank_quantile(smem_keys, pre_idx_count,
                                               smem_hist, s_thr, s_mt_thr,
                                               s_mt_cnt, tidx)
                smem_ptcnt_up[tidx] = cutlass.Int32(0)  # M0=0 default column
                if tidx == 0:
                    s_mstf[0] = v_lo
                    s_mstf[1] = v_hi
                    s_mstf[2] = v_lo
                    s_msti[0] = cutlass.Int32(_INT_MAX)  # best_cnt (M1)
                    s_msti[1] = cutlass.Int32(-1)
                    s_msti[2] = cutlass.Int32(1)
                    s_swf[0] = cutlass.Float32(self.FLT_MAX)  # thr0
                    s_swi[0] = cutlass.Int32(0)   # M0
                    s_swi[1] = cutlass.Int32(-1)  # up col this round
                cute.arch.barrier()

                rr = cutlass.Int32(0)
                while rr < cutlass.Int32(R) and s_msti[2] == cutlass.Int32(1):
                    if tidx == 0:
                        lo = s_mstf[0]
                        hi = s_mstf[1]
                        d = hi - lo
                        if rr == cutlass.Int32(0):
                            if cutlass.const_expr(self.place_mode == 5):
                                pass  # op21: s_mt_thr already placed by phase1b
                            elif cutlass.const_expr(self.place_mode == 3):
                                for m in cutlass.range_constexpr(M):
                                    s_mt_thr[m] = lo + d * cutlass.Float32(self.fracs[m])
                            elif cutlass.const_expr(self.place_mode == 0):
                                for m in cutlass.range_constexpr(M):
                                    s_mt_thr[m] = lo + d * (cutlass.Float32(m) / cutlass.Float32(M))
                            elif cutlass.const_expr(self.place_mode == 1):
                                s_mt_thr[0] = lo
                                for m in cutlass.range_constexpr(M - 1):
                                    s_mt_thr[m + 1] = lo + d * cutlass.Float32(1.0 / (1 << (M - 1 - m)))
                            else:
                                pm = pmean
                                if pm <= lo or pm >= hi:
                                    pm = (lo + hi) * cutlass.Float32(0.5)
                                half = cutlass.const_expr(M // 2)
                                for m in cutlass.range_constexpr(half):
                                    s_mt_thr[m] = lo + (pm - lo) * (cutlass.Float32(m) / cutlass.Float32(half))
                                for m in cutlass.range_constexpr(M - half):
                                    s_mt_thr[half + m] = pm + (hi - pm) * (cutlass.Float32(m) / cutlass.Float32(M - half))
                        else:
                            for m in cutlass.range_constexpr(M):
                                s_mt_thr[m] = lo + d * (cutlass.Float32(m + 1) / cutlass.Float32(M + 1))
                    cute.arch.barrier()

                    if cutlass.const_expr(self.fuse_collect):
                        # R==1 gated: all thresholds known up-front, collect at
                        # the l1 column during the same pass (op20 iter4)
                        if cutlass.const_expr(self.smem_row_elems > 0):
                            if row_in_smem == cutlass.Int32(1):
                                self.block_count_collect_multi_smem(
                                    smem_rowbuf, N, s_mt_thr, smem_ptcnt_multi,
                                    smem_wcnt_multi, s_mt_cnt, smem_slotk,
                                    smem_slotv, tidx, warp_id, lane)
                            else:
                                self.block_count_collect_multi(
                                    input_row, N, s_mt_thr, smem_ptcnt_multi,
                                    smem_wcnt_multi, s_mt_cnt, smem_slotk,
                                    smem_slotv, tidx, warp_id, lane)
                        else:
                            self.block_count_collect_multi(
                                input_row, N, s_mt_thr, smem_ptcnt_multi,
                                smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv,
                                tidx, warp_id, lane)
                    else:
                        self.block_count_ge_multi(input_row, N, s_mt_thr, smem_ptcnt_multi,
                                                  smem_wcnt_multi, s_mt_cnt, tidx, warp_id, lane)
                    cute.arch.barrier()

                    if tidx == 0:
                        best_m = cutlass.Int32(-1)
                        for m in cutlass.range_constexpr(M):
                            if s_mt_cnt[m] >= cutlass.Int32(top_k):
                                best_m = cutlass.Int32(m)
                        # sandwich upper: first column with count < K is
                        # best_m+1 (thresholds ascending => counts descending)
                        up_m = best_m + cutlass.Int32(1)
                        s_swi[1] = cutlass.Int32(-1)
                        if up_m < cutlass.Int32(M):
                            c_up = s_mt_cnt[up_m]
                            if c_up < cutlass.Int32(top_k) and c_up > s_swi[0]:
                                s_swi[0] = c_up
                                s_swf[0] = s_mt_thr[up_m]
                                s_swi[1] = up_m
                        if best_m >= cutlass.Int32(0):
                            c_new = s_mt_cnt[best_m]
                            t_new = s_mt_thr[best_m]
                            if c_new <= s_msti[0]:
                                s_msti[0] = c_new
                                s_mstf[2] = t_new
                                s_msti[1] = best_m
                            else:
                                s_msti[1] = cutlass.Int32(-1)
                            s_mstf[0] = t_new
                            if best_m < cutlass.Int32(M - 1):
                                s_mstf[1] = s_mt_thr[best_m + cutlass.Int32(1)]
                        else:
                            s_msti[1] = cutlass.Int32(-1)
                            s_mstf[1] = s_mt_thr[0]
                        # continue while band > bAcc and bracket nonempty
                        cont = cutlass.Int32(0)
                        if s_msti[0] == cutlass.Int32(_INT_MAX):
                            cont = cutlass.Int32(1)
                        elif (s_msti[0] - s_swi[0]) > cutlass.Int32(bAcc) and s_mstf[1] > s_mstf[0]:
                            cont = cutlass.Int32(1)
                        s_msti[2] = cont
                    cute.arch.barrier()

                    bc = s_msti[1]
                    if bc >= cutlass.Int32(0):
                        smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                    uc = s_swi[1]
                    if uc >= cutlass.Int32(0):
                        smem_ptcnt_up[tidx] = smem_ptcnt_multi[uc * cutlass.Int32(num_threads) + tidx]
                    cute.arch.barrier()
                    rr = rr + cutlass.Int32(1)

                # op20 iter4: slot-overflow reduce (any thread whose l1-column
                # count exceeded slot_cap => slots are incomplete => fall back
                # to the classic P3 rescan). s_iscalars[2] = overflow count.
                if cutlass.const_expr(self.fuse_collect):
                    ofv = cutlass.Int32(0)
                    if smem_ptcnt_multi[cutlass.Int32(self.pred_col) * cutlass.Int32(num_threads) + tidx] > cutlass.Int32(self.slot_cap):
                        ofv = cutlass.Int32(1)
                    ofv = self.warp_reduce_sum_i32(ofv)
                    if lane == 0:
                        smem_wcnt[warp_id] = ofv
                    cute.arch.barrier()
                    if tidx == 0:
                        oft = cutlass.Int32(0)
                        for w7 in cutlass.range_constexpr(num_warps):
                            oft = oft + smem_wcnt[w7]
                        s_iscalars[2] = oft
                # ---- finalize ----
                # done=1 sandwich: pair found and band fits smem
                # done=1 no-pair:  op18 rule (M1 <= kC), M0=0 column = zeros
                # done=2:          band > kC -> baseline retry-shrink
                if tidx == 0:
                    s_thr[0] = s_mstf[2]
                    band_f = s_msti[0] - s_swi[0]
                    if s_msti[0] != cutlass.Int32(_INT_MAX) and band_f <= cutlass.Int32(kC):
                        s_iscalars[0] = band_f
                        s_iscalars[1] = cutlass.Int32(1)
                    else:
                        s_iscalars[1] = cutlass.Int32(2)
                        s_thr[1] = s_mstf[2]
                        s_thr[2] = s_mstf[1]
                        s_swi[0] = cutlass.Int32(0)  # no sandwich on fallback
                        if cutlass.const_expr(self.fb_logfalsi):
                            # iter13 (HLS): stash the ladder-known bracket
                            # counts for the log-falsi fallback. [5] = count
                            # at s_thr[1] (tracked together with s_mstf[2]
                            # across rounds — always consistent). [6] = count
                            # at s_thr[2]: s_mstf[1] comes from the LAST
                            # round (thr[bm+1], or thr[0] when that round had
                            # no >=K column), so its count is still resident
                            # in s_mt_cnt; bm == M-1 leaves s_mstf[1] stale
                            # from an earlier round -> unknown (-1).
                            c_lo_s = cutlass.Int32(-1)
                            if s_msti[0] != cutlass.Int32(_INT_MAX):
                                c_lo_s = s_msti[0]
                            s_iscalars[5] = c_lo_s
                            bm2 = cutlass.Int32(-1)
                            for m2 in cutlass.range_constexpr(M):
                                if s_mt_cnt[m2] >= cutlass.Int32(top_k):
                                    bm2 = cutlass.Int32(m2)
                            c_hi_s = cutlass.Int32(-1)
                            if bm2 == cutlass.Int32(-1):
                                c_hi_s = s_mt_cnt[0]
                            elif bm2 < cutlass.Int32(M - 1):
                                c_hi_s = s_mt_cnt[bm2 + cutlass.Int32(1)]
                            s_iscalars[6] = c_hi_s
                cute.arch.barrier()

                if s_iscalars[1] == cutlass.Int32(1) and s_swi[0] > cutlass.Int32(0):
                    # ---- sandwich path ----
                    use_slots = cutlass.Int32(0)
                    if cutlass.const_expr(self.fuse_collect):
                        # usable iff best column >= l1 collect column (slots
                        # are a superset of the candidates) and no overflow
                        if s_msti[1] >= cutlass.Int32(self.pred_col) and s_iscalars[2] == cutlass.Int32(0):
                            use_slots = cutlass.Int32(1)
                    if cutlass.const_expr(self.fuse_collect):
                        if use_slots == cutlass.Int32(1):
                            self.phase3_from_slots(
                                smem_slotk, smem_slotv, smem_keys, smem_vals,
                                smem_ptcnt, smem_ptcnt_up, smem_ptcnt_multi,
                                smem_wcnt, smem_didx, s_thr, s_swf, s_iscalars,
                                output_indices_row, tidx, warp_id, lane)
                        else:
                            self.phase3_sandwich(input_row, N, smem_keys, smem_vals,
                                                 smem_ptcnt, smem_ptcnt_up, smem_wcnt,
                                                 smem_didx, s_thr, s_swf, s_iscalars,
                                                 output_values_row, output_indices_row,
                                                 tidx, warp_id, lane)
                    else:
                        self.phase3_sandwich(input_row, N, smem_keys, smem_vals,
                                             smem_ptcnt, smem_ptcnt_up, smem_wcnt,
                                             smem_didx, s_thr, s_swf, s_iscalars,
                                             output_values_row, output_indices_row,
                                             tidx, warp_id, lane)
                    band = s_iscalars[0]
                    if band > cutlass.Int32(kC):
                        band = cutlass.Int32(kC)
                    m0 = s_swi[0]
                    k_rem = cutlass.Int32(top_k) - m0
                    self.phase4_band_snap(smem_keys, smem_vals, smem_hist,
                                          smem_wcnt, s_thr, s_swf, s_iscalars,
                                          output_values_row, output_indices_row,
                                          band, k_rem, m0, tidx, warp_id, lane)
                else:
                    # ---- op18/baseline path (M0 == 0 or done=2) ----
                    self.phase3_collect_candidates(input_row, N, smem_keys, smem_vals, smem_ptcnt,
                                                   smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane)
                    cand_count_p4 = s_iscalars[0]
                    if cand_count_p4 > cutlass.Int32(self.kC):
                        cand_count_p4 = cutlass.Int32(self.kC)
                    self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_iscalars,
                                               output_values_row, output_indices_row, cand_count_p4, tidx, warp_id, lane)
        griddepcontrol_launch_dependents()


_compiled = {}
_STRADDLE_TABLE = None


def _load_straddle(K, n, M, dtype_name="fp32"):
    """op19 straddle-fracs, per-dtype table w/ fp32 fallback (nearest N)."""
    global _STRADDLE_TABLE
    if _STRADDLE_TABLE is None:
        import json
        _STRADDLE_TABLE = {}
        for dt in ("fp32", "bf16", "fp16"):
            sfx = "" if dt == "fp32" else f"_{dt}"
            p = _HERE.parent / "results" / f"straddle_fracs{sfx}.json"
            if p.exists():
                _STRADDLE_TABLE[dt] = json.load(open(p))
    tbl = _STRADDLE_TABLE.get(dtype_name) or _STRADDLE_TABLE.get("fp32", {})
    cands = []
    for key, v in tbl.items():
        k_, n_, m_ = (int(x) for x in key.split("_"))
        if k_ == K and m_ == M:
            cands.append((abs(n_ - n), v["fracs"]))
    if not cands:
        raise KeyError(f"no straddle fracs for K={K} M={M}")
    fr = sorted(cands)[0][1]
    while len(fr) < M:
        fr = fr + [min(0.999, fr[-1] + 0.01)]
    return tuple(fr[:M])


# ---------------------------------------------------------------------------
# op25 S1a: per-K static ladder ship table. Host-replay screened 4 rounds
# (op25_hls_expand/screen_qfracs.py + screen_r4_m4.py, 30290 rows/round:
# op22rr fp32 grid + op24 392-combo hr sweep + 29.8k REAL Pro multi-turn
# transitions), then silicon-decomposed (ab_decomp.py):
#   w3a = (0.92, 0.45, 0.048), M_thr=4  ==  ZERO fast-path tax vs stock
#     - 0.92 col fixes the h>0.75 pair01 cliff (Pro real fast 0.31->0.96;
#       the cliff is 67% of real Pro decode steps)
#     - 0.048 tail covers the adversarial all_ge pole (op22rr worst 0->0.78)
#     - M=5 variants (wide4b) were admission-equal but pay +7..19% on fast
#       rows (count-loop column not divided by C) -> rejected on silicon
#   K2048 keeps the stock ladder (deep cols regress the v32 band geometry:
#   real 0.75 -> 0.375 in replay, three independent screens).
# slot_scale=2 rides along N-gated (<65536): free at t=512, +12..21% at
# t=1024 (decomp) — it unlocks the Pro low-h collect overflow (N~9.4K).
# OP25_QFRACS overrides ("base" = stock triple); OP25_SLOTCAP scales slots.
# op27 (2026-07-10, b200-027 ab_decomp same-node paired): K2048 gains a TAIL
# ladder (0.75, 0.45, 0.048) — keeps the stock 0.75 top column (the op25
# rejection above targeted 0.92-top w3a variants), swaps the two lower
# columns for the all_ge tail. Silicon: worst gm 1.96x vs stock (per-cell
# 1.55-2.39; every op22rr K2048 worst loss cell was mode all_ge), best -1.2%
# / real -0.7% (noise-level). OP27_K2048_TAIL=0 restores stock.
# ---------------------------------------------------------------------------
_QFRACS_STOCK = (0.75, 0.5, 0.25)
_QFRACS_SHIP = {512: (0.92, 0.45, 0.048),
                1024: (0.92, 0.45, 0.048)}
_QFRACS_K2048_TAIL = (0.75, 0.45, 0.048)


def _qfracs_for(K):
    env = os.environ.get("OP25_QFRACS")
    if env == "base":
        return _QFRACS_STOCK
    if env:
        return tuple(float(x) for x in env.split(","))
    if int(K) == 2048 and os.environ.get("OP27_K2048_TAIL", "1") == "1":
        return _QFRACS_K2048_TAIL
    return _QFRACS_SHIP.get(int(K), _QFRACS_STOCK)


def _slot_scale():
    return int(os.environ.get("OP25_SLOTCAP", "2"))


def _config(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    # iter5 (b) FALSIFIED: QBINS=64 at bs > NUM_SMS measured a wash (event
    # gm 1.004, 14 P1 cells) — P1b's hist scan is NOT the highBS bottleneck.
    # Rule reverted to a constant; OP21_QBINS stays as the A/B probe knob.
    qb_env = os.environ.get("OP21_QBINS")
    qbins = int(qb_env) if qb_env else 256
    return t, use256, min_bpm, qbins


def _compile(dtype, bs, n, K, cr_val, M, R, band_acc, place_mode, kC, threads, unroll=4,
             fuse=False, smem=False, qfracs=None, slot_scale=1):
    t, use256, min_bpm, qbins = _config(bs, n)
    # op25 decomp (b200-027): slot_scale=2 is FREE at n<65536 (t=512, slot
    # smem 80KB) and a +12..21% systematic tax at n>=65536 (t=1024, 131KB)
    # -> N-gate it to the small-N ms regime, which is also where the real
    # Pro overflow rows live (N~9.4K).
    if n >= 65536:
        slot_scale = 1
    if threads is not None:
        t = threads
    qbins = min(qbins, t)
    # iter5 A/B: OP21_P4_RS=0 falls back to the legacy runtime-k band snap
    p4_rs = os.environ.get("OP21_P4_RS", "1") == "1"
    # iter6/11 A/B: OP21_P4_FAST=0 disables the fast paths (P4 -> exact snap)
    p4_fast = os.environ.get("OP21_P4_FAST", "1") == "1"
    # iter9 A/B: OP21_P2_NATIVE=0 restores the cvt->fp32 ladder (16-bit only)
    p2_nat = os.environ.get("OP21_P2_NATIVE", "1") == "1"
    # iter13 A/B: OP21_FB_LOGFALSI=0 restores the blind-bisection fallback;
    # OP21_FB_ALPHA probes the interior aim exponent (default 0.2)
    fb_lf = os.environ.get("OP21_FB_LOGFALSI", "1") == "1"
    fb_al = float(os.environ.get("OP21_FB_ALPHA", "0.2"))
    # key on DERIVED compile inputs (t/use256/min_bpm), not raw bs — one
    # binary serves every BS in the same bucket (n stays: per-N fracs)
    qfracs = tuple(qfracs) if qfracs else (0.75, 0.5, 0.25)
    key = (dtype, t, use256, min_bpm, n, K, cr_val, M, R, band_acc, place_mode, kC, unroll, fuse, smem, p4_rs, qbins, p4_fast, p2_nat, fb_lf, fb_al, qfracs, slot_scale)
    if key in _compiled:
        return _compiled[key]
    _dtn = {torch.float32: "fp32", torch.bfloat16: "bf16",
            torch.float16: "fp16"}[dtype]
    if place_mode == 5:
        fracs = None
        kernel_place = 5  # op21 rank-quantile: no offline table
    elif place_mode == 4:
        fracs = _load_straddle(K, n, M, _dtn)
        kernel_place = 3  # same codegen: compile-time frac table
    else:
        fracs = _load_fracs(K, n, M) if place_mode == 3 else None
        kernel_place = place_mode
    kobj = GvrSandwichKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                             use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                             min_blocks_per_mp=min_bpm, return_output_values=False,
                             M_thr=M, R_rounds=R, band_accept=band_acc, place_mode=kernel_place,
                             kC_override=kC, fracs=fracs, fuse_collect=fuse,
                             smem_row_elems=(n if smem else 0),
                             p4_rank_scatter=p4_rs, qbins=qbins,
                             p4_smallbin=p4_fast, p2_native=p2_nat,
                             fb_logfalsi=fb_lf, fb_alpha=fb_al,
                             qfracs=qfracs, slot_scale=slot_scale)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if use256 else 16
    in_f = cr.make_fake_compact_tensor(_DT[dtype], (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, stream=fs, options="--enable-tvm-ffi")
    _compiled[key] = c
    return c


def gvr_sw(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
           M=4, R=2, band_acc=64, place_mode=3, kC=None, threads=None, fuse=None,
           smem=None, qfracs=None, slot_scale=1):
    bs, n = logits.shape
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    if fuse is None:
        # op20 iter4 auto-gate: fused P2+P3 needs all thresholds known before
        # the scan (R==1, straddle placement) and 1-CTA/SM residency headroom
        # for the +2*kC*4B slot smem (bs <= NUM_SMS <=> min_bpm == 1).
        fuse = (int(R) == 1 and int(place_mode) in (4, 5) and bs <= NUM_SMS)
    if smem is None:
        # op20 iter6: smem-resident row measured as a perf NO-OP at N<=8192
        # (small-N wall is phase-chain latency, not memory tier) — default
        # OFF; the explicit 's' cfg suffix can still enable it.
        smem = False
    smem = bool(smem) and bool(fuse) and logits.dtype == torch.float32 and n <= 8192
    compiled = _compile(logits.dtype, bs, n, index_topk, compress_ratio, int(M), int(R),
                        int(band_acc), int(place_mode), kC, threads, fuse=bool(fuse),
                        smem=smem, qfracs=qfracs, slot_scale=int(slot_scale))
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


_DISPATCH_TABLE = None
_BS_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _load_dispatch(dtype_name):
    global _DISPATCH_TABLE
    if _DISPATCH_TABLE is None:
        import json
        _DISPATCH_TABLE = {}
        for dt in ("fp32", "bf16", "fp16"):
            p = _HERE.parent / "results" / f"dispatch_table_{dt}.json"
            if p.exists():
                _DISPATCH_TABLE[dt] = json.load(open(p))
    return _DISPATCH_TABLE.get(dtype_name, {})


def gvr_sw_auto(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None):
    """Dispatch entry: per-(K, N-bucket, BS-bucket) best config from the
    sweep-built table; 'baseline' falls back to gvr_cutedsl; 'clusterG'
    dispatches to the Strategy-B cluster op."""
    import re as _re
    from gvr_cutedsl_op import gvr_cutedsl
    bs, n = logits.shape
    dtn = {torch.float32: "fp32", torch.bfloat16: "bf16",
           torch.float16: "fp16"}[logits.dtype]
    tbl = _load_dispatch(dtn) or _load_dispatch("fp32")
    K = index_topk
    ns = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
    nb = min(ns, key=lambda x: abs(x - n))
    bb = min(_BS_GRID, key=lambda x: abs(x - bs))
    ent = tbl.get(f"{K}_{nb}_{bb}")
    cfg = ent["cfg"] if ent else "M4R1p4"
    if cfg == "baseline":
        return gvr_cutedsl(logits, pre_idx, seq_lens, index_topk,
                           compress_ratio, out=out)
    if cfg == "mc" or cfg.startswith("mcC"):
        # op20 iter2/3a: data-parallel cluster GVR (PR#15198) — each CTA scans
        # an N/cluster_size chunk, unlike swc's threshold-parallel O(N)-per-CTA.
        # Decisive at N>=131K low-BS (probe: 131K BS1 20.9us vs radix 24.7;
        # C=8 reaches parity at K512 262K, C=16 regresses via DSM merge cost).
        from gvr_multicta_cutedsl_op import gvr_multicta_cutedsl
        cs = int(cfg[3:]) if cfg.startswith("mcC") else None
        return gvr_multicta_cutedsl(logits, pre_idx, seq_lens, index_topk,
                                    compress_ratio, out=out, cluster_size=cs)
    if cfg.startswith("cluster"):
        from gvr_swc_op import gvr_swc
        return gvr_swc(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                       out=out, G=int(cfg[7:]))
    if cfg.startswith("fusP"):
        # op20 iter5: op17-v2 P-slice x T-threshold fusion cluster (each of P
        # partition groups scans N/P; T threshold slots per group; DSMEM
        # 2-D reduce). Beats mc 1.17-1.20x at N262K BS<=4 (op17 D1 nsys).
        mf = _re.match(r"fusP(\d+)T(\d+)$", cfg)
        _op17 = str(Path(__file__).resolve().parents[2] / "op17_gvr_portfolio" / "v2")
        if _op17 not in sys.path:
            sys.path.insert(0, _op17)
        from gvr_portfolio_fusion_op import gvr_portfolio_fusion
        return gvr_portfolio_fusion(logits, pre_idx, seq_lens, index_topk,
                                    compress_ratio, out=out,
                                    P=int(mf.group(1)), T=int(mf.group(2)))
    # op20 iter4: optional fuse suffix — 'f' forces fused P2+P3, 'nf' forces
    # classic; no suffix = gvr_sw auto-gate (R==1 & p4 & bs<=NUM_SMS).
    # op20 iter6: optional trailing 's' forces the smem-resident row,
    # 'ns' forces it off; no suffix = auto-gate (fuse & fp32 & N<=8192).
    m = _re.match(r"M(\d+)R(\d+)p(\d+)(?:b(\d+))?(f|nf)?(s|ns)?$", cfg)
    fuse = {None: None, "f": True, "nf": False}[m.group(5)]
    smem = {None: None, "s": True, "ns": False}[m.group(6)]
    return gvr_sw(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                  out=out, M=int(m.group(1)), R=int(m.group(2)),
                  place_mode=int(m.group(3)),
                  band_acc=int(m.group(4)) if m.group(4) else 64, fuse=fuse,
                  smem=smem)


if __name__ == "__main__":
    import synth_data
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(f"op19 sandwich smoke (fp32, M={M}, R={R}, band_acc=64)")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            if N <= 2 * K:
                continue
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
            Npad = b["Npad"]
            seq_lens = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
            out = gvr_sw(logits, pre, seq_lens, K, crv, M=M, R=R)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nu = len(set(out[0].tolist()))
            tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
            print(f"  K={K:4d} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")


# ---------------------------------------------------------------------------
# op21 production-shaped entry — NO dispatch table. One rule set:
#   M=4 rank-quantile ladder (place_mode=5, qfracs 0.75/0.5/0.25 of K_valid),
#   R=1, fused P2+P3 slot-collect when bs <= NUM_SMS (one CTA/SM wave),
#   band_accept=64, spec kC. Placement is data-driven (prev-step order
#   statistics) => no offline straddle table, no per-N keys.
# ---------------------------------------------------------------------------
def gvr_ms(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
           threads=None, R=1):
    # fuse rule (production-legal, 2 terms): one CTA wave AND the spec-collect
    # buffer holds >= 4x K (kC default 5120 => fuse for K512/K1024, not K2048
    # where slot overflow makes fused collect a measured 13% loss at large N).
    bs = logits.shape[0]
    fuse = bool(bs <= NUM_SMS and 4 * int(index_topk) <= 5120)
    # op25 S1a: per-K screened ladder + slot-capacity scale (see table above)
    qf = _qfracs_for(index_topk)
    return gvr_sw(logits, pre_idx, seq_lens, index_topk,
                  compress_ratio=compress_ratio, out=out,
                  M=len(qf) + 1, R=int(R), band_acc=64, place_mode=5,
                  threads=threads, fuse=fuse, qfracs=qf,
                  slot_scale=_slot_scale())
