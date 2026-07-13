# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""op26 iter6b: R0 h-space ladder admission for the PR#15198 cluster GVR.

Same three unified facts as gvr_op26_r0_op.py (see that header +
PLAN_ITER6.md), ported to the multi-CTA cluster kernel that owns the
low-BS large-N regime (the 1cta arm's structural wall vs radix's 4-32
SMs/row):

  P1  (vendored, per-CTA redundant over the FULL preIdx) ->
  P1b 256-bin hist over prev-topK values, per-CTA redundant => rungs are
      bit-identical on every CTA of the cluster (zero DSMEM coordination) ->
  R0  ONE M-ary count over THIS CTA's slice (block_count_ge_multi_slice,
      same memory path as the vendored slice count_ge) + a single
      cluster arrive/wait merging all M columns (M-slot s_cluster_partial) ->
  admission: tightest rung with cluster count in [K, kC]; every CTA takes
      the same branch (decision inputs are cluster totals + shared rungs).
      Accepted: the winning column seeds the slice-local smem_ptcnt and
      s_iscalars[5] (the Shift-D contract phase3's collect-write needs);
      P3/P4 proceed exactly as vendored (per-slice collect, leader gather,
      leader P4).
  miss -> R1 inline log-falsi shot between the two MEASURED rungs via the
      vendored cluster block_count_ge (which maintains ptcnt/[5]/[0]) ->
      double-miss falls to the VENDORED retry-shrink fallback = the mc
      anchor's own exactness envelope (op26_mc precedent: per-slice retry +
      leader handoff too risky to fork; fb_fix stays 1cta-only).

Zero vendored edits; subclass GvrTopKClusterKernel directly (the p4/op18
cross-lineage mixin is impossible — see gvr_op26_r0_op.py header).
"""
import math
import os
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass.cute import runtime as cr
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "ops"))                         # cute_vendored

from gvr_op26_op import _resolve_config_mc  # noqa: E402  (host heuristics)
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: E402
    GvrTopKClusterKernel,
)
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: E402
    mapa_shared_cluster, ld_shared_cluster_i32, ld_shared_cluster_f32,
    _fmin_f32_inline, float_as_uint32,
)
from cutlass._mlir.dialects import llvm  # noqa: E402  (p4_rs bitcast)
from cutlass.cutlass_dsl import T, dsl_user_op  # noqa: E402  (p4_coop DSMEM atomics)
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
       torch.float16: cutlass.Float16}

M2D = (0.85, 0.35)   # iter6 v0.2 ship ladder (see gvr_op26_r0_op.py)


# --------------------------------------------------------------------------
# DSMEM atomics for the p4_coop cooperative P4 (iter7 D2). Same inline-PTX
# pattern as the vendored mapa/ld cluster helpers: CuTe DSL smem tensor ops
# do not lower to cluster address space.
# --------------------------------------------------------------------------
@dsl_user_op
def _red_shared_cluster_add_i32(mapped_addr, val, *, loc=None, ip=None):
    """Atomic add (no return) into a cluster-mapped SMEM address."""
    llvm.inline_asm(
        None,
        [mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "red.shared::cluster.add.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def red_shared_cluster_add_i32(mapped_addr, val):
    _red_shared_cluster_add_i32(mapped_addr, val)


@dsl_user_op
def _atom_shared_cluster_add_i32(mapped_addr, val, *, loc=None, ip=None):
    """Atomic fetch-add into a cluster-mapped SMEM address (returns old)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip),
             val.ir_value(loc=loc, ip=ip)],
            "atom.shared::cluster.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def atom_shared_cluster_add_i32(mapped_addr, val):
    return _atom_shared_cluster_add_i32(mapped_addr, val)


class GvrOp26R0ClusterKernel(GvrTopKClusterKernel):
    """Cluster GVR + R0 h-space ladder admission + R1 inline falsi shot."""

    def __init__(self, *a, qfracs=M2D, mt_unroll=4, p1b_cache=False,
                 p4_rs=False, p4_coop=False, r1aim="center",
                 kC_override=None, **kw):
        super().__init__(*a, **kw)
        # kC-diet (backlog-3): narrow the classic candidate window; must be
        # set before the kC-derived aims below. Vendored base has no
        # override hook, so apply post-ctor (kC is const_expr'd at JIT).
        if kC_override is not None:
            self.kC = int(kC_override)
        # compile-time debug printfs (OP26_R0MC_DEBUG=1, fresh process)
        self.dbg = bool(int(os.environ.get("OP26_R0MC_DEBUG", "0")))
        # p4_rs (iter7): leader P4 = op#7 EXACT rank-scatter (verbatim from
        # p4_recursive_digit GvrTopKKernel.phase4_rank_scatter, fixed 256-bin
        # fine level) instead of the vendored histogram_snap. Motivation:
        # ncu on the fin low-BS negative band shows the leader serial tail
        # (peer gather + single-CTA P4) is 57-61% of wall time while 3 CTAs
        # idle at the final cluster barrier (PLAN_ITER7.md §0).
        self.p4_rs = bool(p4_rs)
        # keep the ported body verbatim (it const_exprs on this flag)
        self.enable_p4_rank_scatter_exact = True
        # p4_coop (iter7 D2): cluster-cooperative exact rank-scatter — no
        # leader gather, distributed hist/scatter via DSMEM atomics. Only
        # meaningful at cluster_size>1 (cs=1 falls back to p4_rs/snap).
        # Needs >=2 s_cluster_partial_m slots for the min/max exchange.
        self.p4_coop = bool(p4_coop)
        # p1b_cache (1cta r0f port): P1 stores the K gathered preIdx values
        # into SMEM so the per-CTA-redundant P1b hist skips its second GMEM
        # random gather. Costs top_k*4B extra SMEM per CTA of the cluster.
        # Wrapper default = dispatch_p1bc_mc_op26 (069 A/B: ON all dtypes).
        self.p1b_cache = bool(p1b_cache)
        assert not self.enable_smem_cache, \
            "op26_r0mc v0 supports the production enable_smem_cache=False only"
        assert all(0.0 < q < 1.0 for q in qfracs), qfracs
        assert list(qfracs) == sorted(qfracs, reverse=True), \
            "qfracs must be descending h (ascending threshold value)"
        self.qfracs = tuple(float(q) for q in qfracs)
        self.M_thr = len(self.qfracs)
        if self.p4_coop:
            assert self.M_thr >= 2, "p4_coop needs M>=2 partial slots"
        self.mt_unroll = int(mt_unroll)
        self.qneeds = tuple(max(1, int(math.ceil(q * self.top_k)))
                            for q in self.qfracs)
        # R1 inline-shot aim: "center" = geometric center sqrt(kK*kCC)
        # (shipped); "edge" = log2(kK) (backlog-2 ablation: iter5d 1cta aim
        # table showed edge/center flips by (K,dtype,N), never mc-checked).
        assert r1aim in ("center", "edge"), r1aim
        self.r1aim = r1aim
        self.log2_r1aim = (math.log2(float(self.top_k)) if r1aim == "edge"
                           else math.log2(math.sqrt(self.top_k * self.kC)))
        # fb_fix interior aim (op26 1cta port, fb_alpha=0.2)
        self.log2_mstar = math.log2(
            self.top_k * (self.kC / self.top_k) ** 0.2)

    # ------------------------------------------------------------------
    # P1b — identical to the 1cta version (full-row preIdx, per-CTA
    # redundant so rungs match bit-for-bit across the cluster).
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_hspace_rungs(self, input_row, N, pre_idx_row, pre_idx_count,
                             pre_idx_offset, smem_hist, s_thr, s_mt_thr,
                             tidx, warp_id, lane):
        M = cutlass.const_expr(self.M_thr)
        NB = cutlass.const_expr(256)
        SEG = cutlass.const_expr(8)
        num_threads = cutlass.const_expr(self.num_threads)

        jz = tidx
        while jz < cutlass.Int32(NB):
            smem_hist[jz] = cutlass.Int32(0)
            jz = jz + cutlass.Int32(num_threads)
        cute.arch.barrier()

        v_lo = s_thr[1]
        v_hi = s_thr[2]
        width = (v_hi - v_lo) / cutlass.Float32(NB)
        inv_w = cutlass.Float32(1.0) / width

        ig = tidx
        while ig < cutlass.Int32(pre_idx_count):
            idx = pre_idx_row[ig] + pre_idx_offset
            if idx >= cutlass.Int32(0) and idx < N:
                v = cutlass.Float32(input_row[idx])
                bf = (v - v_lo) * inv_w
                b = cutlass.Int32(bf)
                if b < cutlass.Int32(0):
                    b = cutlass.Int32(0)
                if b > cutlass.Int32(NB - 1):
                    b = cutlass.Int32(NB - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
            ig = ig + cutlass.Int32(num_threads)
        cute.arch.barrier()

        if warp_id == cutlass.Int32(0):
            top = cutlass.Int32(NB - 1) - lane * cutlass.Int32(SEG)
            seg_frag = cute.make_fragment((SEG,), cutlass.Int32)
            part = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                v8 = smem_hist[top - cutlass.Int32(j)]
                seg_frag[j] = v8
                part = part + v8
            tp = part
            for off_i in cutlass.range_constexpr(5):
                off_v = cutlass.const_expr(1 << off_i)
                other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
                if lane >= cutlass.Int32(off_v):
                    tp = tp + other
            excl = tp - part
            total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))
            run = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                run = run + seg_frag[j]
                cum_at = excl + run
                cum_before = cum_at - seg_frag[j]
                for m in cutlass.range_constexpr(M):
                    if (cum_at >= cutlass.Int32(self.qneeds[m])
                            and cum_before < cutlass.Int32(self.qneeds[m])):
                        s_mt_thr[m] = v_lo + cutlass.Float32(top - cutlass.Int32(j)) * width
            if lane == 0:
                for m in cutlass.range_constexpr(M):
                    if total < cutlass.Int32(self.qneeds[m]):
                        s_mt_thr[m] = v_lo
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P1 (p1b_cache variant) — verbatim p4 phase1_preidx_stats plus a
    # per-slot SMEM store of the gathered value (sentinel NEG_FLT_MAX for
    # invalid/out-of-range preIdx) so P1b skips the second GMEM gather.
    # Slot layout matches P1b's stride loop: smem_gath[i] for preIdx i.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1_preidx_stats_cached(self, input_row, N, pre_idx_row,
                                   pre_idx_count, pre_idx_offset, smem_gath,
                                   smem_wmin_f32, smem_wmax_f32,
                                   smem_wsum_f32, smem_wcnt_i32, s_thr,
                                   s_iscalars, tidx, warp_id, lane):
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
                smem_gath[i] = cutlass.Float32(self.NEG_FLT_MAX)
                if idx >= 0 and idx < N:
                    v = self._load_fp32(input_row, idx)
                    smem_gath[i] = v
                    local_max = cute.arch.fmax(local_max, v)
                    local_min = _fmin_f32_inline(local_min, v)
                    local_sum = local_sum + v
                    local_cnt = local_cnt + 1
        else:
            idx = cutlass.Int32(-1)
            if tidx < cutlass.Int32(pre_idx_count):
                idx = pre_idx_row[tidx] + pre_idx_offset
                smem_gath[tidx] = cutlass.Float32(self.NEG_FLT_MAX)
            if idx >= 0 and idx < N:
                v = self._load_fp32(input_row, idx)
                smem_gath[tidx] = v
                local_max = cute.arch.fmax(local_max, v)
                local_min = _fmin_f32_inline(local_min, v)
                local_sum = local_sum + v
                local_cnt = local_cnt + 1

        active_preidx_warps = cutlass.const_expr(
            min(pre_idx_count // self.WARP_SIZE, self.num_warps))
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

        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            if warp_id == cutlass.Int32(0):
                v_min = cutlass.Float32(self.FLT_MAX)
                v_max = cutlass.Float32(self.NEG_FLT_MAX)
                v_sum = cutlass.Float32(0.0)
                v_cnt = cutlass.Int32(0)
                if lane < cutlass.Int32(active_preidx_warps):
                    v_min = smem_wmin_f32[lane]
                    v_max = smem_wmax_f32[lane]
                    v_sum = smem_wsum_f32[lane]
                    v_cnt = smem_wcnt_i32[lane]
                pmin = self.warp_reduce_min_f32(v_min)
                pmax = self.warp_reduce_max_f32(v_max)
                psum = self.warp_reduce_sum_f32(v_sum)
                pcnt = self.warp_reduce_sum_i32(v_cnt)
                if lane == cutlass.Int32(0):
                    pmean = cutlass.Float32(0.0)
                    if pcnt > 0:
                        pmean = psum / cutlass.Float32(pcnt)
                    else:
                        pmean = (pmin + pmax) * cutlass.Float32(0.5)
                    cnt_lo_seed = pre_idx_count + (pre_idx_count >> 2)
                    s_thr[0] = pmean
                    s_thr[1] = pmin
                    s_thr[2] = pmax
                    s_iscalars[0] = cutlass.Int32(0)
                    s_iscalars[1] = cutlass.Int32(0)
                    s_iscalars[2] = cutlass.Int32(cnt_lo_seed)
                    s_iscalars[3] = cutlass.Int32(1)
                    s_iscalars[4] = cutlass.Int32(0)
        else:
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
                s_iscalars[0] = cutlass.Int32(0)
                s_iscalars[1] = cutlass.Int32(0)
                s_iscalars[2] = cutlass.Int32(cnt_lo_seed)
                s_iscalars[3] = cutlass.Int32(1)
                s_iscalars[4] = cutlass.Int32(0)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P1b (p1b_cache variant) — hist from the SMEM-cached gathered values;
    # no GMEM traffic, no idx arithmetic. Sentinel NEG_FLT_MAX = invalid.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_hspace_rungs_cached(self, pre_idx_count, smem_gath,
                                    smem_hist, s_thr, s_mt_thr, tidx,
                                    warp_id, lane):
        M = cutlass.const_expr(self.M_thr)
        NB = cutlass.const_expr(256)
        SEG = cutlass.const_expr(8)
        num_threads = cutlass.const_expr(self.num_threads)

        jz = tidx
        while jz < cutlass.Int32(NB):
            smem_hist[jz] = cutlass.Int32(0)
            jz = jz + cutlass.Int32(num_threads)
        cute.arch.barrier()

        v_lo = s_thr[1]
        v_hi = s_thr[2]
        width = (v_hi - v_lo) / cutlass.Float32(NB)
        inv_w = cutlass.Float32(1.0) / width

        ig = tidx
        while ig < cutlass.Int32(pre_idx_count):
            v = smem_gath[ig]
            if v > cutlass.Float32(self.NEG_FLT_MAX):
                bf = (v - v_lo) * inv_w
                b = cutlass.Int32(bf)
                if b < cutlass.Int32(0):
                    b = cutlass.Int32(0)
                if b > cutlass.Int32(NB - 1):
                    b = cutlass.Int32(NB - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
            ig = ig + cutlass.Int32(num_threads)
        cute.arch.barrier()

        if warp_id == cutlass.Int32(0):
            top = cutlass.Int32(NB - 1) - lane * cutlass.Int32(SEG)
            seg_frag = cute.make_fragment((SEG,), cutlass.Int32)
            part = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                v8 = smem_hist[top - cutlass.Int32(j)]
                seg_frag[j] = v8
                part = part + v8
            tp = part
            for off_i in cutlass.range_constexpr(5):
                off_v = cutlass.const_expr(1 << off_i)
                other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
                if lane >= cutlass.Int32(off_v):
                    tp = tp + other
            excl = tp - part
            total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))
            run = cutlass.Int32(0)
            for j in cutlass.range_constexpr(SEG):
                run = run + seg_frag[j]
                cum_at = excl + run
                cum_before = cum_at - seg_frag[j]
                for m in cutlass.range_constexpr(M):
                    if (cum_at >= cutlass.Int32(self.qneeds[m])
                            and cum_before < cutlass.Int32(self.qneeds[m])):
                        s_mt_thr[m] = v_lo + cutlass.Float32(top - cutlass.Int32(j)) * width
            if lane == 0:
                for m in cutlass.range_constexpr(M):
                    if total < cutlass.Int32(self.qneeds[m]):
                        s_mt_thr[m] = v_lo
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # R0 — M-ary count over THIS CTA's slice + one cluster merge.
    # Writes: smem_ptcnt_multi (M slice-local per-thread columns),
    # s_mt_local[m] (this CTA's slice totals), s_mt_cnt[m] (cluster totals,
    # via the M-slot s_cluster_partial_m + ONE arrive/wait).
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_ge_multi_slice(self, input_row, slice_start, slice_end,
                                   s_mt_thr, smem_ptcnt_multi,
                                   smem_wcnt_multi, s_mt_local, s_mt_cnt,
                                   s_cluster_partial_m, tidx, warp_id, lane):
        M = cutlass.const_expr(self.M_thr)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        cluster_size = cutlass.const_expr(self.cluster_size)
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

        row_addr = input_row.iterator.toint()
        slice_len = slice_end - slice_start
        n_aligned = slice_start + (slice_len // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = slice_start + tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_unroll_4:
            rng_frag = cute.make_fragment((vec_w,), self.dtype)
            big_iters = cutlass.Int32(0)
            if slice_end > i + cutlass.Int32(vec_w - 1):
                big_iters = (slice_end - i - cutlass.Int32(vec_w)) // cutlass.Int32(step_elem) + cutlass.Int32(1)
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
                    for m in cutlass.range_constexpr(M):
                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < slice_end:
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
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + step

        it = n_aligned + tidx
        while it < slice_end:
            v = self._load_fp32(input_row, it)
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
                    s_mt_local[m] = total
        cute.arch.barrier()

        # cluster merge of the M slice totals (one arrive/wait)
        if cutlass.const_expr(cluster_size > 1):
            if tidx == cutlass.Int32(0):
                for m in cutlass.range_constexpr(M):
                    s_cluster_partial_m[m] = s_mt_local[m]
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if tidx == cutlass.Int32(0):
                local_ptr = s_cluster_partial_m.iterator
                for m in cutlass.range_constexpr(M):
                    total = cutlass.Int32(0)
                    for peer in cutlass.range_constexpr(cluster_size):
                        peer_addr = mapa_shared_cluster(
                            local_ptr + cutlass.Int32(m), cutlass.Int32(peer))
                        total = total + ld_shared_cluster_i32(peer_addr)
                    s_mt_cnt[m] = total
            cute.arch.barrier()
        else:
            if tidx == cutlass.Int32(0):
                for m in cutlass.range_constexpr(M):
                    s_mt_cnt[m] = s_mt_local[m]
            cute.arch.barrier()

    # ------------------------------------------------------------------
    # P3 — fb_fix port (GvrOp26Kernel.phase3_collect_candidates, cluster-
    # aggregated counts). WHY (074 first-silicon): the vendored retry-shrink
    # only fixes overshoot (`while count > kCC`); one bisection step over a
    # WIDE R0-miss bracket can skip the [kK, kCC] window and exit undershoot
    # -> P4 Branch C pads -1 (K2048 cr=1 N262144 hr* repro: cand 1654 < 2048,
    # see debug_r0mc_k2048_dbg.py). The anchor never sees this because its P2
    # secant hands over a MEASURED tight bracket; R0's miss bracket is
    # rung-derived and can be 6 decades wide in count space. Cluster-safe:
    # every decision input (s_iscalars[0] after DSMEM all-reduce, s_thr) is
    # cluster-identical, so all CTAs run identical trajectories; each
    # block_count_ge call does its own arrive/wait.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_collect_candidates(self, input_row, N, slice_start, slice_end,
                                  smem_keys, smem_vals, smem_ptcnt, smem_wcnt,
                                  s_thr, s_iscalars, s_cluster_partial, tidx,
                                  warp_id, lane, smem_input=None):
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        if s_iscalars[1] != cutlass.Int32(1):
            # Bracket counts are NOT trustworthy (rung counts, or R1 partial
            # measurements). Mark BOTH end counts unknown; only measured
            # values feed the falsi.
            if tidx == 0:
                s_iscalars[1] = cutlass.Int32(0)
                s_iscalars[2] = cutlass.Int32(-1)  # cnt_lo: unknown
                s_iscalars[3] = cutlass.Int32(-1)  # cnt_hi: unknown
            cute.arch.barrier()
            rs = cutlass.Int32(0)
            while rs < cutlass.Int32(30) and s_iscalars[1] == cutlass.Int32(0):
                if rs > cutlass.Int32(0):
                    if tidx == 0:
                        lo3 = s_thr[1]
                        hi3 = s_thr[2]
                        clo3 = s_iscalars[2]
                        chi3 = s_iscalars[3]
                        cand = (lo3 + hi3) * cutlass.Float32(0.5)
                        if chi3 < cutlass.Int32(0):
                            cand = hi3          # measure the hi end first
                        elif clo3 < cutlass.Int32(0):
                            cand = lo3          # then the lo end
                        else:
                            # both ends measured: log-count regula falsi
                            # aimed at the interior target m*; midpoint
                            # safeguard on degeneracy
                            chic = chi3
                            if chic < cutlass.Int32(1):
                                chic = cutlass.Int32(1)
                            l_lo = cmath.log2(cutlass.Float32(clo3),
                                              fastmath=True)
                            l_hi = cmath.log2(cutlass.Float32(chic),
                                              fastmath=True)
                            den3 = l_lo - l_hi
                            if den3 > cutlass.Float32(0.0):
                                t3 = (cutlass.Float32(self.log2_mstar)
                                      - l_hi) / den3
                                cnd3 = hi3 + t3 * (lo3 - hi3)
                                if cnd3 > lo3 and cnd3 < hi3:
                                    cand = cnd3
                        s_thr[0] = cand
                    cute.arch.barrier()
                self.block_count_ge(
                    input_row, slice_start, slice_end, s_thr[0], smem_ptcnt,
                    smem_wcnt, s_iscalars, s_cluster_partial, tidx, warp_id,
                    lane, smem_input=smem_input)
                cute.arch.barrier()
                if tidx == 0:
                    c3 = s_iscalars[0]
                    t3v = s_thr[0]
                    if c3 >= cutlass.Int32(kK) and c3 <= cutlass.Int32(kCC):
                        s_iscalars[1] = cutlass.Int32(1)  # accept
                    elif c3 > cutlass.Int32(kCC):
                        # overshoot: t3v is a measured lo end
                        s_thr[1] = t3v
                        s_iscalars[2] = c3
                        if t3v >= s_thr[2]:
                            # even the hi end overshoots -> expand upward
                            rng3 = s_thr[2] - s_thr[1]
                            if rng3 < cutlass.Float32(1.0):
                                rng3 = cutlass.Float32(1.0)
                            s_thr[2] = s_thr[2] + rng3 * cutlass.Float32(8.0)
                            s_iscalars[3] = cutlass.Int32(-1)
                    else:
                        # undershoot: t3v is a measured hi end
                        s_thr[2] = t3v
                        s_iscalars[3] = c3
                        if t3v <= s_thr[1]:
                            # even the lo end undershoots (possible with
                            # invalid preIdx entries) -> expand downward
                            rng3 = s_thr[2] - s_thr[1]
                            if rng3 < cutlass.Float32(1.0):
                                rng3 = cutlass.Float32(1.0)
                            s_thr[1] = s_thr[1] - rng3 * cutlass.Float32(8.0)
                            s_iscalars[2] = cutlass.Int32(-1)
                cute.arch.barrier()
                rs = rs + cutlass.Int32(1)
            if s_iscalars[1] != cutlass.Int32(1):
                # exhausted (tie-block): land on the MEASURED undershoot
                # side — fail-soft semantics, count<=kCC so the collect
                # buffer cannot overflow.
                self.block_count_ge(
                    input_row, slice_start, slice_end, s_thr[2], smem_ptcnt,
                    smem_wcnt, s_iscalars, s_cluster_partial, tidx, warp_id,
                    lane, smem_input=smem_input)
                cute.arch.barrier()
                if tidx == 0:
                    s_thr[0] = s_thr[2]
                    s_iscalars[1] = cutlass.Int32(1)
                cute.arch.barrier()
        GvrTopKClusterKernel.phase3_collect_candidates(
            self, input_row, N, slice_start, slice_end, smem_keys, smem_vals,
            smem_ptcnt, smem_wcnt, s_thr, s_iscalars, s_cluster_partial,
            tidx, warp_id, lane, smem_input=smem_input)

    # ------------------------------------------------------------------
    # P4 (p4_rs variant) — op#7 EXACT rank-scatter, verbatim port from
    # p4_recursive_digit/src/gvr_topk_decode_p4.py:phase4_rank_scatter
    # (fixed 256-bin fine recursion on the straddling bin => vdiff=0).
    # Leader-only in the cluster epilogue; same signature/contract as
    # the vendored phase4_histogram_snap it replaces (s_iscalars scratch
    # slots [0]-[4] are dead after P3 handoff, [5] consumed pre-P4).
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_rank_scatter(
        self,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_wcnt,
        s_thr,
        s_iscalars,
        output_values_row,
        output_indices_row,
        cand_count,
        tidx,
        warp_id,
        lane,
    ):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i4] = self.dtype(smem_keys[i4])
                output_indices_row[i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        elif cand_count > cutlass.Int32(kK):
            # ---- block min/max over candidates ----
            local_cmin = cutlass.Float32(self.FLT_MAX)
            local_cmax = cutlass.Float32(self.NEG_FLT_MAX)
            i5 = tidx
            while i5 < cand_count:
                v = smem_keys[i5]
                local_cmin = _fmin_f32_inline(local_cmin, v)
                local_cmax = cute.arch.fmax(local_cmax, v)
                i5 = i5 + cutlass.Int32(num_threads)
            cmin = self.warp_reduce_min_f32(local_cmin)
            cmax = self.warp_reduce_max_f32(local_cmax)
            if lane == cutlass.Int32(0):
                smem_wcnt[warp_id] = float_as_uint32(cmin)
                smem_hist[warp_id] = float_as_uint32(cmax)
            cute.arch.barrier()
            bmin_r = cutlass.Float32(self.FLT_MAX)
            bmax_r = cutlass.Float32(self.NEG_FLT_MAX)
            for w in cutlass.range_constexpr(self.num_warps):
                vmin = cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, smem_wcnt[w].ir_value()))
                vmax = cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, smem_hist[w].ir_value()))
                bmin_r = _fmin_f32_inline(bmin_r, vmin)
                bmax_r = cute.arch.fmax(bmax_r, vmax)
            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            cute.arch.barrier()
            # ---- zero + build histogram ----
            i6 = tidx
            while i6 < cutlass.Int32(kBins):
                smem_hist[i6] = cutlass.Int32(0)
                i6 = i6 + cutlass.Int32(num_threads)
            cute.arch.barrier()
            range1 = bmax_r - bmin_r
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            i7 = tidx
            while i7 < cand_count:
                vk = smem_keys[i7]
                bin_i = cutlass.Int32((vk - bmin_r) * inv1)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()
            # ---- 3-step high→low bin search → straddling bin b* + rank_above ----
            warp_bin_sum = cutlass.Int32(0)
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (cutlass.Int32(kBins - 1) - warp_id * cutlass.Int32(bins_per_warp)
                          - cutlass.Int32(jb))
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            if lane == cutlass.Int32(0):
                smem_wcnt[warp_id] = warp_bin_sum
            cute.arch.barrier()
            if tidx == cutlass.Int32(0):
                cum = cutlass.Int32(0)
                tw = cutlass.Int32(num_warps - 1)
                found = cutlass.Int32(0)
                for w2 in cutlass.range_constexpr(self.num_warps):
                    cum = cum + smem_wcnt[w2]
                    if cum >= cutlass.Int32(kK) and found == cutlass.Int32(0):
                        tw = cutlass.Int32(w2)
                        found = cutlass.Int32(1)
                cum2 = cutlass.Int32(0)
                for w3 in cutlass.range_constexpr(self.num_warps):
                    if cutlass.Int32(w3) < tw:
                        cum2 = cum2 + smem_wcnt[w3]
                s_iscalars[2] = cum2  # prefix-count before target warp
                s_iscalars[3] = tw
            cute.arch.barrier()
            target_warp = s_iscalars[3]
            if warp_id == target_warp and lane == cutlass.Int32(0):
                base_cum = s_iscalars[2]
                b_star = cutlass.Int32(kBins - 1)
                rank_above = base_cum
                set_d = cutlass.Int32(0)
                for jb2 in cutlass.range_constexpr(bins_per_warp):
                    bidx2 = (cutlass.Int32(kBins - 1) - target_warp * cutlass.Int32(bins_per_warp)
                             - cutlass.Int32(jb2))
                    ra_before = base_cum
                    base_cum = base_cum + smem_hist[bidx2]
                    if base_cum >= cutlass.Int32(kK) and set_d == cutlass.Int32(0):
                        b_star = bidx2
                        rank_above = ra_before  # count in bins strictly above b*
                        set_d = cutlass.Int32(1)
                s_iscalars[2] = rank_above
                s_iscalars[3] = b_star
                s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                s_iscalars[1] = cutlass.Int32(0)  # cnt_straddle
            cute.arch.barrier()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]

            # ---- EXACT: one fine-histogram recursion on the straddling bin b* ----
            if cutlass.const_expr(self.enable_p4_rank_scatter_exact):
                # FIXED small fine-bin count (independent of kNumBins) — cuts the
                # re-zero + 3-step cost (esp. K=2048 where kNumBins=2048); 256
                # sub-bins over bin b* gives kNumBins×256 effective resolution,
                # enough to resolve the straddling bin to ≤1 distinct value.
                fbins = cutlass.const_expr(256)
                fbpw = cutlass.const_expr(256 // self.num_warps)
                # bin b* value range under the inv1 binning: [f_lo, f_lo + 1/inv1)
                f_lo = bmin_r + cutlass.Float32(b_star) / inv1
                finv = (cutlass.Float32(fbins - 1) + cutlass.Float32(0.99)) * inv1
                # re-zero (only fbins slots) + build fine sub-hist of bin-b* cands
                iz = tidx
                while iz < cutlass.Int32(fbins):
                    smem_hist[iz] = cutlass.Int32(0)
                    iz = iz + cutlass.Int32(num_threads)
                cute.arch.barrier()
                ifb = tidx
                while ifb < cand_count:
                    vf = smem_keys[ifb]
                    cb = cutlass.Int32((vf - bmin_r) * inv1)
                    if cb < cutlass.Int32(0):
                        cb = cutlass.Int32(0)
                    if cb > cutlass.Int32(kBins - 1):
                        cb = cutlass.Int32(kBins - 1)
                    if cb == b_star:
                        sb = cutlass.Int32((vf - f_lo) * finv)
                        if sb < cutlass.Int32(0):
                            sb = cutlass.Int32(0)
                        if sb > cutlass.Int32(fbins - 1):
                            sb = cutlass.Int32(fbins - 1)
                        atomicAdd(smem_hist.iterator + sb, cutlass.Int32(1))
                    ifb = ifb + cutlass.Int32(num_threads)
                cute.arch.barrier()
                # fine 3-step search seeded at rank_above (over fbins bins)
                fws = cutlass.Int32(0)
                for jbf in cutlass.range_constexpr(fbpw):
                    bif = (cutlass.Int32(fbins - 1) - warp_id * cutlass.Int32(fbpw)
                           - cutlass.Int32(jbf))
                    fws = fws + smem_hist[bif]
                if lane == cutlass.Int32(0):
                    smem_wcnt[warp_id] = fws
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    cumf = rank_above
                    twf = cutlass.Int32(num_warps - 1)
                    fnd = cutlass.Int32(0)
                    for w2 in cutlass.range_constexpr(self.num_warps):
                        cumf = cumf + smem_wcnt[w2]
                        if cumf >= cutlass.Int32(kK) and fnd == cutlass.Int32(0):
                            twf = cutlass.Int32(w2)
                            fnd = cutlass.Int32(1)
                    pre = rank_above
                    for w3 in cutlass.range_constexpr(self.num_warps):
                        if cutlass.Int32(w3) < twf:
                            pre = pre + smem_wcnt[w3]
                    # Stage prefix/target-warp metadata in spare s_iscalars
                    # slots, NOT smem_hist[0]/[1]: the last fine warp's reverse
                    # scan below walks fine bins down to 0/1, so reusing those
                    # histogram bins as scratch would corrupt sb_star/ra_fine
                    # when twf2 == num_warps-1. Slots [4]/[1] are dead here
                    # (re-zeroed at the cnt_above/cnt_strad reset below).
                    s_iscalars[4] = pre   # prefix into target fine warp
                    s_iscalars[1] = twf   # target fine warp
                cute.arch.barrier()
                pre_f = s_iscalars[4]
                twf2 = s_iscalars[1]
                if warp_id == twf2 and lane == cutlass.Int32(0):
                    base_f = pre_f
                    sb_star = cutlass.Int32(fbins - 1)
                    ra_fine = base_f
                    sd = cutlass.Int32(0)
                    for jb3 in cutlass.range_constexpr(fbpw):
                        sbi = (cutlass.Int32(fbins - 1) - twf2 * cutlass.Int32(fbpw)
                               - cutlass.Int32(jb3))
                        ra_b = base_f
                        base_f = base_f + smem_hist[sbi]
                        if base_f >= cutlass.Int32(kK) and sd == cutlass.Int32(0):
                            sb_star = sbi
                            ra_fine = ra_b
                            sd = cutlass.Int32(1)
                    smem_hist[2] = sb_star
                    smem_hist[3] = ra_fine
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                    s_iscalars[0] = cutlass.Int32(0)  # cnt_mid (b*, sub>sb*)
                    s_iscalars[1] = cutlass.Int32(0)  # cnt_strad (b*, sub==sb*)
                cute.arch.barrier()
                sb_star = smem_hist[2]
                rank_above_fine = smem_hist[3]
                isc = tidx
                while isc < cand_count:
                    v = smem_keys[isc]
                    bin_i = cutlass.Int32((v - bmin_r) * inv1)
                    if bin_i < cutlass.Int32(0):
                        bin_i = cutlass.Int32(0)
                    if bin_i > cutlass.Int32(kBins - 1):
                        bin_i = cutlass.Int32(kBins - 1)
                    if bin_i > b_star:
                        pos = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cutlass.Int32(1))
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    elif bin_i == b_star:
                        sb = cutlass.Int32((v - f_lo) * finv)
                        if sb < cutlass.Int32(0):
                            sb = cutlass.Int32(0)
                        if sb > cutlass.Int32(fbins - 1):
                            sb = cutlass.Int32(fbins - 1)
                        if sb > sb_star:
                            o = atomicAdd(s_iscalars.iterator + cutlass.Int32(0), cutlass.Int32(1))
                            pos = rank_above + o
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif sb == sb_star:
                            o = atomicAdd(s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1))
                            pos = rank_above_fine + o
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                    isc = isc + cutlass.Int32(num_threads)
                cute.arch.barrier()
                cnt_strad = s_iscalars[1]
                filled = rank_above_fine + cnt_strad
                if filled > cutlass.Int32(kK):
                    filled = cutlass.Int32(kK)
                ipad = filled + tidx
                while ipad < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[ipad] = cutlass.Int32(-1)
                    ipad = ipad + cutlass.Int32(num_threads)
            else:
                # ---- APPROX rank-and-scatter (single pass), arbitrary straddling order ----
                isc = tidx
                while isc < cand_count:
                    v = smem_keys[isc]
                    bin_i = cutlass.Int32((v - bmin_r) * inv1)
                    if bin_i < cutlass.Int32(0):
                        bin_i = cutlass.Int32(0)
                    if bin_i > cutlass.Int32(kBins - 1):
                        bin_i = cutlass.Int32(kBins - 1)
                    if bin_i > b_star:
                        pos = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cutlass.Int32(1))
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    elif bin_i == b_star:
                        off = atomicAdd(s_iscalars.iterator + cutlass.Int32(1), cutlass.Int32(1))
                        pos = rank_above + off
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    isc = isc + cutlass.Int32(num_threads)
                cute.arch.barrier()
                cnt_strad = s_iscalars[1]
                filled = rank_above + cnt_strad
                if filled > cutlass.Int32(kK):
                    filled = cutlass.Int32(kK)
                ipad = filled + tidx
                while ipad < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[ipad] = cutlass.Int32(-1)
                    ipad = ipad + cutlass.Int32(num_threads)
        else:
            i10 = tidx
            while i10 < cand_count:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i10] = self.dtype(smem_keys[i10])
                output_indices_row[i10] = smem_vals[i10]
                i10 = i10 + cutlass.Int32(num_threads)
            i11 = cand_count + tidx
            while i11 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i11] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[i11] = cutlass.Int32(-1)
                i11 = i11 + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # P4 (p4_coop variant, iter7 D2) — cluster-COOPERATIVE exact rank-
    # scatter: NO leader gather; each CTA rank-scatters its OWN P3
    # candidates. Coarse/fine hists live in the LEADER's smem, built by all
    # CTAs via red.shared::cluster; the three scatter rank counters are
    # cluster fetch-adds on the leader's s_iscalars. ~6 extra balanced
    # cluster syncs (measured sub-us when CTAs arrive together) buy the
    # removal of the leader serial tail (gather + full-cand P4) that ncu
    # put at 51-61% of wall time on the fin low-BS negative band
    # (PLAN_ITER7.md). Binning scalars (bmin/inv1/f_lo) are reduced from
    # cluster-published per-CTA min/max in identical order on every CTA =>
    # bit-identical => same selection semantics as phase4_rank_scatter.
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_rank_scatter_coop(self, smem_keys, smem_vals, smem_hist,
                                 smem_wcnt, s_iscalars, s_cluster_partial_m,
                                 output_values_row, output_indices_row,
                                 tidx, warp_id, lane, cta_rank):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)
        cluster_size = cutlass.const_expr(self.cluster_size)
        leader = cutlass.Int32(0)

        own_cnt = s_iscalars[5]
        # cand_total + my prefix base (peers' [5] valid: entry synced
        # the cluster right after P3).
        if tidx == cutlass.Int32(0):
            tot = cutlass.Int32(0)
            base = cutlass.Int32(0)
            for peer in cutlass.range_constexpr(cluster_size):
                addr5 = mapa_shared_cluster(
                    s_iscalars.iterator + cutlass.Int32(5),
                    cutlass.Int32(peer))
                c5 = ld_shared_cluster_i32(addr5)
                tot = tot + c5
                if cutlass.Int32(peer) < cta_rank:
                    base = base + c5
            s_iscalars[2] = base
            s_iscalars[0] = tot
        cute.arch.barrier()
        cand_total = s_iscalars[0]
        my_base = s_iscalars[2]

        if cand_total <= cutlass.Int32(kK):
            # direct copy at prefix offsets; leader pads the tail with -1
            i0 = tidx
            while i0 < own_cnt:
                pos0 = my_base + i0
                if pos0 < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[pos0] = self.dtype(smem_keys[i0])
                    output_indices_row[pos0] = smem_vals[i0]
                i0 = i0 + cutlass.Int32(num_threads)
            if cta_rank == leader:
                ip0 = cand_total + tidx
                while ip0 < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[ip0] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[ip0] = cutlass.Int32(-1)
                    ip0 = ip0 + cutlass.Int32(num_threads)
        else:
            # ---- A: local block min/max over OWN cands + zero own hist ----
            local_cmin = cutlass.Float32(self.FLT_MAX)
            local_cmax = cutlass.Float32(self.NEG_FLT_MAX)
            i5 = tidx
            while i5 < own_cnt:
                v5 = smem_keys[i5]
                local_cmin = _fmin_f32_inline(local_cmin, v5)
                local_cmax = cute.arch.fmax(local_cmax, v5)
                i5 = i5 + cutlass.Int32(num_threads)
            wmin = self.warp_reduce_min_f32(local_cmin)
            wmax = self.warp_reduce_max_f32(local_cmax)
            if lane == cutlass.Int32(0):
                smem_wcnt[warp_id] = float_as_uint32(wmin)
                smem_hist[warp_id] = float_as_uint32(wmax)
            cute.arch.barrier()
            if tidx == cutlass.Int32(0):
                cmin_c = cutlass.Float32(self.FLT_MAX)
                cmax_c = cutlass.Float32(self.NEG_FLT_MAX)
                for w in cutlass.range_constexpr(self.num_warps):
                    vmn = cutlass.Float32(llvm.bitcast(
                        cutlass.Float32.mlir_type, smem_wcnt[w].ir_value()))
                    vmx = cutlass.Float32(llvm.bitcast(
                        cutlass.Float32.mlir_type, smem_hist[w].ir_value()))
                    cmin_c = _fmin_f32_inline(cmin_c, vmn)
                    cmax_c = cute.arch.fmax(cmax_c, vmx)
                s_cluster_partial_m[0] = float_as_uint32(cmin_c)
                s_cluster_partial_m[1] = float_as_uint32(cmax_c)
            cute.arch.barrier()
            # zero own hist (leader's is the red target; uniform on peers)
            iz0 = tidx
            while iz0 < cutlass.Int32(kBins):
                smem_hist[iz0] = cutlass.Int32(0)
                iz0 = iz0 + cutlass.Int32(num_threads)
            cute.arch.cluster_arrive()   # release; SYNC1: minmax + zero
            cute.arch.cluster_wait()
            # cluster min/max reduce (identical order on every CTA)
            if tidx == cutlass.Int32(0):
                gmin = cutlass.Float32(self.FLT_MAX)
                gmax = cutlass.Float32(self.NEG_FLT_MAX)
                pmp = s_cluster_partial_m.iterator
                for peer in cutlass.range_constexpr(cluster_size):
                    amn = mapa_shared_cluster(pmp, cutlass.Int32(peer))
                    amx = mapa_shared_cluster(pmp + cutlass.Int32(1),
                                              cutlass.Int32(peer))
                    fmn = ld_shared_cluster_f32(amn)
                    fmx = ld_shared_cluster_f32(amx)
                    gmin = _fmin_f32_inline(gmin, fmn)
                    gmax = cute.arch.fmax(gmax, fmx)
                if gmax <= gmin:
                    gmax = gmin + cutlass.Float32(1e-6)
                smem_wcnt[0] = float_as_uint32(gmin)
                smem_wcnt[1] = float_as_uint32(gmax)
            cute.arch.barrier()
            bmin_r = cutlass.Float32(llvm.bitcast(
                cutlass.Float32.mlir_type, smem_wcnt[0].ir_value()))
            bmax_r = cutlass.Float32(llvm.bitcast(
                cutlass.Float32.mlir_type, smem_wcnt[1].ir_value()))
            range1 = bmax_r - bmin_r
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            # ---- B: coarse hist into LEADER smem via cluster red ----
            hbase = smem_hist.iterator
            i7 = tidx
            while i7 < own_cnt:
                vk = smem_keys[i7]
                bin_i = cutlass.Int32((vk - bmin_r) * inv1)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                ah = mapa_shared_cluster(hbase + bin_i, leader)
                red_shared_cluster_add_i32(ah, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.cluster_arrive()   # release; SYNC2: coarse hist done
            cute.arch.cluster_wait()
            # ---- C: leader coarse 3-step search (verbatim) ----
            if cta_rank == leader:
                warp_bin_sum = cutlass.Int32(0)
                for jb in cutlass.range_constexpr(bins_per_warp):
                    bidx_s = (cutlass.Int32(kBins - 1)
                              - warp_id * cutlass.Int32(bins_per_warp)
                              - cutlass.Int32(jb))
                    warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
                if lane == cutlass.Int32(0):
                    smem_wcnt[warp_id] = warp_bin_sum
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    cum = cutlass.Int32(0)
                    tw = cutlass.Int32(num_warps - 1)
                    found = cutlass.Int32(0)
                    for w2 in cutlass.range_constexpr(self.num_warps):
                        cum = cum + smem_wcnt[w2]
                        if cum >= cutlass.Int32(kK) and found == cutlass.Int32(0):
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
                        ra_before = base_cum
                        base_cum = base_cum + smem_hist[bidx2]
                        if base_cum >= cutlass.Int32(kK) and set_d == cutlass.Int32(0):
                            b_star_l = bidx2
                            rank_above_l = ra_before
                            set_d = cutlass.Int32(1)
                    s_iscalars[2] = rank_above_l
                    s_iscalars[3] = b_star_l
                cute.arch.barrier()
                # re-zero the fine window [0:256) AFTER the search consumed it
                izf = tidx
                while izf < cutlass.Int32(256):
                    smem_hist[izf] = cutlass.Int32(0)
                    izf = izf + cutlass.Int32(num_threads)
            cute.arch.cluster_arrive()   # release; SYNC3: b*/rank + fine zero
            cute.arch.cluster_wait()
            # broadcast b*/rank_above (leader s_iscalars[3]/[2]) to regs
            if tidx == cutlass.Int32(0):
                a2 = mapa_shared_cluster(
                    s_iscalars.iterator + cutlass.Int32(2), leader)
                a3 = mapa_shared_cluster(
                    s_iscalars.iterator + cutlass.Int32(3), leader)
                smem_wcnt[2] = ld_shared_cluster_i32(a2)
                smem_wcnt[3] = ld_shared_cluster_i32(a3)
            cute.arch.barrier()
            rank_above = smem_wcnt[2]
            b_star = smem_wcnt[3]
            # ---- D: fine 256-bin hist of bin-b* cands into leader smem ----
            fbins = cutlass.const_expr(256)
            fbpw = cutlass.const_expr(256 // self.num_warps)
            f_lo = bmin_r + cutlass.Float32(b_star) / inv1
            finv = (cutlass.Float32(fbins - 1) + cutlass.Float32(0.99)) * inv1
            ifb = tidx
            while ifb < own_cnt:
                vf = smem_keys[ifb]
                cb = cutlass.Int32((vf - bmin_r) * inv1)
                if cb < cutlass.Int32(0):
                    cb = cutlass.Int32(0)
                if cb > cutlass.Int32(kBins - 1):
                    cb = cutlass.Int32(kBins - 1)
                if cb == b_star:
                    sb = cutlass.Int32((vf - f_lo) * finv)
                    if sb < cutlass.Int32(0):
                        sb = cutlass.Int32(0)
                    if sb > cutlass.Int32(fbins - 1):
                        sb = cutlass.Int32(fbins - 1)
                    af = mapa_shared_cluster(hbase + sb, leader)
                    red_shared_cluster_add_i32(af, cutlass.Int32(1))
                ifb = ifb + cutlass.Int32(num_threads)
            cute.arch.cluster_arrive()   # release; SYNC4: fine hist done
            cute.arch.cluster_wait()
            # ---- E: leader fine 3-step (verbatim) + zero rank counters ----
            if cta_rank == leader:
                fws = cutlass.Int32(0)
                for jbf in cutlass.range_constexpr(fbpw):
                    bif = (cutlass.Int32(fbins - 1)
                           - warp_id * cutlass.Int32(fbpw)
                           - cutlass.Int32(jbf))
                    fws = fws + smem_hist[bif]
                if lane == cutlass.Int32(0):
                    smem_wcnt[warp_id] = fws
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    cumf = rank_above
                    twf = cutlass.Int32(num_warps - 1)
                    fnd = cutlass.Int32(0)
                    for w2 in cutlass.range_constexpr(self.num_warps):
                        cumf = cumf + smem_wcnt[w2]
                        if cumf >= cutlass.Int32(kK) and fnd == cutlass.Int32(0):
                            twf = cutlass.Int32(w2)
                            fnd = cutlass.Int32(1)
                    pre = rank_above
                    for w3 in cutlass.range_constexpr(self.num_warps):
                        if cutlass.Int32(w3) < twf:
                            pre = pre + smem_wcnt[w3]
                    s_iscalars[4] = pre
                    s_iscalars[1] = twf
                cute.arch.barrier()
                pre_f = s_iscalars[4]
                twf2 = s_iscalars[1]
                if warp_id == twf2 and lane == cutlass.Int32(0):
                    base_f = pre_f
                    sb_star_l = cutlass.Int32(fbins - 1)
                    ra_fine_l = base_f
                    sd = cutlass.Int32(0)
                    for jb3 in cutlass.range_constexpr(fbpw):
                        sbi = (cutlass.Int32(fbins - 1)
                               - twf2 * cutlass.Int32(fbpw)
                               - cutlass.Int32(jb3))
                        ra_b = base_f
                        base_f = base_f + smem_hist[sbi]
                        if base_f >= cutlass.Int32(kK) and sd == cutlass.Int32(0):
                            sb_star_l = sbi
                            ra_fine_l = ra_b
                            sd = cutlass.Int32(1)
                    smem_hist[2] = sb_star_l
                    smem_hist[3] = ra_fine_l
                cute.arch.barrier()
                if tidx == cutlass.Int32(0):
                    s_iscalars[4] = cutlass.Int32(0)  # cnt_above
                    s_iscalars[0] = cutlass.Int32(0)  # cnt_mid
                    s_iscalars[1] = cutlass.Int32(0)  # cnt_strad
                cute.arch.barrier()
            cute.arch.cluster_arrive()   # release; SYNC5: fine scalars+ctrs
            cute.arch.cluster_wait()
            if tidx == cutlass.Int32(0):
                ah2 = mapa_shared_cluster(hbase + cutlass.Int32(2), leader)
                ah3 = mapa_shared_cluster(hbase + cutlass.Int32(3), leader)
                smem_wcnt[4] = ld_shared_cluster_i32(ah2)
                smem_wcnt[5] = ld_shared_cluster_i32(ah3)
            cute.arch.barrier()
            sb_star = smem_wcnt[4]
            rank_above_fine = smem_wcnt[5]
            # ---- F: distributed scatter of OWN cands (cluster counters) ----
            a_above = mapa_shared_cluster(
                s_iscalars.iterator + cutlass.Int32(4), leader)
            a_mid = mapa_shared_cluster(
                s_iscalars.iterator + cutlass.Int32(0), leader)
            a_strad = mapa_shared_cluster(
                s_iscalars.iterator + cutlass.Int32(1), leader)
            isc = tidx
            while isc < own_cnt:
                v = smem_keys[isc]
                bin_i = cutlass.Int32((v - bmin_r) * inv1)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                if bin_i > b_star:
                    pos = atom_shared_cluster_add_i32(a_above,
                                                      cutlass.Int32(1))
                    if pos < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[pos] = self.dtype(v)
                        output_indices_row[pos] = smem_vals[isc]
                elif bin_i == b_star:
                    sb = cutlass.Int32((v - f_lo) * finv)
                    if sb < cutlass.Int32(0):
                        sb = cutlass.Int32(0)
                    if sb > cutlass.Int32(fbins - 1):
                        sb = cutlass.Int32(fbins - 1)
                    if sb > sb_star:
                        o = atom_shared_cluster_add_i32(a_mid,
                                                        cutlass.Int32(1))
                        pos = rank_above + o
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                    elif sb == sb_star:
                        o = atom_shared_cluster_add_i32(a_strad,
                                                        cutlass.Int32(1))
                        pos = rank_above_fine + o
                        if pos < cutlass.Int32(kK):
                            if cutlass.const_expr(self.return_output_values):
                                output_values_row[pos] = self.dtype(v)
                            output_indices_row[pos] = smem_vals[isc]
                isc = isc + cutlass.Int32(num_threads)
            cute.arch.cluster_arrive()   # release; SYNC6: scatter done
            cute.arch.cluster_wait()
            # ---- G: leader pads the unfilled tail with -1 ----
            if cta_rank == leader:
                cnt_strad = s_iscalars[1]
                filled = rank_above_fine + cnt_strad
                if filled > cutlass.Int32(kK):
                    filled = cutlass.Int32(kK)
                ipad = filled + tidx
                while ipad < cutlass.Int32(kK):
                    if cutlass.const_expr(self.return_output_values):
                        output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                    output_indices_row[ipad] = cutlass.Int32(-1)
                    ipad = ipad + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Entry — vendored cluster entry with Phase 2 replaced by P1b + R0 (+R1).
    # ------------------------------------------------------------------
    @cute.kernel
    def gvr_topk_kernel(self, input_data, pre_idx, seq_lens, output_values,
                        output_indices):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        next_n = cutlass.const_expr(self.next_n)
        top_k = cutlass.const_expr(self.top_k)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        kC = cutlass.const_expr(self.kC)
        kNumBins = cutlass.const_expr(self.kNumBins)
        cluster_size = cutlass.const_expr(self.cluster_size)
        M = cutlass.const_expr(self.M_thr)

        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        if cutlass.const_expr(cluster_size > 1):
            cta_in_cluster = cute.arch.block_idx_in_cluster()
            row_idx = bidx // cluster_size
        else:
            cta_in_cluster = cutlass.Int32(0)
            row_idx = bidx
        is_leader = cta_in_cluster == cutlass.Int32(0)
        pre_idx_row_idx = row_idx // next_n
        if cutlass.const_expr(self.compress_ratio == 1):
            pre_idx_offset = cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)
        else:
            pre_idx_offset = cutlass.Int32(0)

        seq_len = seq_lens[pre_idx_row_idx]
        actual_kv_len = (
            seq_len - cutlass.Int32(next_n) + cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)
        )
        if cutlass.const_expr(self.compress_ratio == 1):
            N = actual_kv_len
        else:
            N = actual_kv_len // cutlass.Int32(self.compress_ratio)

        if cutlass.const_expr(cluster_size > 1):
            vec_w_const = cutlass.const_expr(self.vec_bits // self.dtype.width)
            raw_base = N // cutlass.Int32(cluster_size)
            slice_base = (raw_base // cutlass.Int32(vec_w_const)) * cutlass.Int32(vec_w_const)
            slice_start = cta_in_cluster * slice_base
            slice_end_normal = slice_start + slice_base
            slice_is_last = cta_in_cluster == cutlass.Int32(cluster_size - 1)
            slice_end = N if slice_is_last else slice_end_normal
        else:
            slice_start = cutlass.Int32(0)
            slice_end = N

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
        smem_hist = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((kNumBins,), order=(0,)), byte_alignment=128)
        smem_ptcnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=128)
        smem_wmin = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wmax = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wsum = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wcnt_p1 = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        s_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        s_iscalars = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((6,), order=(0,)), byte_alignment=16)
        s_cluster_partial = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((1,), order=(0,)), byte_alignment=16)
        # op26_r0mc additions
        smem_ptcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_warps,), order=(0,)), byte_alignment=64)
        s_mt_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        s_mt_cnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        s_mt_local = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        s_cluster_partial_m = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        s_r0col = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((1,), order=(0,)), byte_alignment=16)
        if cutlass.const_expr(self.p1b_cache):
            smem_gath = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((top_k,), order=(0,)), byte_alignment=128)
        else:
            smem_gath = None

        if N <= cutlass.Int32(top_k):
            jd = tidx
            while jd < N:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[jd] = input_row[jd]
                output_indices_row[jd] = cutlass.Int32(jd)
                jd = jd + cutlass.Int32(num_threads)
            jp = N + cutlass.Int32(tidx)
            while jp < cutlass.Int32(top_k):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[jp] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[jp] = cutlass.Int32(-1)
                jp = jp + cutlass.Int32(num_threads)
        else:
            if cutlass.const_expr(self.p1b_cache):
                self.phase1_preidx_stats_cached(
                    input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                    smem_gath, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1,
                    s_thr, s_iscalars, tidx, warp_id, lane)
            else:
                self.phase1_preidx_stats(
                    input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                    smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr,
                    s_iscalars, tidx, warp_id, lane)

            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if tidx == 0:
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[je] = input_row[je]
                        je = je + cutlass.Int32(1)
            else:
                # ---- P1b + R0 (replaces vendored Phase 2) ----
                if cutlass.const_expr(self.p1b_cache):
                    self.phase1b_hspace_rungs_cached(
                        pre_idx_count, smem_gath, smem_hist, s_thr, s_mt_thr,
                        tidx, warp_id, lane)
                else:
                    self.phase1b_hspace_rungs(input_row, N, pre_idx_row,
                                              pre_idx_count, pre_idx_offset,
                                              smem_hist, s_thr, s_mt_thr,
                                              tidx, warp_id, lane)
                self.block_count_ge_multi_slice(
                    input_row, slice_start, slice_end, s_mt_thr,
                    smem_ptcnt_multi, smem_wcnt_multi, s_mt_local, s_mt_cnt,
                    s_cluster_partial_m, tidx, warp_id, lane)

                if tidx == 0:
                    best_m = cutlass.Int32(-1)
                    for m in cutlass.range_constexpr(M):
                        cm = s_mt_cnt[m]
                        if cm >= cutlass.Int32(top_k) and cm <= cutlass.Int32(kC):
                            best_m = cutlass.Int32(m)
                    if best_m >= cutlass.Int32(0):
                        s_thr[0] = s_mt_thr[best_m]
                        s_iscalars[0] = s_mt_cnt[best_m]
                        s_iscalars[5] = s_mt_local[best_m]
                        s_iscalars[1] = cutlass.Int32(1)
                        s_r0col[0] = best_m
                    else:
                        blo = v_lo
                        bhi = v_hi
                        clo = cutlass.Int32(-1)
                        chi = cutlass.Int32(-1)
                        for m in cutlass.range_constexpr(M):
                            if s_mt_cnt[m] > cutlass.Int32(kC):
                                blo = s_mt_thr[m]
                                clo = s_mt_cnt[m]
                        for m in cutlass.range_constexpr(M):
                            mm = cutlass.const_expr(M - 1 - m)
                            if s_mt_cnt[mm] < cutlass.Int32(top_k):
                                bhi = s_mt_thr[mm]
                                chi = s_mt_cnt[mm]
                        s_thr[0] = blo
                        s_thr[1] = blo
                        s_thr[2] = bhi
                        s_iscalars[1] = cutlass.Int32(2)
                        s_r0col[0] = cutlass.Int32(-1)
                        if clo > cutlass.Int32(0) and chi >= cutlass.Int32(0):
                            chic = chi
                            if chic < cutlass.Int32(1):
                                chic = cutlass.Int32(1)
                            l_lo = cmath.log2(cutlass.Float32(clo), fastmath=True)
                            l_hi = cmath.log2(cutlass.Float32(chic), fastmath=True)
                            den = l_lo - l_hi
                            if den > cutlass.Float32(0.0):
                                f = (l_lo - cutlass.Float32(self.log2_r1aim)) / den
                                if f < cutlass.Float32(0.05):
                                    f = cutlass.Float32(0.05)
                                if f > cutlass.Float32(0.95):
                                    f = cutlass.Float32(0.95)
                                nv = blo + (bhi - blo) * f
                                if nv > blo and nv < bhi:
                                    s_thr[0] = nv
                                    s_iscalars[1] = cutlass.Int32(3)
                cute.arch.barrier()

                if cutlass.const_expr(self.dbg):
                    if tidx == 0 and is_leader:
                        cute.printf(
                            "DBG ladder: vlo={} vhi={} thr0={} cnt0={} thr1={} cnt1={} st={} bc={} sthr=[{} {} {}]\n",
                            v_lo, v_hi, s_mt_thr[0], s_mt_cnt[0],
                            s_mt_thr[1], s_mt_cnt[1], s_iscalars[1],
                            s_r0col[0], s_thr[0], s_thr[1], s_thr[2])

                # R1 inline shot: vendored cluster count (keeps the Shift-D
                # [5] snapshot + cluster aggregation + slice ptcnt contract).
                # All CTAs take the same branch: the decision inputs
                # (s_mt_cnt, rungs) are cluster-identical.
                if s_iscalars[1] == cutlass.Int32(3):
                    self.block_count_ge(
                        input_row, slice_start, slice_end, s_thr[0],
                        smem_ptcnt, smem_wcnt, s_iscalars, s_cluster_partial,
                        tidx, warp_id, lane, smem_input=None)
                    cute.arch.barrier()
                    if tidx == 0:
                        c1 = s_iscalars[0]
                        t1 = s_thr[0]
                        if c1 >= cutlass.Int32(top_k) and c1 <= cutlass.Int32(kC):
                            s_iscalars[1] = cutlass.Int32(1)
                        else:
                            if c1 > cutlass.Int32(kC):
                                s_thr[1] = t1
                            else:
                                s_thr[2] = t1
                            s_thr[0] = s_thr[1]
                            s_iscalars[1] = cutlass.Int32(2)
                    cute.arch.barrier()

                # admitted at R0: winning column seeds the slice-local ptcnt
                bc = s_r0col[0]
                if bc >= cutlass.Int32(0):
                    smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                cute.arch.barrier()

                # ---- vendored epilogue: handoff + P3 + gather + P4 ----
                if cutlass.const_expr(cluster_size > 1):
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()

                self.phase3_collect_candidates(
                    input_row, N, slice_start, slice_end, smem_keys,
                    smem_vals, smem_ptcnt, smem_wcnt, s_thr, s_iscalars,
                    s_cluster_partial, tidx, warp_id, lane, smem_input=None)

                if cutlass.const_expr(cluster_size > 1):
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()

                if cutlass.const_expr(self.dbg):
                    if tidx == 0 and is_leader:
                        cute.printf(
                            "DBG postP3: st={} cand={} local5={} thr0={} sthr12=[{} {}]\n",
                            s_iscalars[1], s_iscalars[0], s_iscalars[5],
                            s_thr[0], s_thr[1], s_thr[2])

                if cutlass.const_expr(self.p4_coop and self.cluster_size > 1):
                    self.phase4_rank_scatter_coop(
                        smem_keys, smem_vals, smem_hist, smem_wcnt,
                        s_iscalars, s_cluster_partial_m, output_values_row,
                        output_indices_row, tidx, warp_id, lane,
                        cta_in_cluster)
                else:
                    if cluster_size == 1 or is_leader:
                        if cutlass.const_expr(cluster_size > 1):
                            local_cnt_self = s_iscalars[5]
                            local_iscalars_ptr = s_iscalars.iterator + cutlass.Int32(5)
                            smem_keys_iter = smem_keys.iterator
                            smem_vals_iter = smem_vals.iterator
                            base_offset = local_cnt_self
                            for peer in cutlass.range_constexpr(1, cluster_size):
                                peer_iscalars_addr = mapa_shared_cluster(
                                    local_iscalars_ptr, cutlass.Int32(peer))
                                peer_cnt = ld_shared_cluster_i32(peer_iscalars_addr)
                                i_gather = tidx
                                while i_gather < peer_cnt:
                                    peer_key_addr = mapa_shared_cluster(
                                        smem_keys_iter + i_gather, cutlass.Int32(peer))
                                    peer_val_addr = mapa_shared_cluster(
                                        smem_vals_iter + i_gather, cutlass.Int32(peer))
                                    k_val = ld_shared_cluster_f32(peer_key_addr)
                                    v_val = ld_shared_cluster_i32(peer_val_addr)
                                    dst = base_offset + i_gather
                                    if dst < cutlass.Int32(self.kC):
                                        smem_keys[dst] = k_val
                                        smem_vals[dst] = v_val
                                    i_gather = i_gather + cutlass.Int32(num_threads)
                                base_offset = base_offset + peer_cnt
                            if tidx == cutlass.Int32(0):
                                s_iscalars[0] = base_offset
                            cute.arch.barrier()

                        cand_count_p4 = s_iscalars[0]
                        if cand_count_p4 > cutlass.Int32(self.kC):
                            cand_count_p4 = cutlass.Int32(self.kC)

                        if cutlass.const_expr(self.p4_rs):
                            self.phase4_rank_scatter(
                                smem_keys, smem_vals, smem_hist, smem_wcnt,
                                s_thr, s_iscalars, output_values_row,
                                output_indices_row, cand_count_p4, tidx,
                                warp_id, lane)
                        else:
                            self.phase4_histogram_snap(
                                smem_keys, smem_vals, smem_hist, smem_wcnt,
                                s_thr, s_iscalars, output_values_row,
                                output_indices_row, cand_count_p4, tidx,
                                warp_id, lane)

        if cutlass.const_expr(self.cluster_size > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Wrapper — host heuristics mirror gvr_multicta_op26/_resolve_config_mc.
# ---------------------------------------------------------------------------
_compiled_r0mc = {}


def dispatch_p1bc_mc_op26(dt):
    """p1b_cache dispatch for the mc port (069 r0mcc A/B, 54 nsys batches,
    1812 paired cells): ALL dtypes positive in the mc dispatch region —
    gm 1.003 (K512) -> 1.017 (K1024) -> 1.02-1.034 (K2048), zero loss
    cells <0.98; the 1cta fp32-K2048 occupancy regression does NOT
    reproduce in the cluster kernel (different SMEM budget, and the mc
    region is latency-bound where occupancy is not the limiter). So the
    gate is unconditional ON, unlike the 1cta dtype split."""
    return True


def dispatch_p4rs_mc_op26(dt, top_k):
    """p4_rs (leader rank-scatter P4) dispatch for the mc port — iter7
    A/B verdict (092, 54 nsys batches, 1812 paired cells): mc dispatch
    region gm 1.038 overall; fp32 K1024/K2048 1.093 (max 1.23), fp16 all K
    1.016-1.052, bf16 K1024/K2048 1.024-1.031. Single loss band =
    (bf16, K512) gm 0.992 with all 18 <0.98 cells -> gated OFF there.
    NOTE: differs from the 1cta dispatch_rs_op26 (fp32-anywhere ∪ BS>=256)
    — in the latency-bound mc domain 16-bit rank-scatter also wins; yet
    another port-must-rejudge instance (cf. p1b_cache)."""
    return not (dt == torch.bfloat16 and top_k == 512)


def dispatch_p4co_mc_op26(dt, top_k):
    """p4_coop (cluster-cooperative P4) dispatch. Default OFF pending the
    iter7 D2 mc-domain A/B (arm op26_r0mcp forces ON)."""
    return False


def gvr_r0_mc_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                   next_n=1, out=None, cluster_size=None, qfracs=None,
                   p1b_cache=None, p4_rs=None, p4_coop=None, r1aim=None,
                   kc_override=None):
    dt = logits.dtype
    if p1b_cache is None:
        p1b_cache = dispatch_p1bc_mc_op26(dt)
    if p4_rs is None:
        p4_rs = dispatch_p4rs_mc_op26(dt, index_topk)
    if p4_coop is None:
        p4_coop = dispatch_p4co_mc_op26(dt, index_topk)
    qf = tuple(qfracs) if qfracs is not None else M2D
    ra = r1aim if r1aim is not None else "center"
    cfg = _resolve_config_mc(logits, NUM_SMS, cluster_size)
    key = (dt, index_topk, next_n, compress_ratio, qf, ra, kc_override,
           cfg["min_blocks_per_mp"], cfg["use_256bit_load"],
           cfg["num_threads_per_block"], cfg["enable_warp_parallel_reduce"],
           cfg["cluster_size"], bool(p1b_cache), bool(p4_rs), bool(p4_coop))
    compiled = _compiled_r0mc.get(key)
    if compiled is None:
        kobj = GvrOp26R0ClusterKernel(
            dtype=_DT[dt], top_k=index_topk, next_n=next_n,
            num_threads=cfg["num_threads_per_block"],
            enable_unroll_4=True, enable_phase3_unroll=True,
            use_constant_hint=False,
            min_blocks_per_mp=cfg["min_blocks_per_mp"],
            use_256bit_load=cfg["use_256bit_load"],
            enable_warp_parallel_reduce=cfg["enable_warp_parallel_reduce"],
            compress_ratio=compress_ratio, return_output_values=False,
            cluster_size=cfg["cluster_size"], enable_smem_cache=False,
            smem_cache_elems=32768,
            qfracs=qf, p1b_cache=bool(p1b_cache), p4_rs=bool(p4_rs),
            p4_coop=bool(p4_coop), r1aim=ra, kC_override=kc_override,
        )
        n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
        in_align = 32 if cfg["use_256bit_load"] else 16
        input_fake = cr.make_fake_compact_tensor(
            _DT[dt], (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align)
        pre_idx_fake = cr.make_fake_compact_tensor(
            cutlass.Int32, (n_batch, index_topk), stride_order=(1, 0), assumed_align=16)
        seq_lens_fake = cr.make_fake_compact_tensor(
            cutlass.Int32, (n_batch,), stride_order=(0,))
        out_idx_fake = cr.make_fake_compact_tensor(
            cutlass.Int32, (n_rows, index_topk), stride_order=(1, 0), assumed_align=16)
        fake_stream = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(kobj, input_fake, pre_idx_fake, seq_lens_fake,
                                None, out_idx_fake, stream=fake_stream,
                                options="--enable-tvm-ffi")
        _compiled_r0mc[key] = compiled
    if out is None:
        out = torch.empty(logits.shape[0], index_topk, dtype=torch.int32,
                          device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


def gvr_r0mcc_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                   out=None, qfracs=None):
    """op26_r0mcc = op26_r0mc with p1b_cache (mc-port P1-fused gather
    ablation arm; 1cta twin is op26_r0f)."""
    return gvr_r0_mc_op26(logits, pre_idx, seq_lens, index_topk,
                          compress_ratio=compress_ratio, out=out,
                          qfracs=qfracs, p1b_cache=True)


def gvr_r0mcr_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                   out=None, qfracs=None):
    """op26_r0mcr = op26_r0mc with p4_rs forced ON (iter7 leader-tail
    rank-scatter P4 ablation arm; 1cta twin is the dispatch_rs_op26 path)."""
    return gvr_r0_mc_op26(logits, pre_idx, seq_lens, index_topk,
                          compress_ratio=compress_ratio, out=out,
                          qfracs=qfracs, p4_rs=True)


def gvr_r0mcp_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                   out=None, qfracs=None):
    """op26_r0mcp = op26_r0mc with p4_coop forced ON (iter7 D2 cluster-
    cooperative rank-scatter P4 ablation arm; falls back to the production
    p4_rs/snap dispatch wherever cluster_size==1)."""
    return gvr_r0_mc_op26(logits, pre_idx, seq_lens, index_topk,
                          compress_ratio=compress_ratio, out=out,
                          qfracs=qfracs, p4_coop=True)


def picked_cluster_size_r0mc(logits, index_topk, compress_ratio=1):
    return _resolve_config_mc(logits, NUM_SMS)["cluster_size"]


def dispatch_r0_arm_op26(bs, n):
    """1cta-vs-mc arm dispatch (074 mcab grid, r0mc/r0 gm pooled over
    scenario x K x dtype): mc wins N>=65536 at BS<=64 (1.07-2.6x, growing
    with N), washes at BS>=128 (rows saturate the SMs, cluster splits hurt),
    and slightly loses at N<=8192 (cluster tax). 1cta additionally keeps the
    op#7 rank-scatter P4."""
    return "mc" if (n >= 65536 and bs <= 64) else "1cta"


def dispatch_r0_smalln_op26(dt, n):
    """Small-N R0-ladder gate — 07-13 smalln A/B (umbriel-b200-039 8-GPU,
    27 nsys batches, 324 paired cells: N in {16384, 32768} x BS 1..1024 x
    3K x 3 scenarios; plain = op26_1cta, ladder = op26_r0, metric =
    us_cold(r0)/us_cold(1cta), >1 = the ladder is a net tax):
      fp32: 16K gm 1.138 / 32K gm 1.096 -> ladder OFF for N < 65536.
            Residual band (K512, N32K) gm 0.979 (R0 +1.02, scenario-split:
            worst gm 1.25 plain-favoring) — right at the regression
            threshold and not addressable at (dt, n) granularity.
      bf16: 16K gm 0.976 / 32K gm 0.932 with systematic R0-win bands
            ((512,32K) +16%, (512,16K) +10%) -> ladder ON from 16384.
      fp16: 16K gm 0.953 / 32K gm 0.965, band (1024,16K) +13% ->
            ladder ON from 16384.
    N=4096/8192: OFF for all dtypes on the fin full-grid history
    (anchor-transfer gm 0.971/0.877, op22rr_op26{,r}_raw.csv) — that
    evidence is from the full-grid anchor transfer, not this A/B.
    Returns True when the auto arm must route plain 1cta (op26_1cta)."""
    n_r0_min = 65536 if dt == torch.float32 else 16384
    return n < n_r0_min


def gvr_r0_auto_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                     out=None, qfracs=None):
    """Production-facing R0 arm: routes to op26_r0mc (big-N low-BS),
    plain 1cta op26_1cta (below the per-dtype small-N R0 gate), or
    op26_r0 (1cta R0 ladder) otherwise."""
    from gvr_op26_r0_op import gvr_r0_op26  # local import avoids cycle
    from gvr_op26_op import gvr_cutedsl_op26  # small-N plain route
    bs, n = logits.shape
    if dispatch_r0_arm_op26(bs, n) == "mc":
        return gvr_r0_mc_op26(logits, pre_idx, seq_lens, index_topk,
                              compress_ratio=compress_ratio, out=out,
                              qfracs=qfracs)
    # qfracs forced = ablation call: keep it on the R0 ladder unrerouted.
    if qfracs is None and dispatch_r0_smalln_op26(logits.dtype, n):
        return gvr_cutedsl_op26(logits, pre_idx, seq_lens, index_topk,
                                compress_ratio=compress_ratio, out=out)
    return gvr_r0_op26(logits, pre_idx, seq_lens, index_topk,
                       compress_ratio=compress_ratio, out=out, qfracs=qfracs)


if __name__ == "__main__":
    torch.manual_seed(0)
    print("== op26_r0mc smoke (cluster R0 ladder; exactness vs torch.topk) ==")

    # "1" forces the p1b_cache path, "0" forces it off; unset exercises the
    # production default (dispatch_p1bc_mc_op26: ON all dtypes).
    _env_p1bc = os.environ.get("OP26_R0MC_SMOKE_P1BC")
    P1BC = None if _env_p1bc is None else _env_p1bc == "1"
    # "1" forces the leader rank-scatter P4 (iter7 p4_rs); unset exercises
    # the production default (dispatch_p4rs_mc_op26).
    _env_p4rs = os.environ.get("OP26_R0MC_SMOKE_P4RS")
    P4RS = None if _env_p4rs is None else _env_p4rs == "1"
    # "1" forces the cluster-cooperative P4 (iter7 D2 p4_coop).
    _env_p4co = os.environ.get("OP26_R0MC_SMOKE_P4CO")
    P4CO = None if _env_p4co is None else _env_p4co == "1"

    def check(logits, pre_idx, K, crv, tag):
        N = logits.shape[1]
        seq_lens = torch.full((logits.shape[0],), N * crv, dtype=torch.int32,
                              device="cuda")
        cs = picked_cluster_size_r0mc(logits, K, crv)
        out = gvr_r0_mc_op26(logits, pre_idx, seq_lens, K, crv,
                             p1b_cache=P1BC, p4_rs=P4RS, p4_coop=P4CO)
        torch.cuda.synchronize()
        for r in (0, logits.shape[0] - 1):
            idx = out[r].clamp(min=0).long()
            v = logits[r].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[r].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[r].tolist()))
            ok = (d == 0.0 and nuniq == K
                  and int(out[r].min()) >= 0 and int(out[r].max()) < N)
            print(f"  {tag} cs={cs} row{r}: uniq={nuniq}/{K} valdiff={d:.2e}"
                  f"{'' if ok else '  << FAIL'}")
            assert ok, f"op26_r0mc NOT exact: {tag}"

    for dt in (torch.float32, torch.bfloat16, torch.float16):
        # cs>1 cells (BS=1, N>=65536 -> cluster 4) + cs=1 cells (N<65536)
        for K, crv, N in ((512, 4, 32768), (512, 4, 131072), (1024, 4, 65536),
                          (1024, 4, 262144), (2048, 1, 131072), (2048, 1, 262144)):
            logits = torch.randn(1, N, dtype=dt, device="cuda")
            row = logits[0].float()
            pre_hit = torch.topk(row, K).indices.int().view(1, K).contiguous()
            check(logits, pre_hit, K, crv, f"{str(dt):14s} K={K:4d} N={N:6d} hr1")
            noisy = row + 0.8 * row.std() * torch.randn_like(row)
            pre_mid = torch.topk(noisy, K).indices.int().view(1, K).contiguous()
            check(logits, pre_mid, K, crv, f"{str(dt):14s} K={K:4d} N={N:6d} hr~")
            # hr0 in two strengths. Both need the ported fb_fix (the vendored
            # anchor FAILS bottom-K: its P1 bracket [pmin, pmax] sits entirely
            # below the true K-th value, valdiff 8.56e-01 — debug_r0mc_hr0.py;
            # fb_fix's expand-upward guard recovers it, 1cta-parity envelope).
            topk_idx = torch.topk(row, 2 * K).indices
            mask = torch.ones(N, dtype=torch.bool)
            mask[topk_idx.cpu()] = False
            rest = torch.arange(N)[mask]
            pre_miss = rest[torch.randperm(rest.numel())[:K]].int().cuda()
            pre_miss = pre_miss.view(1, K).contiguous()
            check(logits, pre_miss, K, crv, f"{str(dt):14s} K={K:4d} N={N:6d} hr0")
            pre_bk = torch.topk(-row, K).indices.int().view(1, K).contiguous()
            check(logits, pre_bk, K, crv, f"{str(dt):14s} K={K:4d} N={N:6d} hr0bk")
    print("op26_r0mc smoke OK")
