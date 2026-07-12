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
)
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
       torch.float16: cutlass.Float16}

M2D = (0.85, 0.35)   # iter6 v0.2 ship ladder (see gvr_op26_r0_op.py)


class GvrOp26R0ClusterKernel(GvrTopKClusterKernel):
    """Cluster GVR + R0 h-space ladder admission + R1 inline falsi shot."""

    def __init__(self, *a, qfracs=M2D, mt_unroll=4, **kw):
        super().__init__(*a, **kw)
        assert not self.enable_smem_cache, \
            "op26_r0mc v0 supports the production enable_smem_cache=False only"
        assert all(0.0 < q < 1.0 for q in qfracs), qfracs
        assert list(qfracs) == sorted(qfracs, reverse=True), \
            "qfracs must be descending h (ascending threshold value)"
        self.qfracs = tuple(float(q) for q in qfracs)
        self.M_thr = len(self.qfracs)
        self.mt_unroll = int(mt_unroll)
        self.qneeds = tuple(max(1, int(math.ceil(q * self.top_k)))
                            for q in self.qfracs)
        self.log2_r1aim = math.log2(math.sqrt(self.top_k * self.kC))

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
                self.phase1b_hspace_rungs(input_row, N, pre_idx_row,
                                          pre_idx_count, pre_idx_offset,
                                          smem_hist, s_thr, s_mt_thr, tidx,
                                          warp_id, lane)
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

                    self.phase4_histogram_snap(
                        smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr,
                        s_iscalars, output_values_row, output_indices_row,
                        cand_count_p4, tidx, warp_id, lane)

        if cutlass.const_expr(self.cluster_size > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Wrapper — host heuristics mirror gvr_multicta_op26/_resolve_config_mc.
# ---------------------------------------------------------------------------
_compiled_r0mc = {}


def gvr_r0_mc_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                   next_n=1, out=None, cluster_size=None, qfracs=None):
    dt = logits.dtype
    qf = tuple(qfracs) if qfracs is not None else M2D
    cfg = _resolve_config_mc(logits, NUM_SMS, cluster_size)
    key = (dt, index_topk, next_n, compress_ratio, qf,
           cfg["min_blocks_per_mp"], cfg["use_256bit_load"],
           cfg["num_threads_per_block"], cfg["enable_warp_parallel_reduce"],
           cfg["cluster_size"])
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
            qfracs=qf,
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


def picked_cluster_size_r0mc(logits, index_topk, compress_ratio=1):
    return _resolve_config_mc(logits, NUM_SMS)["cluster_size"]


if __name__ == "__main__":
    torch.manual_seed(0)
    print("== op26_r0mc smoke (cluster R0 ladder; exactness vs torch.topk) ==")

    def check(logits, pre_idx, K, crv, tag):
        N = logits.shape[1]
        seq_lens = torch.full((logits.shape[0],), N * crv, dtype=torch.int32,
                              device="cuda")
        cs = picked_cluster_size_r0mc(logits, K, crv)
        out = gvr_r0_mc_op26(logits, pre_idx, seq_lens, K, crv)
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
            pre_miss = torch.topk(-row, K).indices.int().view(1, K).contiguous()
            check(logits, pre_miss, K, crv, f"{str(dt):14s} K={K:4d} N={N:6d} hr0")
    print("op26_r0mc smoke OK")
