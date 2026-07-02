# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op19: sandwich two-threshold GVR top-K (single-CTA, Strategy-A).

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
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
       torch.float16: cutlass.Float16}
_INT_MAX = 0x7FFFFFFF


class GvrSandwichKernel(GvrMultiThreshKernel):
    """Sandwich two-threshold single-CTA kernel. New tunable: band_accept
    (stop refining once band <= band_accept; replaces op18 c_accept)."""

    def __init__(self, *a, band_accept=64, **kw):
        super().__init__(*a, **kw)
        self.band_accept = int(band_accept)
        # deferred direct-write stores values nowhere; indices-only op
        assert not self.return_output_values, "sandwich is indices-only"

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

        # ---- prefix sum #1: direct-write positions (thr0 column) ----
        my_up = smem_ptcnt_up[tidx]
        tp0 = my_up
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp0, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp0 = tp0 + other
        my_excl0 = tp0 - my_up
        warp_tot0 = cute.arch.shuffle_sync(tp0, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_tot0
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                c = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + c
            s_iscalars[4] = tot  # M0 total (deferred-flush bound)
        cute.arch.barrier()
        my_pos0 = smem_wcnt[warp_id] + my_excl0
        cute.arch.barrier()  # all reads of smem_wcnt done before reuse

        # ---- prefix sum #2: band positions (thr1 col - thr0 col) ----
        my_band = smem_ptcnt[tidx] - my_up
        tpb = my_band
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tpb, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tpb = tpb + other
        my_exclb = tpb - my_band
        warp_totb = cute.arch.shuffle_sync(tpb, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_totb
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                c = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + c
            s_iscalars[0] = tot  # band count
        cute.arch.barrier()
        my_posb = smem_wcnt[warp_id] + my_exclb

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
                        if vj >= thr0:
                            if wc0 < cutlass.Int32(self.top_k):
                                smem_didx[wc0] = ic_local + cutlass.Int32(j)
                                wc0 = wc0 + cutlass.Int32(1)
                        elif vj >= thr1 and wcb < cutlass.Int32(kCC):
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
                if vj >= thr0:
                    if wc0 < cutlass.Int32(self.top_k):
                        smem_didx[wc0] = ic + cutlass.Int32(j)
                        wc0 = wc0 + cutlass.Int32(1)
                elif vj >= thr1 and wcb < cutlass.Int32(kCC):
                    smem_keys[wcb] = vj
                    smem_vals[wcb] = ic + cutlass.Int32(j)
                    wcb = wcb + cutlass.Int32(1)
            ic = ic + step

        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= thr0:
                if wc0 < cutlass.Int32(self.top_k):
                    smem_didx[wc0] = it
                    wc0 = wc0 + cutlass.Int32(1)
            elif v >= thr1 and wcb < cutlass.Int32(kCC):
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
    # Phase-4 band snap: pick k_rem winners from band candidates in smem.
    # Histogram range seeded [thr1, thr0). out_count starts at M0 so the
    # writeback fills output positions M0..K-1.
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_band_snap(
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
        smem_hist = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((kNumBins,), order=(0,)), byte_alignment=128)
        smem_ptcnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=128)
        smem_wmin = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wmax = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wsum = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wcnt_p1 = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        s_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        s_iscalars = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((5,), order=(0,)), byte_alignment=16)
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
                            if cutlass.const_expr(self.place_mode == 3):
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
                cute.arch.barrier()

                if s_iscalars[1] == cutlass.Int32(1) and s_swi[0] > cutlass.Int32(0):
                    # ---- sandwich path ----
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


def _load_straddle(K, n, M):
    """op19 straddle-fracs from results/straddle_fracs.json (nearest N)."""
    global _STRADDLE_TABLE
    if _STRADDLE_TABLE is None:
        import json
        p = _HERE.parent / "results" / "straddle_fracs.json"
        _STRADDLE_TABLE = json.load(open(p)) if p.exists() else {}
    cands = []
    for key, v in _STRADDLE_TABLE.items():
        k_, n_, m_ = (int(x) for x in key.split("_"))
        if k_ == K and m_ == M:
            cands.append((abs(n_ - n), v["fracs"]))
    if not cands:
        raise KeyError(f"no straddle fracs for K={K} M={M}")
    fr = sorted(cands)[0][1]
    while len(fr) < M:
        fr = fr + [min(0.999, fr[-1] + 0.01)]
    return tuple(fr[:M])


def _config(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def _compile(dtype, bs, n, K, cr_val, M, R, band_acc, place_mode, kC, threads, unroll=4):
    key = (dtype, bs, n, K, cr_val, M, R, band_acc, place_mode, kC, threads, unroll)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    if threads is not None:
        t = threads
    if place_mode == 4:
        fracs = _load_straddle(K, n, M)
        kernel_place = 3  # same codegen: compile-time frac table
    else:
        fracs = _load_fracs(K, n, M) if place_mode == 3 else None
        kernel_place = place_mode
    kobj = GvrSandwichKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                             use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                             min_blocks_per_mp=min_bpm, return_output_values=False,
                             M_thr=M, R_rounds=R, band_accept=band_acc, place_mode=kernel_place,
                             kC_override=kC, fracs=fracs)
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
           M=4, R=2, band_acc=64, place_mode=3, kC=None, threads=None):
    bs, n = logits.shape
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled = _compile(logits.dtype, bs, n, index_topk, compress_ratio, int(M), int(R),
                        int(band_acc), int(place_mode), kC, threads)
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


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
