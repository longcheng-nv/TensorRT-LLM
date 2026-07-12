# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""op26 iter6: R0 h-space ladder admission for the classic 1-CTA GVR kernel.

Unifies three silicon-validated facts (PLAN_ITER6.md):
  (1) multi-threshold count_ge is cheap (M=2 ~free, M=4 ~x1.25-1.40 —
      count_ge_multi_bench);
  (2) indexer logits CCDF is near-exponential -> any interpolation must live
      in log space (op13/op21/op26 iter5);
  (3) the pmean P1 seed has no distribution robustness (host screen: real
      3/72, best 0/72 first-pass admission) -> the robust equally-cheap
      replacement is a 256-bin histogram over the prev-topK values +
      h-space quantile rungs (HLS-op25 w3a lineage; op25 screening
      methodology re-run on the CLASSIC window [K, kC] picked
      uh4 = (0.90, 0.65, 0.40, 0.15): 216/216 static admission on the
      op22rr real/best/worst grid vs w3a's 0.894 — the classic window is
      much wider than the HLS ms slot window, so even coverage beats the
      deep-edge blade).

Kernel = GvrOp26R0Kernel ⊂ GvrOp26Kernel ⊂ op#7 _P4Kernel (all
subclass-only, vendored files untouched). op18's block_count_ge_multi is
COPIED IN verbatim rather than inherited: gvr_topk_decode_p4.GvrTopKKernel
is a standalone full copy of the vendored kernel (NOT its subclass), so the
op18 lineage cannot diamond-compose with the op#7 lineage — attempting the
mixin sends MT's super().__init__ straight to the vendored base (smoke
falsified 2026-07-12).

Flow: P1 (vendored pmin/pmax/pmean) -> P1b: 256-bin smem histogram over the
K prev-topK gathered values (K loads, L2-hot; HLS "P1b hist == exact
quantiles") -> M rung thresholds at compile-time h-space qfracs -> R0: ONE
block_count_ge_multi pass -> tightest rung with count in [K, kC] accepted,
its cached per-thread column seeds P3 with ZERO recount. Miss -> done=2
with the measured rung bracket in s_thr[1]/s_thr[2] -> fb_fix (which
re-measures ends and log-falsi aims; R2-class unmeasured-seed bugs are
structurally impossible).
"""
import math
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "ops"))                         # cute_vendored
sys.path.insert(0, str(_BENCH / "p4_recursive_digit" / "src"))  # op#7 P4

from gvr_op26_op import GvrOp26Kernel, dispatch_rs_op26  # noqa: E402
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
       torch.float16: cutlass.Float16}

# host-screen verdict (screen_r0_qfracs.py, 2026-07-12): descending h-fracs
# (=> ascending threshold values). uh4 = 216/216 static admission across
# real/best/worst x K x dtype x N on the classic [K, kC] window.
UH4 = (0.90, 0.65, 0.40, 0.15)
M3A = (0.90, 0.55, 0.20)     # 97.7% admission, cheaper pass at large N


class GvrOp26R0Kernel(GvrOp26Kernel):
    """R0 h-space ladder + cached-column P3 + fb_fix safety + gated RS P4."""

    def __init__(self, *a, qfracs=UH4, mt_unroll=4, **kw):
        super().__init__(*a, **kw)
        assert all(0.0 < q < 1.0 for q in qfracs), qfracs
        assert list(qfracs) == sorted(qfracs, reverse=True), \
            "qfracs must be descending h (ascending threshold value)"
        self.qfracs = tuple(float(q) for q in qfracs)
        self.M_thr = len(self.qfracs)
        self.mt_unroll = int(mt_unroll)
        # rung rank targets: need[m] = ceil(q_m * K) prev-topK values >= rung
        self.qneeds = tuple(max(1, int(math.ceil(q * self.top_k)))
                            for q in self.qfracs)

    # ------------------------------------------------------------------
    # block_count_ge_multi<M> — VERBATIM copy of op18 gvr_mt_op.py (same
    # vectorized/4-way-unrolled memory path as block_count_ge; M static
    # register counters; caches all M per-thread count columns). Copied
    # instead of inherited: the op18 base is the *vendored* GvrTopKKernel,
    # this lineage is the p4 full-copy — no common subclassable ancestor.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_ge_multi(
        self, input_row, N, s_mt_thr, smem_ptcnt_multi, smem_wcnt_multi,
        s_mt_cnt, tidx, warp_id, lane,
    ):
        M = cutlass.const_expr(self.M_thr)
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

        row_addr = input_row.iterator.toint()
        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_unroll_4:
            rng_frag = cute.make_fragment((vec_w,), self.dtype)
            big_iters = cutlass.Int32(0)
            if N > i + cutlass.Int32(vec_w - 1):
                big_iters = (N - i - cutlass.Int32(vec_w)) // cutlass.Int32(step_elem) + cutlass.Int32(1)
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
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + step

        it = n_aligned + tidx
        while it < N:
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
                    s_mt_cnt[m] = total

    # ------------------------------------------------------------------
    # P1b — 256-bin smem histogram over the prev-topK gathered values
    # (band [v_lo, v_hi] = P1's pmin/pmax), then M h-space quantile rungs
    # into s_mt_thr (ascending value order). Reuses the P4 smem_hist buffer
    # (kNumBins >= 512 >= 256 in every spec; P4 re-zeroes it later).
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_hspace_rungs(self, input_row, N, pre_idx_row, pre_idx_count,
                             pre_idx_offset, smem_hist, s_thr, s_mt_thr,
                             tidx, warp_id, lane):
        M = cutlass.const_expr(self.M_thr)
        NB = cutlass.const_expr(256)
        SEG = cutlass.const_expr(8)          # NB / WARP_SIZE bins per lane
        num_threads = cutlass.const_expr(self.num_threads)

        jz = tidx
        while jz < cutlass.Int32(NB):
            smem_hist[jz] = cutlass.Int32(0)
            jz = jz + cutlass.Int32(num_threads)
        cute.arch.barrier()

        v_lo = s_thr[1]
        v_hi = s_thr[2]
        width = (v_hi - v_lo) / cutlass.Float32(NB)  # caller guards v_hi > v_lo
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

        # Warp-0-parallel rung extraction (v0's tid0 256-bin serial walk was
        # a ~10-15us per-CTA dependency chain — the dominant tax in the first
        # silicon A/B). Lane l owns the SEG consecutive bins descending from
        # bin NB-1-l*SEG; segment sums -> 5-step shfl_up inclusive scan gives
        # each lane the cumulative count of all higher-value bins; each lane
        # then walks its SEG bins once and fires rung m at the unique
        # crossing bin (cum_before < qneeds[m] <= cum_at). Rung m fires when
        # the from-the-top cumulative first reaches qneeds[m]; qfracs are
        # descending h => thresholds come out ascending in m automatically.
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
            excl = tp - part                 # cum of all bins above my segment
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
            # unfired rungs (heavy invalid-preIdx rows: total < need): v_lo
            if lane == 0:
                for m in cutlass.range_constexpr(M):
                    if total < cutlass.Int32(self.qneeds[m]):
                        s_mt_thr[m] = v_lo
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Entry — op18's kernel body with the R-round M-ary search replaced by
    # the single R0 ladder round (placement from P1b, admission into the
    # classic [K, kC] window, miss -> measured bracket for fb_fix).
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
        M = cutlass.const_expr(self.M_thr)
        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        row_idx = bidx
        pre_idx_row_idx = row_idx // next_n
        if cutlass.const_expr(self.compress_ratio == 1):
            pre_idx_offset = cutlass.Int32(row_idx % next_n) + cutlass.Int32(1)
        else:
            pre_idx_offset = cutlass.Int32(0)
        seq_len = seq_lens[pre_idx_row_idx]
        actual_kv_len = (seq_len - cutlass.Int32(next_n)
                         + cutlass.Int32(row_idx % next_n) + cutlass.Int32(1))
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
            self.phase1_preidx_stats(input_row, N, pre_idx_row, pre_idx_count,
                                     pre_idx_offset, smem_wmin, smem_wmax,
                                     smem_wsum, smem_wcnt_p1, s_thr,
                                     s_iscalars, tidx, warp_id, lane)
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
                # ---- P1b: h-space quantile rungs from the prev-topK hist ----
                self.phase1b_hspace_rungs(input_row, N, pre_idx_row,
                                          pre_idx_count, pre_idx_offset,
                                          smem_hist, s_thr, s_mt_thr, tidx,
                                          warp_id, lane)

                # ---- R0: ONE M-ary pass ----
                self.block_count_ge_multi(input_row, N, s_mt_thr,
                                          smem_ptcnt_multi, smem_wcnt_multi,
                                          s_mt_cnt, tidx, warp_id, lane)
                cute.arch.barrier()

                if tidx == 0:
                    # tightest admissible rung = LAST m with count in [K, kC]
                    # (thresholds ascending => counts non-increasing in m)
                    best_m = cutlass.Int32(-1)
                    for m in cutlass.range_constexpr(M):
                        cm = s_mt_cnt[m]
                        if cm >= cutlass.Int32(top_k) and cm <= cutlass.Int32(kC):
                            best_m = cutlass.Int32(m)
                    if best_m >= cutlass.Int32(0):
                        s_thr[0] = s_mt_thr[best_m]
                        s_iscalars[0] = s_mt_cnt[best_m]
                        s_iscalars[1] = cutlass.Int32(1)   # done=1: admitted
                        s_iscalars[4] = best_m             # column to reuse
                    else:
                        # miss -> measured bracket for fb_fix (done=2).
                        # lo end = deepest rung with count > kC (overshoot),
                        # else P1's pmin; hi end = shallowest rung with
                        # count < K (undershoot), else P1's pmax.
                        blo = v_lo
                        bhi = v_hi
                        for m in cutlass.range_constexpr(M):
                            if s_mt_cnt[m] > cutlass.Int32(kC):
                                blo = s_mt_thr[m]          # ends at deepest >kC
                        for m in cutlass.range_constexpr(M):
                            mm = cutlass.const_expr(M - 1 - m)
                            if s_mt_cnt[mm] < cutlass.Int32(top_k):
                                bhi = s_mt_thr[mm]         # ends at shallowest <K
                        s_thr[0] = blo
                        s_thr[1] = blo
                        s_thr[2] = bhi
                        s_iscalars[1] = cutlass.Int32(2)   # done=2 -> fb_fix
                        s_iscalars[4] = cutlass.Int32(-1)
                cute.arch.barrier()

                # admitted: seed P3 with the cached column (zero recount)
                bc = s_iscalars[4]
                if bc >= cutlass.Int32(0):
                    smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                cute.arch.barrier()

                # fb_fix (GvrOp26Kernel) runs only when done!=1, then the
                # op#7 collect; P4 = rank-scatter/snap per flags via MRO.
                self.phase3_collect_candidates(input_row, N, smem_keys,
                                               smem_vals, smem_ptcnt,
                                               smem_wcnt, s_thr, s_iscalars,
                                               tidx, warp_id, lane)
                cand_count_p4 = s_iscalars[0]
                if cand_count_p4 > cutlass.Int32(self.kC):
                    cand_count_p4 = cutlass.Int32(self.kC)
                self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist,
                                           smem_wcnt, s_thr, s_iscalars,
                                           output_values_row,
                                           output_indices_row, cand_count_p4,
                                           tidx, warp_id, lane)
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values,
                 output_indices, stream):
        num_rows = input_data.shape[0]
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values,
                             output_indices).launch(
            grid=(num_rows, 1, 1), block=(self.num_threads, 1, 1),
            stream=stream, use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp)


# ---------------------------------------------------------------------------
# Wrapper — launch heuristics mirror gvr_cutedsl_op / gvr_op26_op exactly.
# ---------------------------------------------------------------------------
_compiled_r0 = {}


def dispatch_r0_op26(dtype, K, n):
    """Ladder per (dtype, K, N). v0: uh4 everywhere (100% static admission);
    m3_a reserved as the large-N cheaper-pass alternative for silicon A/B."""
    return UH4


def _config_1cta(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def gvr_r0_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                out=None, qfracs=None):
    bs, n = logits.shape
    dt = logits.dtype
    qf = tuple(qfracs) if qfracs is not None else dispatch_r0_op26(dt, index_topk, n)
    rs_on = dispatch_rs_op26(dt, bs)
    key = (dt, bs, n, index_topk, compress_ratio, qf, rs_on)
    compiled = _compiled_r0.get(key)
    if compiled is None:
        t, use256, min_bpm = _config_1cta(bs, n)
        kobj = GvrOp26R0Kernel(
            dtype=_DT[dt], top_k=index_topk, next_n=1, num_threads=t,
            compress_ratio=compress_ratio, use_256bit_load=use256,
            enable_unroll_4=True, enable_phase3_unroll=True,
            min_blocks_per_mp=min_bpm, return_output_values=False,
            enable_p4_rank_scatter=rs_on, enable_p4_rank_scatter_exact=rs_on,
            qfracs=qf, fb_fix=True,
        )
        n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
        in_align = 32 if use256 else 16
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
        _compiled_r0[key] = compiled
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    print("== op26_r0 smoke (h-space ladder admission; exactness vs torch.topk) ==")

    def check(logits, pre_idx, K, crv, tag):
        N = logits.shape[1]
        seq_lens = torch.full((logits.shape[0],), N * crv, dtype=torch.int32,
                              device="cuda")
        out = gvr_r0_op26(logits, pre_idx, seq_lens, K, crv)
        torch.cuda.synchronize()
        for r in (0, logits.shape[0] - 1):
            idx = out[r].clamp(min=0).long()
            v = logits[r].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[r].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[r].tolist()))
            ok = (d == 0.0 and nuniq == K
                  and int(out[r].min()) >= 0 and int(out[r].max()) < N)
            print(f"  {tag} row{r}: uniq={nuniq}/{K} valdiff={d:.2e}"
                  f"{'' if ok else '  << FAIL'}")
            assert ok, f"op26_r0 NOT exact: {tag}"

    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for K, crv, N in ((512, 4, 16384), (512, 4, 262144), (1024, 4, 32768),
                          (1024, 4, 131072), (2048, 1, 32768), (2048, 1, 262144)):
            logits = torch.randn(1, N, dtype=dt, device="cuda")
            row = logits[0].float()
            # hr=1 pole (exact prev top-K: R0 all_below -> fb path)
            pre_hit = torch.topk(row, K).indices.int().view(1, K).contiguous()
            check(logits, pre_hit, K, crv, f"{str(dt):14s} K={K:4d} N={N:6d} hr1")
            # realistic hr~0.5: prev top-K of a noise-perturbed row
            noisy = row + 0.8 * row.std() * torch.randn_like(row)
            pre_mid = torch.topk(noisy, K).indices.int().view(1, K).contiguous()
            check(logits, pre_mid, K, crv, f"{str(dt):14s} K={K:4d} N={N:6d} hr~")
            # hr=0 pole (disjoint bottom-K preIdx: all_above -> fb path)
            pre_miss = torch.topk(-row, K).indices.int().view(1, K).contiguous()
            check(logits, pre_miss, K, crv, f"{str(dt):14s} K={K:4d} N={N:6d} hr0")
    # BS>1 replication path (rank-scatter 16-bit gate at BS>=256 exercised
    # separately in gate_op26)
    logits = torch.randn(16, 65536, dtype=torch.float32, device="cuda")
    pre = torch.topk(logits.float() + 0.5 * torch.randn_like(logits.float()),
                     1024, dim=1).indices.int().contiguous()
    seq = torch.full((16,), 65536 * 4, dtype=torch.int32, device="cuda")
    out = gvr_r0_op26(logits, pre, seq, 1024, 4)
    torch.cuda.synchronize()
    for r in range(16):
        idx = out[r].clamp(min=0).long()
        v = logits[r].gather(0, idx).sort(descending=True).values
        ref = torch.topk(logits[r], 1024).values
        assert (v - ref).abs().max().item() == 0.0 and len(set(out[r].tolist())) == 1024
    print("  BS=16 fp32 K1024 N65536: 16/16 exact")
    print("op26_r0 smoke OK")
