# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op21 iter2: row-chunked multi-CTA (cluster) rank-quantile sandwich.

C CTAs cooperate on ONE row (aggregate L2 bandwidth — the P0 lever proven
by op8's 17ns/Kelt N-slope vs single-CTA's 87):
- P1 stash + P1b rank-quantile seeding run REPLICATED on every CTA (same
  inputs -> bit-identical thresholds; zero cross-CTA traffic). Cost: C*K
  extra gather loads, negligible vs N at the C>1 dispatch sizes.
- ONE fused M-threshold ladder pass per CTA over its 64-elt-aligned slice
  (slice-aware count+collect; slot indices stored GLOBAL via base offset).
- DSMEM count merge (M ints per CTA, one cluster barrier) -> every CTA
  picks the same sandwich pair.
- P3 distributed: per-CTA direct-write of >=thr0 winners STRAIGHT to the
  output row at a rank-prefix offset (output GMEM is shared; no staging);
  band candidates compact into each CTA's local smem_keys.
- Leader (rank 0) DSMEM-gathers the <=kC band entries from peers, runs the
  unchanged exact phase4_band_snap for the remaining K-m0 slots.
- Fallback (no pair / band > kC / slot overflow / non-slot column): leader
  re-runs the exact classic P3+P4 over the FULL row; peers idle. Rare
  (iter0.5: all_ge 5.5% worst-case model) and exact.

Exactness authority unchanged: counts + band refine, tie handling in P4.
Red lines respected: no gmem scratch in the hot path (band moves via
DSMEM), counts merged once (single cluster barrier chain), no warp
collectives in the streaming loop.
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
sys.path.insert(0, str(_HERE))
from gvr_ms_op import GvrSandwichKernel, gvr_ms, NUM_SMS, _DT, _INT_MAX  # noqa: E402
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: E402
    ld_shared_cluster_f32, ld_shared_cluster_i32, mapa_shared_cluster,
)
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402


class GvrMsClusterKernel(GvrSandwichKernel):
    """C-CTA row-chunked cluster around the mode-5 sandwich. Requires
    place_mode=5, R=1, fuse_collect=True (thresholds known pre-scan)."""

    def __init__(self, *a, C_cta=4, **kw):
        super().__init__(*a, **kw)
        self.C_cta = int(C_cta)
        assert self.place_mode == 5 and self.R_rounds == 1 and self.fuse_collect
        assert self.C_cta >= 2

    # ------------------------------------------------------------------
    # slice-aware fused M-count + slot-collect: identical to the parent's
    # block_count_collect_multi but scans input_row[base : base+Ns) and
    # records GLOBAL indices (local + base). Counts returned are slice-local.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_collect_multi_base(
        self, input_row, base, Ns, s_mt_thr, smem_ptcnt_multi,
        smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv, tidx, warp_id,
        lane,
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

        # base is 64-elt aligned => vector alignment preserved
        row_addr = input_row.iterator.toint() + cutlass.Int64(base) * cutlass.Int64(elem_bytes)
        n_aligned = (Ns // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_unroll_4:
            rng_frag = cute.make_fragment((vec_w,), self.dtype)
            big_iters = cutlass.Int32(0)
            if Ns > i + cutlass.Int32(vec_w - 1):
                big_iters = (Ns - i - cutlass.Int32(vec_w)) // cutlass.Int32(step_elem) + cutlass.Int32(1)
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
                            smem_slotv[slot_base + cpos] = base + i_local + cutlass.Int32(j)
                    for m in cutlass.range_constexpr(M):
                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < Ns:
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
                        smem_slotv[slot_base + cpos] = base + i + cutlass.Int32(j)
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + step

        it = n_aligned + tidx
        while it < Ns:
            v = self._load_fp32_base(input_row, base, it)
            if v >= thr_frag[PC]:
                cpos = cnt_frag[PC]
                if cpos < cutlass.Int32(S):
                    smem_slotk[slot_base + cpos] = v
                    smem_slotv[slot_base + cpos] = base + it
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
        if tidx == 0:
            for m in cutlass.range_constexpr(M):
                v = cutlass.Int32(0)
                for w in cutlass.range_constexpr(num_warps):
                    v = v + smem_wcnt_multi[m * num_warps + w]
                s_mt_cnt[m] = v
        cute.arch.barrier()

    @cute.jit
    def _load_fp32_base(self, input_row, base, i):
        return self._load_fp32(input_row, base + i)

    # ------------------------------------------------------------------
    # distributed P3 from slots: direct-write >=thr0 winners straight to
    # output_indices_row[d_off + ...]; band [thr1, thr0) compacts into the
    # LOCAL smem_keys/vals starting at 0. Assumes per-thread thr1/thr0 slot
    # columns already copied into smem_ptcnt / smem_ptcnt_up.
    # After: s_iscalars[4] = local direct count, s_iscalars[0] = local band.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_from_slots_mc(
        self, smem_slotk, smem_slotv, smem_keys, smem_vals, smem_ptcnt,
        smem_ptcnt_up, smem_ptcnt_multi, smem_wcnt, s_thr, s_swf,
        s_iscalars, output_indices_row, d_off, tidx, warp_id, lane,
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
        wc0 = d_off + (base_pk >> cutlass.Int32(16))
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
                        output_indices_row[wc0] = smem_slotv[slot_base + sw]
                        wc0 = wc0 + cutlass.Int32(1)
                elif wcb < cutlass.Int32(kCC):
                    smem_keys[wcb] = v
                    smem_vals[wcb] = smem_slotv[slot_base + sw]
                    wcb = wcb + cutlass.Int32(1)
            sw = sw + cutlass.Int32(1)
        cute.arch.barrier()

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
        C = cutlass.const_expr(self.C_cta)
        PCc = cutlass.const_expr(self.pred_col)
        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        rank = bidx % cutlass.Int32(C)
        row_idx = bidx // cutlass.Int32(C)
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
        smem_ptcnt_up = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_didx = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((top_k,), order=(0,)), byte_alignment=128)
        s_swf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((1,), order=(0,)), byte_alignment=16)
        s_swi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((2,), order=(0,)), byte_alignment=16)
        slot_elems = cutlass.const_expr(self.slot_cap * self.num_threads)
        smem_slotk = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((slot_elems,), order=(0,)), byte_alignment=128)
        smem_slotv = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((slot_elems,), order=(0,)), byte_alignment=128)
        # cluster exchange: [0..M) local ladder counts; [M] pk(direct<<16|band);
        # [M+1] overflow flag; [M+2] ok flag; [M+3] d_off; [M+4] b_off;
        # [M+5] merged m1g. (M+3.. are CTA-local scratch — NEVER stash these
        # in s_iscalars: the vendored P3/P4 own its done/cnt_lo/cnt_hi slots.)
        s_cluster = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M + 6,), order=(0,)), byte_alignment=16)

        if N <= cutlass.Int32(top_k):
            if rank == cutlass.Int32(0):
                jd = tidx
                while jd < N:
                    output_indices_row[jd] = cutlass.Int32(jd)
                    jd = jd + cutlass.Int32(num_threads)
                jp = N + cutlass.Int32(tidx)
                while jp < cutlass.Int32(top_k):
                    output_indices_row[jp] = cutlass.Int32(-1)
                    jp = jp + cutlass.Int32(num_threads)
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            # ---- replicated P1 (stash) + P1b (rank-quantile seeds) ----
            self.phase1_stats_stash(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                    smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if rank == cutlass.Int32(0) and tidx == 0:
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        je = je + cutlass.Int32(1)
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            else:
                self.phase1b_rank_quantile(smem_keys, pre_idx_count,
                                           smem_hist, s_thr, s_mt_thr,
                                           s_mt_cnt, tidx)
                cute.arch.barrier()

                # ---- fused ladder over my 64-elt-aligned slice ----
                chunk = ((N + cutlass.Int32(C * 64 - 1)) // cutlass.Int32(C * 64)) * cutlass.Int32(64)
                sl_start = rank * chunk
                if sl_start > N:
                    sl_start = N
                sl_end = sl_start + chunk
                if sl_end > N:
                    sl_end = N
                Ns = sl_end - sl_start
                self.block_count_collect_multi_base(
                    input_row, sl_start, Ns, s_mt_thr, smem_ptcnt_multi,
                    smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv,
                    tidx, warp_id, lane)

                # local overflow check on the collect column
                ofv = cutlass.Int32(0)
                if smem_ptcnt_multi[cutlass.Int32(PCc) * cutlass.Int32(num_threads) + tidx] > cutlass.Int32(self.slot_cap):
                    ofv = cutlass.Int32(1)
                ofv = self.warp_reduce_sum_i32(ofv)
                if lane == 0:
                    smem_wcnt[warp_id] = ofv
                cute.arch.barrier()
                if tidx == 0:
                    oft = cutlass.Int32(0)
                    for w7 in cutlass.range_constexpr(num_warps):
                        oft = oft + smem_wcnt[w7]
                    for m in cutlass.range_constexpr(M):
                        s_cluster[m] = s_mt_cnt[m]
                    s_cluster[M + 1] = oft
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

                # ---- merge counts + pick pair (replicated, deterministic) ----
                if tidx == 0:
                    of_g = cutlass.Int32(0)
                    for m in cutlass.range_constexpr(M):
                        tot_m = cutlass.Int32(0)
                        for peer in cutlass.range_constexpr(C):
                            pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(m), cutlass.Int32(peer))
                            tot_m = tot_m + ld_shared_cluster_i32(pa)
                        s_mt_cnt[m] = tot_m
                    for peer in cutlass.range_constexpr(C):
                        pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M + 1), cutlass.Int32(peer))
                        of_g = of_g + ld_shared_cluster_i32(pa)
                    best_m = cutlass.Int32(-1)
                    for m in cutlass.range_constexpr(M):
                        if s_mt_cnt[m] >= cutlass.Int32(top_k):
                            best_m = cutlass.Int32(m)
                    up_m = best_m + cutlass.Int32(1)
                    m1g = cutlass.Int32(_INT_MAX)
                    m0g = cutlass.Int32(0)
                    if best_m >= cutlass.Int32(0):
                        m1g = s_mt_cnt[best_m]
                    if up_m >= cutlass.Int32(1) and up_m < cutlass.Int32(M):
                        m0g = s_mt_cnt[up_m]
                    ok = cutlass.Int32(0)
                    if (best_m >= cutlass.Int32(PCc) and up_m < cutlass.Int32(M)
                            and m0g > cutlass.Int32(0) and m0g < cutlass.Int32(top_k)
                            and (m1g - m0g) <= cutlass.Int32(kC)
                            and of_g == cutlass.Int32(0)):
                        ok = cutlass.Int32(1)
                    s_swi[0] = m0g
                    s_swi[1] = best_m
                    s_cluster[M + 5] = m1g
                    s_cluster[M + 2] = ok
                    if best_m >= cutlass.Int32(0):
                        s_thr[0] = s_mt_thr[best_m]
                    if up_m < cutlass.Int32(M):
                        s_swf[0] = s_mt_thr[up_m]
                    else:
                        s_swf[0] = cutlass.Float32(self.FLT_MAX)
                cute.arch.barrier()
                ok = s_cluster[M + 2]
                best_m = s_swi[1]
                m0g = s_swi[0]
                m1g = s_cluster[M + 5]

                if ok == cutlass.Int32(1):
                    # ---- distributed P3: rank prefix over peers' pk ----
                    bc = best_m
                    up_c = bc + cutlass.Int32(1)
                    smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                    smem_ptcnt_up[tidx] = smem_ptcnt_multi[up_c * cutlass.Int32(num_threads) + tidx]
                    cute.arch.barrier()
                    # local direct/band totals -> s_cluster[M] for peers
                    if tidx == 0:
                        ld = cutlass.Int32(0)
                        lb = cutlass.Int32(0)
                        # cheap: local col totals already in ptcnt copies? need
                        # block totals: reuse s_cluster local ladder counts
                        ld = s_cluster[up_c]
                        lb = s_cluster[bc] - s_cluster[up_c]
                        s_cluster[M] = (ld << cutlass.Int32(16)) | lb
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    d_off = cutlass.Int32(0)
                    b_off = cutlass.Int32(0)
                    if tidx == 0:
                        for peer in cutlass.range_constexpr(C):
                            if cutlass.Int32(peer) < rank:
                                pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M), cutlass.Int32(peer))
                                pk = ld_shared_cluster_i32(pa)
                                d_off = d_off + (pk >> cutlass.Int32(16))
                                b_off = b_off + (pk & cutlass.Int32(0xFFFF))
                        s_cluster[M + 3] = d_off
                        s_cluster[M + 4] = b_off
                    cute.arch.barrier()
                    d_off = s_cluster[M + 3]
                    self.phase3_from_slots_mc(
                        smem_slotk, smem_slotv, smem_keys, smem_vals,
                        smem_ptcnt, smem_ptcnt_up, smem_ptcnt_multi,
                        smem_wcnt, s_thr, s_swf, s_iscalars,
                        output_indices_row, d_off, tidx, warp_id, lane)
                    # publish local band count for the leader gather
                    if tidx == 0:
                        s_cluster[M] = (s_cluster[M + 4] << cutlass.Int32(16)) | s_iscalars[0]
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

                    # ---- leader: gather peers' band entries via DSMEM ----
                    if rank == cutlass.Int32(0):
                        for peer in cutlass.range_constexpr(C):
                            if cutlass.const_expr(peer > 0):
                                pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M), cutlass.Int32(peer))
                                pk = ld_shared_cluster_i32(pa)
                                p_off = pk >> cutlass.Int32(16)
                                p_cnt = pk & cutlass.Int32(0xFFFF)
                                ig = tidx
                                while ig < p_cnt:
                                    ka = mapa_shared_cluster(smem_keys.iterator + ig, cutlass.Int32(peer))
                                    va = mapa_shared_cluster(smem_vals.iterator + ig, cutlass.Int32(peer))
                                    smem_keys[p_off + ig] = ld_shared_cluster_f32(ka)
                                    smem_vals[p_off + ig] = ld_shared_cluster_i32(va)
                                    ig = ig + cutlass.Int32(num_threads)
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    if rank == cutlass.Int32(0):
                        band_g = m1g - m0g
                        if band_g > cutlass.Int32(kC):
                            band_g = cutlass.Int32(kC)
                        k_rem = cutlass.Int32(top_k) - m0g
                        if tidx == 0:
                            s_iscalars[0] = band_g
                        cute.arch.barrier()
                        self.phase4_band_snap(smem_keys, smem_vals, smem_hist,
                                              smem_wcnt, s_thr, s_swf, s_iscalars,
                                              output_values_row, output_indices_row,
                                              band_g, k_rem, m0g, tidx, warp_id, lane)
                else:
                    # ---- fallback: leader-only exact classic path, full row.
                    # phase3_collect_candidates PREFIXES over smem_ptcnt (per-
                    # thread counts at s_thr[0] over the SAME [0,N) striding),
                    # so the leader must recount the full row first — its
                    # ladder counts cover only its slice (found the hard way:
                    # 646/1024 holes on pro L4). Vendored done/cnt seeds in
                    # s_iscalars are restored exactly as phase1 leaves them.
                    if rank == cutlass.Int32(0):
                        thr_c = s_mt_thr[0]
                        if best_m >= cutlass.Int32(0):
                            thr_c = s_mt_thr[best_m]
                        if tidx == cutlass.Int32(0):
                            s_thr[0] = thr_c
                            if best_m >= cutlass.Int32(0) and m1g <= cutlass.Int32(kC):
                                s_iscalars[1] = cutlass.Int32(1)
                            else:
                                s_iscalars[1] = cutlass.Int32(2)
                                if best_m >= cutlass.Int32(0):
                                    s_thr[1] = s_mt_thr[best_m]
                                    if best_m < cutlass.Int32(M - 1):
                                        s_thr[2] = s_mt_thr[best_m + cutlass.Int32(1)]
                                    else:
                                        s_thr[2] = v_hi
                                else:
                                    s_thr[1] = v_lo
                                    s_thr[2] = s_mt_thr[0]
                            s_iscalars[2] = pre_idx_count + (pre_idx_count >> 2)
                            s_iscalars[3] = cutlass.Int32(1)
                            s_iscalars[4] = cutlass.Int32(0)
                            s_swi[0] = cutlass.Int32(0)  # no sandwich
                        smem_ptcnt_up[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()
                        # full-row recount at thr_c -> smem_ptcnt + cand_count
                        self.block_count_ge(input_row, N, thr_c, smem_ptcnt,
                                            smem_wcnt, s_iscalars, tidx,
                                            warp_id, lane)
                        cute.arch.barrier()
                        self.phase3_collect_candidates(input_row, N, smem_keys, smem_vals, smem_ptcnt,
                                                       smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane)
                        cand_count_p4 = s_iscalars[0]
                        if cand_count_p4 > cutlass.Int32(self.kC):
                            cand_count_p4 = cutlass.Int32(self.kC)
                        self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_iscalars,
                                                   output_values_row, output_indices_row, cand_count_p4, tidx, warp_id, lane)
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values, output_indices, stream):
        num_rows = input_data.shape[0]
        C = cutlass.const_expr(self.C_cta)
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values, output_indices).launch(
            grid=(num_rows * C, 1, 1), block=(self.num_threads, 1, 1),
            cluster=(C, 1, 1), stream=stream, use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp)


_compiled = {}


def _compile(dtype, n, K, cr_val, C, threads):
    key = (dtype, n, K, cr_val, C, threads)
    if key in _compiled:
        return _compiled[key]
    use256 = (n >= 16384)
    kobj = GvrMsClusterKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=threads,
                              compress_ratio=cr_val, use_256bit_load=use256,
                              enable_unroll_4=True, enable_phase3_unroll=True,
                              min_blocks_per_mp=1, return_output_values=False,
                              M_thr=4, R_rounds=1, band_accept=64, place_mode=5,
                              fuse_collect=True, C_cta=C)
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


def gvr_msc(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
            C=4, threads=1024):
    """Row-chunked C-CTA cluster mode-5 sandwich. C must be >= 2; use gvr_ms
    for C == 1."""
    bs, n = logits.shape
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled = _compile(logits.dtype, n, index_topk, compress_ratio, int(C), int(threads))
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


# production entry: ONE extra dispatch rule on (BS, max-N buffer) only.
# C=4 measured best or tied-best on 15/17 P0 cells (event screen 2026-07-05);
# C=8 gains <=5% at N262K BS1 but collapses at BS16 (43.8 vs 28.5us) — not
# worth a second tier.
def gvr_ms_auto(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                out=None):
    bs, n = logits.shape
    if n >= 65536 and bs * 4 <= NUM_SMS:
        return gvr_msc(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                       out=out, C=4)
    return gvr_ms(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                  out=out)


if __name__ == "__main__":
    import synth_data
    C = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"op21 iter2 cluster smoke (fp32, C={C})")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (65536, 131072, 262144):
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
            Npad = b["Npad"]
            seq_lens = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
            out = gvr_msc(logits, pre, seq_lens, K, crv, C=C)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nu = len(set(out[0].tolist()))
            tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
            print(f"  K={K:4d} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")
