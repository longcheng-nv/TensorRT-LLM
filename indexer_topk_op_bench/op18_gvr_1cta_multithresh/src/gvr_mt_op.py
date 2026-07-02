# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op18: single-CTA multi-threshold GVR top-K (M-ary P2, cached M count columns).

One CTA per row (baseline grid). P2's one-threshold-per-scan secant is replaced
by an adaptive M-ary search: each round, ONE full-N scan evaluates M sorted
thresholds (block_count_ge_multi<M>, same vectorized/4-way-unrolled memory path
as production block_count_ge; M static register counters, M unrolled predicated
adds — count_ge_multi_bench: M=2 ~free, M=4 ~x1.25 mean) and caches ALL M
per-thread count columns in smem. The tightest threshold with count>=K seeds
Phase-3 with ZERO recount (its column is copied to smem_ptcnt), closing the
"1 extra scan" gap in the count_ge_multi report design.

Win sources (per op17 iter1b / op16 LEARNINGS):
  - tight cand -> cand-linear P4 shrink, WITHOUT op16's serial-secant tax
    (this is op16's "only real lever: cheaper P2");
  - fewer full-N passes than the secant at K1024/2048 large-N.

Threshold band: [pmin, pmax] of preIdx values (op17: count(pmin)>=K always).
Round 0 places M points per `place_mode` (0=uniform, 1=dyadic-low, 2=pmean-
anchored); later rounds refine the sub-bracket (thr_best, thr_above) uniformly.
Accept when best count <= c_accept (tunable, replaces kFTarget targeting).
Exactness: done=1 only when count in [K, kC]; else done=2 -> baseline P3
retry-shrink. P3/P4 unchanged (exact).
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "ops"))
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_INT_MAX = 0x7FFFFFFF


class GvrMultiThreshKernel(GvrTopKKernel):
    """M-ary multi-threshold P2 in one CTA. Tunables: M_thr, R_rounds,
    c_accept (absolute count), place_mode, kC_override, num_threads."""

    def __init__(self, *a, M_thr=4, R_rounds=2, c_accept=1024, place_mode=0,
                 kC_override=None, mt_unroll=4, fracs=None, **kw):
        super().__init__(*a, **kw)
        self.M_thr = int(M_thr)
        self.R_rounds = int(R_rounds)
        self.c_accept = int(c_accept)
        self.place_mode = int(place_mode)
        self.mt_unroll = int(mt_unroll)  # LSU-ILP unroll depth of the M-ary scan
        # place_mode=3: CDF-aware compile-time frac table (round-1 thresholds
        # thr_m = pmin + fracs[m]*(pmax-pmin); fracs[0] must be 0.0 = safety anchor)
        self.fracs = tuple(float(f) for f in fracs) if fracs is not None else None
        if kC_override is not None:
            self.kC = int(kC_override)

    # ------------------------------------------------------------------
    # block_count_ge_multi<M>: ONE full-N scan, M sorted thresholds.
    # Identical memory path to block_count_ge (vec load + 4-way unroll);
    # M static register counters, M unrolled predicated adds per element.
    # Writes per-thread counts column-major smem_ptcnt_multi[m*T + tid]
    # and block totals s_mt_cnt[m]. Ends with the internal barrier of the
    # staged block reduce (ptcnt_multi visible to all threads after).
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
                        # branchless predicated add (FSETP+IADD, no divergence)
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

        # Cache all M per-thread columns (P3 seed for the winning column).
        for m in cutlass.range_constexpr(M):
            smem_ptcnt_multi[m * num_threads + tidx] = cnt_frag[m]

        # M warp reduces -> staged smem -> ONE barrier -> warp0 reduces M cols.
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
        cAcc = cutlass.const_expr(self.c_accept)
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
        # op18 additions
        smem_ptcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_warps,), order=(0,)), byte_alignment=64)
        s_mt_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        s_mt_cnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        # [0]=br_lo, [1]=br_hi, [2]=best_thr
        s_mstf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        # [0]=best_cnt, [1]=best_col_this_round(-1 none), [2]=continue_flag
        s_msti = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)

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
                # ---- op18 P2: adaptive M-ary multi-threshold search ----
                pmean = s_thr[0]
                if tidx == 0:
                    s_mstf[0] = v_lo   # bracket lo (count >= K guaranteed)
                    s_mstf[1] = v_hi   # bracket hi
                    s_mstf[2] = v_lo   # best_thr (safe fallback = pmin)
                    s_msti[0] = cutlass.Int32(_INT_MAX)  # best_cnt
                    s_msti[1] = cutlass.Int32(-1)
                    s_msti[2] = cutlass.Int32(1)  # continue
                cute.arch.barrier()

                rr = cutlass.Int32(0)
                while rr < cutlass.Int32(R) and s_msti[2] == cutlass.Int32(1):
                    if tidx == 0:
                        lo = s_mstf[0]
                        hi = s_mstf[1]
                        d = hi - lo
                        if rr == cutlass.Int32(0):
                            if cutlass.const_expr(self.place_mode == 3):
                                # CDF-aware compile-time fracs (per K,N,M table)
                                for m in cutlass.range_constexpr(M):
                                    s_mt_thr[m] = lo + d * cutlass.Float32(self.fracs[m])
                            elif cutlass.const_expr(self.place_mode == 0):
                                # uniform [pmin, pmax): frac m/M (m=0 -> pmin)
                                for m in cutlass.range_constexpr(M):
                                    s_mt_thr[m] = lo + d * (cutlass.Float32(m) / cutlass.Float32(M))
                            elif cutlass.const_expr(self.place_mode == 1):
                                # dyadic-low: {0, 1/2^(M-1), ..., 1/4, 1/2}
                                s_mt_thr[0] = lo
                                for m in cutlass.range_constexpr(M - 1):
                                    s_mt_thr[m + 1] = lo + d * cutlass.Float32(1.0 / (1 << (M - 1 - m)))
                            else:
                                # pmean-anchored: split [lo,pmean] and [pmean,hi]
                                pm = pmean
                                if pm <= lo or pm >= hi:
                                    pm = (lo + hi) * cutlass.Float32(0.5)
                                half = cutlass.const_expr(M // 2)
                                for m in cutlass.range_constexpr(half):
                                    s_mt_thr[m] = lo + (pm - lo) * (cutlass.Float32(m) / cutlass.Float32(half))
                                for m in cutlass.range_constexpr(M - half):
                                    s_mt_thr[half + m] = pm + (hi - pm) * (cutlass.Float32(m) / cutlass.Float32(M - half))
                        else:
                            # refine rounds: strictly inside (lo, hi)
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
                        if best_m >= cutlass.Int32(0):
                            c_new = s_mt_cnt[best_m]
                            t_new = s_mt_thr[best_m]
                            if c_new <= s_msti[0]:
                                s_msti[0] = c_new
                                s_mstf[2] = t_new
                                s_msti[1] = best_m
                            else:
                                s_msti[1] = cutlass.Int32(-1)
                            # shrink bracket: (t_new, thr[best_m+1] or old hi)
                            s_mstf[0] = t_new
                            if best_m < cutlass.Int32(M - 1):
                                s_mstf[1] = s_mt_thr[best_m + cutlass.Int32(1)]
                        else:
                            # all counts < K -> K-th value in (lo, thr[0])
                            s_msti[1] = cutlass.Int32(-1)
                            s_mstf[1] = s_mt_thr[0]
                        # continue?
                        cont = cutlass.Int32(0)
                        if s_msti[0] > cutlass.Int32(cAcc) and s_mstf[1] > s_mstf[0]:
                            cont = cutlass.Int32(1)
                        s_msti[2] = cont
                    cute.arch.barrier()

                    # keep smem_ptcnt = column of the current best threshold
                    bc = s_msti[1]
                    if bc >= cutlass.Int32(0):
                        smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                    cute.arch.barrier()
                    rr = rr + cutlass.Int32(1)

                # ---- finalize: done=1 (tight, no recount) or done=2 (shrink) ----
                if tidx == 0:
                    s_thr[0] = s_mstf[2]
                    if s_msti[0] <= cutlass.Int32(kC):
                        s_iscalars[0] = s_msti[0]
                        s_iscalars[1] = cutlass.Int32(1)
                    else:
                        # too many even at the tightest count>=K threshold
                        s_iscalars[1] = cutlass.Int32(2)
                        s_thr[1] = s_mstf[2]
                        s_thr[2] = s_mstf[1]
                cute.arch.barrier()

                self.phase3_collect_candidates(input_row, N, smem_keys, smem_vals, smem_ptcnt,
                                               smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane)
                cand_count_p4 = s_iscalars[0]
                if cand_count_p4 > cutlass.Int32(self.kC):
                    cand_count_p4 = cutlass.Int32(self.kC)
                self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_iscalars,
                                           output_values_row, output_indices_row, cand_count_p4, tidx, warp_id, lane)
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values, output_indices, stream):
        num_rows = input_data.shape[0]
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values, output_indices).launch(
            grid=(num_rows, 1, 1), block=(self.num_threads, 1, 1),
            stream=stream, use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp)


_compiled = {}
_FRACS_TABLE = None


def _load_fracs(K, n, M):
    """CDF-aware round-1 fracs from results/fracs_table.json (nearest N)."""
    global _FRACS_TABLE
    if _FRACS_TABLE is None:
        import json
        p = _HERE.parent / "results" / "fracs_table.json"
        _FRACS_TABLE = json.load(open(p)) if p.exists() else {}
    cands = []
    for key, v in _FRACS_TABLE.items():
        k_, n_, m_ = (int(x) for x in key.split("_"))
        if k_ == K and m_ == M:
            cands.append((abs(n_ - n), v["fracs"]))
    if not cands:
        raise KeyError(f"no fracs for K={K} M={M}")
    fr = sorted(cands)[0][1]
    # pad to exactly M entries (dedup in the optimizer may shorten the list)
    while len(fr) < M:
        fr = fr + [min(0.999, fr[-1] + 0.01)]
    return tuple(fr[:M])


def _config(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def _compile(dtype, bs, n, K, cr_val, M, R, c_accept, place_mode, kC, threads, unroll=4):
    key = (dtype, bs, n, K, cr_val, M, R, c_accept, place_mode, kC, threads, unroll)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    if threads is not None:
        t = threads
    fracs = _load_fracs(K, n, M) if place_mode == 3 else None
    kobj = GvrMultiThreshKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                                use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                                min_blocks_per_mp=min_bpm, return_output_values=False,
                                M_thr=M, R_rounds=R, c_accept=c_accept, place_mode=place_mode,
                                kC_override=kC, mt_unroll=unroll, fracs=fracs)
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


def gvr_mt(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
           M=4, R=2, accept_mult=2.0, place_mode=0, kC=None, threads=None, unroll=4):
    bs, n = logits.shape
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    c_accept = int(index_topk * accept_mult)
    compiled = _compile(logits.dtype, bs, n, index_topk, compress_ratio, int(M), int(R),
                        c_accept, int(place_mode), kC, threads, int(unroll))
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    import synth_data
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(f"op18 single-CTA multi-threshold smoke (fp32, M={M}, R={R})")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (4096, 8192, 16384, 65536, 262144):
            if N <= 2 * K:
                continue
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
            Npad = b["Npad"]
            seq_lens = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
            out = gvr_mt(logits, pre, seq_lens, K, crv, M=M, R=R)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nu = len(set(out[0].tolist()))
            tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
            print(f"  K={K:4d} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")
