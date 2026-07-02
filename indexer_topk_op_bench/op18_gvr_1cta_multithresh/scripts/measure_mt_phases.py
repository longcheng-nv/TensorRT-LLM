# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Phase breakdown of op18 gvr_mt vs baseline: clock64 stamps around
# P1 / P2(multi) / P3 / P4. Baseline phases via harness/measure_cute_phases.py
# (GvrTopKKernelTimed); mt phases via GvrMultiThreshKernelTimed below.
# Usage: python3 measure_mt_phases.py <K> <N> [M] [R] [acc] [place]
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr
from cutlass.utils.smem_allocator import SmemAllocator

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
from gvr_mt_op import GvrMultiThreshKernel, _config, _DT  # noqa: E402
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
import measure_cute_phases as mcp  # noqa: E402

_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")


class GvrMultiThreshKernelTimed(GvrMultiThreshKernel):
    @cute.kernel
    def gvr_topk_kernel(self, input_data, pre_idx, seq_lens, output_values,
                        output_indices, phase_ts):
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
        output_values_row = None
        output_indices_row = output_indices[row_idx, None]
        pre_idx_count = pre_idx.shape[1]
        phase_ts_row = phase_ts[row_idx, None]
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
        s_mstf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        s_msti = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)

        if tidx == 0:
            phase_ts_row[0] = cute.arch.clock64()

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
            if tidx == 0:
                phase_ts_row[1] = cute.arch.clock64()
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
                pmean = s_thr[0]
                if tidx == 0:
                    s_mstf[0] = v_lo
                    s_mstf[1] = v_hi
                    s_mstf[2] = v_lo
                    s_msti[0] = cutlass.Int32(0x7FFFFFFF)
                    s_msti[1] = cutlass.Int32(-1)
                    s_msti[2] = cutlass.Int32(1)
                cute.arch.barrier()

                rr = cutlass.Int32(0)
                while rr < cutlass.Int32(R) and s_msti[2] == cutlass.Int32(1):
                    if tidx == 0:
                        lo = s_mstf[0]
                        hi = s_mstf[1]
                        d = hi - lo
                        if rr == cutlass.Int32(0):
                            if cutlass.const_expr(self.place_mode == 0):
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
                        cont = cutlass.Int32(0)
                        if s_msti[0] > cutlass.Int32(cAcc) and s_mstf[1] > s_mstf[0]:
                            cont = cutlass.Int32(1)
                        s_msti[2] = cont
                    cute.arch.barrier()

                    bc = s_msti[1]
                    if bc >= cutlass.Int32(0):
                        smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                    cute.arch.barrier()
                    rr = rr + cutlass.Int32(1)

                if tidx == 0:
                    s_thr[0] = s_mstf[2]
                    if s_msti[0] <= cutlass.Int32(kC):
                        s_iscalars[0] = s_msti[0]
                        s_iscalars[1] = cutlass.Int32(1)
                    else:
                        s_iscalars[1] = cutlass.Int32(2)
                        s_thr[1] = s_mstf[2]
                        s_thr[2] = s_mstf[1]
                cute.arch.barrier()
                if tidx == 0:
                    phase_ts_row[2] = cute.arch.clock64()

                self.phase3_collect_candidates(input_row, N, smem_keys, smem_vals, smem_ptcnt,
                                               smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane)
                if tidx == 0:
                    phase_ts_row[3] = cute.arch.clock64()
                cand_count_p4 = s_iscalars[0]
                if cand_count_p4 > cutlass.Int32(self.kC):
                    cand_count_p4 = cutlass.Int32(self.kC)
                self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_iscalars,
                                           output_values_row, output_indices_row, cand_count_p4, tidx, warp_id, lane)
                if tidx == 0:
                    phase_ts_row[4] = cute.arch.clock64()
                    # stash cand_count for host readback
                    phase_ts_row[5] = cutlass.Int64(cand_count_p4)
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values, output_indices,
                 phase_ts, stream):
        num_rows = input_data.shape[0]
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values, output_indices,
                             phase_ts).launch(
            grid=(num_rows, 1, 1), block=(self.num_threads, 1, 1),
            stream=stream, use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp)


_timed_cache = {}


def _compile_timed(dtype, bs, n, K, cr_val, M, R, c_accept, place_mode, unroll=4):
    key = (dtype, bs, n, K, cr_val, M, R, c_accept, place_mode, unroll)
    if key in _timed_cache:
        return _timed_cache[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrMultiThreshKernelTimed(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                                     use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                                     min_blocks_per_mp=min_bpm, return_output_values=False,
                                     M_thr=M, R_rounds=R, c_accept=c_accept, place_mode=place_mode,
                                     mt_unroll=unroll)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if use256 else 16
    in_f = cr.make_fake_compact_tensor(_DT[dtype], (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    ts_f = cr.make_fake_compact_tensor(cutlass.Int64, (nr, 6), stride_order=(1, 0), assumed_align=16)
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, ts_f, stream=fs, options="--enable-tvm-ffi")
    _timed_cache[key] = c
    return c


def phases_mt(K, N, M, R, acc, place, reps=30, unroll=4):
    dt = torch.float32
    b = synth_data.get_bundle(K, dt, N)
    lo, pr = b["logits"].cuda(), b["preIdx"].cuda()
    crv = b["cr"]
    sl = torch.full((1,), b["Npad"] * crv, dtype=torch.int32, device="cuda")
    out = torch.empty(1, K, dtype=torch.int32, device="cuda")
    ts = torch.zeros(1, 6, dtype=torch.int64, device="cuda")
    comp = _compile_timed(dt, 1, N, K, crv, M, R, int(K * acc), place, unroll)
    call = lambda: comp(lo, pr, sl, None, out, ts)
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    acc_cyc = []
    for _ in range(reps):
        _EVICT.uniform_(0, 1)
        torch.cuda.synchronize()
        call()
        torch.cuda.synchronize()
        h = ts[0].tolist()
        cand = h[5]
        p1, p2, p3, p4 = h[1] - h[0], h[2] - h[1], h[3] - h[2], h[4] - h[3]
        acc_cyc.append((p1, p2, p3, p4, cand))
    acc_cyc.sort(key=lambda x: x[0] + x[1] + x[2] + x[3])
    return acc_cyc[len(acc_cyc) // 2]


def phases_base(K, N, reps=30):
    r = mcp.measure(K, torch.float32, N, {512: 4, 1024: 4, 2048: 1}[K], reps=reps)
    return r


if __name__ == "__main__":
    K, N = int(sys.argv[1]), int(sys.argv[2])
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    acc = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    place = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    unroll = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    p1, p2, p3, p4, cand = phases_mt(K, N, M, R, acc, place, unroll=unroll)
    tot = p1 + p2 + p3 + p4
    print(f"mt   K={K} N={N} M={M} R={R} acc={acc} pl={place} u={unroll}: "
          f"P1={p1} P2={p2} P3={p3} P4={p4} tot={tot} cyc cand={cand}")
    try:
        rb = phases_base(K, N)
        print(f"base K={K} N={N}: {rb}")
    except Exception as e:
        print(f"base measure failed: {e}")
