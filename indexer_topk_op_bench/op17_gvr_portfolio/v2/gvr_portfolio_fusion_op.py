# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op17 D1: partition x threshold FUSION kernel (P slices x T threshold-slots).

Cluster = P*T CTAs per row (<=16, HW cap). CTA (p, t) scans slice p of the row
(bandwidth-parallel, like the PR#15198 multicta cluster) and counts >= thr_t
(speculative portfolio, like op17 v1/v2). One DSMEM round-trip gives every CTA
all P*T slice-counts; column sums reproduce count(thr_t) for all T thresholds
at once -> P2's secant loop (6-10 slice-passes in multicta) collapses to ONE
slice-pass (+ rare multi-round shrink, inherited from v2b for D0 exactness).
The winning t-column's P CTAs already hold slice-local smem_ptcnt caches at
thr* -> cluster-parallel P3 collect on the winning column only; the column
leader DSMEM-gathers the column's candidates and runs P4 (same gather protocol
as the vendored cluster kernel).

Target regime: N>=131K where v1 portfolio (full-N redundant scans) loses to
multicta's partitioned scan; fusion should reclaim it while keeping the
tight-cand P4 and the 1-pass threshold search.
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
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: E402
    GvrTopKClusterKernel, mapa_shared_cluster, ld_shared_cluster_i32,
    ld_shared_cluster_f32,
)
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


class GvrPortfolioFusionKernel(GvrTopKClusterKernel):
    def __init__(self, *a, P=4, T=4, kC_override=None, **kw):
        kw["cluster_size"] = int(P) * int(T)
        super().__init__(*a, **kw)
        self.P = int(P)
        self.T = int(T)
        if kC_override is not None:
            self.kC = int(kC_override)

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
        P = cutlass.const_expr(self.P)
        T = cutlass.const_expr(self.T)
        G = cutlass.const_expr(self.P * self.T)
        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        cta = cute.arch.block_idx_in_cluster()
        row_idx = bidx // cutlass.Int32(G)
        t_col = cta // cutlass.Int32(P)
        p_row = cta % cutlass.Int32(P)

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

        # Slice p of P (vec_w-aligned base; last slice absorbs the tail).
        vec_w_const = cutlass.const_expr(self.vec_bits // self.dtype.width)
        raw_base = N // cutlass.Int32(P)
        slice_base = (raw_base // cutlass.Int32(vec_w_const)) * cutlass.Int32(vec_w_const)
        slice_start = p_row * slice_base
        slice_is_last = p_row == cutlass.Int32(P - 1)
        slice_end = N if slice_is_last else (slice_start + slice_base)

        input_row = input_data[row_idx, None]
        pre_idx_row = pre_idx[pre_idx_row_idx, None]
        if cutlass.const_expr(self.return_output_values):
            output_values_row = output_values[row_idx, None]
        else:
            output_values_row = None
        output_indices_row = output_indices[row_idx, None]
        pre_idx_count = pre_idx.shape[1]

        griddepcontrol_wait()

        # ---- Shared memory (identical layout on every CTA; DSMEM relies on it) ----
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
        # fusion extras: [0]=best_t broadcast; [1..T]=column totals;
        # [1+T..1+T+G)=raw peer counts (warp-parallel DSMEM read landing zone)
        s_sel = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((1 + T + G,), order=(0,)), byte_alignment=16)

        if N <= cutlass.Int32(top_k):
            if cta == cutlass.Int32(0):
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
            # P1: preIdx band (redundant per CTA; scattered GMEM reads, cheap)
            self.phase1_preidx_stats(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                     smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars,
                                     tidx, warp_id, lane)
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if cta == cutlass.Int32(0) and tidx == 0:
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[je] = input_row[je]
                        je = je + cutlass.Int32(1)
            else:
                # ---- multi-round T-way portfolio over P-sliced counts ----
                denomT = cutlass.Float32(1.0) / cutlass.Float32(T - 1)
                if tidx == cutlass.Int32(0):
                    s_iscalars[1] = cutlass.Int32(0)  # done=0
                cute.arch.barrier()
                sweep_lo = s_thr[1]
                sweep_hi = s_thr[2]
                thr_t = sweep_lo
                rounds = cutlass.Int32(0)
                while s_iscalars[1] == cutlass.Int32(0) and rounds < cutlass.Int32(4):
                    thr_t = sweep_lo + (sweep_hi - sweep_lo) * (cutlass.Float32(t_col) * denomT)
                    self.block_count_ge(input_row, slice_start, slice_end, thr_t,
                                        smem_ptcnt, smem_wcnt, s_iscalars, s_cluster_partial,
                                        tidx, warp_id, lane, do_cluster_aggregation=False)
                    if tidx == cutlass.Int32(0):
                        s_cluster_partial[0] = s_iscalars[0]
                    cute.arch.barrier()
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()
                    # warp-parallel peer read: lane g reads peer g's slice count
                    # (ONE DSMEM round-trip instead of G serial dependent reads)
                    if warp_id == cutlass.Int32(0) and lane < cutlass.Int32(G):
                        part_ptr = s_cluster_partial.iterator + cutlass.Int32(0)
                        s_sel[1 + T + lane] = ld_shared_cluster_i32(mapa_shared_cluster(part_ptr, lane))
                    cute.arch.barrier()
                    if tidx == cutlass.Int32(0):
                        # column sums from the LOCAL copies (every CTA computes the same result)
                        bt = cutlass.Int32(0)
                        for c in cutlass.range_constexpr(T):
                            tot_c = cutlass.Int32(0)
                            for pp in cutlass.range_constexpr(P):
                                tot_c = tot_c + s_sel[1 + T + c * P + pp]
                            s_sel[1 + c] = tot_c
                            if tot_c >= cutlass.Int32(top_k):
                                bt = cutlass.Int32(c)
                        s_sel[0] = bt
                        bt_tot = s_sel[1 + bt]
                        if bt_tot <= cutlass.Int32(kC) or bt == cutlass.Int32(T - 1):
                            s_iscalars[1] = cutlass.Int32(1)  # window hit (or cap-degenerate)
                        else:
                            nlo = sweep_lo + (sweep_hi - sweep_lo) * (cutlass.Float32(bt) * denomT)
                            nhi = sweep_lo + (sweep_hi - sweep_lo) * (cutlass.Float32(bt + 1) * denomT)
                            s_thr[1] = nlo
                            s_thr[2] = nhi
                            if nhi <= nlo:
                                s_iscalars[1] = cutlass.Int32(1)  # fp-degenerate: cap
                    cute.arch.barrier()
                    sweep_lo = s_thr[1]
                    sweep_hi = s_thr[2]
                    rounds = rounds + cutlass.Int32(1)
                    if s_iscalars[1] == cutlass.Int32(0):
                        cute.arch.cluster_arrive_relaxed()
                        cute.arch.cluster_wait()
                best_t = s_sel[0]
                thr_star = sweep_lo + (sweep_hi - sweep_lo) * (cutlass.Float32(best_t) * denomT)

                # ---- winning column: cluster-parallel P3 on slice-local caches ----
                if t_col == best_t:
                    if tidx == cutlass.Int32(0):
                        s_thr[0] = thr_star
                        s_iscalars[1] = cutlass.Int32(1)  # done (also on rounds-exhausted cap)
                    cute.arch.barrier()
                    self.phase3_collect_candidates(input_row, N, slice_start, slice_end,
                                                   smem_keys, smem_vals, smem_ptcnt, smem_wcnt,
                                                   s_thr, s_iscalars, s_cluster_partial,
                                                   tidx, warp_id, lane)
                    # publish this CTA's local candidate count for the gather
                    if tidx == cutlass.Int32(0):
                        s_iscalars[5] = s_iscalars[0]
                    cute.arch.barrier()

                # handoff: winning column's P3 must finish before the gather reads
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

                if cta == best_t * cutlass.Int32(P):
                    # column leader: DSMEM-gather the other P-1 slice CTAs' candidates
                    base_offset = s_iscalars[5]
                    local_iscalars_ptr = s_iscalars.iterator + cutlass.Int32(5)
                    smem_keys_iter = smem_keys.iterator
                    smem_vals_iter = smem_vals.iterator
                    for ppi in cutlass.range_constexpr(1, P):
                        peer = best_t * cutlass.Int32(P) + cutlass.Int32(ppi)
                        peer_cnt = ld_shared_cluster_i32(mapa_shared_cluster(local_iscalars_ptr, peer))
                        i_gather = tidx
                        while i_gather < peer_cnt:
                            k_val = ld_shared_cluster_f32(mapa_shared_cluster(smem_keys_iter + i_gather, peer))
                            v_val = ld_shared_cluster_i32(mapa_shared_cluster(smem_vals_iter + i_gather, peer))
                            dst = base_offset + i_gather
                            if dst < cutlass.Int32(kC):
                                smem_keys[dst] = k_val
                                smem_vals[dst] = v_val
                            i_gather = i_gather + cutlass.Int32(num_threads)
                        base_offset = base_offset + peer_cnt
                    if tidx == cutlass.Int32(0):
                        s_iscalars[0] = base_offset
                    cute.arch.barrier()
                    cand_count_p4 = s_iscalars[0]
                    if cand_count_p4 > cutlass.Int32(kC):
                        cand_count_p4 = cutlass.Int32(kC)
                    self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt,
                                               s_thr, s_iscalars, output_values_row, output_indices_row,
                                               cand_count_p4, tidx, warp_id, lane)

        # final cluster barrier: keep all CTAs (and their SMEM) alive through the gather/P4
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values, output_indices, stream):
        num_rows = input_data.shape[0]
        G = cutlass.const_expr(self.P * self.T)
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values, output_indices).launch(
            grid=(num_rows * G, 1, 1), block=(self.num_threads, 1, 1),
            cluster=(G, 1, 1), stream=stream, use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp)


_compiled = {}


def _config(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def _compile(dtype, bs, n, K, cr_val, P, T, kC):
    key = (dtype, bs, n, K, cr_val, P, T, kC)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrPortfolioFusionKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                                    use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                                    min_blocks_per_mp=min_bpm, return_output_values=False,
                                    P=P, T=T, kC_override=kC)
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


def gvr_portfolio_fusion(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None, P=4, T=4, kC=None):
    bs, n = logits.shape
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled = _compile(logits.dtype, bs, n, index_topk, compress_ratio, P, T, kC)
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(_HERE.parents[1] / "harness"))
    import synth_data
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"D1 fusion smoke (fp32, P={P} T={T})")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (16384, 65536, 131072, 262144):
            if N <= 2 * K:
                continue
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
            seq_lens = torch.full((1,), b["Npad"] * crv, dtype=torch.int32, device="cuda")
            out = gvr_portfolio_fusion(logits, pre, seq_lens, K, crv, P=P, T=T)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nu = len(set(out[0].tolist()))
            tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
            print(f"  K={K:4d} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")
