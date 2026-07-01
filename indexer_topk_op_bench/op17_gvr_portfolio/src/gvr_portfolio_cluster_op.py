# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op17 cooperative-cluster threshold portfolio (single kernel, fp32).

A cluster of G CTAs cooperates on ONE row. Each CTA r:
  P1 (redundant, cheap) -> band [pmin,pmax];  thr_r = pmin + r*(pmax-pmin)/(G-1)
  block_count_ge over FULL N at thr_r  -> fills ITS smem_ptcnt + count_r  (free:
    G redundant memory-bound scans ~= 1 scan at BS=1, the crux). This IS Phase-2
    (the sweep REPLACES the secant — no extra pass).
DSMEM-share the G counts; all CTAs pick best_m = highest r with count_r >= K
(tightest). The WINNER (r==best_m) already holds smem_ptcnt at thr_r=thr* -> it
runs P3 collect (done=1, no recount) + P4 with a TIGHT candidate set. Others exit.

Net vs single-CTA baseline = baseline - P4_shrink + cluster(launch+1 barrier), with
ZERO extra full-N pass (unlike the iter5 2-kernel). Exact: winner's smem_ptcnt at
thr* + P3/P4 unchanged; count(thr*)>=K guaranteed (best_m has count>=K).
"""
import sys
from pathlib import Path

import torch
import triton  # noqa: F401  (kept for parity; unused here)
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "ops"))
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: E402
    mapa_shared_cluster, ld_shared_cluster_i32,
)
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


class GvrClusterPortfolioKernel(GvrTopKKernel):
    def __init__(self, *a, G_thr=8, kC_override=None, **kw):
        super().__init__(*a, **kw)
        self.G_thr = int(G_thr)
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
        G = cutlass.const_expr(self.G_thr)
        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        rank = bidx % cutlass.Int32(G)
        row_idx = bidx // cutlass.Int32(G)
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
        # cluster scratch: [0]=this CTA's count (DSMEM-shared), [1]=best_m broadcast
        s_cluster = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((2,), order=(0,)), byte_alignment=16)

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
        else:
            self.phase1_preidx_stats(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                     smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if rank == cutlass.Int32(0) and tidx == 0:
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        je = je + cutlass.Int32(1)
            else:
                # thr_r = pmin + rank*(pmax-pmin)/(G-1); count full N at thr_r
                band_lo = s_thr[1]
                band_hi = s_thr[2]
                denom = cutlass.Float32(1.0) / cutlass.Float32(G - 1)
                thr_r = band_lo + (band_hi - band_lo) * (cutlass.Float32(rank) * denom)
                self.block_count_ge(input_row, N, thr_r, smem_ptcnt, smem_wcnt, s_iscalars, tidx, warp_id, lane)
                cute.arch.barrier()
                # DSMEM-share this CTA's count, then pick best_m (tightest count>=K)
                if tidx == cutlass.Int32(0):
                    s_cluster[0] = s_iscalars[0]
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
                if tidx == cutlass.Int32(0):
                    best_m = cutlass.Int32(0)
                    local_ptr = s_cluster.iterator + cutlass.Int32(0)
                    for peer in cutlass.range_constexpr(G):
                        peer_addr = mapa_shared_cluster(local_ptr, cutlass.Int32(peer))
                        peer_cnt = ld_shared_cluster_i32(peer_addr)
                        if peer_cnt >= cutlass.Int32(top_k):
                            best_m = cutlass.Int32(peer)
                    s_cluster[1] = best_m
                cute.arch.barrier()
                best_m = s_cluster[1]
                if rank == best_m:
                    # my smem_ptcnt is already at thr_r == thr* -> P3(done=1) reuses it
                    if tidx == cutlass.Int32(0):
                        s_thr[0] = thr_r
                        s_iscalars[1] = cutlass.Int32(1)  # done -> P3 skips recount
                    cute.arch.barrier()
                    self.phase3_collect_candidates(input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane)
                    cand_count_p4 = s_iscalars[0]
                    if cand_count_p4 > cutlass.Int32(self.kC):
                        cand_count_p4 = cutlass.Int32(self.kC)
                    self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_iscalars,
                                               output_values_row, output_indices_row, cand_count_p4, tidx, warp_id, lane)
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values, output_indices, stream):
        num_rows = input_data.shape[0]
        G = cutlass.const_expr(self.G_thr)
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


def _compile(dtype, bs, n, K, cr_val, G, kC):
    key = (dtype, bs, n, K, cr_val, G, kC)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrClusterPortfolioKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                                     use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                                     min_blocks_per_mp=min_bpm, return_output_values=False, G_thr=G, kC_override=kC)
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


def gvr_portfolio_cluster(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None, G=8, kC=None):
    bs, n = logits.shape
    compiled = _compile(logits.dtype, bs, n, index_topk, compress_ratio, G, kC)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(_HERE.parents[1] / "harness"))
    import synth_data
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"cooperative-cluster portfolio smoke (fp32, G={G})")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (4096, 8192, 16384, 65536, 262144):
            if N <= 2 * K:
                continue
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
            Npad = b["Npad"]
            seq_lens = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
            out = gvr_portfolio_cluster(logits, pre, seq_lens, K, crv, G=G)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nu = len(set(out[0].tolist()))
            tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
            print(f"  K={K:4d} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")
