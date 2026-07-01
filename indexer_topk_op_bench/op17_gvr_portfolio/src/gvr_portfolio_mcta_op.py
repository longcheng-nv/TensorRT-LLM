# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op17 multi-CTA redundant-threshold portfolio (2-kernel), fp32.

Kernel A (Triton, grid=G): each program computes band=[pmean,pmax] from preIdx
logits, then ONE count_ge over full N at thr[pid] → counts[G] (+ band[2]). Free
at BS=1 (memory-bound, 1 compare/elem; G redundant scans ~= 1 pass — the crux).

Kernel B (cuteDSL, GvrTopKKernel body copied + counts_g/band_g params): P1 kept;
phase2_seeded picks the tightest threshold with count>=K from counts[G] and seeds
s_thr[0] (ZERO count passes — no serial-secant tax, no single-CTA M-way ALU tax);
P3/P4 unchanged → exact top-K with a tight candidate set (small P4).
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "ops"))
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


# --------------------------------------------------------------------------
# Kernel A: Triton portfolio sweep
# --------------------------------------------------------------------------
@triton.jit
def portfolio_sweep(logits_ptr, preidx_ptr, counts_ptr, band_ptr, offset,
                    NC: tl.constexpr, KC: tl.constexpr, G: tl.constexpr,
                    BLOCK: tl.constexpr, KB: tl.constexpr):
    pid = tl.program_id(0)
    # band = [pmin, pmax] of preIdx logits. Since all K preIdx >= pmin,
    # count(pmin) >= K  <=>  v_K >= pmin, so [pmin,pmax] always brackets v_K
    # (redundant across programs; K small, L2-served).
    mn = 3.0e38
    mx = -3.0e38
    for ks in tl.static_range(0, KC, KB):
        o = ks + tl.arange(0, KB)
        m = o < KC
        idx = tl.load(preidx_ptr + o, mask=m, other=0) + offset
        val = tl.load(logits_ptr + idx, mask=m, other=0.0)
        mn = tl.minimum(mn, tl.min(tl.where(m, val, 3.0e38)))
        mx = tl.maximum(mx, tl.max(tl.where(m, val, -3.0e38)))
    band_lo = mn
    band_hi = mx
    thr = band_lo + pid * (band_hi - band_lo) / (G - 1)
    acc = tl.zeros((), tl.int32)
    for start in tl.static_range(0, NC, BLOCK):
        o = start + tl.arange(0, BLOCK)
        m = o < NC
        x = tl.load(logits_ptr + o, mask=m, other=-3.0e38)
        acc += tl.sum((x >= thr).to(tl.int32))
    tl.store(counts_ptr + pid, acc)
    if pid == 0:
        tl.store(band_ptr + 0, band_lo)
        tl.store(band_ptr + 1, band_hi)


# --------------------------------------------------------------------------
# Kernel B: seeded-threshold GVR tail (copied kernel body + phase2_seeded)
# --------------------------------------------------------------------------
class GvrSeededKernel(GvrTopKKernel):
    def __init__(self, *a, G_thr=NUM_SMS, kC_override=None, **kw):
        super().__init__(*a, **kw)
        self.G_thr = int(G_thr)
        if kC_override is not None:
            self.kC = int(kC_override)

    @cute.jit
    def phase2_seeded(self, counts_g, band_g, s_thr, s_iscalars, tidx):
        kK = cutlass.const_expr(self.top_k)
        G = cutlass.const_expr(self.G_thr)
        if tidx == 0:
            band_lo = band_g[0]
            band_hi = band_g[1]
            denom = cutlass.Float32(1.0) / cutlass.Float32(G - 1)
            span = band_hi - band_lo
            best_m = cutlass.Int32(-1)
            m = cutlass.Int32(G - 1)
            while m >= cutlass.Int32(0) and best_m < cutlass.Int32(0):
                if counts_g[m] >= cutlass.Int32(kK):
                    best_m = m
                m = m - cutlass.Int32(1)
            if best_m < cutlass.Int32(0):
                s_thr[0] = band_lo
            else:
                s_thr[0] = band_lo + span * (cutlass.Float32(best_m) * denom)
            # done=2 → P3 recounts at thr* (populates smem_ptcnt + cand_count);
            # tight cand ≤ kC ⇒ P3 retry-shrink does 0 iters.
            s_iscalars[1] = cutlass.Int32(2)
        cute.arch.barrier()

    @cute.kernel
    def gvr_topk_kernel(self, input_data, pre_idx, seq_lens, output_values,
                        output_indices, counts_g, band_g):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        next_n = cutlass.const_expr(self.next_n)
        top_k = cutlass.const_expr(self.top_k)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        kC = cutlass.const_expr(self.kC)
        kNumBins = cutlass.const_expr(self.kNumBins)
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
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[je] = input_row[je]
                        je = je + cutlass.Int32(1)
            else:
                # ---- Phase 2 REPLACED: seed threshold from the portfolio sweep ----
                self.phase2_seeded(counts_g, band_g, s_thr, s_iscalars, tidx)
                self.phase3_collect_candidates(input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane)
                cand_count_p4 = s_iscalars[0]
                if cand_count_p4 > cutlass.Int32(self.kC):
                    cand_count_p4 = cutlass.Int32(self.kC)
                self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_iscalars,
                                           output_values_row, output_indices_row, cand_count_p4, tidx, warp_id, lane)
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values, output_indices, counts_g, band_g, stream):
        num_rows = input_data.shape[0]
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values, output_indices, counts_g, band_g).launch(
            grid=(num_rows, 1, 1), block=(self.num_threads, 1, 1), stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL, min_blocks_per_mp=self.min_blocks_per_mp)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
_compiledB = {}


def _config(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def _compile_B(dtype, bs, n, K, cr_val, G, kC):
    key = (dtype, bs, n, K, cr_val, G, kC)
    if key in _compiledB:
        return _compiledB[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrSeededKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                           use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                           min_blocks_per_mp=min_bpm, return_output_values=False, G_thr=G, kC_override=kC)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if use256 else 16
    in_f = cr.make_fake_compact_tensor(_DT[dtype], (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    ct_f = cr.make_fake_compact_tensor(cutlass.Int32, (G,), stride_order=(0,))
    bd_f = cr.make_fake_compact_tensor(cutlass.Float32, (2,), stride_order=(0,))
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, ct_f, bd_f, stream=fs, options="--enable-tvm-ffi")
    _compiledB[key] = c
    return c


def gvr_portfolio_mcta(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
                       G=NUM_SMS, kC=None, _scratch={}):
    bs, n = logits.shape
    K = index_topk
    offset = 1 if compress_ratio == 1 else 0
    sk = (bs, n, K, G)
    if sk not in _scratch:
        _scratch[sk] = (torch.empty(G, dtype=torch.int32, device="cuda"),
                        torch.empty(2, dtype=torch.float32, device="cuda"))
    counts_g, band_g = _scratch[sk]
    BLOCK = 8192
    KB = 512 if K >= 512 else 128
    portfolio_sweep[(G,)](logits, pre_idx, counts_g, band_g, offset,
                          NC=n, KC=K, G=G, BLOCK=BLOCK, KB=KB, num_warps=32)
    compiled = _compile_B(logits.dtype, bs, n, K, compress_ratio, G, kC)
    if out is None:
        out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out, counts_g, band_g)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(_HERE.parents[1] / "harness"))
    import synth_data
    print("multi-CTA portfolio smoke (fp32, report synth)")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (4096, 8192, 16384, 65536, 262144):
            if N <= 2 * K:
                continue
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
            Npad = b["Npad"]
            seq_lens = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
            out = gvr_portfolio_mcta(logits, pre, seq_lens, K, crv)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nu = len(set(out[0].tolist()))
            tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
            print(f"  K={K:4d} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")
