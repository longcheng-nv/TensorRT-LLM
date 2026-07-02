# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op17 GVR cuteDSL threshold-portfolio op (single-CTA M-way multi-threshold P2).

Realizes the "threshold portfolio" as an M-way multi-threshold count folded into
ONE Phase-2 pass (memory-bound; the M extra per-element compares hide under the
HBM stalls — this is why the falsified 2-way/k=4 multi-threshold were a wash, not
a loss). We use the M counts to pick the TIGHTEST threshold with count in [K, kC]
(count ≈ K+ε), which shrinks the P4 candidate working set (P4 is cand-linear;
iter1b: P4 −58% at N=16K) WITHOUT the serial-secant P2 tax that cancelled op13.

Structure preserved: preIdx (P1) → threshold search (P2, now 1 M-way pass) → P3
collect → P4 snap. Exact: P3/P4 are unchanged and produce exact top-K from the
elements ≥ the chosen threshold, as long as the final count ≥ K (guaranteed: we
pick a threshold with count≥K, else fall back to the stock secant).

Subclass override only — the vendored kernel is NOT edited. The main kernel calls
self.phase2_secant_search, so overriding it (+ adding block_count_ge_multi) is the
production-equivalent path (same as GvrP2C in op13).
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "ops"))
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from cute_vendored.blackwell.top_k.gvr_topk_decode import _fmin_f32_inline  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_compiled = {}


class GvrPortfolioKernel(GvrTopKKernel):
    """GVR kernel whose P2 is a single M-way multi-threshold pass + tight pick."""

    def __init__(self, *a, M_thr=16, kC_override=None, **kw):
        super().__init__(*a, **kw)
        self.M_thr = int(M_thr)
        if kC_override is not None:
            self.kC = int(kC_override)

    @cute.jit
    def block_count_ge_multi(
        self, input_row, N, thr_lo, thr_hi,
        smem_ptcnt,  # [num_threads] int32 scratch (reused: warp partials)
        smem_wcnt,   # [num_warps]  int32 scratch (reused: final M counts)
        tidx, warp_id, lane,
    ):
        """Count input[i] >= thr[m] for m in 0..M-1, thr[m] uniform in [thr_lo,thr_hi].

        Writes the M block-total counts into smem_wcnt[0:M]. One barrier."""
        M = cutlass.const_expr(self.M_thr)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        step_elem = cutlass.const_expr(num_threads * vec_w)
        copy_atom = self._make_load_copy_atom()

        # Per-thread M counters (constexpr-indexed → stay in registers).
        c = [cutlass.Int32(0) for _ in range(M)]
        # Precompute the M thresholds (runtime floats).
        denom = cutlass.Float32(1.0) / cutlass.Float32(M - 1)
        span = thr_hi - thr_lo
        thr = [thr_lo + span * (cutlass.Float32(m) * denom) for m in range(M)]

        row_addr = input_row.iterator.toint()
        i = tidx * cutlass.Int32(vec_w)
        big_iters = cutlass.Int32(0)
        if N > i + cutlass.Int32(vec_w - 1):
            big_iters = (N - i - cutlass.Int32(vec_w)) // cutlass.Int32(step_elem) + cutlass.Int32(1)
        frag = cute.make_fragment((vec_w,), self.dtype)
        for k in cutlass.range(big_iters, unroll=4):
            i_local = i + k * cutlass.Int32(step_elem)
            src_ptr = cute.make_ptr(self.dtype,
                                    row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                                    cute.AddressSpace.gmem, assumed_align=vec_align)
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = frag[j]
                else:
                    vj = cutlass.Float32(frag[j])
                for m in cutlass.range_constexpr(M):
                    if vj >= thr[m]:
                        c[m] = c[m] + cutlass.Int32(1)
        i = i + big_iters * cutlass.Int32(step_elem)
        # scalar tail
        it = ((N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)) + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            for m in cutlass.range_constexpr(M):
                if v >= thr[m]:
                    c[m] = c[m] + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)

        # Warp-reduce each of the M counters, lane0 writes to smem_ptcnt[warp*M+m].
        for m in cutlass.range_constexpr(M):
            wc = self.warp_reduce_sum_i32(c[m])
            if lane == 0:
                smem_ptcnt[warp_id * cutlass.Int32(M) + cutlass.Int32(m)] = wc
        cute.arch.barrier()
        # tid0 sums warp partials → smem_wcnt[m].
        if tidx == 0:
            for m in cutlass.range_constexpr(M):
                tot = cutlass.Int32(0)
                for w in cutlass.range_constexpr(num_warps):
                    tot = tot + smem_ptcnt[w * cutlass.Int32(M) + cutlass.Int32(m)]
                smem_wcnt[m] = tot
        cute.arch.barrier()

    @cute.jit
    def phase2_secant_search(
        self, input_row, N, smem_ptcnt, smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane,
    ):
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        M = cutlass.const_expr(self.M_thr)

        # Band [pmean, pmax] brackets v_K (count(pmean) ~ 3-4K > K, count(pmax) < K).
        band_lo = s_thr[0]  # pmean
        band_hi = s_thr[2]  # pmax
        self.block_count_ge_multi(input_row, N, band_lo, band_hi,
                                  smem_ptcnt, smem_wcnt, tidx, warp_id, lane)

        # tid0 picks the TIGHTEST threshold (highest m) with count in [kK, kCC].
        if tidx == 0:
            denom = cutlass.Float32(1.0) / cutlass.Float32(M - 1)
            span = band_hi - band_lo
            best_m = cutlass.Int32(-1)
            best_cnt = cutlass.Int32(0)
            # highest m with count >= kK (walk high→low, take first valid)
            m = cutlass.Int32(M - 1)
            while m >= cutlass.Int32(0) and best_m < cutlass.Int32(0):
                cm = smem_wcnt[m]
                if cm >= cutlass.Int32(kK):
                    best_m = m
                    best_cnt = cm
                m = m - cutlass.Int32(1)
            if best_m >= cutlass.Int32(0) and best_cnt <= cutlass.Int32(kCC):
                # tight pick lands in-window → done, set threshold
                fm = cutlass.Float32(best_m) * denom
                s_thr[0] = band_lo + span * fm
                s_iscalars[0] = best_cnt
                s_iscalars[1] = cutlass.Int32(1)  # done
            else:
                # fall back to stock secant: leave s_thr[0]=pmean (unchanged),
                # done=0 so the secant loop below runs. Seed the bracket.
                s_iscalars[1] = cutlass.Int32(0)
                if best_m >= cutlass.Int32(0):
                    # count>kCC at tightest valid → threshold too low, search higher
                    fm = cutlass.Float32(best_m) * denom
                    s_thr[1] = band_lo + span * fm  # val_lo
                    s_iscalars[2] = best_cnt         # cnt_lo
                    s_thr[0] = band_lo + span * fm
        cute.arch.barrier()

        # ---- Fallback secant (stock) if not done ----
        it = cutlass.Int32(0)
        while it < cutlass.Int32(self.MAX_REFINE_ITERS) and s_iscalars[1] == cutlass.Int32(0):
            if tidx == 0:
                vlo = s_thr[1]; vhi = s_thr[2]
                clo = s_iscalars[2]; chi = s_iscalars[3]
                rng = vhi - vlo
                nv = cutlass.Float32(0.0)
                kFTarget = cutlass.const_expr(self.kFTarget)
                if clo > chi and rng > cutlass.Float32(1e-10):
                    f = cutlass.Float32(clo - cutlass.Int32(kFTarget)) / cutlass.Float32(clo - chi)
                    f = cute.arch.fmax(cutlass.Float32(0.05), f)
                    f = _fmin_f32_inline(f, cutlass.Float32(0.95))
                    nv = vlo + rng * f
                else:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)
                if nv <= vlo:
                    nv = vlo + rng * cutlass.Float32(0.05)
                if nv >= vhi:
                    nv = vhi - rng * cutlass.Float32(0.05)
                if nv == vlo or nv == vhi:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)
                    if nv == vlo or nv == vhi:
                        s_thr[0] = vlo
                        s_iscalars[1] = cutlass.Int32(2)
                    else:
                        s_thr[0] = nv
                else:
                    s_thr[0] = nv
            cute.arch.barrier()
            if s_iscalars[1] == cutlass.Int32(0):
                new_thr = s_thr[0]
                self.block_count_ge(input_row, N, new_thr, smem_ptcnt, smem_wcnt,
                                    s_iscalars, tidx, warp_id, lane)
                if tidx == 0:
                    c_new = s_iscalars[0]; t_new = s_thr[0]
                    if c_new >= cutlass.Int32(kK) and c_new <= cutlass.Int32(kCC):
                        s_iscalars[1] = cutlass.Int32(1)
                    elif c_new > cutlass.Int32(kCC):
                        s_thr[1] = t_new; s_iscalars[2] = c_new
                    else:
                        s_thr[2] = t_new; s_iscalars[3] = c_new
                cute.arch.barrier()
            it = it + cutlass.Int32(1)

        if tidx == 0:
            if s_iscalars[1] == cutlass.Int32(0):
                if s_iscalars[2] <= cutlass.Int32(kCC * 2):
                    s_thr[0] = s_thr[1]
                else:
                    s_thr[0] = s_thr[2]
                s_iscalars[1] = cutlass.Int32(2)
        cute.arch.barrier()


def _config(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def compile_portfolio(dtype, bs, n, K, cr_val, M_thr=16, kC=None):
    key = (dtype, bs, n, K, cr_val, M_thr, kC)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrPortfolioKernel(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False, M_thr=M_thr, kC_override=kC,
    )
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


def gvr_portfolio(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
                  M_thr=16, kC=None):
    bs, n = logits.shape
    compiled = compile_portfolio(logits.dtype, bs, n, index_topk, compress_ratio, M_thr, kC)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(_HERE.parents[1] / "harness"))
    import synth_data
    print(f"M_thr=16 smoke (report synth data)")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (4096, 8192, 16384, 65536):
            if N <= 2 * K:
                continue
            for dt in (torch.float32,):
                b = synth_data.get_bundle(K, dt, N)
                logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
                Npad = b["Npad"]
                seq_lens = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
                out = gvr_portfolio(logits, pre, seq_lens, K, crv)
                torch.cuda.synchronize()
                idx = out[0].clamp(min=0).long()
                v = logits[0].float().gather(0, idx).sort(descending=True).values
                ref = torch.topk(logits[0].float(), K).values
                d = (v - ref).abs().max().item()
                nu = len(set(out[0].tolist()))
                tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
                print(f"  K={K:4d} {str(dt):9s} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")
