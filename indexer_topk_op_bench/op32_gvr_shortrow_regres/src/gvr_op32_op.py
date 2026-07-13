# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op32 path-A: barrier-cheapened secant for op26_r0auto's fp32 BS=1 short-N route.

Subclass of GvrOp26Kernel. ONLY override = phase2_secant_search, rewritten so
the per-refine-iter control flow (bracket update + interpolation) runs
REDUNDANTLY on ALL threads from registers instead of on tid0-only behind two
block barriers. This removes barrier B (bracket-visibility) and barrier A
(nv-broadcast) per refine iter, keeping only block_count_ge's internal reduce
barrier + one protect barrier before the next count. The secant MATH is copied
bit-for-bit (secant2 / p2_log / clamps) → every threshold and count is
identical → exactness is preserved by construction. Baseline byte-identical
when redundant_secant=False (falls back to super()).

Aligns with the iter4 decomposition (cost = K-independent barrier chain) and
the rank-scatter precedent (cut barriers -> ~19% for one phase). Ceiling for
the secant phase alone is modest (~1 barrier/iter x ~1.46 iters); nsys is the
arbiter.
"""
import math
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2] / "op26_gvr_logfalsi_rs" / "src"))
sys.path.insert(0, str(_HERE.parents[2] / "p4_recursive_digit" / "src"))
from gvr_op26_op import GvrOp26Kernel  # noqa: E402
from gvr_topk_decode_p4 import _fmin_f32_inline  # noqa: E402


class GvrOp32Kernel(GvrOp26Kernel):
    def __init__(self, *a, redundant_secant=True, **kw):
        super().__init__(*a, **kw)
        self.redundant_secant = bool(redundant_secant)

    @cute.jit
    def phase2_secant_search(self, input_row, N, smem_ptcnt, smem_wcnt,
                             s_thr, s_iscalars, tidx, warp_id, lane):
        if cutlass.const_expr(not self.redundant_secant):
            super().phase2_secant_search(input_row, N, smem_ptcnt, smem_wcnt,
                                         s_thr, s_iscalars, tidx, warp_id, lane)
            return

        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        kFTarget = cutlass.const_expr(self.kFTarget)
        num_warps = cutlass.const_expr(self.num_warps)

        thr_init = s_thr[0]
        # initial count: block_count_ge writes smem_wcnt[warp]=partial + its
        # own internal barrier; we re-sum smem_wcnt redundantly on all threads.
        self.block_count_ge(input_row, N, thr_init, smem_ptcnt, smem_wcnt,
                            s_iscalars, tidx, warp_id, lane)
        c0 = cutlass.Int32(0)
        for w in cutlass.range_constexpr(num_warps):
            c0 = c0 + smem_wcnt[w]

        # bracket / done in ALL-THREAD registers (deterministic, block-uniform)
        done = cutlass.Int32(0)
        v_lo = thr_init
        v_hi = thr_init
        c_lo = cutlass.Int32(-1)
        c_hi = cutlass.Int32(-1)
        adopted = thr_init
        if c0 >= cutlass.Int32(kK) and c0 <= cutlass.Int32(kCC):
            done = cutlass.Int32(1)
            adopted = thr_init
        elif c0 > cutlass.Int32(kCC):
            v_lo = thr_init
            c_lo = c0
        else:
            v_hi = thr_init
            c_hi = c0

        v_last = thr_init
        c_last = c0
        v_prev = thr_init
        c_prev = cutlass.Int32(-1)

        it = cutlass.Int32(0)
        while it < cutlass.Int32(self.MAX_REFINE_ITERS) and done == cutlass.Int32(0):
            # ---- interpolation, ALL threads, identical (was tid0-only) ----
            vlo = v_lo
            vhi = v_hi
            clo = c_lo
            chi = c_hi
            rng = vhi - vlo
            nv = cutlass.Float32(0.0)
            use_sec = cutlass.Int32(0)
            if cutlass.const_expr(self.p2_secant2):
                if c_prev >= cutlass.Int32(0):
                    l1 = cmath.log2(cute.arch.fmax(cutlass.Float32(c_prev),
                                                   cutlass.Float32(1.0)),
                                    fastmath=True)
                    l2 = cmath.log2(cute.arch.fmax(cutlass.Float32(c_last),
                                                   cutlass.Float32(1.0)),
                                    fastmath=True)
                    dl = l1 - l2
                    dv = v_prev - v_last
                    adl = cute.arch.fmax(dl, cutlass.Float32(0.0) - dl)
                    adv = cute.arch.fmax(dv, cutlass.Float32(0.0) - dv)
                    if adl > cutlass.Float32(1e-6) and adv > cutlass.Float32(1e-10):
                        t = (cutlass.Float32(self.log2_kft) - l2) / dl
                        nv = v_last + t * dv
                        use_sec = cutlass.Int32(1)
            if use_sec == cutlass.Int32(0):
                if clo > chi and rng > cutlass.Float32(1e-10):
                    f = cutlass.Float32(0.0)
                    if cutlass.const_expr(self.p2_log):
                        clo_f = cutlass.Float32(clo)
                        chi_f = cute.arch.fmax(cutlass.Float32(chi),
                                               cutlass.Float32(1.0))
                        den = cmath.log2(clo_f / chi_f, fastmath=True)
                        if den > cutlass.Float32(0.0):
                            f = cmath.log2(clo_f / cutlass.Float32(kFTarget),
                                           fastmath=True) / den
                        else:
                            f = (cutlass.Float32(clo - cutlass.Int32(kFTarget))
                                 / cutlass.Float32(clo - chi))
                    else:
                        f = (cutlass.Float32(clo - cutlass.Int32(kFTarget))
                             / cutlass.Float32(clo - chi))
                    f = cute.arch.fmax(cutlass.Float32(0.05), f)
                    f = _fmin_f32_inline(f, cutlass.Float32(0.95))
                    if it == cutlass.Int32(0):
                        f = _fmin_f32_inline(f, cutlass.Float32(0.5))
                    nv = vlo + rng * f
                else:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)

            if nv <= vlo:
                nv = vlo + rng * cutlass.Float32(0.05)
            if nv >= vhi:
                nv = vhi - rng * cutlass.Float32(0.05)

            give_up = cutlass.Int32(0)
            if nv == vlo or nv == vhi:
                nv = (vlo + vhi) * cutlass.Float32(0.5)
                if nv == vlo or nv == vhi:
                    adopted = vlo
                    done = cutlass.Int32(2)
                    give_up = cutlass.Int32(1)

            if give_up == cutlass.Int32(0):
                # protect smem_wcnt: all-thread read above is done; the next
                # count won't touch smem_wcnt until after its scan loop, but
                # fence for correctness before it overwrites the slots.
                cute.arch.barrier()
                self.block_count_ge(input_row, N, nv, smem_ptcnt, smem_wcnt,
                                    s_iscalars, tidx, warp_id, lane)
                c_new = cutlass.Int32(0)
                for w in cutlass.range_constexpr(num_warps):
                    c_new = c_new + smem_wcnt[w]
                if c_new >= cutlass.Int32(kK) and c_new <= cutlass.Int32(kCC):
                    done = cutlass.Int32(1)
                    adopted = nv
                elif c_new > cutlass.Int32(kCC):
                    v_lo = nv
                    c_lo = c_new
                else:
                    v_hi = nv
                    c_hi = c_new
                v_prev = v_last
                c_prev = c_last
                v_last = nv
                c_last = c_new
            it = it + cutlass.Int32(1)

        # post-loop force (all-thread register decision)
        if done == cutlass.Int32(0):
            if c_lo <= cutlass.Int32(kCC * 2):
                adopted = v_lo
            else:
                adopted = v_hi
            done = cutlass.Int32(2)

        # ---- publish the smem contract for P3/fb_fix (one writer + barrier) ----
        # smem_ptcnt already caches the LAST block_count_ge's per-thread counts
        # (at thr=adopted when done==1 via the accepting count; when done==2 the
        # last count was at v_last — fb_fix recounts at s_thr[0] anyway).
        if tidx == 0:
            s_thr[0] = adopted
            s_thr[1] = v_lo
            s_thr[2] = v_hi
            s_iscalars[0] = c_last
            s_iscalars[1] = done
            s_iscalars[2] = c_lo
            s_iscalars[3] = c_hi
        cute.arch.barrier()
