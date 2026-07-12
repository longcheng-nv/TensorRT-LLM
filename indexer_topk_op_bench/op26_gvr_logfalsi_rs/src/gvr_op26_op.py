# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""op26: HLS-op25 ideas ported back to the classic GVR kernels.

Two arms, both subclass-only (no vendored file edits, op13 GvrP2C pattern):

op26_1cta — GvrOp26Kernel ⊂ p4_recursive_digit's GvrTopKKernel (the op#7
  kernel that already carries the EXACT rank-scatter P4 behind
  enable_p4_rank_scatter[_exact]):
  * P2: gated log-count secant interpolation (op13 iter8 formula) + optional
    kC/kFTarget window override; dispatched per (dtype, K, N) by the op13
    iter8c ship table `dispatch_p2c_v2` (fp32 only; 16-bit keeps baseline P2
    per op13's no-evidence rule). iter5: wherever the log path is on, the
    aim moves to the window's geometric center (fixes the K1024@131K
    edge-aim rejection creep, R1). iter5b: the log-secant through the last
    two measured points (p2_secant2, R2 fix) is silicon-falsified as a
    default (pass savings < loop overhead) and survives only on K2048
    16-bit n>=262144; N bands where the log path still lost to the linear
    anchor are pruned back to stock P2 (see dispatch_p2_op26).
  * P3 fallback (fb_fix, always on): replaces the vendored one-sided
    retry-shrink (exits on the FIRST count<=kCC INCLUDING count<kK ⇒ the
    report.html §5 real-data red card: -1 slots) with a correct bounded
    refine — verify/expand the hi end (P1 seeds cnt_hi=1 at v_hi=pmax
    WITHOUT measuring; hit~0 rows can have >=K elements above pmax), then
    log-count regula-falsi aim at m* = K*(kC/K)^0.2 (HLS Theorem 3), accept
    ONLY count in [kK, kCC]; 30-iter exhaustion lands on a MEASURED
    undershoot side (fail-soft, never an unguarded truncation, and never a
    count>kCC smem overflow).
  * P4: exact rank-scatter gated to the op#7 production-win region
    (fp32 anywhere, 16-bit only at BS>=256); histogram-snap elsewhere.

op26_mc — GvrOp26ClusterKernel ⊂ GvrTopKClusterKernel (PR#15198): P2
  log-count interpolation only (fp32; stock windows — cluster P3 handoff and
  leader-only P4 are left untouched this campaign; per-slice retry semantics
  make the fb_fix port a separate, riskier change).

Wrappers mirror harness/gvr_cutedsl_op.py and gvr_multicta_cutedsl_op.py
launch-config heuristics exactly so A/B deltas are algorithmic only.
"""
import math
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "ops"))                       # cute_vendored
sys.path.insert(0, str(_BENCH / "p4_recursive_digit" / "src"))  # op#7 P4 kernel

from gvr_topk_decode_p4 import GvrTopKKernel as _P4Kernel  # noqa: E402
from cute_vendored.blackwell.top_k.gvr_topk_decode import (  # noqa: E402
    _fmin_f32_inline,
)
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: E402
    GvrTopKClusterKernel,
)

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
       torch.float16: cutlass.Float16}


# ---------------------------------------------------------------------------
# op13 iter8c ship table (dispatch_p2c_v2, @390c99c3e4), iter5b revision.
# -> (use_log, kCC_override, kFTarget_override, secant2).
# (False, None, None, False) = stock.
#
# iter5 (ROOTCAUSE_P2.md fix A): everywhere log-interp is on, the aim point
# moves from the acceptance-band EDGE (kFT=kK, R1: ~half the exact shots
# land below kK and get rejected -> one-sided falsi creep) to the GEOMETRIC
# CENTER sqrt(kK*kCC) of the EFFECTIVE window. K512 fp32 keeps the iter7
# linear-narrow entry unchanged (no regression there; log falsified).
#
# iter5b (silicon ablation, diag_iter5_silicon.py + iter5 single-cell nsys):
# secant2 is silicon-FALSIFIED as a default — the host replay's pass savings
# are real, but the loop-carried secant costs more than it saves on silicon
# (K2048 fp16 65536: iter4 28.7us == V1 28.7us < V3 32.9us; K1024 fp32
# 131072 BS256: V1 112.6us < V3 114.7us) and the ENTIRE K1024@131K recovery
# comes from the center aim alone (38.98 -> 32.77us). The one place secant2
# stays on is K2048 16-bit large N, where iter5 nsys flipped 262144 from
# 0.996 (iter4) to 1.11-1.21 and the V3 effect is monotone-improving in N
# (65536 0.71 -> 131072 0.97 -> 262144 1.15). Cells where the log path
# still lost to the linear anchor with the fix in (K1024@131K all dtypes
# 0.83-0.91; K2048 16-bit 16K-131K 0.84-0.94; K1024 16-bit 131K+/4096)
# are pruned back to stock P2.
# ---------------------------------------------------------------------------
def dispatch_p2_op26(dtype, K, n):
    if dtype != torch.float32:
        # 16-bit: stock windows (kCC=5120), center aim; K512 keeps the
        # baseline P2 (the K512 log variant is the one hard op13
        # falsification).
        if K == 1024:
            # iter4 real: wins 8K-64K (up to 1.46), loses 4096 (fp16 0.92),
            # 131072 (0.69-0.74, R1 edge-aim) and 1M (0.71) -> band-gate.
            # iter5c: aim is N-split — at 8192 the stock edge aim (kFT=1024)
            # beats center by 15-30% (iter4 1.26/1.46 vs iter5b 1.09/1.12);
            # at 16K-64K center wins (bf16 16384 0.92 -> 0.98, 65536
            # 1.15 -> 1.23). The interp bias direction flips with N.
            if n == 8192:
                return True, None, None, False    # stock edge aim
            # iter5d: 16384 pruned — persistent loss cell under BOTH aims
            # (iter4 edge 0.92-1.00, iter5c center real 0.96-0.98 but
            # mixed-scenario 0.928 in the full grid).
            if 32768 <= n <= 65536:
                return True, None, 2289, False    # sqrt(1024*5120)
            return False, None, None, False
        if K == 2048:
            # 8192: iter4 1.01-1.04 / iter5 high-BS 1.17-1.18; 16K-131K
            # loss band both iterations -> stock; >=262144 the V3
            # (center+secant2) arm is the measured win (1.11-1.21 @262144).
            if n == 8192:
                return True, None, 3238, False    # sqrt(2048*5120)
            # iter5d: the 262144 secant2 win (1.14, both dtypes) does NOT
            # extrapolate upward — full grid measured 524288 at 0.878/0.880
            # (iter4 stock-aim log was ~0.96-1.01 there) -> exact-N gate.
            if n == 262144:
                return True, None, 3238, True
            return False, None, None, False
        return False, None, None, False
    if K == 512:
        if n <= 65536:
            return False, 1536, 1280, False   # iter7 lin-narrow (log falsified)
        return False, None, None, False
    if K == 1024:
        # 131072 pruned (iter4 0.69 real; still 0.83-0.91 with center aim —
        # the narrow window's pass/cand exchange rate stays inverted there).
        if n <= 32768:
            return True, 2048, 1448, False    # narrow; sqrt(1024*2048)
        return False, None, None, False
    if K == 2048:
        # iter4 real: log wins the whole 8K-512K band (1.03-1.28); 1M wash.
        # iter5c: the iter4 edge aim (kFT=2048=kK) beats the geometric
        # center at EVERY N here (iter5b center: 32768 0.93 vs 1.03,
        # 65536 0.95 vs 1.10, ties at >=131K) — K2048 fp32's linear-tail
        # overshoot lands center-band when aimed at the edge. Keep iter4.
        if n >= 8192:
            return True, 4096, 2048, False    # narrow; iter4 edge aim
        return False, None, None, False
    return False, None, None, False


def dispatch_rs_op26(dtype, bs):
    """op#7 production-win region: fp32 anywhere; 16-bit only BS>=256."""
    return dtype == torch.float32 or bs >= 256


class GvrOp26Kernel(_P4Kernel):
    """Single-CTA GVR + gated log-P2 + corrected fallback + rank-scatter P4."""

    def __init__(self, *a, p2_log=False, kC_override=None,
                 kFTarget_override=None, p2_secant2=False, fb_fix=True,
                 fb_alpha=0.2, **kw):
        super().__init__(*a, **kw)
        self.p2_log = bool(p2_log)
        if kC_override is not None:
            self.kC = int(kC_override)
        if kFTarget_override is not None:
            self.kFTarget = int(kFTarget_override)
        self.p2_secant2 = bool(p2_secant2)
        # secant aim target in log2-count space (kFTarget_override already
        # applied above, so this is the dispatched center for iter5 cells)
        self.log2_kft = math.log2(max(self.kFTarget, 1))
        self.fb_fix = bool(fb_fix)
        # interior aim for the corrected fallback (HLS Theorem 3 grid optimum)
        self.log2_mstar = math.log2(
            self.top_k * (self.kC / self.top_k) ** float(fb_alpha))

    # ------------------------------------------------------------------
    # P2 — verbatim vendored phase2_secant_search except the interpolation
    # block: const_expr(p2_log) selects log-count interpolation
    # f = log2(clo/kFT) / log2(clo/chi) (op13 iter8), else the vendored
    # linear formula. const_expr(p2_secant2) (iter5, ROOTCAUSE_P2 V3)
    # additionally replaces the bracket-endpoint interpolant with a
    # log-secant through the last two MEASURED (thr,count) points — immune
    # to the chi=1 unmeasured P1 seed and to regula-falsi endpoint
    # freezing; degenerate pairs fall back to the bracket path. The secant
    # proposal still passes the vendored bracket clamps (5% margins +
    # midpoint safeguard), and bracket/window/fallback logic, f clamps
    # (f in [0.05,0.95], iter0 cap 0.5) and barriers are unchanged, so the
    # exactness guard (done==1 <=> count in [kK,kCC]) is untouched.
    # ------------------------------------------------------------------
    @cute.jit
    def phase2_secant_search(self, input_row, N, smem_ptcnt, smem_wcnt,
                             s_thr, s_iscalars, tidx, warp_id, lane):
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        kFTarget = cutlass.const_expr(self.kFTarget)

        thr_init = s_thr[0]
        self.block_count_ge(input_row, N, thr_init, smem_ptcnt, smem_wcnt,
                            s_iscalars, tidx, warp_id, lane)

        if tidx == 0:
            c0 = s_iscalars[0]
            t0 = s_thr[0]
            if c0 >= cutlass.Int32(kK) and c0 <= cutlass.Int32(kCC):
                s_iscalars[1] = cutlass.Int32(1)  # done = 1 (converged)
            elif c0 > cutlass.Int32(kCC):
                s_thr[1] = t0
                s_iscalars[2] = c0
            else:
                s_thr[2] = t0
                s_iscalars[3] = c0
        cute.arch.barrier()

        # iter5-V3 loop-carried registers: the last two MEASURED (thr,count)
        # points. Maintained UNIFORMLY by every thread — s_thr[0] and
        # s_iscalars[0] reads after the barriers are block-uniform — so no
        # extra smem slot is needed and thread 0's interpolant sees them
        # as plain registers. c_prev = -1 flags "<2 measured points yet".
        v_last = thr_init
        c_last = s_iscalars[0]
        v_prev = thr_init
        c_prev = cutlass.Int32(-1)

        it = cutlass.Int32(0)
        while it < cutlass.Int32(self.MAX_REFINE_ITERS) and s_iscalars[1] == cutlass.Int32(0):
            if tidx == 0:
                vlo = s_thr[1]
                vhi = s_thr[2]
                clo = s_iscalars[2]
                chi = s_iscalars[3]
                rng = vhi - vlo
                nv = cutlass.Float32(0.0)
                use_sec = cutlass.Int32(0)
                if cutlass.const_expr(self.p2_secant2):
                    # log-secant through the last two measured points,
                    # aimed at log2(kFTarget) (= the window's geometric
                    # center under the iter5 dispatch table).
                    if c_prev >= cutlass.Int32(0):
                        l1 = cmath.log2(
                            cute.arch.fmax(cutlass.Float32(c_prev),
                                           cutlass.Float32(1.0)),
                            fastmath=True)
                        l2 = cmath.log2(
                            cute.arch.fmax(cutlass.Float32(c_last),
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
                            # log-count interpolation: count(v) ~ exp(a - b*v)
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

                if nv == vlo or nv == vhi:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)
                    if nv == vlo or nv == vhi:
                        s_thr[0] = vlo
                        s_iscalars[1] = cutlass.Int32(2)  # done = 2 (give up)
                    else:
                        s_thr[0] = nv
                else:
                    s_thr[0] = nv
            cute.arch.barrier()

            if s_iscalars[1] == cutlass.Int32(0):
                new_thr = s_thr[0]
                self.block_count_ge(input_row, N, new_thr, smem_ptcnt,
                                    smem_wcnt, s_iscalars, tidx, warp_id, lane)
                if tidx == 0:
                    c_new = s_iscalars[0]
                    t_new = s_thr[0]
                    if c_new >= cutlass.Int32(kK) and c_new <= cutlass.Int32(kCC):
                        s_iscalars[1] = cutlass.Int32(1)
                    elif c_new > cutlass.Int32(kCC):
                        s_thr[1] = t_new
                        s_iscalars[2] = c_new
                    else:
                        s_thr[2] = t_new
                        s_iscalars[3] = c_new
                cute.arch.barrier()
                v_prev = v_last
                c_prev = c_last
                v_last = new_thr
                c_last = s_iscalars[0]
            it = it + cutlass.Int32(1)

        # ---- Post-loop fallback: if still not done, force threshold ----
        if tidx == 0:
            if s_iscalars[1] == cutlass.Int32(0):
                if s_iscalars[2] <= cutlass.Int32(kCC * 2):
                    s_thr[0] = s_thr[1]  # threshold = val_lo
                else:
                    s_thr[0] = s_thr[2]  # threshold = val_hi
                s_iscalars[1] = cutlass.Int32(2)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # P3 — fb_fix: correct bounded refine replacing the vendored one-sided
    # retry-shrink, then delegate to the parent P3 (prefix-sum +
    # stream-write; its own retry block is skipped because done==1).
    # Contract preserved: smem_ptcnt caches the per-thread counts of the
    # LAST block_count_ge at the final adopted s_thr[0].
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_collect_candidates(self, input_row, N, smem_keys, smem_vals,
                                  smem_ptcnt, smem_wcnt, s_thr, s_iscalars,
                                  tidx, warp_id, lane):
        if cutlass.const_expr(self.fb_fix):
            kK = cutlass.const_expr(self.top_k)
            kCC = cutlass.const_expr(self.kC)
            if s_iscalars[1] != cutlass.Int32(1):
                # P2 gave up (done=2). Its bracket counts are NOT trustworthy:
                # cnt_lo may still be the P1 SEED (1.25*K at v_lo=pmin, never
                # measured — the host replay shows exactly this poisoning the
                # interpolant into an undershoot creep). Mark BOTH end counts
                # unknown; only measured values feed the falsi.
                if tidx == 0:
                    s_iscalars[1] = cutlass.Int32(0)
                    s_iscalars[2] = cutlass.Int32(-1)  # cnt_lo: unknown
                    s_iscalars[3] = cutlass.Int32(-1)  # cnt_hi: unknown
                cute.arch.barrier()
                # Entry recount at the P2-forced threshold (vendored-
                # compatible; accepts the common undershoot-creep row in ONE
                # pass: e.g. forced thr=pmin whose true count==K).
                rs = cutlass.Int32(0)
                while rs < cutlass.Int32(30) and s_iscalars[1] == cutlass.Int32(0):
                    if rs > cutlass.Int32(0):
                        if tidx == 0:
                            lo3 = s_thr[1]
                            hi3 = s_thr[2]
                            clo3 = s_iscalars[2]
                            chi3 = s_iscalars[3]
                            cand = (lo3 + hi3) * cutlass.Float32(0.5)
                            if chi3 < cutlass.Int32(0):
                                cand = hi3          # measure the hi end first
                            elif clo3 < cutlass.Int32(0):
                                cand = lo3          # then the lo end
                            else:
                                # both ends measured: log-count regula falsi
                                # aimed at the interior target m*; midpoint
                                # safeguard on degeneracy
                                chic = chi3
                                if chic < cutlass.Int32(1):
                                    chic = cutlass.Int32(1)
                                l_lo = cmath.log2(cutlass.Float32(clo3),
                                                  fastmath=True)
                                l_hi = cmath.log2(cutlass.Float32(chic),
                                                  fastmath=True)
                                den3 = l_lo - l_hi
                                if den3 > cutlass.Float32(0.0):
                                    t3 = (cutlass.Float32(self.log2_mstar)
                                          - l_hi) / den3
                                    cnd3 = hi3 + t3 * (lo3 - hi3)
                                    if cnd3 > lo3 and cnd3 < hi3:
                                        cand = cnd3
                            s_thr[0] = cand
                        cute.arch.barrier()
                    self.block_count_ge(input_row, N, s_thr[0], smem_ptcnt,
                                        smem_wcnt, s_iscalars, tidx,
                                        warp_id, lane)
                    cute.arch.barrier()
                    if tidx == 0:
                        c3 = s_iscalars[0]
                        t3v = s_thr[0]
                        if c3 >= cutlass.Int32(kK) and c3 <= cutlass.Int32(kCC):
                            s_iscalars[1] = cutlass.Int32(1)  # accept
                        elif c3 > cutlass.Int32(kCC):
                            # overshoot: t3v is a measured lo end
                            s_thr[1] = t3v
                            s_iscalars[2] = c3
                            if t3v >= s_thr[2]:
                                # even the hi end overshoots -> expand upward
                                rng3 = s_thr[2] - s_thr[1]
                                if rng3 < cutlass.Float32(1.0):
                                    rng3 = cutlass.Float32(1.0)
                                s_thr[2] = s_thr[2] + rng3 * cutlass.Float32(8.0)
                                s_iscalars[3] = cutlass.Int32(-1)
                        else:
                            # undershoot: t3v is a measured hi end
                            s_thr[2] = t3v
                            s_iscalars[3] = c3
                            if t3v <= s_thr[1]:
                                # even the lo end undershoots (possible with
                                # invalid preIdx entries) -> expand downward
                                rng3 = s_thr[2] - s_thr[1]
                                if rng3 < cutlass.Float32(1.0):
                                    rng3 = cutlass.Float32(1.0)
                                s_thr[1] = s_thr[1] - rng3 * cutlass.Float32(8.0)
                                s_iscalars[2] = cutlass.Int32(-1)
                    cute.arch.barrier()
                    rs = rs + cutlass.Int32(1)
                if s_iscalars[1] != cutlass.Int32(1):
                    # exhausted (tie-block): land on the MEASURED undershoot
                    # side — fail-soft semantics, and count<=kCC guaranteed so
                    # the collect buffer cannot overflow.
                    self.block_count_ge(input_row, N, s_thr[2], smem_ptcnt,
                                        smem_wcnt, s_iscalars, tidx,
                                        warp_id, lane)
                    cute.arch.barrier()
                    if tidx == 0:
                        s_thr[0] = s_thr[2]
                        s_iscalars[1] = cutlass.Int32(1)
                    cute.arch.barrier()
        _P4Kernel.phase3_collect_candidates(
            self, input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_wcnt,
            s_thr, s_iscalars, tidx, warp_id, lane)


class GvrOp26ClusterKernel(GvrTopKClusterKernel):
    """PR#15198 cluster GVR + gated log-count P2 interpolation (stock windows)."""

    def __init__(self, *a, p2_log=False, **kw):
        super().__init__(*a, **kw)
        self.p2_log = bool(p2_log)

    @cute.jit
    def phase2_secant_search(self, input_row, N, slice_start, slice_end,
                             smem_ptcnt, smem_wcnt, s_thr, s_iscalars,
                             s_cluster_partial, tidx, warp_id, lane,
                             smem_input=None):
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        kFTarget = cutlass.const_expr(self.kFTarget)

        thr_init = s_thr[0]
        self.block_count_ge(input_row, slice_start, slice_end, thr_init,
                            smem_ptcnt, smem_wcnt, s_iscalars,
                            s_cluster_partial, tidx, warp_id, lane,
                            smem_input=smem_input)

        if tidx == 0:
            c0 = s_iscalars[0]
            t0 = s_thr[0]
            if c0 >= cutlass.Int32(kK) and c0 <= cutlass.Int32(kCC):
                s_iscalars[1] = cutlass.Int32(1)  # done = 1 (converged)
            elif c0 > cutlass.Int32(kCC):
                s_thr[1] = t0
                s_iscalars[2] = c0
            else:
                s_thr[2] = t0
                s_iscalars[3] = c0
        cute.arch.barrier()

        it = cutlass.Int32(0)
        while it < cutlass.Int32(self.MAX_REFINE_ITERS) and s_iscalars[1] == cutlass.Int32(0):
            if tidx == 0:
                vlo = s_thr[1]
                vhi = s_thr[2]
                clo = s_iscalars[2]
                chi = s_iscalars[3]
                rng = vhi - vlo
                nv = cutlass.Float32(0.0)
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

                if nv == vlo or nv == vhi:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)
                    if nv == vlo or nv == vhi:
                        s_thr[0] = vlo
                        s_iscalars[1] = cutlass.Int32(2)  # done = 2 (give up)
                    else:
                        s_thr[0] = nv
                else:
                    s_thr[0] = nv
            cute.arch.barrier()

            if s_iscalars[1] == cutlass.Int32(0):
                new_thr = s_thr[0]
                self.block_count_ge(input_row, slice_start, slice_end, new_thr,
                                    smem_ptcnt, smem_wcnt, s_iscalars,
                                    s_cluster_partial, tidx, warp_id, lane,
                                    smem_input=smem_input)
                if tidx == 0:
                    c_new = s_iscalars[0]
                    t_new = s_thr[0]
                    if c_new >= cutlass.Int32(kK) and c_new <= cutlass.Int32(kCC):
                        s_iscalars[1] = cutlass.Int32(1)
                    elif c_new > cutlass.Int32(kCC):
                        s_thr[1] = t_new
                        s_iscalars[2] = c_new
                    else:
                        s_thr[2] = t_new
                        s_iscalars[3] = c_new
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


# ---------------------------------------------------------------------------
# Wrappers — launch heuristics mirror the baseline op drivers exactly.
# ---------------------------------------------------------------------------
_compiled_1cta = {}
_compiled_mc = {}


def _config_1cta(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def gvr_cutedsl_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                     out=None):
    bs, n = logits.shape
    dt = logits.dtype
    use_log, kcc, kft, sec2 = dispatch_p2_op26(dt, index_topk, n)
    rs_on = dispatch_rs_op26(dt, bs)
    key = (dt, bs, n, index_topk, compress_ratio, use_log, kcc, kft, sec2,
           rs_on)
    compiled = _compiled_1cta.get(key)
    if compiled is None:
        t, use256, min_bpm = _config_1cta(bs, n)
        kobj = GvrOp26Kernel(
            dtype=_DT[dt], top_k=index_topk, next_n=1, num_threads=t,
            compress_ratio=compress_ratio, use_256bit_load=use256,
            enable_unroll_4=True, enable_phase3_unroll=True,
            min_blocks_per_mp=min_bpm, return_output_values=False,
            enable_p4_rank_scatter=rs_on, enable_p4_rank_scatter_exact=rs_on,
            p2_log=use_log, kC_override=kcc, kFTarget_override=kft,
            p2_secant2=sec2, fb_fix=True,
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
        _compiled_1cta[key] = compiled
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


def _resolve_config_mc(logits, num_sms, cluster_size=None):
    """Verbatim copy of gvr_multicta_cutedsl_op._resolve_config heuristics."""
    num_rows, N_dec = logits.shape
    if cluster_size is None:
        if N_dec < 65536:
            cluster_size = 1
        elif num_rows <= 16 and num_rows * 4 <= num_sms:
            cluster_size = 4
        elif num_rows * 2 <= num_sms:
            cluster_size = 2
        else:
            cluster_size = 1
    num_threads_per_block = 1024 if (num_rows <= num_sms and N_dec >= 65536) else 512
    use_256bit_load = logits.dtype == torch.float32 and N_dec >= 16384
    enable_warp_parallel_reduce = num_threads_per_block == 1024
    vec_bits_host = 256 if use_256bit_load else 128
    vec_w_host = vec_bits_host // (32 if logits.dtype == torch.float32 else 16)
    n_vec_iters = max(1, N_dec // (num_threads_per_block * vec_w_host))
    if logits.dtype == torch.float32:
        if n_vec_iters < 4:
            min_blocks_per_mp = 0
        elif num_rows <= num_sms:
            min_blocks_per_mp = 1
        elif num_sms * 2 < num_rows <= num_sms * 3 and N_dec <= 32768:
            min_blocks_per_mp = 3
        else:
            min_blocks_per_mp = 2
    else:
        if num_rows > num_sms:
            min_blocks_per_mp = 3
        elif n_vec_iters < 4:
            min_blocks_per_mp = 0
        else:
            min_blocks_per_mp = 1
    return dict(min_blocks_per_mp=min_blocks_per_mp,
                use_256bit_load=use_256bit_load,
                num_threads_per_block=num_threads_per_block,
                enable_warp_parallel_reduce=enable_warp_parallel_reduce,
                cluster_size=cluster_size)


def gvr_multicta_op26(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                      next_n=1, out=None, cluster_size=None):
    dt = logits.dtype
    # log-interp on K1024/K2048, all dtypes; K512 stock — the op26_dry batch
    # corroborated op13's K512-log falsification at large N (131K 0.91,
    # 1M 0.85 vs the mc anchor), and K512 fp32 stock cells double as a
    # same-process drift-QA anchor.
    p2_log = (index_topk in (1024, 2048))
    cfg = _resolve_config_mc(logits, NUM_SMS, cluster_size)
    key = (dt, index_topk, next_n, compress_ratio, p2_log,
           cfg["min_blocks_per_mp"], cfg["use_256bit_load"],
           cfg["num_threads_per_block"], cfg["enable_warp_parallel_reduce"],
           cfg["cluster_size"])
    compiled = _compiled_mc.get(key)
    if compiled is None:
        kobj = GvrOp26ClusterKernel(
            dtype=_DT[dt], top_k=index_topk, next_n=next_n,
            num_threads=cfg["num_threads_per_block"],
            enable_unroll_4=True, enable_phase3_unroll=True,
            use_constant_hint=False,
            min_blocks_per_mp=cfg["min_blocks_per_mp"],
            use_256bit_load=cfg["use_256bit_load"],
            enable_warp_parallel_reduce=cfg["enable_warp_parallel_reduce"],
            compress_ratio=compress_ratio, return_output_values=False,
            cluster_size=cfg["cluster_size"], enable_smem_cache=False,
            smem_cache_elems=32768,
            p2_log=p2_log,
        )
        n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
        in_align = 32 if cfg["use_256bit_load"] else 16
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
        _compiled_mc[key] = compiled
    if out is None:
        out = torch.empty(logits.shape[0], index_topk, dtype=torch.int32,
                          device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


def picked_cluster_size_op26(logits, index_topk, compress_ratio=1):
    return _resolve_config_mc(logits, NUM_SMS)["cluster_size"]


if __name__ == "__main__":
    torch.manual_seed(0)
    print("== op26_1cta smoke (dispatch + exactness vs torch.topk) ==")
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for K, crv, N in ((512, 4, 16384), (512, 4, 262144), (1024, 4, 32768),
                          (1024, 4, 131072), (2048, 1, 32768), (2048, 1, 262144)):
            logits = torch.randn(1, N, dtype=dt, device="cuda")
            pre_idx = torch.topk(logits[0].float(), K).indices.int().view(1, K).contiguous()
            seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
            out = gvr_cutedsl_op26(logits, pre_idx, seq_lens, K, crv)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[0].tolist()))
            ul, kcc, kft, sec2 = dispatch_p2_op26(dt, K, N)
            print(f"  {str(dt):14s} K={K:4d} N={N:6d} log={int(ul)} kCC={kcc} "
                  f"kFT={kft} sec2={int(sec2)}: uniq={nuniq}/{K} valdiff={d:.2e}")
            assert d == 0.0 and nuniq == K, "op26_1cta NOT exact"
    print("== op26_mc smoke ==")
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for K, crv, N in ((512, 4, 65536), (1024, 4, 131072), (2048, 1, 131072)):
            logits = torch.randn(1, N, dtype=dt, device="cuda")
            pre_idx = torch.topk(logits[0].float(), K).indices.int().view(1, K).contiguous()
            seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
            cs = picked_cluster_size_op26(logits, K, crv)
            out = gvr_multicta_op26(logits, pre_idx, seq_lens, K, crv)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[0].tolist()))
            print(f"  {str(dt):14s} K={K:4d} N={N:6d} cs={cs}: uniq={nuniq}/{K} valdiff={d:.2e}")
            assert d == 0.0 and nuniq == K, "op26_mc NOT exact"
    print("op26 smoke OK")
