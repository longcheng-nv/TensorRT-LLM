# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ct_regclus.py — op46 gvr_reg_clus CuTeDSL translation.

CUDA ground truth: src_cuda/kernel.cu L2359-2648 (CLUSTERED register-resident
GVR: the register algorithm — T = GMIN directly, one float-space histogram,
one register sweep — run across a cluster of CS CTAs; per-CTA instruction
stream intentionally identical to the single-CTA reg path plus two hardware
cluster barriers and CS DSMEM reads per bin). Contract: TRANSLATION_SPEC.md
§5.4. Probe verdicts (probes/PROBE_RESULTS.md) binding; op43 lessons L1-L5
applied; ct_main G1 signedness mitigation applied at every unsigned
compare/shift in/after dynamic loops.

Template knobs (CUDA `gvr_reg_clus<BLK,VPT,CS>`, all instantiations
BLK=BLKC=1024): ctor args of :class:`GvrRegClusKernel`. Runtime args mirror
the CUDA `(n, npad, k)` — npad/k come from tensor shapes, so only `n` crosses
the ABI. Launch: grid=(CS, b), cluster=(CS,1,1), block=1024, dynamic smem
45,056 B (+512 B static-mirror prelude), `__launch_bounds__(1024,1)` ==
min_blocks_per_mp=1 -> 64-register wall.

Shared-memory map (single dynamic window, word offsets; the CUDA static
__shared__ block folded into the first 512 B — byte-identical layout in every
CTA, a mapa/DSMEM requirement):

    [0..5]        s_res   (ct_common slot map RES_B/M/ABOVE/TOT/B2/B3)
    [6..7]        s_cnt   (s_o1, s_o2 — kernel.cu L2392)
    [8..9]        s_kmm   (s_kmin, s_kmax — Uint32, L2393)
    [16..16+32)   ws      (scan_cross_w workspace, L2390)
    [48..48+32)   wmn     (Uint32 warp min partials)
    [80..80+32)   wmx     (Uint32 warp max partials)
    [128..1152)   hist    (this CTA's raw counts, L2384)
    [1152..2176)  mrg     (cluster totals -> per-CTA global write cursors)
    [2176..3200)  hoff    (this CTA's rank-exclusive bin offset)
    [3200..7296)  ck      (crossing keys, Uint32, CMPC=4096 slots)
    [7296..11392) ci      (crossing indices, Int32, CMPC slots)

Launch smem = 45,568 B (compile-time constant -> plain int at .launch();
MINB==1 so the _build_kernel_attrs carveout path is not taken and ct_reg's
_no_carveout workaround is unnecessary here).
"""

import os
import sys

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import dsl_user_op

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_common import (  # noqa: E402
    RES_ABOVE,
    RES_B,
    RES_M,
    SENT_HI,
    SENT_LO,
    _cluster_sync_aligned,
    _ld_shared_cluster_i32,
    _mapa_shared_cluster_addr,
    _st_shared_cluster_i32,
    atomic_add_cta,
    atomic_max_cta,
    atomic_min_cta,
    ballot,
    clz_i32,
    f2u_rz,
    find_cross,
    fkey,
    g2r_atom_f32,
    invkey,
    ld_g_f32x4,
    ld_g_i32,
    ldg_f32,
    popc,
    scan_cross_w,
    warp_max_u32,
    warp_min_u32,
)

# ---- constants (kernel.cu lines) -------------------------------------------
NB = 1024                      # L16  (histogram bins; == BLKC here)
LNB = 10                       # L1267 log2(NB) — reg_clus narrowing shift
QUADC = 96                     # L21  O(mc^2) rank gate (L2534)
CMPC = 4096                    # L2372 crossing slots PER CTA (pow2)
LCMPC = 12                     # L2373 log2(CMPC)
BLKC = 1024                    # L2374 CTA size

STATIC_WORDS = 128             # DSL smem prelude (static-__shared__ mirror)
STATIC_BYTES = STATIC_WORDS * 4
DYN_SMEM_BYTES = (3 * NB + 2 * CMPC) * 4        # 45,056 (L2926)
SMEM_BYTES = STATIC_BYTES + DYN_SMEM_BYTES      # 45,568

# word offsets into the shared window (module docstring)
W_HIST = STATIC_WORDS
W_MRG = STATIC_WORDS + NB
W_HOFF = STATIC_WORDS + 2 * NB
W_CK = STATIC_WORDS + 3 * NB
W_CI = STATIC_WORDS + 3 * NB + CMPC

_NEG_INF = float("-inf")
_POS_INF = float("inf")


# ---------------------------------------------------------------------------
# module-local FP/util spellings (copied from frozen sibling ct_reg.py —
# probe P6 fma discipline; kept local so this module is self-contained)
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf(a, b, c, *, loc=None, ip=None):
    """CUDA fmaf: single fma.rn.f32 (P6 spelling; classify == emit bit-exact)."""
    return cutlass.Float32(mlir_math.fma(
        a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
        c.ir_value(loc=loc, ip=ip),
        fastmath=mlir_arith.FastMathFlags.none, loc=loc, ip=ip))


@cute.jit
def _umin_u32(a, b):
    """unsigned min(a, b) — CUDA min() on the bin clamp (IMNMX)."""
    r = a
    if b < a:
        r = b
    return r


@cute.jit
def _fabsf(x):
    """|x| via sign-bit clear (exact, matches fabsf)."""
    from ct_common import f32_of_u32, u32_of_f32
    return f32_of_u32(u32_of_f32(x) & cutlass.Uint32(0x7FFFFFFF))


def _smem_view(dtype, sbase, word_off: int, length: int, align: int = 16):
    """Typed tensor view at a constexpr word offset into the smem window."""
    p = cute.make_ptr(dtype, sbase + cutlass.Int32(word_off * 4),
                      cute.AddressSpace.smem, assumed_align=align)
    return cute.make_tensor(p, cute.make_layout((length,)))


def _val(frags, s: int):
    """val[s] accessor over the float4[VPT] register batch (constexpr s)."""
    return frags[s // 4][s % 4]


class GvrRegClusKernel:
    """gvr_reg_clus<BLK, VPT, CS> (kernel.cu L2376-2379)."""

    def __init__(self, blk: int, vpt: int, cs: int, pdl: bool = False):
        assert blk == BLKC, "all instantiations BLK=BLKC=1024 (spec §4b)"
        assert vpt in (1, 2, 4) and cs in (2, 4, 8)
        self.blk = blk
        self.vpt = vpt
        self.cs = cs
        self.pdl = bool(pdl)
        self.S = vpt * 4                    # L2381
        self.span = blk * vpt               # L2382 (float4 per CTA)

    # ------------------------------------------------------------------
    @cute.kernel
    def kern(self, logits: cute.Tensor, pre_idx: cute.Tensor, out: cute.Tensor,
             n: cutlass.Int32):
        BLK = cutlass.const_expr(self.blk)
        VPT = cutlass.const_expr(self.vpt)
        CS = cutlass.const_expr(self.cs)
        S = cutlass.const_expr(self.S)
        NW = cutlass.const_expr(self.blk // 32)

        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()          # L2380 (knob default off)

        tid, _, _ = cute.arch.thread_idx()
        rank, row, _ = cute.arch.block_idx()         # L2396-2397 (P11: bx=rank)
        lane = tid & cutlass.Int32(31)

        # ------------------------------------------------------------------
        # Predeclarations (DSL AST rule: every scalar (re)assigned under a
        # dynamic if/while must pre-exist with a stable type; constant inits
        # are dead-coded — ct_reg precedent, reg-audited free).
        # ------------------------------------------------------------------
        i = cutlass.Int32(0)
        j = cutlass.Int32(0)
        rnk = cutlass.Int32(0)
        tinc = cutlass.Int32(0)
        mc = cutlass.Int32(0)
        p = cutlass.Int32(0)
        q2i = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        lim1 = cutlass.Int32(0)
        aboveC = cutlass.Int32(0)
        needC = cutlass.Int32(0)
        mm = cutlass.Int32(0)
        lev = cutlass.Int32(0)
        done = cutlass.Int32(0)
        b2w = cutlass.Int32(0)
        sh2 = cutlass.Int32(0)
        b_lv = cutlass.Int32(0)
        it = cutlass.Int32(0)
        it2 = cutlass.Int32(0)
        idv = cutlass.Int32(0)
        q1f = cutlass.Int32(0)
        q2f = cutlass.Int32(0)
        n1 = cutlass.Int32(0)
        n2 = cutlass.Int32(0)
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        p1e = cutlass.Int32(0)
        p2e = cutlass.Int32(0)
        lml = cutlass.Int32(0)
        nA = cutlass.Int32(0)
        nT = cutlass.Int32(0)
        tieM = cutlass.Int32(0)
        pv0 = cutlass.Int32(-1)
        okc = cutlass.Int32(0)
        whole = cutlass.Int32(0)
        degen = cutlass.Int32(0)
        pre_a = cutlass.Int32(0)
        tot_a = cutlass.Int32(0)
        uk = cutlass.Uint32(0)
        uq = cutlass.Uint32(0)
        vq = cutlass.Uint32(0)
        kv = cutlass.Uint32(0)
        rlo = cutlass.Uint32(0)
        rhi = cutlass.Uint32(0)
        d2 = cutlass.Uint32(0)
        unar = cutlass.Uint32(0)
        bnn = cutlass.Uint32(0)
        nlo = cutlass.Uint32(0)
        uke = cutlass.Uint32(0)
        bn = cutlass.Uint32(0)
        ethr = cutlass.Int64(0)
        tval = cutlass.Float32(_NEG_INF)
        LOQ = cutlass.Float32(0.0)
        qv = cutlass.Float32(0.0)

        npad = cutlass.Int32(logits.shape[1])
        k = cutlass.Int32(pre_idx.shape[1])
        out_row = out[row, None]
        x_addr = logits[row, None].iterator.toint()      # Int64 gmem byte base
        p_addr = pre_idx[row, None].iterator.toint()

        # ---- shared-memory window (map in module docstring) ----
        sptr = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
        sbase = sptr.toint()                              # Int32 shared addr

        s_res = _smem_view(cutlass.Int32, sbase, 0, 6)
        s_cnt = _smem_view(cutlass.Int32, sbase, 6, 2)    # [0]=s_o1 [1]=s_o2
        s_kmm = _smem_view(cutlass.Uint32, sbase, 8, 2)   # [0]=s_kmin [1]=s_kmax
        s_ws = _smem_view(cutlass.Int32, sbase, 16, 32)
        s_wmn = _smem_view(cutlass.Uint32, sbase, 48, 32)
        s_wmx = _smem_view(cutlass.Uint32, sbase, 80, 32)
        s_hist = _smem_view(cutlass.Int32, sbase, W_HIST, NB)
        s_mrg = _smem_view(cutlass.Int32, sbase, W_MRG, NB)
        s_hoff = _smem_view(cutlass.Int32, sbase, W_HOFF, NB)
        s_ck = _smem_view(cutlass.Uint32, sbase, W_CK, CMPC)
        s_ci = _smem_view(cutlass.Int32, sbase, W_CI, CMPC, align=4)
        # raw byte bases for DSMEM (mapa) addressing
        hist_addr = sbase + cutlass.Int32(W_HIST * 4)
        ck_addr = sbase + cutlass.Int32(W_CK * 4)
        ci_addr = sbase + cutlass.Int32(W_CI * 4)

        n4 = n >> cutlass.Int32(2)                        # L2405
        ntail = n - (n4 << cutlass.Int32(2))              # L2406
        base4 = rank * cutlass.Int32(self.span)           # L2407
        tix = (n4 << cutlass.Int32(2)) + tid              # CUDA `tidx` L2425

        # ---- P0: redundant hint gather, EVERY CTA (L2410-2413; k<=BLK by
        # dispatch gate L2897). One coalesced word per thread, NO cluster
        # barrier — GMIN/GMAX identical everywhere by construction.
        if tid < k:
            pv0 = ld_g_i32(p_addr, tid)

        # ---- P1: row load — predicated flat float4[VPT] batch (L2415-2424;
        # the CUDA has NO exact-fit peel here, guard is per-load). Issue all
        # loads first (op43 L1), then -INFINITY-fill missed slots (op43 L2).
        atom128 = g2r_atom_f32(128, invariant=True)
        frags = [cute.make_fragment((4,), cutlass.Float32) for _ in range(VPT)]
        for u in cutlass.range_constexpr(VPT):
            i = base4 + tid + cutlass.Int32(u * self.blk)
            if i < n4:
                ld_g_f32x4(atom128, x_addr, i, frags[u])
        for u in cutlass.range_constexpr(VPT):
            i = base4 + tid + cutlass.Int32(u * self.blk)
            if i >= n4:                                   # -INFINITY fill L2421
                for z in cutlass.range_constexpr(4):
                    frags[u][z] = cutlass.Float32(_NEG_INF)
        # tail element: rank 0 only (L2425-2426)
        if rank == cutlass.Int32(0):
            if tid < ntail:
                tval = ldg_f32(x_addr, tix)

        # ---- P2: init (L2428-2429). NB == BLK -> single-pass hist clear.
        if tid == cutlass.Int32(0):
            s_cnt[0] = cutlass.Int32(0)
            s_cnt[1] = cutlass.Int32(0)
        for z in cutlass.range_constexpr(NB // self.blk):
            s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)

        # ---- P3: GMIN/GMAX from the hint (L2431-2445), ONE barrier fold.
        lmin = cutlass.Uint32(0xFFFFFFFF)
        lmax = cutlass.Uint32(0)
        if cutlass.Uint32(pv0) < cutlass.Uint32(n):
            uk = fkey(ldg_f32(x_addr, pv0))               # __ldg(X+pv0) L2433
            lmin = uk
            lmax = uk
        lmin = warp_min_u32(lmin)
        lmax = warp_max_u32(lmax)
        if lane == cutlass.Int32(0):
            s_wmn[tid >> cutlass.Int32(5)] = lmin
            s_wmx[tid >> cutlass.Int32(5)] = lmax
        cute.arch.barrier()                               # L2438
        a = cutlass.Uint32(0xFFFFFFFF)
        c = cutlass.Uint32(0)
        if lane < cutlass.Int32(NW):
            a = cutlass.Uint32(s_wmn[lane])
            c = cutlass.Uint32(s_wmx[lane])
        lmin = warp_min_u32(a)
        lmax = warp_max_u32(c)
        Tv = invkey(lmin)
        GMAX = invkey(lmax)

        # ---- collapse guard, NaN-safe (L2446-2453)
        okc = cutlass.Int32(0)
        if Tv < GMAX:
            if (GMAX - Tv) > cutlass.Float32(1e-30):
                okc = cutlass.Int32(1)
        if okc == cutlass.Int32(0):
            Tv = cutlass.Float32(SENT_LO)
            GMAX = cutlass.Float32(SENT_HI)

        # ---- bin transform constants (L2454-2467): branchless trash bin.
        WD = (GMAX - Tv) * cutlass.Float32(1.0 / float(NB - 2))
        wsel = cutlass.Float32(1e-30)
        if WD > cutlass.Float32(0.0):
            wsel = WD
        SC = cutlass.Float32(1.0) / wsel
        CQ0 = cutlass.Float32(1.0) - Tv * SC
        CQ = CQ0 + cutlass.Float32(1e-6) * (_fabsf(CQ0) + cutlass.Float32(1.0))

        # ---- P4: histogram (L2469-2472); tval add UNCONDITIONAL (trash bin
        # swallows -INFINITY via the saturating cvt).
        for s in cutlass.range_constexpr(S):
            qv = _fmaf(_val(frags, s), SC, CQ)
            bn = _umin_u32(f2u_rz(qv), cutlass.Uint32(NB - 1))
            atomic_add_cta(s_hist.iterator + cutlass.Int32(bn),
                           cutlass.Int32(1))
        qv = _fmaf(tval, SC, CQ)
        bn = _umin_u32(f2u_rz(qv), cutlass.Uint32(NB - 1))
        atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))

        # ---- P5: cluster merge (L2474-2484)
        _cluster_sync_aligned()                           # L2474
        for z in cutlass.range_constexpr(NB // self.blk):
            i = tid + cutlass.Int32(z * self.blk)
            # CS-unrolled remote u32 loads: batch-issue, then fold (#pragma
            # unroll L2477; one mapa per (i, r) exactly like map_shared_rank)
            hvals = []
            for r in cutlass.range_constexpr(CS):
                ma = _mapa_shared_cluster_addr(
                    hist_addr + (i << cutlass.Int32(2)), cutlass.Int32(r))
                hvals.append(_ld_shared_cluster_i32(ma))
            tot_a = cutlass.Int32(0)
            pre_a = cutlass.Int32(0)
            for r in cutlass.range_constexpr(CS):
                if cutlass.Int32(r) < rank:
                    pre_a = pre_a + hvals[r]              # rank-exclusive
                tot_a = tot_a + hvals[r]
            s_mrg[i] = tot_a
            s_hoff[i] = pre_a

        # ---- P6: scan (L2485-2492)
        cute.arch.barrier()                               # L2485
        scan_cross_w(s_mrg, s_ws, k, tid, s_res, blk=self.blk, nb=NB)
        cute.arch.barrier()                               # L2487
        above = s_res[RES_ABOVE]
        m = s_res[RES_M]
        Bv = s_res[RES_B]
        need = k - above
        whole = cutlass.Int32(0)
        if need >= m:
            whole = cutlass.Int32(1)
        degen = cutlass.Int32(0)
        if m > cutlass.Int32(CS * CMPC):
            degen = cutlass.Int32(1)
        for z in cutlass.range_constexpr(NB // self.blk):
            i = tid + cutlass.Int32(z * self.blk)
            s_mrg[i] = s_mrg[i] + s_hoff[i]               # L2491 global cursor
        cute.arch.barrier()                               # L2492

        # ---- P7: register sweep emit (L2494-2527, !degen)
        if degen == cutlass.Int32(0):
            LOQ = cutlass.Float32(Bv)                     # L2495
            lim1 = above
            if whole == cutlass.Int32(1):
                lim1 = above + m                          # L2496
            for s in cutlass.range_constexpr(S):
                qv = _fmaf(_val(frags, s), SC, CQ)        # bit-identical L2499
                if qv >= LOQ:
                    bn = _umin_u32(f2u_rz(qv), cutlass.Uint32(NB - 1))
                    p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn),
                                       cutlass.Int32(1))
                    idx = ((base4 + tid + cutlass.Int32((s // 4) * self.blk))
                           << cutlass.Int32(2)) + cutlass.Int32(s % 4)
                    if p < lim1:
                        out_row[p] = idx
                    else:
                        if whole == cutlass.Int32(0):
                            # crossing overflow -> striped DSMEM slabs; TWO
                            # separate u32 remote stores (NOT packed, L2507-10)
                            q2i = p - above
                            rnk = q2i >> cutlass.Int32(LCMPC)
                            j = (q2i & cutlass.Int32(CMPC - 1)) \
                                << cutlass.Int32(2)
                            _st_shared_cluster_i32(
                                _mapa_shared_cluster_addr(ck_addr + j, rnk),
                                fkey(_val(frags, s)))
                            _st_shared_cluster_i32(
                                _mapa_shared_cluster_addr(ci_addr + j, rnk),
                                idx)
            # tail element (L2514-2526): tval == -INF fails q>=LOQ elsewhere
            qv = _fmaf(tval, SC, CQ)
            if qv >= LOQ:
                bn = _umin_u32(f2u_rz(qv), cutlass.Uint32(NB - 1))
                p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn),
                                   cutlass.Int32(1))
                if p < lim1:
                    out_row[p] = tix
                else:
                    if whole == cutlass.Int32(0):
                        q2i = p - above
                        rnk = q2i >> cutlass.Int32(LCMPC)
                        j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                        _st_shared_cluster_i32(
                            _mapa_shared_cluster_addr(ck_addr + j, rnk),
                            fkey(tval))
                        _st_shared_cluster_i32(
                            _mapa_shared_cluster_addr(ci_addr + j, rnk), tix)

        # ---- P8 (L2529-2530): release staging to rank 0
        cute.arch.barrier()                               # L2529
        _cluster_sync_aligned()                           # L2530

        # ---- P9: rank-0 selection (L2532-2647)
        if rank == cutlass.Int32(0):
            if whole == cutlass.Int32(0):
                mc = m
                if degen == cutlass.Int32(1):
                    mc = cutlass.Int32(0)                 # L2533
                if degen == cutlass.Int32(0):
                    if mc <= cutlass.Int32(QUADC):
                        # (1) quad-96: all candidates LOCAL (96 < CMPC),
                        # O(mc^2) slot-order tie-broken rank (L2535-2543)
                        i = tid
                        while i < mc:
                            uq = cutlass.Uint32(s_ck[i])
                            rnk = cutlass.Int32(0)
                            j = cutlass.Int32(0)
                            while j < mc:
                                vq = cutlass.Uint32(s_ck[j])
                                tinc = cutlass.Int32(0)
                                if vq > uq:
                                    tinc = cutlass.Int32(1)
                                if vq == uq:
                                    if j < i:
                                        tinc = cutlass.Int32(1)
                                rnk = rnk + tinc
                                j = j + cutlass.Int32(1)
                            if rnk < need:
                                out_row[above + rnk] = s_ci[i]
                            i = i + cutlass.Int32(BLK)
                    else:
                        # (2) key-space narrowing over striped DSMEM slabs
                        # (L2544-2596): slot = i & (CMPC-1), rank = i >> LCMPC
                        if tid == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                            s_kmm[1] = cutlass.Uint32(0)
                        cute.arch.barrier()               # L2546
                        i = tid
                        while i < mc:
                            kv = cutlass.Uint32(_ld_shared_cluster_i32(
                                _mapa_shared_cluster_addr(
                                    ck_addr + ((i & cutlass.Int32(CMPC - 1))
                                               << cutlass.Int32(2)),
                                    i >> cutlass.Int32(LCMPC))))
                            atomic_min_cta(s_kmm.iterator, kv)
                            atomic_max_cta(s_kmm.iterator + 1, kv)
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()               # L2551
                        rlo = cutlass.Uint32(s_kmm[0])
                        rhi = cutlass.Uint32(s_kmm[1])
                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                        aboveC = cutlass.Int32(0)
                        needC = need
                        mm = mc
                        lev = cutlass.Int32(0)
                        done = cutlass.Int32(0)
                        while done == cutlass.Int32(0):   # <=6 levels L2553
                            if needC == mm:               # L2554
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) \
                                    - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                if cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    done = cutlass.Int32(1)
                                if lev >= cutlass.Int32(6):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2w = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1)))
                                sh2 = cutlass.Int32(0)
                                if b2w > cutlass.Int32(LNB):
                                    sh2 = b2w - cutlass.Int32(LNB)
                                for z in cutlass.range_constexpr(NB // self.blk):
                                    s_hist[tid + cutlass.Int32(z * self.blk)] \
                                        = cutlass.Int32(0)
                                cute.arch.barrier()       # L2563
                                i = tid
                                while i < mc:
                                    unar = cutlass.Uint32(_ld_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(
                                            ck_addr
                                            + ((i & cutlass.Int32(CMPC - 1))
                                               << cutlass.Int32(2)),
                                            i >> cutlass.Int32(LCMPC))))
                                    if cutlass.Uint32(unar) >= cutlass.Uint32(rlo):
                                        if cutlass.Uint32(unar) <= cutlass.Uint32(rhi):
                                            bnn = (cutlass.Uint32(unar)
                                                   - cutlass.Uint32(rlo)) \
                                                >> cutlass.Uint32(sh2)
                                            bnn = _umin_u32(
                                                bnn, cutlass.Uint32(NB - 1))
                                            atomic_add_cta(
                                                s_hist.iterator
                                                + cutlass.Int32(bnn),
                                                cutlass.Int32(1))
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()       # L2568
                                find_cross(s_hist, needC, tid, s_res, nb=NB)
                                cute.arch.barrier()       # L2570
                                aboveC = aboveC + s_res[RES_ABOVE]
                                needC = needC - s_res[RES_ABOVE]
                                mm = s_res[RES_M]
                                b_lv = s_res[RES_B]
                                nlo = cutlass.Uint32(rlo) \
                                    + (cutlass.Uint32(b_lv)
                                       << cutlass.Uint32(sh2))
                                if b_lv != cutlass.Int32(NB - 1):
                                    rhi = nlo + ((cutlass.Uint32(1)
                                                  << cutlass.Uint32(sh2))
                                                 - cutlass.Uint32(1))
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        cute.arch.barrier()               # L2576
                        # two-predicate ballot emit over the striped slabs
                        lml = cutlass.Int32(cute.arch.lanemask_lt())
                        it2 = (mc + cutlass.Int32(self.blk - 1)) \
                            // cutlass.Int32(self.blk)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tid
                            uke = cutlass.Uint32(0)
                            idv = cutlass.Int32(0)
                            if i < mc:                    # predicated remote
                                uke = cutlass.Uint32(_ld_shared_cluster_i32(
                                    _mapa_shared_cluster_addr(
                                        ck_addr
                                        + ((i & cutlass.Int32(CMPC - 1))
                                           << cutlass.Int32(2)),
                                        i >> cutlass.Int32(LCMPC))))
                                idv = _ld_shared_cluster_i32(
                                    _mapa_shared_cluster_addr(
                                        ci_addr
                                        + ((i & cutlass.Int32(CMPC - 1))
                                           << cutlass.Int32(2)),
                                        i >> cutlass.Int32(LCMPC)))
                            q1f = cutlass.Int32(0)
                            q2f = cutlass.Int32(0)
                            if i < mc:
                                if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                    q1f = cutlass.Int32(1)
                                if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
                                    q2f = cutlass.Int32(1)
                            n1 = ballot(q1f == cutlass.Int32(1))
                            n2 = ballot(q2f == cutlass.Int32(1))
                            b1 = cutlass.Int32(0)
                            b2 = cutlass.Int32(0)
                            if lane == cutlass.Int32(0):
                                if n1 != cutlass.Int32(0):
                                    b1 = atomic_add_cta(s_cnt.iterator,
                                                        popc(n1))
                                if n2 != cutlass.Int32(0):
                                    b2 = atomic_add_cta(s_cnt.iterator + 1,
                                                        popc(n2))
                            b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                            b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                            p1e = b1 + popc(n1 & lml)
                            p2e = b2 + popc(n2 & lml)
                            if q1f == cutlass.Int32(1):
                                if p1e < aboveC:
                                    out_row[above + p1e] = idv
                            if q2f == cutlass.Int32(1):
                                if p2e < needC:
                                    out_row[above + aboveC + p2e] = idv
                            it = it + cutlass.Int32(1)
                else:
                    # (3) degen safety net (L2597-2645): crossing bin larger
                    # than the whole cluster buffer -> exact whole-row
                    # key-space narrowing by rank 0 alone, <=8 levels.
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    aboveC = cutlass.Int32(0)             # above2
                    needC = k                             # need2
                    mm = n                                # m2
                    ethr = cutlass.Int64(0)
                    tieM = cutlass.Int32(1)
                    lev = cutlass.Int32(0)
                    done = cutlass.Int32(0)
                    while done == cutlass.Int32(0):
                        if needC == mm:                   # L2603
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) \
                                - cutlass.Int64(1)
                            aboveC = aboveC + mm
                            needC = cutlass.Int32(0)
                            tieM = cutlass.Int32(0)
                            done = cutlass.Int32(1)
                        if done == cutlass.Int32(0):
                            if cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                done = cutlass.Int32(1)
                            if lev >= cutlass.Int32(8):   # L2605
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                done = cutlass.Int32(1)
                        if done == cutlass.Int32(0):
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2w = cutlass.Int32(32) - clz_i32(
                                cutlass.Int32(d2 | cutlass.Uint32(1)))
                            sh2 = cutlass.Int32(0)
                            if b2w > cutlass.Int32(LNB):
                                sh2 = b2w - cutlass.Int32(LNB)
                            for z in cutlass.range_constexpr(NB // self.blk):
                                s_hist[tid + cutlass.Int32(z * self.blk)] = \
                                    cutlass.Int32(0)
                            cute.arch.barrier()           # L2612
                            i = tid
                            while i < n:                  # whole-row bin L2613
                                unar = fkey(ldg_f32(x_addr, i))
                                if cutlass.Uint32(unar) >= cutlass.Uint32(rlo):
                                    if cutlass.Uint32(unar) <= cutlass.Uint32(rhi):
                                        bnn = (cutlass.Uint32(unar)
                                               - cutlass.Uint32(rlo)) \
                                            >> cutlass.Uint32(sh2)
                                        bnn = _umin_u32(
                                            bnn, cutlass.Uint32(NB - 1))
                                        atomic_add_cta(
                                            s_hist.iterator
                                            + cutlass.Int32(bnn),
                                            cutlass.Int32(1))
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()           # L2618
                            find_cross(s_hist, needC, tid, s_res, nb=NB)
                            cute.arch.barrier()           # L2620
                            aboveC = aboveC + s_res[RES_ABOVE]
                            needC = needC - s_res[RES_ABOVE]
                            mm = s_res[RES_M]
                            b_lv = s_res[RES_B]
                            nlo = cutlass.Uint32(rlo) \
                                + (cutlass.Uint32(b_lv)
                                   << cutlass.Uint32(sh2))
                            if b_lv != cutlass.Int32(NB - 1):
                                rhi = nlo + ((cutlass.Uint32(1)
                                              << cutlass.Uint32(sh2))
                                             - cutlass.Uint32(1))
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    cute.arch.barrier()                   # L2626
                    nA = k                                # tieM ? above2 : k
                    if tieM == cutlass.Int32(1):
                        nA = aboveC
                    nT = cutlass.Int32(0)
                    if tieM == cutlass.Int32(1):
                        nT = needC
                    lml = cutlass.Int32(cute.arch.lanemask_lt())
                    it2 = (n + cutlass.Int32(self.blk - 1)) \
                        // cutlass.Int32(self.blk)
                    it = cutlass.Int32(0)
                    while it < it2:                       # L2628-2645
                        i = it * cutlass.Int32(BLK) + tid
                        uke = cutlass.Uint32(0)
                        if i < n:
                            uke = fkey(ldg_f32(x_addr, i))
                        q1f = cutlass.Int32(0)
                        q2f = cutlass.Int32(0)
                        if i < n:
                            if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                q1f = cutlass.Int32(1)
                            if tieM == cutlass.Int32(1):
                                if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
                                    q2f = cutlass.Int32(1)
                        n1 = ballot(q1f == cutlass.Int32(1))
                        n2 = ballot(q2f == cutlass.Int32(1))
                        b1 = cutlass.Int32(0)
                        b2 = cutlass.Int32(0)
                        if lane == cutlass.Int32(0):
                            if n1 != cutlass.Int32(0):
                                b1 = atomic_add_cta(s_cnt.iterator, popc(n1))
                            if n2 != cutlass.Int32(0):
                                b2 = atomic_add_cta(s_cnt.iterator + 1,
                                                    popc(n2))
                        b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                        b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                        p1e = b1 + popc(n1 & lml)
                        p2e = b2 + popc(n2 & lml)
                        if q1f == cutlass.Int32(1):
                            if p1e < nA:
                                out_row[p1e] = i
                        if q2f == cutlass.Int32(1):
                            if p2e < nT:
                                out_row[nA + p2e] = i
                        it = it + cutlass.Int32(1)

        # ---- P10: FINAL cluster rendezvous (L2648) — ALL ranks reach it;
        # keeps peers resident until rank 0 has read their ck/ci.
        _cluster_sync_aligned()

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, logits: cute.Tensor, pre_idx: cute.Tensor,
                 out: cute.Tensor, n: cutlass.Int32, stream):
        b = logits.shape[0]
        self.kern(logits, pre_idx, out, n).launch(
            grid=(self.cs, b, 1), block=(self.blk, 1, 1),
            cluster=(self.cs, 1, 1), stream=stream,
            smem=SMEM_BYTES, min_blocks_per_mp=1,
            use_pdl=self.pdl)


# ---------------------------------------------------------------------------
# host wrapper: compile cache + route()-driven entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE: dict = {}


def get_compiled(tpl, dump_dir=None, pdl=False):
    """Compile (or fetch) the variant for constexpr tuple (BLK, VPT, CS)."""
    key = (tuple(tpl), bool(pdl))
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt
        blk, vpt, cs = tpl
        kernel = GvrRegClusKernel(blk, vpt, cs, pdl=pdl)
        nb_, nc_ = cute.sym_int(), cute.sym_int()
        nb2_, nc2_ = cute.sym_int(), cute.sym_int()
        nb3_, nc3_ = cute.sym_int(), cute.sym_int()
        lg_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (nb_, nc_), stride_order=(1, 0), assumed_align=16)
        pi_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb2_, nc2_), stride_order=(1, 0), assumed_align=16)
        out_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb3_, nc3_), stride_order=(1, 0), assumed_align=16)
        fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
        opts = "--enable-tvm-ffi"
        if dump_dir:
            opts += f" --keep-ptx --keep-cubin --dump-dir {dump_dir}"
        compiled = cute.compile(
            kernel, lg_fake, pi_fake, out_fake, cutlass.Int32(0),
            stream=fake_stream, options=opts)
        _COMPILE_CACHE[key] = compiled
    return compiled


def regclus_topk(logits, pre_idx, n, out, rd=None):
    """torch-facing entry for the clustered register family.

    logits [b, npad] f32, pre_idx [b, k] i32, out [b, >=k] i32, n = valid len.
    rd: optional pre-computed ct_dispatch.route() dict (must be reg_clus).
    """
    if rd is None:
        from ct_dispatch import route
        rd = route(logits.shape[0], int(n), logits.shape[1],
                   pre_idx.shape[1])
    assert rd['kernel'] == 'reg_clus', rd['kernel']
    tpl = tuple(rd['tpl'])
    assert pre_idx.shape[1] <= tpl[0], "k <= BLK enforced by dispatch L2897"
    assert rd['smem'] == DYN_SMEM_BYTES
    compiled = get_compiled(tpl)
    compiled(logits, pre_idx, out, int(n))
    return out


__all__ = ["GvrRegClusKernel", "get_compiled", "regclus_topk",
           "SMEM_BYTES", "DYN_SMEM_BYTES", "STATIC_BYTES"]
