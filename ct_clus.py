# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ct_clus.py — op46 gvr_clus (clustered streaming GVR) CuTeDSL port.

Ground truth: src_cuda/kernel.cu L1793-2356 (frozen); phase contract, smem
map and barrier inventory per TRANSLATION_SPEC.md §5.3; DSL spellings pinned
by probes/PROBE_RESULTS.md (BINDING) + op43 lessons L1..L5; per-CTA stream
mirrors gvr_main (idioms reused from the proven src/ct_main.py).

Ctor knobs (compile-time, mirror of the CUDA template params, spec §4c):
    BLK = 1024, U ∈ {1,2,4,8}, MINB = 1, NBS = 256, CS ∈ {2,4,8}
    (+ scap/cmp smem-extent knobs: every reachable route has 8192/2048 —
     SCAP/CMP stay LIVE runtime args for all value logic, ABI parity).
Derived: HB=NBS, STEPC=BLK*U, PFD=min(U,4) (kernel.cu L1905).

Signature (ABI parity with kernel.cu L1797-1799; Q is dead in-kernel):
    run(logits[b,npad] f32, pre_idx[b,k] i32, out[b,k] i32) via
    kern(..., n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2)
Grid dim3(CS, b) native 2-D + cluster (CS,1,1) (probe P11); block 1024;
min_blocks_per_mp=1 (64-reg wall, probes P2/P15); smem one SmemAllocator
blob mirroring the CUDA dynamic map hist|cbuf|ck64c|mrg (L1802-1814),
dyn-equivalent bytes == host smc formula L3130 (asserted in run()).

int2 staging convention (same as ct_main): int2(value bits, index) is ONE
little-endian Uint64 = (idx << 32) | value_bits — single u64 smem ld/st.

Barrier / cluster-op inventory implemented (kernel.cu line cites, op43 L5):
    L1902 (sample redux publish), L1948 (sample hist), L1956 (scan publish),
    [degenerate sample: 2 inside gather_hint],
    retry preamble: clus.sync L2016 + __syncthreads L2024,
    clus.sync L2135 (merge), __syncthreads L2143 (merge publish),
    [ladder gather: 2 inside gather_hint],
    clus.sync L2226 (EXIT RENDEZVOUS — the only one; rank!=0 falls through),
    narrowing: L2255, L2257, per-level L2270/L2273, L2281,
    degen: per-level L2321/L2325/L2327 (+1 INSIDE scan_cross), L2337.
    NO loop-tail ladder barriers (gvr_clus has none — unlike gvr_main).
    All clus.sync = releasing aligned arrive+wait (risk R3, never relaxed).
Cluster ops: merge = _merge_scan0_local, a LOCAL patched copy of the frozen
    ct_common.merge_scan0 that rematerializes mapa per (q, r) like the CUDA
    L135-137 (register-pressure fix, see notes; DSMEM v4 spelling = probe
    P5b via ct_common ops); ONE packed u64 st.shared::cluster candidate push
    to rank-0 ck64c (L2185/L2205/L2219 — never split 4B, op43 L3); mapa of
    ck64c to rank 0. PTX inventory audited: 3 arrive+3 wait (aligned), 3
    st.shared::cluster.u64 sites, 2*CS ld.shared::cluster.v4 sites, 4
    prefetch.global.L2 sites on U=8 only, zero griddepcontrol.
Every rung/ladder decision is cluster-uniform by construction (identical
sample locations on every rank; merged tot; block-uniform gather) — the
conditional retry clus.sync at L2016 cannot deadlock (spec §5.3).
"""

import os
import sys

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator
from cutlass.cute import runtime as _crt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import ct_common as C  # noqa: E402  (FROZEN sibling — import only)

QUADC_CLUS = C.QUADC_CLUS

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# single-rounding fma.rn.f32 (probe P6 emit spelling; local — ct_common has
# no fma export). Sites: T (L1957), Tk/T3 (L1966/1977/1992), HIC (L1981).
# (x-TF)*SC classify shapes stay plain sub+mul (P6: uncontractible).
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(mlir_math.fma(
        a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
        c.ir_value(loc=loc, ip=ip),
        fastmath=mlir_arith.FastMathFlags.none, loc=loc, ip=ip))


class GvrClusKernel:
    """CuTeDSL port of gvr_clus<BLK, U, MINB, NBS, CS> (kernel.cu L1795)."""

    def __init__(self, blk: int, u: int, minb: int, nbs: int, cs: int,
                 scap: int = 8192, cmp_: int = 2048):
        assert blk == 1024, "gvr_clus is always BLK=1024 (dispatch L3132)"
        assert minb == 1, "gvr_clus is __launch_bounds__(BLK, 1) (L1796)"
        assert nbs == 256, "SNB must stay 256 (kernel.cu L170-177)"
        assert u in (1, 2, 4, 8) and cs in (2, 4, 8)
        self.blk = blk
        self.u = u
        self.minb = minb
        self.nbs = nbs
        self.cs = cs
        self.scap = scap                                # smem extents only —
        self.cmp = cmp_                                 # value logic uses rt args
        self.hb = nbs                                   # L1801
        self.stepc = blk * u                            # L1836
        self.pfd = u if u < 4 else 4                    # L1905 PFD=min(U,4)
        self.lb = nbs.bit_length() - 1                  # log2(NBS)=8
        # dynamic-region byte map (L1802-1814): hist | cbuf(int2) | ck64c | mrg
        self.cbuf_bytes = (scap + 4) * 8
        assert self.cbuf_bytes % 16 == 0
        self.ck_off = self.cbuf_bytes                   # inside the blob
        self.dyn_bytes = nbs * 4 + self.cbuf_bytes + cmp_ * 8 + nbs * 4
        # == host smc = SNB*8 + (SCAP+4)*8 + CMP*8 (L3130)

    # ------------------------------------------------------------------
    # GVR_EMITK (kernel.cu L2096-2104): classify+stage one survivor.
    # bn via UNSIGNED saturating convert (f2u_rz, P4); staging store is ONE
    # u64; branchless trash slot min(pos, SCAP) (runtime SCAP). Returns pos+1.
    # ------------------------------------------------------------------
    @cute.jit
    def _emitk(self, xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2):
        NBS = self.nbs
        bn_u = C.f2u_rz((xv - TF) * SC)
        if bn_u > cutlass.Uint32(NBS - 1):
            bn_u = cutlass.Uint32(NBS - 1)
        bn = cutlass.Int32(bn_u)
        C.atomic_add_cta(s_hist.iterator + bn, cutlass.Int32(1))
        ps = pos
        if ps > SCAP:
            ps = SCAP                                    # trash slot (IMNMX)
        s_cbuf2[ps] = ((cutlass.Uint64(cutlass.Uint32(idx))
                        << cutlass.Uint64(32))
                       | cutlass.Uint64(C.u32_of_f32(xv)))
        return pos + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # P5 emit step (kernel.cu L2178-2186 == L2196-2204 == L2211-2219):
    # bn via SIGNED rz convert (__float2int_rz, L2180); bn>=B gate; LOCAL
    # mrg atomicAdd whose result is a CLUSTER-GLOBAL position (prefix-biased
    # cursors from merge_scan0); overflow -> ONE packed u64 DSMEM store to
    # rank-0 ck64c (probe P5b, op43 L3 — never split 4B).
    # ------------------------------------------------------------------
    @cute.jit
    def _p5_emit(self, xv, idv, TF, SC, B, above, lim1, whole, CMP,
                 s_mrg, out_row, rk64):
        NBS = self.nbs
        bn = C.f2s_rz((xv - TF) * SC)
        if bn > cutlass.Int32(NBS - 1):
            bn = cutlass.Int32(NBS - 1)
        if bn >= B:
            p = C.atomic_add_cta(s_mrg.iterator + bn, cutlass.Int32(1))
            if p < lim1:
                out_row[p] = idv
            else:
                if whole == cutlass.Int32(0):
                    q2 = p - above
                    if q2 < CMP:
                        C._st_shared_cluster_u64(
                            rk64 + q2 * cutlass.Int32(8),
                            (cutlass.Uint64(C.fkey(xv)) << cutlass.Uint64(32))
                            | cutlass.Uint64(cutlass.Uint32(idv)))

    # ------------------------------------------------------------------
    # LOCAL patched copy of ct_common.merge_scan0 (frozen sibling — not
    # edited): rematerializes mapa per (q, r) exactly like the CUDA L135-137
    # instead of holding CS mapped base addresses across the whole merge.
    # The hoisted-array form costs CS extra long-lived registers; with the
    # U>=4 sixteen-register pf prime batch it tips ptxas into spilling the
    # batch across the rung phase (reg-audit finding, notes). Semantics,
    # the DSMEM v4 load spelling (probe P5b), the register accumulation and
    # the prefix-biased STS.128 cursor write are IDENTICAL to ct_common.
    # NO barrier inside (caller pays L2143).
    # ------------------------------------------------------------------
    @cute.jit
    def _merge_scan0_local(self, s_hist, s_mrg, rank, target, tidx, s_res):
        NBS = self.nbs
        CS = self.cs
        BPT = NBS // 32
        NV = BPT // 4
        if tidx < cutlass.Int32(32):
            lane = tidx
            atom = C.smem_atom_i32_128()
            hbase = s_hist.iterator.toint()
            # pass 1 (L131-143): remote v4 accumulation of tot/pre per vector
            tot_r = []
            pre_r = []
            sm = cutlass.Int32(0)
            for q in cutlass.range_constexpr(NV):
                boff = (lane * cutlass.Int32(BPT)
                        + cutlass.Int32(4 * q)) * cutlass.Int32(4)
                t = [cutlass.Int32(0)] * 4
                p = [cutlass.Int32(0)] * 4
                for r in cutlass.range_constexpr(CS):
                    mapped = C._mapa_shared_cluster_addr(
                        hbase + boff, cutlass.Int32(r))   # per-use mapa (L136)
                    v0, v1, v2, v3 = C._ld_shared_cluster_v4_u32(mapped)
                    t[0] = t[0] + v0
                    t[1] = t[1] + v1
                    t[2] = t[2] + v2
                    t[3] = t[3] + v3
                    if cutlass.Int32(r) < rank:           # L140 predicated adds
                        p[0] = p[0] + v0
                        p[1] = p[1] + v1
                        p[2] = p[2] + v2
                        p[3] = p[3] + v3
                tot_r.append(t)
                pre_r.append(p)
                sm = sm + t[0] + t[1] + t[2] + t[3]
            # inclusive scan + totals (L144-148)
            w = C.warp_incl_scan_add(sm, lane)
            tt = cute.arch.shuffle_sync(w, cutlass.Int32(31))
            after = tt - w
            if lane == cutlass.Int32(0):
                s_res[C.RES_TOT] = tt
            base = lane * cutlass.Int32(BPT)
            # descending walk: crossing pin + prefix-biased cursors into mrg
            for q in cutlass.range_constexpr(NV - 1, -1, -1):   # L151-165
                o4 = cute.make_fragment((4,), cutlass.Int32)
                for j in cutlass.range_constexpr(3, -1, -1):
                    cq = tot_r[q][j]
                    o4[j] = after + pre_r[q][j]
                    gb = base + cutlass.Int32(4 * q + j)
                    cross = cutlass.Int32(0)
                    if after < target:
                        if (after + cq) >= target:
                            cross = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross = cutlass.Int32(1)
                    if cross != cutlass.Int32(0):
                        s_res[C.RES_B] = gb
                        s_res[C.RES_ABOVE] = after
                        s_res[C.RES_M] = cq
                    after = after + cq
                boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) \
                    * cutlass.Int32(16)
                C.sts128_i32(atom, o4, s_mrg.iterator.toint(), boff)

    # ------------------------------------------------------------------
    # two-predicate warp-ballot emit step (narrowing L2279-2301 and degen
    # L2338-2354) — same helper as ct_main. s_scal[1]=s_o1, s_scal[2]=s_o2.
    # ------------------------------------------------------------------
    @cute.jit
    def _ballot_pair_emit(self, p1, p2, idv, base1, cap1, base2, cap2,
                          out_row, s_scal, lane):
        n1 = C.ballot(p1 != cutlass.Int32(0))
        n2 = C.ballot(p2 != cutlass.Int32(0))
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        if lane == cutlass.Int32(0):
            if n1 != cutlass.Int32(0):
                b1 = C.atomic_add_cta(s_scal.iterator + 1,
                                      cutlass.Int32(C.popc(n1)))
            if n2 != cutlass.Int32(0):
                b2 = C.atomic_add_cta(s_scal.iterator + 2,
                                      cutlass.Int32(C.popc(n2)))
        b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
        b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
        lm = cutlass.Int32(cute.arch.lanemask_lt())
        if p1 != cutlass.Int32(0):
            p = b1 + cutlass.Int32(C.popc(n1 & lm))
            if p < cap1:
                out_row[base1 + p] = idv
        if p2 != cutlass.Int32(0):
            p = b2 + cutlass.Int32(C.popc(n2 & lm))
            if p < cap2:
                out_row[base2 + p] = idv

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kern(self, logits: cute.Tensor, pre_idx: cute.Tensor,
             out: cute.Tensor,
             n: cutlass.Int32, npad: cutlass.Int32, k: cutlass.Int32,
             SCAP: cutlass.Int32, CMP: cutlass.Int32, SMP: cutlass.Int32,
             TGT: cutlass.Int32, Q: cutlass.Int32, SS2: cutlass.Int32,
             TGT2: cutlass.Int32):
        BLK = self.blk
        U = self.u
        NBS = self.nbs
        CS = self.cs
        PFD = self.pfd
        STEPC = self.stepc
        NW = BLK // 32

        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()               # (rank, row) L1824-1825
        rank = bx
        row = by
        lane = tidx & cutlass.Int32(31)

        # ---- shared memory (CUDA dynamic map order L1802-1814, then statics) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(                   # hist[NBS] @ blob start
            cutlass.Int32, cute.make_ordered_layout((self.hb,), order=(0,)),
            byte_alignment=128)
        blob = smem.allocate_tensor(                     # cbuf(int2) | ck64c
            cutlass.Int8,
            cute.make_ordered_layout((self.cbuf_bytes + self.cmp * 8,),
                                     order=(0,)),
            byte_alignment=16)
        s_mrg = smem.allocate_tensor(                    # mrg[NBS] (L1814)
            cutlass.Int32, cute.make_ordered_layout((self.nbs,), order=(0,)),
            byte_alignment=16)
        s_ws = smem.allocate_tensor(                     # L1817 (degen scan only)
            cutlass.Int32, cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16)                           # Int32: ct_common
        # scan_cross predeclares its second-stage partial as Int32 and reads
        # s_ws inside a dynamic if — a Uint32 ws tensor trips the DSL type-
        # stability check (frozen sibling; counts < 2^31 so Int32 is exact).
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16)
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16)
        s_res = smem.allocate_tensor(                    # ct_common slot map
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=16)
        # scalar block: [0]=s_bufn [1]=s_o1 [2]=s_o2 (L1818)
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)),
            byte_alignment=16)
        s_tsh = smem.allocate_tensor(                    # L1820
            cutlass.Float32, cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=4)
        s_kmm = smem.allocate_tensor(                    # L1821 [0]=kmin [1]=kmax
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=8)
        sbase = blob.iterator.toint()
        s_cbuf2 = cute.make_tensor(                      # int2 staged as u64
            cute.make_ptr(cutlass.Uint64, sbase, cute.AddressSpace.smem,
                          assumed_align=16),
            cute.make_layout((self.scap + 4,)))
        ck_addr = sbase + cutlass.Int32(self.ck_off)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, ck_addr, cute.AddressSpace.smem,
                          assumed_align=16),
            cute.make_layout((self.cmp,)))

        # ---- row bases (L1830-1833) ----
        row64 = cutlass.Int64(row)
        x_addr = logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(4)
        p_addr = pre_idx.iterator.toint() + row64 * cutlass.Int64(k) * cutlass.Int64(4)
        out_row = out[row, None]

        # ---- interleaved chunk ownership (L1835-1846) ----
        n4 = n >> cutlass.Int32(2)
        nCh = (n4 + cutlass.Int32(STEPC - 1)) // cutlass.Int32(STEPC)
        nFullG = n4 // cutlass.Int32(STEPC)
        tail0 = n4 << cutlass.Int32(2)
        tailn = cutlass.Int32(0)
        if rank == cutlass.Int32(0):
            tailn = n - tail0

        if tidx == cutlass.Int32(0):                     # L1848
            s_res[C.RES_B2] = cutlass.Int32(-1)
            s_res[C.RES_B3] = cutlass.Int32(-1)
            s_scal[0] = cutlass.Int32(0)                 # s_bufn
        if tidx < cutlass.Int32(self.hb):                # L1849 (HB<=BLK)
            s_hist[tidx] = cutlass.Int32(0)

        # ============ P1: QUAD sample (hint gather LAZY, L1851-1878) ========
        # one 64B line = 4 float4 per location, TWO threads: tid takes the
        # lower pair at p4, tid+SMP the upper pair at p4+2 (L1868-1869).
        atom128 = C.g2r_atom_f32(128, invariant=True)
        fsa = cute.make_fragment((4,), cutlass.Float32)
        fsb = cute.make_fragment((4,), cutlass.Float32)
        smp2 = SMP * cutlass.Int32(2)
        shas = cutlass.Int32(0)
        if tidx < smp2:
            shas = cutlass.Int32(1)
        if shas != cutlass.Int32(0):
            p4 = tidx * SS2 * cutlass.Int32(4)
            if tidx >= SMP:
                p4 = (tidx - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
            C.ld_g_f32x4(atom128, x_addr, p4, fsa)
            C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)

        # ============ P2: quantile rung, redundant per CTA (L1883-1997) =====
        smn = cutlass.Float32(float("inf"))
        smx = cutlass.Float32(float("-inf"))
        if shas != cutlass.Int32(0):                     # L1928-1932
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fsa[t])
                smx = C.fmax_f32(smx, fsa[t])
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fsb[t])
                smx = C.fmax_f32(smx, fsb[t])
        fma_ = cute.make_fragment((4,), cutlass.Float32)  # mop-up pair bufs
        fmb_ = cute.make_fragment((4,), cutlass.Float32)
        j = tidx + cutlass.Int32(BLK)                    # mop-up L1891-1897
        while j < smp2:
            p4 = j * SS2 * cutlass.Int32(4)
            if j >= SMP:
                p4 = (j - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
            C.ld_g_f32x4(atom128, x_addr, p4, fma_)
            C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fma_[t])
                smx = C.fmax_f32(smx, fma_[t])
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fmb_[t])
                smx = C.fmax_f32(smx, fmb_[t])
            j = j + cutlass.Int32(BLK)
        a0 = C.warp_min_u32(C.fkey(smn))                 # L1898-1901
        c0m = C.warp_max_u32(C.fkey(smx))
        if lane == cutlass.Int32(0):
            s_wmn[tidx >> cutlass.Int32(5)] = a0
            s_wmx[tidx >> cutlass.Int32(5)] = c0m
        cute.arch.barrier()                              # ---- barrier L1902 ----

        # PRIME-LATE (L1903-1916): every rank's sample has landed; prime NOW.
        lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)   # L1904
        pf = [cute.make_fragment((4,), cutlass.Float32) for _ in range(PFD)]
        for uu in cutlass.range_constexpr(PFD):          # clamped prime L1906
            i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
            ic = i_
            if ic >= n4:
                ic = lim4
            C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
        # asm prefetch gate (L1912-1916): DEEP rows only; empty for U<=PFD
        if cutlass.const_expr(U > PFD):
            gpp = cutlass.Int32(0)
            if n4 >= cutlass.Int32(32768):
                if (rank + cutlass.Int32(1)) * cutlass.Int32(STEPC) <= n4:
                    gpp = cutlass.Int32(1)
            if gpp != cutlass.Int32(0):
                for uu in cutlass.range_constexpr(PFD, U):
                    C._prefetch_l2(x_addr + cutlass.Int64(
                        rank * cutlass.Int32(STEPC) + tidx
                        + cutlass.Int32(uu * BLK)) * cutlass.Int64(16))

        # cross-warp sample reduce (L1917-1923)
        av = cutlass.Uint32(0xFFFFFFFF)
        cv = cutlass.Uint32(0)
        if lane < cutlass.Int32(NW):
            av = s_wmn[lane]
            cv = s_wmx[lane]
        SMIN = C.invkey(C.warp_min_u32(av))
        SMAX = C.invkey(C.warp_max_u32(cv))

        GMIN = cutlass.Float32(C.SENT_LO)                # L1925-1926
        GMAX = cutlass.Float32(C.SENT_HI)
        T = cutlass.Float32(_NEG_INF)
        HIC = cutlass.Float32(_NEG_INF)
        w = cutlass.Float32(0.0)
        sok = cutlass.Int32(0)                           # L1930
        if SMP > cutlass.Int32(0):
            if SMAX > SMIN:
                sok = cutlass.Int32(1)
        if sok != cutlass.Int32(0):                      # L1932-1947 sample hist
            w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
            sc_s = cutlass.Float32(1.0) / w
            if shas != cutlass.Int32(0):
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fsa[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fsb[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
            j = tidx + cutlass.Int32(BLK)                # mop-up reloads
            while j < smp2:
                p4 = j * SS2 * cutlass.Int32(4)
                if j >= SMP:
                    p4 = (j - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                C.ld_g_f32x4(atom128, x_addr, p4, fma_)
                C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fma_[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fmb_[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                j = j + cutlass.Int32(BLK)
        cute.arch.barrier()                              # ---- barrier L1948 ----
        # triple-target ZERO scan (L1952-1955): TGT / TGT2 / 2*TGT
        C.scan_cross0(s_hist, TGT, tidx, s_res, TGT2, TGT * cutlass.Int32(2),
                      s_hist, nb=NBS, zero=True, two=True, three=True)
        cute.arch.barrier()                              # ---- barrier L1956 ----

        tot0 = s_res[C.RES_TOT]
        b1v = s_res[C.RES_B]
        if sok != cutlass.Int32(0):                      # L1957
            if tot0 >= TGT:
                T = _fmaf(cutlass.Float32(b1v), w, SMIN)
        needg = cutlass.Int32(1)                         # L1958-1963
        if T > cutlass.Float32(_NEG_INF):
            needg = cutlass.Int32(0)
        if needg != cutlass.Int32(0):
            # degenerate sample: identical on every rank of the cluster
            GMIN, GMAX = C.gather_hint(x_addr, p_addr, k, n, tidx, s_wmn,
                                       s_wmx, blk=BLK, kpt=1)  # 2 barriers
            T = GMIN
        if sok != cutlass.Int32(0):                      # L1964-1982 HIC
            if tot0 >= TGT:
                b2v = s_res[C.RES_B2]
                if b2v >= cutlass.Int32(0):
                    Tk = _fmaf(cutlass.Float32(b2v), w, SMIN)
                    up = C.fmax_f32(Tk - T, cutlass.Float32(0.0))
                    # heavy-tail cap by T - T3 (rank-TGT..rank-2TGT distance)
                    if tot0 >= TGT * cutlass.Int32(2):
                        b3v = s_res[C.RES_B3]
                        if b3v >= cutlass.Int32(0):
                            T3 = _fmaf(cutlass.Float32(b3v), w, SMIN)
                            if T > T3:
                                up = C.fmin_f32(
                                    up, cutlass.Float32(2.0) * (T - T3))
                    HIC = C.fmax_f32(
                        _fmaf(cutlass.Float32(4.0), up, T),
                        _fmaf(cutlass.Float32(8.0), w, T))
        # r4 (a000) ladder floor in SHARED (64-reg wall, L1983-1996)
        if tidx == cutlass.Int32(0):
            t5 = cutlass.Float32(_NEG_INF)
            if sok != cutlass.Int32(0):
                if tot0 >= TGT * cutlass.Int32(2):
                    b3v = s_res[C.RES_B3]
                    if b3v >= cutlass.Int32(0):
                        if T > GMIN:
                            T3 = _fmaf(cutlass.Float32(b3v), w, SMIN)
                            if T3 < T:
                                t5 = T3
            s_tsh[0] = t5

        # ============ attempt loop (L1999-2162) — MUST NOT unroll ===========
        listN = cutlass.Int32(0)
        above = cutlass.Int32(0)
        m = cutlass.Int32(0)
        need = cutlass.Int32(0)
        B = cutlass.Int32(0)
        SC = cutlass.Float32(1.0)
        TF = T
        complete = cutlass.Int32(0)
        valid = cutlass.Int32(0)

        fr = [cute.make_fragment((4,), cutlass.Float32)
              for _ in range(U - PFD)]                   # explicit batch (op43 L1)
        # (empty for U<=PFD — every row-pass float4 then comes from pf[])
        att = cutlass.Int32(0)
        running = cutlass.Int32(1)
        while running != cutlass.Int32(0):
            if att > cutlass.Int32(0):                   # retry preamble L2005-2024
                # EXACTNESS: re-prime pf[] (stale roll data, L2006-2015)
                if rank < nFullG:
                    for uu in cutlass.range_constexpr(PFD):
                        C.ld_g_f32x4(
                            atom128, x_addr,
                            rank * cutlass.Int32(STEPC) + tidx
                            + cutlass.Int32(uu * BLK), pf[uu])
                else:
                    for uu in cutlass.range_constexpr(PFD):
                        i_ = (rank * cutlass.Int32(STEPC) + tidx
                              + cutlass.Int32(uu * BLK))
                        ic = i_
                        if ic >= n4:
                            ic = lim4
                        C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                C._cluster_sync_aligned()                # ==== clus.sync L2016 ====
                if tidx < cutlass.Int32(NBS):            # L2021-2022
                    s_hist[tidx] = cutlass.Int32(0)
                if tidx == cutlass.Int32(0):
                    s_scal[0] = cutlass.Int32(0)
                cute.arch.barrier()                      # ---- barrier L2024 ----

            TF = T                                       # window L2026-2031
            hi = C.fmax_f32(GMAX, T)
            if HIC > T:
                if HIC < hi:
                    hi = HIC
            WD = (hi - T) * cutlass.Float32(1.0 / 256.0)
            wdok = cutlass.Int32(0)
            if WD > cutlass.Float32(0.0):
                wdok = cutlass.Int32(1)
            if wdok == cutlass.Int32(0):
                WD = cutlass.Float32(1e-30)
            SC = cutlass.Float32(1.0) / WD

            # ---- P3 row pass over OWNED CHUNKS (L2033-2121) ----
            g = rank + cutlass.Int32(0)
            while g < nCh:
                i0 = g * cutlass.Int32(STEPC) + tidx
                M = cutlass.Int32(0)
                isfull = cutlass.Int32(0)
                if g < nFullG:
                    isfull = cutlass.Int32(1)
                if isfull != cutlass.Int32(0):           # full body L2042-2049
                    for uu in cutlass.range_constexpr(PFD, U):
                        C.ld_g_f32x4(atom128, x_addr,
                                     i0 + cutlass.Int32(uu * BLK), fr[uu - PFD])
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        for q in cutlass.range_constexpr(4):
                            M = M | (cutlass.Int32(vv[q] >= TF)
                                     << cutlass.Int32(uu * 4 + q))
                else:                                    # partial body L2051-2064
                    for uu in cutlass.range_constexpr(PFD, U):
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= n4:
                            ic = lim4                    # clamp in [n, npad)
                        C.ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        okq = cutlass.Int32(0)
                        if i_ < n4:
                            okq = cutlass.Int32(1)
                        if okq != cutlass.Int32(0):      # +inf-pad escape, ok-gated
                            for q in cutlass.range_constexpr(4):
                                M = M | (cutlass.Int32(vv[q] >= TF)
                                         << cutlass.Int32(uu * 4 + q))
                # ROLL THE PREFETCH FORWARD (L2066-2081): next OWNED chunk,
                # issued before the reservation and the survivor walk.
                g2 = g + cutlass.Int32(CS)
                if g2 < nCh:
                    j0 = g2 * cutlass.Int32(STEPC) + tidx
                    infull = cutlass.Int32(0)
                    if g2 < nFullG:
                        infull = cutlass.Int32(1)
                    if infull != cutlass.Int32(0):
                        for uu in cutlass.range_constexpr(PFD):
                            C.ld_g_f32x4(atom128, x_addr,
                                         j0 + cutlass.Int32(uu * BLK), pf[uu])
                    else:
                        for uu in cutlass.range_constexpr(PFD):
                            j_ = j0 + cutlass.Int32(uu * BLK)
                            jc = j_
                            if jc >= n4:
                                jc = lim4
                            C.ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                # warp-aggregated slot reservation (L2082-2095)
                cnt = cutlass.Int32(C.popc(M))
                inc = C.warp_incl_scan_add(cnt, lane)
                bpos = cutlass.Int32(0)
                if lane == cutlass.Int32(31):
                    if inc != cutlass.Int32(0):
                        bpos = C.atomic_add_cta(s_scal.iterator + 0, inc)
                pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) \
                    + (inc - cnt)
                # survivor bit-walk, software-pipelined ONE deep (L2105-2119);
                # reload X[idx] — never hold the U float4s across the walk
                if M != cutlass.Int32(0):
                    bp = C.ffs_m1(M)
                    M = M & (M - cutlass.Int32(1))
                    idx = ((i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                           << cutlass.Int32(2)) + (bp & cutlass.Int32(3))
                    xv = C.ldg_f32(x_addr, idx)
                    while M != cutlass.Int32(0):
                        bp2 = C.ffs_m1(M)
                        M = M & (M - cutlass.Int32(1))
                        idx2 = ((i0 + (bp2 >> cutlass.Int32(2))
                                 * cutlass.Int32(BLK))
                                << cutlass.Int32(2)) + (bp2 & cutlass.Int32(3))
                        xv2 = C.ldg_f32(x_addr, idx2)
                        pos = self._emitk(xv, idx, pos, TF, SC, SCAP,
                                          s_hist, s_cbuf2)
                        idx = idx2
                        xv = xv2
                    pos = self._emitk(xv, idx, pos, TF, SC, SCAP,
                                      s_hist, s_cbuf2)
                g = g + cutlass.Int32(CS)
            # rank-0 scalar tail (L2122-2130): per-thread atomics, bound-check
            i = tidx
            while i < tailn:
                x = C.ldg_f32(x_addr, tail0 + i)
                if x >= TF:
                    bq = C.f2s_rz((x - TF) * SC)         # signed form L2125
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                    post = C.atomic_add_cta(s_scal.iterator + 0,
                                            cutlass.Int32(1))
                    if post < SCAP:
                        s_cbuf2[post] = ((cutlass.Uint64(
                            cutlass.Uint32(tail0 + i)) << cutlass.Uint64(32))
                            | cutlass.Uint64(C.u32_of_f32(x)))
                i = i + cutlass.Int32(BLK)

            # ---- cluster merge (L2132-2148) ----
            C._cluster_sync_aligned()                    # ==== clus.sync L2135 ====
            myn = s_scal[0]                              # L2140
            self._merge_scan0_local(s_hist, s_mrg, rank, k, tidx, s_res)
            cute.arch.barrier()                          # ---- barrier L2143 ----
            tot = s_res[C.RES_TOT]
            acc = cutlass.Int32(0)
            if tot >= k:
                acc = cutlass.Int32(1)
            if acc != cutlass.Int32(0):                  # L2145-2148 accept
                valid = cutlass.Int32(1)
                complete = cutlass.Int32(0)
                if myn <= SCAP:
                    complete = cutlass.Int32(1)
                listN = myn
                above = s_res[C.RES_ABOVE]
                m = s_res[C.RES_M]
                need = k - s_res[C.RES_ABOVE]
                B = s_res[C.RES_B]
                running = cutlass.Int32(0)
            else:
                if att == cutlass.Int32(2):              # L2149
                    running = cutlass.Int32(0)
                else:
                    # rung ladder (L2150-2161) — cluster-uniform on every arm
                    tshtaken = cutlass.Int32(0)
                    if att == cutlass.Int32(0):
                        T5 = s_tsh[0]
                        if T5 > cutlass.Float32(_NEG_INF):
                            if T5 < TF:
                                T = T5
                                tshtaken = cutlass.Int32(1)
                    if tshtaken == cutlass.Int32(0):
                        # LAZY GATHER — every rank computes identical GMIN
                        if GMIN == cutlass.Float32(C.SENT_LO):
                            GMIN, GMAX = C.gather_hint(
                                x_addr, p_addr, k, n, tidx, s_wmn, s_wmx,
                                blk=BLK, kpt=1)          # 2 barriers inside
                        floorhit = cutlass.Int32(1)      # L2160
                        if T > GMIN:
                            floorhit = cutlass.Int32(0)
                        if floorhit != cutlass.Int32(0):
                            running = cutlass.Int32(0)
                        else:
                            T = GMIN                     # L2161
            att = att + cutlass.Int32(1)

        # ============ classification (L2165-2173) ============
        whole = cutlass.Int32(0)
        if valid != cutlass.Int32(0):
            if need >= m:
                whole = cutlass.Int32(1)
        lim1 = above
        if whole != cutlass.Int32(0):
            lim1 = above + m
        degen = cutlass.Int32(0)
        if valid == cutlass.Int32(0):
            degen = cutlass.Int32(1)
        if m > CMP:
            degen = cutlass.Int32(1)
        mc = cutlass.Int32(0)
        if degen == cutlass.Int32(0):
            mc = m
        # crossing candidates land in RANK 0's ck64c via DSMEM (L2173)
        rk64 = C._mapa_shared_cluster_addr(ck_addr, cutlass.Int32(0))

        if degen == cutlass.Int32(0):
            if complete != cutlass.Int32(0):
                # ---- P5 emit from staged cbuf (L2176-2187) ----
                i = tidx
                while i < listN:
                    pk64 = s_cbuf2[i]
                    vx = cutlass.Int32(cutlass.Uint32(
                        pk64 & cutlass.Uint64(0xFFFFFFFF)))
                    idv = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                    xv = C.f32_of_i32(vx)
                    self._p5_emit(xv, idv, TF, SC, B, above, lim1, whole,
                                  CMP, s_mrg, out_row, rk64)
                    i = i + cutlass.Int32(BLK)
            else:
                # ---- EXACTNESS re-sweep: OWNED CHUNKS + rank-0 true tail
                # (L2188-2222) ----
                g = rank + cutlass.Int32(0)
                while g < nCh:
                    lo2 = (g * cutlass.Int32(STEPC)) << cutlass.Int32(2)
                    e4 = (g + cutlass.Int32(1)) * cutlass.Int32(STEPC)
                    if e4 > n4:
                        e4 = n4
                    hi2 = e4 << cutlass.Int32(2)
                    i = lo2 + tidx
                    while i < hi2:
                        x = C.ldg_f32(x_addr, i)
                        if x >= TF:
                            self._p5_emit(x, i, TF, SC, B, above, lim1,
                                          whole, CMP, s_mrg, out_row, rk64)
                        i = i + cutlass.Int32(BLK)
                    g = g + cutlass.Int32(CS)
                t2 = tidx
                while t2 < tailn:
                    ii = tail0 + t2
                    x = C.ldg_f32(x_addr, ii)
                    if x >= TF:
                        self._p5_emit(x, ii, TF, SC, B, above, lim1,
                                      whole, CMP, s_mrg, out_row, rk64)
                    t2 = t2 + cutlass.Int32(BLK)

        # ============ EXIT RENDEZVOUS (L2226) ============
        # all DSMEM traffic retired; the ONLY exit rendezvous. rank!=0 falls
        # through to the kernel end (post-barrier asymmetric exit L2227);
        # NO later cluster barrier.
        C._cluster_sync_aligned()                        # ==== clus.sync L2226 ====

        if rank == cutlass.Int32(0):
            if degen == cutlass.Int32(0):
                if whole == cutlass.Int32(0):
                    # ---- P6 rank-0 refine (L2229-2303) ----
                    if mc <= cutlass.Int32(QUADC_CLUS):  # L2232-2247 O(mc^2)
                        mc2 = mc & cutlass.Int32(~1)
                        i = tidx
                        while i < mc:
                            # G1: re-assert Uint64 at every unsigned compare
                            # in/after dynamic loops (ct_main notes).
                            u64v = s_ck64[i]
                            r_ = cutlass.Int32(0)
                            jq = cutlass.Int32(0)
                            while jq < mc2:              # ulonglong2 16B reads
                                vlo, vhi = C._lds_v2_u64(
                                    ck_addr + jq * cutlass.Int32(8))
                                r_ = r_ + cutlass.Int32(vlo > cutlass.Uint64(u64v)) \
                                    + cutlass.Int32(vhi > cutlass.Uint64(u64v))
                                jq = jq + cutlass.Int32(2)
                            if mc2 < mc:                 # odd tail L2244
                                r_ = r_ + cutlass.Int32(
                                    cutlass.Uint64(s_ck64[mc2])
                                    > cutlass.Uint64(u64v))
                            if r_ < need:
                                out_row[above + r_] = cutlass.Int32(
                                    cutlass.Uint32(cutlass.Uint64(u64v)
                                                   & cutlass.Uint64(0xFFFFFFFF)))
                            i = i + cutlass.Int32(BLK)
                    else:
                        # key-space narrowing over ck64c (L2249-2278)
                        if tidx == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                            s_kmm[1] = cutlass.Uint32(0)
                        if tidx < cutlass.Int32(NBS):    # cleared ONCE L2251
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()              # ---- barrier L2255 ----
                        i = tidx
                        while i < mc:
                            kk = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                            C.atomic_min_cta(s_kmm.iterator + 0, kk)
                            C.atomic_max_cta(s_kmm.iterator + 1, kk)
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()              # ---- barrier L2257 ----
                        rlo = s_kmm[0]
                        rhi = s_kmm[1]
                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                        aboveC = cutlass.Int32(0)
                        needC = need
                        mm = mc
                        brk = cutlass.Int32(0)
                        lev = cutlass.Int32(0)
                        while brk == cutlass.Int32(0):   # L2258-2278 (<=6 lvl)
                            if needC == mm:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) \
                                    - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                brk = cutlass.Int32(1)
                            elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            elif lev >= cutlass.Int32(6):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            else:
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2_ = cutlass.Int32(32) - C.clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1)))
                                sh2 = b2_ - cutlass.Int32(self.lb)
                                if sh2 < cutlass.Int32(0):
                                    sh2 = cutlass.Int32(0)
                                sh2u = cutlass.Uint32(sh2)
                                i = tidx
                                while i < mc:            # L2266-2269 re-bin
                                    uq = cutlass.Uint32(
                                        s_ck64[i] >> cutlass.Uint64(32))
                                    if uq >= cutlass.Uint32(rlo):
                                        if uq <= cutlass.Uint32(rhi):
                                            du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                            if du > cutlass.Uint32(NBS - 1):
                                                du = cutlass.Uint32(NBS - 1)
                                            C.atomic_add_cta(
                                                s_hist.iterator
                                                + cutlass.Int32(du),
                                                cutlass.Int32(1))
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()      # ---- barrier L2270 ----
                                C.scan_cross0(s_hist, needC, tidx, s_res,
                                              cutlass.Int32(0),
                                              cutlass.Int32(0),
                                              s_hist, nb=NBS, zero=True)
                                cute.arch.barrier()      # ---- barrier L2273 ----
                                aboveC = aboveC + s_res[C.RES_ABOVE]
                                needC = needC - s_res[C.RES_ABOVE]
                                mm = s_res[C.RES_M]
                                sB = s_res[C.RES_B]
                                nlo = cutlass.Uint32(rlo) \
                                    + (cutlass.Uint32(sB) << sh2u)
                                if sB != cutlass.Int32(NBS - 1):   # L2276
                                    rhi = nlo + ((cutlass.Uint32(1) << sh2u)
                                                 - cutlass.Uint32(1))
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        if tidx == cutlass.Int32(0):     # L2279
                            s_scal[1] = cutlass.Int32(0)
                            s_scal[2] = cutlass.Int32(0)
                        cute.arch.barrier()              # ---- barrier L2281 ----
                        it2 = (mc + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                        it = cutlass.Int32(0)
                        while it < it2:                  # L2282-2301 ballot emit
                            i = it * cutlass.Int32(BLK) + tidx
                            p1 = cutlass.Int32(0)
                            p2 = cutlass.Int32(0)
                            idv = cutlass.Int32(0)
                            if i < mc:
                                w64 = s_ck64[i]
                                iu = cutlass.Int64(cutlass.Uint32(
                                    w64 >> cutlass.Uint64(32)))
                                idv = cutlass.Int32(cutlass.Uint32(
                                    w64 & cutlass.Uint64(0xFFFFFFFF)))
                                if iu > ethr:
                                    p1 = cutlass.Int32(1)
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                            self._ballot_pair_emit(
                                p1, p2, idv, above, aboveC,
                                above + aboveC, needC, out_row, s_scal, lane)
                            it = it + cutlass.Int32(1)
            else:
                # ---- degen fallback: whole-row key-space narrowing
                # (L2305-2355; per-level clear + scan_cross w/ ws) ----
                rlo = cutlass.Uint32(0)
                rhi = cutlass.Uint32(0xFFFFFFFF)
                above2 = cutlass.Int32(0)
                need2 = k
                m2 = n
                ethr = cutlass.Int64(0)
                tieM = cutlass.Int32(1)
                brk = cutlass.Int32(0)
                lev = cutlass.Int32(0)
                while brk == cutlass.Int32(0):           # L2312-2331 (<=8 lvl)
                    if need2 == m2:
                        ethr = cutlass.Int64(cutlass.Uint32(rlo)) \
                            - cutlass.Int64(1)
                        above2 = above2 + m2
                        need2 = cutlass.Int32(0)
                        tieM = cutlass.Int32(0)
                        brk = cutlass.Int32(1)
                    elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                        brk = cutlass.Int32(1)
                    elif lev >= cutlass.Int32(8):
                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                        brk = cutlass.Int32(1)
                    else:
                        d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                        b2_ = cutlass.Int32(32) - C.clz_i32(
                            cutlass.Int32(d2 | cutlass.Uint32(1)))
                        sh2 = b2_ - cutlass.Int32(self.lb)
                        if sh2 < cutlass.Int32(0):
                            sh2 = cutlass.Int32(0)
                        sh2u = cutlass.Uint32(sh2)
                        if tidx < cutlass.Int32(NBS):    # per-level clear L2320
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()              # ---- barrier L2321 ----
                        i = tidx
                        while i < n:                     # L2322-2324 whole row
                            uq = C.fkey(C.ldg_f32(x_addr, i))
                            if uq >= cutlass.Uint32(rlo):
                                if uq <= cutlass.Uint32(rhi):
                                    du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                    if du > cutlass.Uint32(NBS - 1):
                                        du = cutlass.Uint32(NBS - 1)
                                    C.atomic_add_cta(
                                        s_hist.iterator + cutlass.Int32(du),
                                        cutlass.Int32(1))
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()              # ---- barrier L2325 ----
                        # block-parallel scan (ONE internal barrier; only use
                        # of ws in this kernel, L2326)
                        C.scan_cross(s_hist, s_ws, need2, tidx, s_res,
                                     cutlass.Int32(0), blk=BLK, nb=NBS,
                                     two=False)
                        cute.arch.barrier()              # ---- barrier L2327 ----
                        above2 = above2 + s_res[C.RES_ABOVE]
                        need2 = need2 - s_res[C.RES_ABOVE]
                        m2 = s_res[C.RES_M]
                        sB = s_res[C.RES_B]
                        nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                        if sB != cutlass.Int32(NBS - 1):
                            rhi = nlo + ((cutlass.Uint32(1) << sh2u)
                                         - cutlass.Uint32(1))
                        rlo = nlo
                        lev = lev + cutlass.Int32(1)
                if tidx == cutlass.Int32(0):             # L2336
                    s_scal[1] = cutlass.Int32(0)
                    s_scal[2] = cutlass.Int32(0)
                cute.arch.barrier()                      # ---- barrier L2337 ----
                nA = k                                   # L2338
                nT = cutlass.Int32(0)
                if tieM != cutlass.Int32(0):
                    nA = above2
                    nT = need2
                it2 = (n + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                it = cutlass.Int32(0)
                while it < it2:                          # L2340-2354
                    i = it * cutlass.Int32(BLK) + tidx
                    p1 = cutlass.Int32(0)
                    p2 = cutlass.Int32(0)
                    if i < n:
                        uq = C.fkey(C.ldg_f32(x_addr, i))
                        iu = cutlass.Int64(uq)
                        if iu > ethr:
                            p1 = cutlass.Int32(1)
                        if tieM != cutlass.Int32(0):
                            if iu == ethr:
                                p2 = cutlass.Int32(1)
                    self._ballot_pair_emit(p1, p2, i, cutlass.Int32(0),
                                           nA, nA, nT, out_row, s_scal, lane)
                    it = it + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # host launcher: grid dim3(CS, b) + cluster (CS,1,1) (probe P11);
    # min_blocks_per_mp=1 == __launch_bounds__(1024, 1) 64-reg wall.
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, logits: cute.Tensor, pre_idx: cute.Tensor,
                 out: cute.Tensor,
                 n: cutlass.Int32, npad: cutlass.Int32, k: cutlass.Int32,
                 SCAP: cutlass.Int32, CMP: cutlass.Int32, SMP: cutlass.Int32,
                 TGT: cutlass.Int32, Q: cutlass.Int32, SS2: cutlass.Int32,
                 TGT2: cutlass.Int32, stream):
        b = logits.shape[0]
        self.kern(logits, pre_idx, out, n, npad, k, SCAP, CMP, SMP,
                  TGT, Q, SS2, TGT2).launch(
            grid=(self.cs, b, 1),
            block=(self.blk, 1, 1),
            cluster=(self.cs, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.minb)


# ---------------------------------------------------------------------------
# compile cache + torch-facing entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE = {}


def get_compiled(tpl, scap: int = 8192, cmp_: int = 2048,
                 options_extra: str = ""):
    """Compile (or fetch) the gvr_clus variant for constexpr tuple
    tpl = (BLK, U, MINB, NBS, CS); scap/cmp are smem-extent keys (every
    reachable route has 8192/2048 — asserted by run())."""
    key = (tuple(tpl), scap, cmp_, options_extra)
    hit = _COMPILE_CACHE.get(key)
    if hit is not None:
        return hit
    blk, u, minb, nbs, cs = tpl
    kern = GvrClusKernel(blk, u, minb, nbs, cs, scap=scap, cmp_=cmp_)
    r0, c0 = cute.sym_int(), cute.sym_int()
    r1, c1 = cute.sym_int(), cute.sym_int()
    r2, c2 = cute.sym_int(), cute.sym_int()
    logits_fake = _crt.make_fake_compact_tensor(
        cutlass.Float32, (r0, c0), stride_order=(1, 0), assumed_align=16)
    pre_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r1, c1), stride_order=(1, 0), assumed_align=16)
    out_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r2, c2), stride_order=(1, 0), assumed_align=16)
    fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kern, logits_fake, pre_fake, out_fake,
                            *([cutlass.Int32(0)] * 10),
                            stream=fake_stream,
                            options=("--enable-tvm-ffi " + options_extra).strip())
    _COMPILE_CACHE[key] = compiled
    return compiled


def run(logits, pre_idx, n: int, out):
    """torch-facing single-call entry: routes (b, n, k) through ct_dispatch,
    asserts the shape lands on gvr_clus, launches the matching variant.
    gvr_clus takes NO workspace (spec §4c)."""
    import ct_dispatch
    b, npad = logits.shape
    k = pre_idx.shape[1]
    r = ct_dispatch.route(b, int(n), npad, k)
    assert r['kernel'] == 'clus', f"shape routes to {r['kernel']}, not gvr_clus"
    rt = r['rt']
    kobj = GvrClusKernel(*r['tpl'], scap=rt['SCAP'], cmp_=rt['CMP'])
    assert r['smem'] == kobj.dyn_bytes, (r['smem'], kobj.dyn_bytes)
    fn = get_compiled(tuple(r['tpl']), scap=rt['SCAP'], cmp_=rt['CMP'])
    fn(logits, pre_idx, out,
       rt['n'], rt['npad'], rt['k'], rt['SCAP'], rt['CMP'], rt['SMP'],
       rt['TGT'], rt['Q'], rt['SS2'], rt['TGT2'])
    return r


def run_manual(logits, pre_idx, n: int, out, tpl, rt):
    """Manual-lattice entry for route()-unreachable (U, CS) members: launches
    tpl with caller-supplied runtime scalars (must be route()-consistent for
    the same CS; U only changes the chunk geometry)."""
    fn = get_compiled(tuple(tpl), scap=rt['SCAP'], cmp_=rt['CMP'])
    fn(logits, pre_idx, out,
       rt['n'], rt['npad'], rt['k'], rt['SCAP'], rt['CMP'], rt['SMP'],
       rt['TGT'], rt['Q'], rt['SS2'], rt['TGT2'])
