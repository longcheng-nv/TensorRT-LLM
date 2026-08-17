# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ct_main.py — op46 gvr_main (streaming self-sampling GVR) CuTeDSL port.

Ground truth: src_cuda/kernel.cu L377-1265 (frozen); phase contract, smem map
and barrier inventory per TRANSLATION_SPEC.md §5.1; every DSL spelling pinned
by probes/PROBE_RESULTS.md (BINDING) and op43 idioms (L1..L5 lessons).

Ctor knobs (compile-time, mirror of the CUDA template params, spec §4d):
    BLK ∈ {1024, 512, 256}, U ∈ {1,2,4,8}, MINB ∈ {1,2,4}, NBS = 256,
    KPT ∈ {1,2,4,8}, SPLIT ∈ {True, False}
Derived constexprs (kernel.cu L394-523, bit-identical):
    HB=NBS; KBIG=(KPT>=2 && KPT*BLK>=2048); SCPB=(BLK>=1024)?(SPLIT?8192:16384)
    :(KBIG?8192:4096); CMPB=(BLK>=1024)?(KBIG?4096:2048):1024; SHD=!SPLIT;
    VSTG=SPLIT||BLK>=512; PFD=(MINB<=2)?min(U,4):0; PF=PFD>0; NATT=SPLIT?1:3.

Signature (ABI parity with kernel.cu L379-382 incl. dead SCAP_/CMP_):
    run(logits[b,npad] f32, pre_idx[b,k] i32, out[b,k] i32,
        n, npad, k, SCAP_, CMP_, R, SMP, TGT, Q, SS2, TGT2, ws)
Grid dim3(R, b) native 2-D (probe P11); block BLK; min_blocks_per_mp=MINB is
the 64-register wall (probes P2/P15); smem via one SmemAllocator blob (all
extents compile-time), dynamic-equivalent region byte-identical to the host
formula L3149: (SCPB+4)*(VSTG?8:4) + (CMPB+1)*8.

int2 staging convention: an int2 (x=value bits, y=index) is ONE little-endian
Uint64 = (idx << 32) | value_bits, so cbuf2 / g_buf traffic is single u64
ld/st (mirrors the CUDA ST.64/LD.64; __ldcg = ct_common._ldcg_v2_i32).

Barrier inventory implemented (kernel.cu line cites, checklist per op43 L5):
    L555, L655, L664, [retry: L753], L909,
    SPLIT: L924, [overflow: L935], L956, L961, [last: L983, L985]
    non-split: L992, [ladder: L1008 | L1017]
    P6: L1075, L1097, L1100, per-level L1116/L1118, L1125
    degen A: L1162, per-level L1177/L1179, L1186
    degen B: L1221, per-level L1234/L1236, L1243
    + exactly 2 inside each gather_hint expansion (L349/L357, ct_common).
scan_cross0 contains NO barrier (probe P14 protocol).
"""

import os
import sys

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm as mlir_llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator
from cutlass.cute import runtime as _crt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import ct_common as C  # noqa: E402  (FROZEN sibling — import only)

MAXC = C.MAXC
GCAP = C.GCAP
IDXB = C.IDXB
IDXM = C.IDXM
QUADC_CLUS = C.QUADC_CLUS
WS_BYTES = C.GVR_WS_BUF_OFF + MAXC * GCAP * 8  # 20,973,568 (kernel.cu L44-46)

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# single-rounding fma.rn.f32 (probe P6 emit spelling; ct_common has no fma).
# Used at every CUDA fmaf() site: T/Tk/T3 rung math (L665/L689/L722), HIC
# (L708), window terms. (x-T)*SC classify shapes stay plain sub+mul (P6:
# structurally uncontractible).
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(mlir_math.fma(
        a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
        c.ir_value(loc=loc, ip=ip),
        fastmath=mlir_arith.FastMathFlags.none, loc=loc, ip=ip))


def _st_g_u64(addr_i64, val_u64):
    """plain st.global.u64 (slab publish L929/945/952, g_don restore L969)."""
    p = cute.make_ptr(cutlass.Uint64, addr_i64, cute.AddressSpace.gmem,
                      assumed_align=8)
    t = cute.make_tensor(p, cute.make_layout((1,)))
    t[0] = val_u64


def _st_g_u32(addr_i64, val_i32):
    """plain st.global.u32 (g_off restore L969)."""
    p = cute.make_ptr(cutlass.Int32, addr_i64, cute.AddressSpace.gmem,
                      assumed_align=4)
    t = cute.make_tensor(p, cute.make_layout((1,)))
    t[0] = val_i32


@dsl_user_op
def _st_s_v2_u32(saddr_i32, lo_u32, hi_u32, *, loc=None, ip=None):
    """st.shared.v2.u32 [saddr], {lo, hi} — the CUDA make_int2 STS.64 spelling
    (kernel.cu L1018-1019). Byte-identical to the little-endian u64 pack
    ((hi << 32) | lo) but keeps the two words as independent 32-bit registers,
    so ptxas can coalesce the emission bit-walk's loop-carried (xv, idx) pair
    straight into the store pair (drops 2 IMAD.MOV/iter; op46 SASS diff)."""
    mlir_llvm.inline_asm(
        res=None,
        operands_=[saddr_i32.ir_value(loc=loc, ip=ip),
                   lo_u32.ir_value(loc=loc, ip=ip),
                   hi_u32.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared.v2.u32 [$0], {$1, $2};",
        constraints="r,r,r",
        has_side_effects=True,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip)


@dsl_user_op
def _pin_i64(v, *, loc=None, ip=None):
    """Opaque identity mov.b64: pins a loop-invariant Int64 so NVVM cannot
    rematerialize its defining chain (param ld.const + %ctaid reads + mul/add)
    into every scf region body (PTX $L__BB0_123 evidence, op46 SASS diff)."""
    return cutlass.Int64(
        mlir_llvm.inline_asm(
            T.i64(),
            [v.ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, $1;",
            "=l,l", has_side_effects=False, is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _pin_i32(v, *, loc=None, ip=None):
    """Opaque identity mov.b32 (Int32 twin of _pin_i64)."""
    return cutlass.Int32(
        mlir_llvm.inline_asm(
            T.i32(),
            [v.ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


class GvrMainKernel:
    """CuTeDSL port of gvr_main<BLK, U, MINB, NBS, KPT, SPLIT> (kernel.cu L377)."""

    def __init__(self, blk: int, u: int, minb: int, nbs: int, kpt: int,
                 split: bool, tshg: bool = False):
        assert nbs == 256, "SNB must stay 256 (kernel.cu L170-177, measured)"
        assert blk in (256, 512, 1024) and u in (1, 2, 4, 8)
        assert kpt in (1, 2, 4, 8) and minb in (1, 2, 4)
        self.blk = blk
        self.u = u
        self.minb = minb
        self.nbs = nbs
        self.kpt = kpt
        self.split = bool(split)
        # knife5 (layer 7): TSH-floor staging arm.  SPLIT-only compile-time
        # key; the CUDA form is a grid-uniform runtime gate over the same
        # predicate (b > 15 && k <= 1024 && n4 <= 32768).
        self.tshg = bool(tshg) and bool(split)
        # derived constexprs (kernel.cu L394-523)
        self.hb = nbs                                            # L394
        self.kbig = (kpt >= 2) and (kpt * blk >= 2048)           # L413
        self.scpb = ((8192 if split else 16384) if blk >= 1024
                     else (8192 if self.kbig else 4096))         # L422-423
        self.cmpb = ((4096 if self.kbig else 2048) if blk >= 1024
                     else 1024)                                  # L424
        self.shd = not split                                     # L438
        self.vstg = split or blk >= 512                          # L445
        self.pfd = (u if u < 4 else 4) if minb <= 2 else 0       # L522
        self.pf = self.pfd > 0                                   # L523
        self.natt = 1 if split else 3                            # L733
        # smem blob byte map (kernel.cu L447-456): cbuf/cbuf2 alias @0,
        # ck64 @ 4*(VSTG ? 2*(SCPB+4) : SCPB+4), size (CMPB+1)*8
        self.ck_off = 4 * ((2 * (self.scpb + 4)) if self.vstg
                           else (self.scpb + 4))
        assert self.ck_off % 16 == 0, "ck64 must stay 16B aligned (ulonglong2)"
        self.dyn_bytes = self.ck_off + (self.cmpb + 1) * 8
        self.lb = self.nbs.bit_length() - 1                      # log2(NBS)=8

    # ------------------------------------------------------------------
    # GVR_EMITC (kernel.cu L869-883): classify+stage one survivor.
    # Returns pos+1. Branchless trash slot min(pos, SCPB) (L866-868).
    # ------------------------------------------------------------------
    @cute.jit
    def _emitc(self, xv, idx, pos, TF, SC, s_hist, s_cbuf, s_cbuf2):
        SCPB = self.scpb
        NBS = self.nbs
        if cutlass.const_expr(not self.split):
            bn_u = C.f2u_rz((xv - TF) * SC)          # saturating cvt.rzi (P4)
            if bn_u > cutlass.Uint32(NBS - 1):
                bn_u = cutlass.Uint32(NBS - 1)
            bn = cutlass.Int32(bn_u)
            C.atomic_add_cta(s_hist.iterator + bn, cutlass.Int32(1))
            if cutlass.const_expr(not self.vstg):
                ps = pos
                if ps > cutlass.Int32(SCPB):
                    ps = cutlass.Int32(SCPB)         # trash slot (IMNMX)
                s_cbuf[ps] = cutlass.Int32(
                    (bn_u << cutlass.Uint32(IDXB)) | cutlass.Uint32(idx))
        if cutlass.const_expr(self.vstg):
            ps = pos
            if ps > cutlass.Int32(SCPB):
                ps = cutlass.Int32(SCPB)
            # int2 {value bits, idx} via st.shared.v2.u32 — same bytes as the
            # former (idx << 32) | bits u64 pack (+0=bits, +4=idx), but no i64
            # materialization inside the bit-walk (kernel.cu L1018-1019 parity)
            _st_s_v2_u32(s_cbuf2.iterator.toint() + ps * cutlass.Int32(8),
                         C.u32_of_f32(xv), cutlass.Uint32(idx))
        return pos + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # two-predicate warp-ballot emit step (shared by P6 L1124-1146 and both
    # degen emits L1185-1210 / L1242-1263): q1 winners to out[base1+p] p<cap1,
    # q2 ties to out[base2+p] p<cap2. s_scal[1]=s_o1, s_scal[2]=s_o2.
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
             out: cute.Tensor, ws: cute.Tensor,
             n: cutlass.Int32, npad: cutlass.Int32, k: cutlass.Int32,
             scap_dead: cutlass.Int32, cmp_dead: cutlass.Int32,
             R: cutlass.Int32, SMP: cutlass.Int32, TGT: cutlass.Int32,
             Q: cutlass.Int32, SS2: cutlass.Int32, TGT2: cutlass.Int32):
        BLK = self.blk
        U = self.u
        NBS = self.nbs
        KPT = self.kpt
        SCPB = self.scpb
        CMPB = self.cmpb
        PFD = self.pfd
        NATT = self.natt
        NW = BLK // 32

        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()               # 2-D grid (part,row) L467-468
        row = by
        part = cutlass.Int32(0)
        if cutlass.const_expr(self.split):
            part = bx
        lane = tidx & cutlass.Int32(31)

        # ---- shared memory (one blob, compile-time offsets; spec §5.1 map) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((self.hb,), order=(0,)),
            byte_alignment=128)                          # L446 __align__(16)
        s_ws = smem.allocate_tensor(                     # L458 (unused; byte parity)
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16)
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16)
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)),
            byte_alignment=16)
        # ct_common crossing-scan result slots (RES_B/M/ABOVE/TOT/B2/B3)
        s_res = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)),
            byte_alignment=16)
        # scalar block: [0]=s_bufn [1]=s_o1 [2]=s_o2 [3]=s_base (L459)
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)),
            byte_alignment=16)
        s_pk = smem.allocate_tensor(                     # L460
            cutlass.Int64, cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=8)
        s_tsh = smem.allocate_tensor(                    # L462
            cutlass.Float32, cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=4)
        s_kmm = smem.allocate_tensor(                    # L463 [0]=kmin [1]=kmax
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)),
            byte_alignment=8)
        blob = smem.allocate_tensor(                     # dynamic-equivalent L447-456
            cutlass.Int8, cute.make_ordered_layout((self.dyn_bytes,), order=(0,)),
            byte_alignment=16)
        sbase = blob.iterator.toint()
        s_cbuf = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, sbase, cute.AddressSpace.smem,
                          assumed_align=16),
            cute.make_layout((SCPB + 4,)))
        s_cbuf2 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, sbase, cute.AddressSpace.smem,
                          assumed_align=16),
            cute.make_layout((SCPB + 4,)))
        ck_addr = sbase + cutlass.Int32(self.ck_off)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, ck_addr, cute.AddressSpace.smem,
                          assumed_align=16),
            cute.make_layout((CMPB + 1,)))

        # ---- row bases (L472-475) ----
        row64 = cutlass.Int64(row)
        # _pin_i64: keep the row base a REGISTER across the attempt/tile scf
        # regions (NVVM otherwise re-derives ld.param+%ctaid.y+mul per region)
        x_addr = _pin_i64(logits.iterator.toint()
                          + row64 * cutlass.Int64(npad) * cutlass.Int64(4))
        p_addr = pre_idx.iterator.toint() + row64 * cutlass.Int64(k) * cutlass.Int64(4)
        out_row = out[row, None]
        ws_addr = ws.iterator.toint()
        gdon_addr = ws_addr                              # L386-388 slab views
        goff_addr = ws_addr + cutlass.Int64(C.GVR_WS_OFF_OFF)
        gbuf_addr = ws_addr + cutlass.Int64(C.GVR_WS_BUF_OFF)

        n4 = n >> cutlass.Int32(2)                       # L477
        c0 = cutlass.Int32(0)
        c1 = n4
        if cutlass.const_expr(self.split):               # L478-479
            c0 = part * Q
            c1 = c0 + Q
            if c1 > n4:
                c1 = n4
        tail0 = n4 << cutlass.Int32(2)                   # L480
        tailn = cutlass.Int32(0)
        if part == cutlass.Int32(0):                     # L481
            tailn = n - tail0

        if tidx == cutlass.Int32(0):                     # L483
            s_scal[0] = cutlass.Int32(0)                 # s_bufn
            s_res[C.RES_B2] = cutlass.Int32(-1)
            s_res[C.RES_B3] = cutlass.Int32(-1)
        if tidx < cutlass.Int32(self.hb):                # L484-487 (HB<=BLK always)
            s_hist[tidx] = cutlass.Int32(0)

        # ============ P1: sample prefetch (hint gather LAZY, L489-529) =======
        atom128 = C.g2r_atom_f32(128, invariant=True)
        fsa = cute.make_fragment((4,), cutlass.Float32)
        fsb = cute.make_fragment((4,), cutlass.Float32)
        shas = cutlass.Int32(0)
        if tidx < SMP:
            shas = cutlass.Int32(1)
        if shas != cutlass.Int32(0):                     # L502-504
            p4 = tidx * SS2 * cutlass.Int32(2)
            C.ld_g_f32x4(atom128, x_addr, p4, fsa)
            C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)

        # ============ P2: quantile rung from the sample (L531-727) ===========
        smn = cutlass.Float32(float("inf"))              # L538
        smx = cutlass.Float32(float("-inf"))
        if shas != cutlass.Int32(0):                     # L539-543
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fsa[t])
                smx = C.fmax_f32(smx, fsa[t])
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fsb[t])
                smx = C.fmax_f32(smx, fsb[t])
        fma_ = cute.make_fragment((4,), cutlass.Float32)  # strided-tail pair bufs
        fmb_ = cute.make_fragment((4,), cutlass.Float32)
        j = tidx + cutlass.Int32(BLK)                    # L544-550 strided tail
        while j < SMP:
            p4 = j * SS2 * cutlass.Int32(2)
            C.ld_g_f32x4(atom128, x_addr, p4, fma_)
            C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fma_[t])
                smx = C.fmax_f32(smx, fma_[t])
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fmb_[t])
                smx = C.fmax_f32(smx, fmb_[t])
            j = j + cutlass.Int32(BLK)
        a0 = C.warp_min_u32(C.fkey(smn))                 # L551-554
        c0m = C.warp_max_u32(C.fkey(smx))
        if lane == cutlass.Int32(0):
            s_wmn[tidx >> cutlass.Int32(5)] = a0
            s_wmx[tidx >> cutlass.Int32(5)] = c0m
        cute.arch.barrier()                              # ---- barrier L555 ----

        # PRIME-LATE prefetch block (L556-616): strictly after the barrier.
        lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)   # L524
        pf = [cute.make_fragment((4,), cutlass.Float32) for _ in range(max(PFD, 1))]
        if cutlass.const_expr(self.pf):
            fullsl = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(BLK * U):
                fullsl = cutlass.Int32(1)
            if fullsl != cutlass.Int32(0):               # L557-559 prime, full slice
                for uu in cutlass.range_constexpr(PFD):
                    C.ld_g_f32x4(atom128, x_addr,
                                 c0 + tidx + cutlass.Int32(uu * BLK), pf[uu])
            else:                                        # L561-562 clamped prime
                for uu in cutlass.range_constexpr(PFD):
                    i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                    ic = i_
                    if ic >= c1:
                        ic = lim4
                    C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
            # asm prefetch site #1 (L573-577): gate (c1-c0)>=2*BLK*U && SMP>=160
            g1 = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(2 * BLK * U):
                if SMP >= cutlass.Int32(160):
                    g1 = cutlass.Int32(1)
            if g1 != cutlass.Int32(0):
                for uu in cutlass.range_constexpr(PFD, U):
                    C._prefetch_l2(x_addr + cutlass.Int64(
                        c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16))
        if cutlass.const_expr((not self.pf) and (not self.split)):
            fullsl = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(BLK * U):
                fullsl = cutlass.Int32(1)
            if fullsl != cutlass.Int32(0):               # site #2 (L589-592)
                for uu in cutlass.range_constexpr(U):
                    C._prefetch_l2(x_addr + cutlass.Int64(
                        c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16))
            else:
                if SMP > cutlass.Int32(0):               # site #3 knife4-L1 (L593-615)
                    for uu in cutlass.range_constexpr(U):
                        i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= c1:
                            ic = lim4
                        C._prefetch_l2(x_addr + cutlass.Int64(ic) * cutlass.Int64(16))

        # cross-warp sample reduce (L617-623)
        av = cutlass.Uint32(0xFFFFFFFF)
        cv = cutlass.Uint32(0)
        if lane < cutlass.Int32(NW):
            av = s_wmn[lane]
            cv = s_wmx[lane]
        SMIN = C.invkey(C.warp_min_u32(av))
        SMAX = C.invkey(C.warp_max_u32(cv))

        GMIN = cutlass.Float32(C.SENT_LO)                # L629 sentinels
        GMAX = cutlass.Float32(C.SENT_HI)
        T = cutlass.Float32(_NEG_INF)
        HIC = cutlass.Float32(_NEG_INF)
        w = cutlass.Float32(0.0)
        sok = cutlass.Int32(0)                           # L633
        if SMP > cutlass.Int32(0):
            if SMAX > SMIN:
                sok = cutlass.Int32(1)
        if sok != cutlass.Int32(0):                      # L635-654 sample histogram
            w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
            # rcp.approx.ftz.f32 = the CUDA arm's --use_fast_math 1.0f/w
            # (bare MUFU.RCP, no Newton refinement) — bitwise-aligned scale
            sc_s = cute.arch.rcp_approx(w)
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
            j = tidx + cutlass.Int32(BLK)                # L646-653 tail re-loads
            while j < SMP:
                p4 = j * SS2 * cutlass.Int32(2)
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
        cute.arch.barrier()                              # ---- barrier L655 ----
        # triple-target ZERO scan (L659-660): TGT / TGT2 / 2*TGT
        # (THREE = SHD || gated-SPLIT, knife5 layer 7)
        C.scan_cross0(s_hist, TGT, tidx, s_res, TGT2, TGT * cutlass.Int32(2),
                      s_hist, nb=NBS, zero=True, two=True,
                      three=(self.shd or self.tshg))
        cute.arch.barrier()                              # ---- barrier L664 ----

        tot0 = s_res[C.RES_TOT]
        b1v = s_res[C.RES_B]
        if sok != cutlass.Int32(0):                      # L665
            if tot0 >= TGT:
                T = _fmaf(cutlass.Float32(b1v), w, SMIN)
        Trung = T                                        # L666 snapshot
        needg = cutlass.Int32(1)                         # L667-675 degenerate sample
        if T > cutlass.Float32(_NEG_INF):
            needg = cutlass.Int32(0)
        if needg != cutlass.Int32(0):
            GMIN, GMAX = C.gather_hint(x_addr, p_addr, k, n, tidx, s_wmn,
                                       s_wmx, blk=BLK, kpt=KPT)  # 2 barriers inside
            T = GMIN
        if sok != cutlass.Int32(0):                      # L688-709 HIC tighten
            if tot0 >= TGT:
                b2v = s_res[C.RES_B2]
                if b2v >= cutlass.Int32(0):
                    Tk = _fmaf(cutlass.Float32(b2v), w, SMIN)
                    anc = T
                    if cutlass.const_expr(not self.split):
                        anc = C.fmin_f32(T, Trung)       # L703
                    d_ = C.fmax_f32(Tk - anc, cutlass.Float32(0.0))
                    HIC = C.fmax_f32(
                        _fmaf(cutlass.Float32(4.0), d_, T),
                        _fmaf(cutlass.Float32(8.0), w, T))   # L708
        if cutlass.const_expr(self.shd or self.tshg):    # TSH floor (knife5: +gated SPLIT)
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

        if cutlass.const_expr(self.tshg):
            # knife5 (layer 7) TSH-FLOOR STAGING: SPLIT has no retry ladder,
            # so a rung overshoot (count(>=T) < k) used to hand the LAST CTA
            # a single-CTA whole-row narrowing.  Stage at the sample's
            # rank-(2*TGT) floor instead: staged population ~aim -> ~2*aim,
            # and the merged histogram contains the k-crossing whenever
            # count(>=TSH) >= k.  TSH miss falls to GMIN/degen unchanged.
            cute.arch.barrier()
            t5s = s_tsh[0]
            if t5s > cutlass.Float32(_NEG_INF):
                if t5s < T:
                    T = t5s

        # ============ attempt loop (L729-1019) — MUST NOT unroll ============
        listN = cutlass.Int32(0)
        above = cutlass.Int32(0)
        m = cutlass.Int32(0)
        need = cutlass.Int32(0)
        B = cutlass.Int32(0)
        SC = cutlass.Float32(1.0)
        TF = T
        complete = cutlass.Int32(0)
        valid = cutlass.Int32(0)
        fromg = cutlass.Int32(0)
        alive = cutlass.Int32(1)

        fr = [cute.make_fragment((4,), cutlass.Float32)
              for _ in range(max(U - PFD, 1))]           # explicit batch (op43 L1)
        att = cutlass.Int32(0)
        running = cutlass.Int32(1)
        while running != cutlass.Int32(0):
            if cutlass.const_expr(not self.split):       # SPLIT never retries (NATT=1)
                if att > cutlass.Int32(0):               # retry reset (L737-754)
                    if cutlass.const_expr(self.pf):
                        # EXACTNESS: re-prime pf[] (stale roll data, L738-749)
                        fullsl = cutlass.Int32(0)
                        if (c1 - c0) >= cutlass.Int32(BLK * U):
                            fullsl = cutlass.Int32(1)
                        if fullsl != cutlass.Int32(0):
                            for uu in cutlass.range_constexpr(PFD):
                                C.ld_g_f32x4(atom128, x_addr,
                                             c0 + tidx + cutlass.Int32(uu * BLK), pf[uu])
                        else:
                            for uu in cutlass.range_constexpr(PFD):
                                i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                                ic = i_
                                if ic >= c1:
                                    ic = lim4
                                C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                    if tidx < cutlass.Int32(NBS):        # L750-751
                        s_hist[tidx] = cutlass.Int32(0)
                    if tidx == cutlass.Int32(0):         # L752
                        s_scal[0] = cutlass.Int32(0)
                    cute.arch.barrier()                  # ---- barrier L753 ----

            TF = T                                       # window (L756-761)
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
            # NOT rcp_approx: on (256,8,4,·,4) the single-inst rcp shifts SC's
            # live range and spills 5 LDL/STL at the 64-reg wall (bisected);
            # div.rn here is the original validated-exact spelling
            SC = cutlass.Float32(1.0) / WD

            # ---- P3 row pass (L763-908) ----
            span = c1 - c0
            step = cutlass.Int32(BLK * U)
            nFull = cutlass.Int32(0)
            rem = cutlass.Int32(0)
            if span > cutlass.Int32(0):                  # L776-779 peel
                nFull = span // step
                rem = span - nFull * step
            # _pin_i32: the isfull peel predicate reads nFull every tile iter;
            # unpinned, NVVM re-derives the whole ld.param+shr/sel div chain
            # at the loop head (v3 SASS evidence)
            nFull = _pin_i32(nFull)
            nIt = nFull
            if rem > cutlass.Int32(0):
                nIt = nIt + cutlass.Int32(1)
            # _pin_i32: stop NVVM re-deriving the ceil-div bound (ld.param n +
            # shr/sel chain) inside the tile-loop condition region per iter
            nIt = _pin_i32(nIt)

            it = cutlass.Int32(0)
            while it < nIt:
                i0 = c0 + it * step + tidx
                M = cutlass.Int32(0)
                isfull = cutlass.Int32(0)
                if it < nFull:
                    isfull = cutlass.Int32(1)
                if isfull != cutlass.Int32(0):           # full body (L783-795)
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
                else:                                    # partial body (L796-815)
                    for uu in cutlass.range_constexpr(PFD, U):
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= c1:
                            ic = lim4                    # clamped address (P10)
                        C.ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        okq = cutlass.Int32(0)
                        if i_ < c1:
                            okq = cutlass.Int32(1)
                        if okq != cutlass.Int32(0):      # ok-gated (+inf pad fix L804-813)
                            for q in cutlass.range_constexpr(4):
                                M = M | (cutlass.Int32(vv[q] >= TF)
                                         << cutlass.Int32(uu * 4 + q))
                # prefetch roll-forward BEFORE reservation/walk (L816-842)
                if cutlass.const_expr(self.pf):
                    hasnext = cutlass.Int32(0)
                    if it + cutlass.Int32(1) < nIt:
                        hasnext = cutlass.Int32(1)
                    if hasnext != cutlass.Int32(0):
                        j0 = i0 + step
                        infull = cutlass.Int32(0)        # warp-uniform peel L835
                        if it + cutlass.Int32(1) < nFull:
                            infull = cutlass.Int32(1)
                        if infull != cutlass.Int32(0):
                            for uu in cutlass.range_constexpr(PFD):
                                C.ld_g_f32x4(atom128, x_addr,
                                             j0 + cutlass.Int32(uu * BLK), pf[uu])
                        else:
                            for uu in cutlass.range_constexpr(PFD):
                                j_ = j0 + cutlass.Int32(uu * BLK)
                                jc = j_
                                if jc >= c1:
                                    jc = lim4
                                C.ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                # warp-aggregated reservation (L843-854)
                cnt = cutlass.Int32(C.popc(M))
                inc = C.warp_incl_scan_add(cnt, lane)
                bpos = cutlass.Int32(0)
                if lane == cutlass.Int32(31):
                    if inc != cutlass.Int32(0):
                        bpos = C.atomic_add_cta(s_scal.iterator + 0, inc)
                pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                # survivor bit-walk, software-pipelined ONE deep (L884-898);
                # reload X[idx] — do NOT hold the U float4s (+18% spill L855-859)
                if M != cutlass.Int32(0):
                    bp = C.ffs_m1(M)
                    M = M & (M - cutlass.Int32(1))
                    idx = ((i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                           << cutlass.Int32(2)) + (bp & cutlass.Int32(3))
                    xv = C.ldg_f32(x_addr, idx)
                    while M != cutlass.Int32(0):
                        bp2 = C.ffs_m1(M)
                        M = M & (M - cutlass.Int32(1))
                        idx2 = ((i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                << cutlass.Int32(2)) + (bp2 & cutlass.Int32(3))
                        xv2 = C.ldg_f32(x_addr, idx2)
                        pos = self._emitc(xv, idx, pos, TF, SC,
                                          s_hist, s_cbuf, s_cbuf2)
                        idx = idx2
                        xv = xv2
                    pos = self._emitc(xv, idx, pos, TF, SC,
                                      s_hist, s_cbuf, s_cbuf2)
                it = it + cutlass.Int32(1)
            # scalar tail, part 0 only (L900-906)
            i = tidx
            while i < tailn:
                x = C.ldg_f32(x_addr, tail0 + i)
                if x >= TF:
                    post = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                    post = self._emitc(x, tail0 + i, post, TF, SC,
                                       s_hist, s_cbuf, s_cbuf2)
                i = i + cutlass.Int32(BLK)
            cute.arch.barrier()                          # ---- barrier L909 ----
            myn = s_scal[0]                              # L911

            if cutlass.const_expr(self.split):
                # ---- SLAB HAND-OFF (L913-989); exactly ONE attempt ----
                if tidx == cutlass.Int32(0):             # L923
                    pgo = cute.make_ptr(cutlass.Int32,
                                        goff_addr + row64 * cutlass.Int64(4),
                                        cute.AddressSpace.gmem, assumed_align=4)
                    s_scal[3] = cutlass.Int32(cute.arch.atomic_add(pgo, myn))
                cute.arch.barrier()                      # ---- barrier L924 ----
                base = s_scal[3]
                if myn <= cutlass.Int32(SCPB):           # L926-930 coalesced publish
                    i = tidx
                    while i < myn:
                        p = base + i
                        if p < cutlass.Int32(GCAP):
                            _st_g_u64(gbuf_addr + (row64 * cutlass.Int64(GCAP)
                                                   + cutlass.Int64(p)) * cutlass.Int64(8),
                                      s_cbuf2[i])
                        i = i + cutlass.Int32(BLK)
                else:                                    # L931-955 overflow re-sweep
                    if tidx == cutlass.Int32(0):
                        s_scal[0] = cutlass.Int32(0)
                    cute.arch.barrier()                  # ---- barrier L935 ----
                    lo2 = c0 << cutlass.Int32(2)
                    hi2 = c1 << cutlass.Int32(2)
                    i = lo2 + tidx
                    while i < hi2:
                        x = C.ldg_f32(x_addr, i)
                        if x >= TF:
                            pq = C.atomic_add_cta(s_scal.iterator + 0,
                                                  cutlass.Int32(1))
                            p = base + pq
                            if p < cutlass.Int32(GCAP):
                                _st_g_u64(gbuf_addr + (row64 * cutlass.Int64(GCAP)
                                                       + cutlass.Int64(p)) * cutlass.Int64(8),
                                          (cutlass.Uint64(cutlass.Uint32(i))
                                           << cutlass.Uint64(32))
                                          | cutlass.Uint64(C.u32_of_f32(x)))
                        i = i + cutlass.Int32(BLK)
                    i = tidx                             # true tail (L948-954)
                    while i < tailn:
                        x = C.ldg_f32(x_addr, tail0 + i)
                        if x >= TF:
                            pq = C.atomic_add_cta(s_scal.iterator + 0,
                                                  cutlass.Int32(1))
                            p = base + pq
                            if p < cutlass.Int32(GCAP):
                                _st_g_u64(gbuf_addr + (row64 * cutlass.Int64(GCAP)
                                                       + cutlass.Int64(p)) * cutlass.Int64(8),
                                          (cutlass.Uint64(cutlass.Uint32(tail0 + i))
                                           << cutlass.Uint64(32))
                                          | cutlass.Uint64(C.u32_of_f32(x)))
                        i = i + cutlass.Int32(BLK)
                cute.arch.barrier()                      # ---- barrier L956 ----
                if tidx == cutlass.Int32(0):             # L959-960 release + RMW
                    C.threadfence_gpu()
                    pdon = cute.make_ptr(cutlass.Int64,
                                         gdon_addr + row64 * cutlass.Int64(8),
                                         cute.AddressSpace.gmem, assumed_align=8)
                    s_pk[0] = C.atomic_add_u64_gpu(
                        pdon, cutlass.Int64(1 << 32) + cutlass.Int64(myn))
                cute.arch.barrier()                      # ---- barrier L961 ----
                pk = s_pk[0]
                alive = cutlass.Int32(0)                 # L963 last-CTA test
                if cutlass.Int32(pk >> cutlass.Int64(32)) == R - cutlass.Int32(1):
                    alive = cutlass.Int32(1)
                if alive != cutlass.Int32(0):
                    C.threadfence_gpu()                  # L968 acquire
                    if tidx == cutlass.Int32(0):         # L969 ZERO-RESTORE
                        _st_g_u32(goff_addr + row64 * cutlass.Int64(4),
                                  cutlass.Int32(0))
                        _st_g_u64(gdon_addr + row64 * cutlass.Int64(8),
                                  cutlass.Uint64(0))
                    total = cutlass.Int32(pk & cutlass.Int64(0xFFFFFFFF)) + myn
                    if total <= cutlass.Int32(GCAP):     # L971-988 one-pass consume
                        listN = total
                        if total > cutlass.Int32(SCPB):
                            fromg = cutlass.Int32(1)
                        i = tidx
                        while i < listN:
                            gvx, gvy = C._ldcg_v2_i32(
                                gbuf_addr + (row64 * cutlass.Int64(GCAP)
                                             + cutlass.Int64(i)) * cutlass.Int64(8))
                            if fromg == cutlass.Int32(0):
                                s_cbuf2[i] = ((cutlass.Uint64(cutlass.Uint32(gvy))
                                               << cutlass.Uint64(32))
                                              | cutlass.Uint64(cutlass.Uint32(gvx)))
                            bq = C.f2s_rz((C.f32_of_i32(gvx) - TF) * SC)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()              # ---- barrier L983 ----
                        C.scan_cross0(s_hist, k, tidx, s_res, cutlass.Int32(0),
                                      cutlass.Int32(0), s_hist, nb=NBS, zero=False)
                        cute.arch.barrier()              # ---- barrier L985 ----
                        if s_res[C.RES_TOT] >= k:        # L986-987
                            valid = cutlass.Int32(1)
                            complete = cutlass.Int32(1)
                            above = s_res[C.RES_ABOVE]
                            m = s_res[C.RES_M]
                            need = k - above
                            B = s_res[C.RES_B]
                running = cutlass.Int32(0)               # L989 break (NATT==1)
            else:
                # ---- non-split verify + rung ladder (L990-1018) ----
                C.scan_cross0(s_hist, k, tidx, s_res, cutlass.Int32(0),
                              cutlass.Int32(0), s_hist, nb=NBS, zero=False)
                cute.arch.barrier()                      # ---- barrier L992 ----
                tot = s_res[C.RES_TOT]
                acc = cutlass.Int32(0)
                if tot >= k:
                    acc = cutlass.Int32(1)
                if acc != cutlass.Int32(0):              # L994-998 accept
                    valid = cutlass.Int32(1)
                    complete = cutlass.Int32(0)
                    if myn <= cutlass.Int32(SCPB):
                        complete = cutlass.Int32(1)
                    listN = myn
                    above = s_res[C.RES_ABOVE]
                    m = s_res[C.RES_M]
                    need = k - above
                    B = s_res[C.RES_B]
                    running = cutlass.Int32(0)
                else:
                    if att == cutlass.Int32(NATT - 1):   # L999 ladder exhausted
                        running = cutlass.Int32(0)
                    else:
                        tshtaken = cutlass.Int32(0)      # L1005-1010 TSH retry
                        if cutlass.const_expr(self.shd):
                            if att == cutlass.Int32(0):
                                T5 = s_tsh[0]
                                if T5 > cutlass.Float32(_NEG_INF):
                                    if T5 < TF:
                                        T = T5
                                        tshtaken = cutlass.Int32(1)
                        if tshtaken != cutlass.Int32(0):
                            cute.arch.barrier()          # ---- barrier L1008 ----
                        else:
                            # LAZY GATHER (L1014, sentinel equality flag)
                            if GMIN == cutlass.Float32(C.SENT_LO):
                                GMIN, GMAX = C.gather_hint(
                                    x_addr, p_addr, k, n, tidx, s_wmn, s_wmx,
                                    blk=BLK, kpt=KPT)
                            floorhit = cutlass.Int32(1)  # L1015
                            if T > GMIN:
                                floorhit = cutlass.Int32(0)
                            if floorhit != cutlass.Int32(0):
                                running = cutlass.Int32(0)
                            else:
                                T = GMIN                 # L1016
                                cute.arch.barrier()      # ---- barrier L1017 ----
            att = att + cutlass.Int32(1)

        # ============ classification (L1021-1024) ============
        if alive != cutlass.Int32(0):
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
            if m > cutlass.Int32(CMPB):
                degen = cutlass.Int32(1)
            mc = cutlass.Int32(0)
            if degen == cutlass.Int32(0):
                mc = m

            if degen == cutlass.Int32(0):
                # ---- P5 cursor emit (L1026-1071) ----
                if complete != cutlass.Int32(0):         # L1028-1048
                    i = tidx
                    while i < listN:
                        idv = cutlass.Int32(0)
                        bq = cutlass.Int32(0)
                        xv = cutlass.Float32(0.0)
                        if cutlass.const_expr(self.vstg):
                            vx = cutlass.Int32(0)
                            vy = cutlass.Int32(0)
                            if cutlass.const_expr(self.split):
                                if fromg != cutlass.Int32(0):
                                    vx, vy = C._ldcg_v2_i32(
                                        gbuf_addr + (row64 * cutlass.Int64(GCAP)
                                                     + cutlass.Int64(i)) * cutlass.Int64(8))
                                else:
                                    pk64 = s_cbuf2[i]
                                    vx = cutlass.Int32(cutlass.Uint32(
                                        pk64 & cutlass.Uint64(0xFFFFFFFF)))
                                    vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                            else:
                                pk64 = s_cbuf2[i]
                                vx = cutlass.Int32(cutlass.Uint32(
                                    pk64 & cutlass.Uint64(0xFFFFFFFF)))
                                vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                            xv = C.f32_of_i32(vx)
                            idv = vy
                            bq = C.f2s_rz((xv - TF) * SC)   # L1034
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                        else:
                            wpk = cutlass.Uint32(s_cbuf[i])  # L1036-1037
                            idv = cutlass.Int32(wpk & cutlass.Uint32(IDXM))
                            bq = cutlass.Int32(wpk >> cutlass.Uint32(IDXB))
                        if bq >= B:                      # L1039-1047
                            p = C.atomic_add_cta(s_hist.iterator + bq,
                                                 cutlass.Int32(1))
                            if p < lim1:
                                out_row[p] = idv
                            else:
                                if whole == cutlass.Int32(0):
                                    q2 = p - above
                                    if q2 < cutlass.Int32(CMPB):
                                        if cutlass.const_expr(self.vstg):
                                            kk = C.fkey(xv)
                                        else:
                                            kk = C.fkey(C.ldg_f32(x_addr, idv))
                                        s_ck64[q2] = ((cutlass.Uint64(kk)
                                                       << cutlass.Uint64(32))
                                                      | cutlass.Uint64(
                                                          cutlass.Uint32(idv)))
                        i = i + cutlass.Int32(BLK)
                else:
                    # collect overflow: scalar re-sweep, exact tail remap
                    # (L1049-1070) — zero extra live registers by design
                    lo2 = c0 << cutlass.Int32(2)
                    hi2 = c1 << cutlass.Int32(2)
                    i0_ = lo2 + tidx
                    while i0_ < hi2 + tailn:
                        i_ = i0_
                        if i0_ >= hi2:
                            i_ = tail0 + (i0_ - hi2)     # L1057-1058
                        x = C.ldg_f32(x_addr, i_)
                        if x >= TF:
                            bq = C.f2s_rz((x - TF) * SC)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            if bq >= B:
                                p = C.atomic_add_cta(s_hist.iterator + bq,
                                                     cutlass.Int32(1))
                                if p < lim1:
                                    out_row[p] = i_
                                else:
                                    if whole == cutlass.Int32(0):
                                        q2 = p - above
                                        if q2 < cutlass.Int32(CMPB):
                                            s_ck64[q2] = ((cutlass.Uint64(C.fkey(x))
                                                           << cutlass.Uint64(32))
                                                          | cutlass.Uint64(
                                                              cutlass.Uint32(i_)))
                        i0_ = i0_ + cutlass.Int32(BLK)

                # ---- P6 refine (L1073-1147) ----
                if whole == cutlass.Int32(0):
                    cute.arch.barrier()                  # ---- barrier L1075 ----
                    if mc <= cutlass.Int32(QUADC_CLUS):  # L1077-1092 O(mc^2) rank
                        mc2 = mc & cutlass.Int32(~1)
                        i = tidx
                        while i < mc:
                            # NOTE: values crossing a dynamic-while region are
                            # re-wrapped SIGNED by the DSL — every u64 compare
                            # must re-assert Uint64 at the USE site (found via
                            # odd-tail rank corruption; see notes G1).
                            u64v = s_ck64[i]
                            r_ = cutlass.Int32(0)
                            jq = cutlass.Int32(0)
                            while jq < mc2:              # ulonglong2 16B reads
                                vlo, vhi = C._lds_v2_u64(
                                    ck_addr + jq * cutlass.Int32(8))
                                r_ = r_ + cutlass.Int32(vlo > cutlass.Uint64(u64v)) \
                                    + cutlass.Int32(vhi > cutlass.Uint64(u64v))
                                jq = jq + cutlass.Int32(2)
                            if mc2 < mc:                 # odd tail L1089
                                r_ = r_ + cutlass.Int32(
                                    cutlass.Uint64(s_ck64[mc2])
                                    > cutlass.Uint64(u64v))
                            if r_ < need:
                                out_row[above + r_] = cutlass.Int32(
                                    cutlass.Uint32(cutlass.Uint64(u64v)
                                                   & cutlass.Uint64(0xFFFFFFFF)))
                            i = i + cutlass.Int32(BLK)
                    else:
                        # key-space narrowing over ck64 (L1094-1123)
                        if tidx == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                            s_kmm[1] = cutlass.Uint32(0)
                        if tidx < cutlass.Int32(NBS):    # cleared ONCE (L1096)
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()              # ---- barrier L1097 ----
                        i = tidx
                        while i < mc:
                            kk = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                            C.atomic_min_cta(s_kmm.iterator + 0, kk)
                            C.atomic_max_cta(s_kmm.iterator + 1, kk)
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()              # ---- barrier L1100 ----
                        rlo = s_kmm[0]
                        rhi = s_kmm[1]
                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                        aboveC = cutlass.Int32(0)
                        needC = need
                        mm = mc
                        brk = cutlass.Int32(0)
                        lev = cutlass.Int32(0)
                        while brk == cutlass.Int32(0):   # L1103-1123 (<=6 levels)
                            if needC == mm:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
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
                                while i < mc:            # L1111-1115 re-bin
                                    uq = cutlass.Uint32(
                                        s_ck64[i] >> cutlass.Uint64(32))
                                    if uq >= cutlass.Uint32(rlo):
                                        if uq <= cutlass.Uint32(rhi):
                                            du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                            if du > cutlass.Uint32(NBS - 1):
                                                du = cutlass.Uint32(NBS - 1)
                                            C.atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(du),
                                                cutlass.Int32(1))
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()      # ---- barrier L1116 ----
                                C.scan_cross0(s_hist, needC, tidx, s_res,
                                              cutlass.Int32(0), cutlass.Int32(0),
                                              s_hist, nb=NBS, zero=True)
                                cute.arch.barrier()      # ---- barrier L1118 ----
                                aboveC = aboveC + s_res[C.RES_ABOVE]
                                needC = needC - s_res[C.RES_ABOVE]
                                mm = s_res[C.RES_M]
                                sB = s_res[C.RES_B]
                                nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                if sB != cutlass.Int32(NBS - 1):   # L1121
                                    rhi = nlo + ((cutlass.Uint32(1) << sh2u)
                                                 - cutlass.Uint32(1))
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        if tidx == cutlass.Int32(0):     # L1124
                            s_scal[1] = cutlass.Int32(0)
                            s_scal[2] = cutlass.Int32(0)
                        cute.arch.barrier()              # ---- barrier L1125 ----
                        it2 = (mc + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                        it = cutlass.Int32(0)
                        while it < it2:                  # L1127-1146 ballot emit
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
                dga = cutlass.Int32(0)                   # L1156 gate: valid && complete
                if valid != cutlass.Int32(0):
                    if complete != cutlass.Int32(0):
                        dga = cutlass.Int32(1)
                if dga != cutlass.Int32(0):
                    # ---- degen A: narrowing over STAGED candidates
                    # (L1150-1212) ----
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    m2 = listN
                    ethr = cutlass.Int64(0)
                    tieM = cutlass.Int32(1)
                    if tidx < cutlass.Int32(NBS):        # L1161
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()                  # ---- barrier L1162 ----
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == cutlass.Int32(0):       # L1163-1184 (<=8 levels)
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
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
                            i = tidx
                            while i < listN:             # L1170-1176
                                uq = cutlass.Uint32(0)
                                if cutlass.const_expr(self.vstg):
                                    vx = cutlass.Int32(0)
                                    vy = cutlass.Int32(0)
                                    if cutlass.const_expr(self.split):
                                        if fromg != cutlass.Int32(0):
                                            vx, vy = C._ldcg_v2_i32(
                                                gbuf_addr
                                                + (row64 * cutlass.Int64(GCAP)
                                                   + cutlass.Int64(i))
                                                * cutlass.Int64(8))
                                        else:
                                            pk64 = s_cbuf2[i]
                                            vx = cutlass.Int32(cutlass.Uint32(
                                                pk64 & cutlass.Uint64(0xFFFFFFFF)))
                                    else:
                                        pk64 = s_cbuf2[i]
                                        vx = cutlass.Int32(cutlass.Uint32(
                                            pk64 & cutlass.Uint64(0xFFFFFFFF)))
                                    uq = C.fkey_bits(cutlass.Uint32(vx))
                                else:
                                    id0 = cutlass.Int32(
                                        cutlass.Uint32(s_cbuf[i])
                                        & cutlass.Uint32(IDXM))
                                    uq = C.fkey(C.ldg_f32(x_addr, id0))
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        C.atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du),
                                            cutlass.Int32(1))
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()          # ---- barrier L1177 ----
                            C.scan_cross0(s_hist, need2, tidx, s_res,
                                          cutlass.Int32(0), cutlass.Int32(0),
                                          s_hist, nb=NBS, zero=True)
                            cute.arch.barrier()          # ---- barrier L1179 ----
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
                    if tidx == cutlass.Int32(0):         # L1185
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()                  # ---- barrier L1186 ----
                    nA = k                               # L1187
                    nT = cutlass.Int32(0)
                    if tieM != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (listN + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:                      # L1189-1210
                        i = it * cutlass.Int32(BLK) + tidx
                        p1 = cutlass.Int32(0)
                        p2 = cutlass.Int32(0)
                        idv = cutlass.Int32(0)
                        if i < listN:
                            uq = cutlass.Uint32(0)
                            if cutlass.const_expr(self.vstg):
                                vx = cutlass.Int32(0)
                                vy = cutlass.Int32(0)
                                if cutlass.const_expr(self.split):
                                    if fromg != cutlass.Int32(0):
                                        vx, vy = C._ldcg_v2_i32(
                                            gbuf_addr
                                            + (row64 * cutlass.Int64(GCAP)
                                               + cutlass.Int64(i))
                                            * cutlass.Int64(8))
                                    else:
                                        pk64 = s_cbuf2[i]
                                        vx = cutlass.Int32(cutlass.Uint32(
                                            pk64 & cutlass.Uint64(0xFFFFFFFF)))
                                        vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                else:
                                    pk64 = s_cbuf2[i]
                                    vx = cutlass.Int32(cutlass.Uint32(
                                        pk64 & cutlass.Uint64(0xFFFFFFFF)))
                                    vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                uq = C.fkey_bits(cutlass.Uint32(vx))
                                idv = vy
                            else:
                                idv = cutlass.Int32(cutlass.Uint32(s_cbuf[i])
                                                    & cutlass.Uint32(IDXM))
                                uq = C.fkey(C.ldg_f32(x_addr, idv))
                            iu = cutlass.Int64(uq)
                            if iu > ethr:
                                p1 = cutlass.Int32(1)
                            if tieM != cutlass.Int32(0):
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                        self._ballot_pair_emit(p1, p2, idv, cutlass.Int32(0),
                                               nA, nA, nT, out_row, s_scal, lane)
                        it = it + cutlass.Int32(1)
                else:
                    # ---- degen B: whole-row narrowing (L1214-1264) ----
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    m2 = n
                    ethr = cutlass.Int64(0)
                    tieM = cutlass.Int32(1)
                    if tidx < cutlass.Int32(NBS):        # L1220
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()                  # ---- barrier L1221 ----
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == cutlass.Int32(0):       # L1222-1241 (<=8 levels)
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
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
                            i = tidx
                            while i < n:                 # L1229-1233 whole row
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
                            cute.arch.barrier()          # ---- barrier L1234 ----
                            C.scan_cross0(s_hist, need2, tidx, s_res,
                                          cutlass.Int32(0), cutlass.Int32(0),
                                          s_hist, nb=NBS, zero=True)
                            cute.arch.barrier()          # ---- barrier L1236 ----
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
                    if tidx == cutlass.Int32(0):         # L1242
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()                  # ---- barrier L1243 ----
                    nA = k
                    nT = cutlass.Int32(0)
                    if tieM != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (n + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:                      # L1246-1263
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
    # host launcher (grid dim3(R, b) L2750; MINB wall via min_blocks_per_mp)
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, logits: cute.Tensor, pre_idx: cute.Tensor,
                 out: cute.Tensor, ws: cute.Tensor,
                 n: cutlass.Int32, npad: cutlass.Int32, k: cutlass.Int32,
                 scap_dead: cutlass.Int32, cmp_dead: cutlass.Int32,
                 R: cutlass.Int32, SMP: cutlass.Int32, TGT: cutlass.Int32,
                 Q: cutlass.Int32, SS2: cutlass.Int32, TGT2: cutlass.Int32,
                 stream):
        b = logits.shape[0]
        self.kern(logits, pre_idx, out, ws, n, npad, k, scap_dead, cmp_dead,
                  R, SMP, TGT, Q, SS2, TGT2).launch(
            grid=(R, b, 1),
            block=(self.blk, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.minb)


# ---------------------------------------------------------------------------
# compile cache + torch-facing entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE = {}


def get_compiled(tpl, options_extra: str = ""):
    """Compile (or fetch) the gvr_main variant for constexpr tuple
    tpl = (BLK, U, MINB, NBS, KPT, SPLIT, TSHG)."""
    key = (tuple(tpl), options_extra)
    hit = _COMPILE_CACHE.get(key)
    if hit is not None:
        return hit
    blk, u, minb, nbs, kpt, split, tshg = tpl
    kern = GvrMainKernel(blk, u, minb, nbs, kpt, bool(split), bool(tshg))
    r0, c0 = cute.sym_int(), cute.sym_int()
    r1, c1 = cute.sym_int(), cute.sym_int()
    r2, c2 = cute.sym_int(), cute.sym_int()
    w0 = cute.sym_int()
    logits_fake = _crt.make_fake_compact_tensor(
        cutlass.Float32, (r0, c0), stride_order=(1, 0), assumed_align=16)
    pre_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r1, c1), stride_order=(1, 0), assumed_align=16)
    out_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r2, c2), stride_order=(1, 0), assumed_align=16)
    ws_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (w0,), stride_order=(0,), assumed_align=16)
    fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kern, logits_fake, pre_fake, out_fake, ws_fake,
                            *([cutlass.Int32(0)] * 11),
                            stream=fake_stream,
                            options=("--enable-tvm-ffi " + options_extra).strip())
    _COMPILE_CACHE[key] = compiled
    return compiled


def workspace_bytes() -> int:
    return WS_BYTES


def run(logits, pre_idx, n: int, out, ws):
    """torch-facing single-call entry: routes (b, n, k) through ct_dispatch,
    asserts the shape lands on gvr_main, launches the matching variant.
    ws: zero-initialised >=20,973,568-B CUDA buffer (reused across launches;
    the kernel restores the zeros it consumes)."""
    import ct_dispatch
    b, npad = logits.shape
    k = pre_idx.shape[1]
    r = ct_dispatch.route(b, int(n), npad, k)
    assert r['kernel'] == 'main', f"shape routes to {r['kernel']}, not gvr_main"
    assert ws.numel() * ws.element_size() >= WS_BYTES
    rt = r['rt']
    fn = get_compiled(tuple(r['tpl']))
    fn(logits, pre_idx, out, ws,
       rt['n'], rt['npad'], rt['k'], rt['SCAP_'], rt['CMP_'], rt['R'],
       rt['SMP'], rt['TGT'], rt['Q'], rt['SS2'], rt['TGT2'])
    return r
