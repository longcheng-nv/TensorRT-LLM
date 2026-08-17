# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ct_reg.py — op46 gvr_topk_reg CuTeDSL translation.

CUDA ground truth: src_cuda/kernel.cu L1269-1776 (register-resident exact
top-K, one CTA per row, histogram bins in FLOAT space). Contract:
TRANSLATION_SPEC.md §5.2. All probe verdicts (probes/PROBE_RESULTS.md) are
binding; op43 lessons L1-L5 applied throughout.

Template knobs (CUDA `gvr_topk_reg<BLK,VPT,MINB,KPT,CUR,DEG,IMGF,NBH>`):
ctor args of :class:`GvrTopkRegKernel`, read as `cutlass.const_expr(self.x)`
inside the kernel. Runtime args mirror the CUDA `(n, npad, k, CMP, IMGOFF,
QC)` — npad/k come from tensor shapes, IMGOFF is dropped (dispatch pins
IMGOFF == NBSEL == NBH at every site, asserted in the host wrapper).

Shared-memory map (single dynamic window, word offsets; the CUDA static
__shared__ block is folded into the first 512 B so occupancy accounting
matches nvcc's static+dynamic sum):

    [0..5]        s_res    (ct_common slot map RES_B/M/ABOVE/TOT/B2/B3)
    [6..7]        s_cnt    (s_o1, s_oc)
    [8..9]        s_kmm    (s_kmin, s_kmax — Uint32)
    [10..11]      s_e12    (s_e1, s_e2)
    [16..16+NW)   ws       (scan_cross_w workspace)
    [48..48+NW)   wmn      (Uint32 warp min partials)
    [80..80+NW)   wmx      (Uint32 warp max partials)
    [128..128+NBH)          hist   (kernel.cu L1296)
    [128+NBH..128+NBH+CMP)  ck     (Uint32 crossing keys; CMP dynamic)
    [128+NBH+CMP..+2CMP)    ci     (Int32 crossing indices)
    img/bm alias ck at word 128+NBH (kernel.cu L1299/L1409; IMGOFF==NBH)

Launch smem = 512 + dispatch_smem_bytes (dynamic Int32).

TOOLCHAIN GOTCHA (documented in notes/ct_reg_NOTES.md): dynamic launch smem
with min_blocks_per_mp>1 crashes cutlass_dsl._build_kernel_attrs (host ceil()
on a dynamic value while computing the PREFERRED_SHARED_MEMORY_CARVEOUT
hint). `_no_carveout()` scopes a monkeypatch around cute.compile dropping
ONLY that hint (CUDA __launch_bounds__ sets no carveout either);
`.reqntid`/`.minnctapersm` (the register wall) are unaffected — verified.
"""

import contextlib
import os
import sys

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cutlass_dsl import T, dsl_user_op

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_common import (  # noqa: E402
    RES_ABOVE,
    RES_B,
    RES_M,
    SENT_HI,
    SENT_LO,
    atomic_add_cta,
    atomic_max_cta,
    atomic_min_cta,
    atomic_or_cta,
    ballot,
    clz_i32,
    f2s_rz,
    f2u_rz,
    ffs_m1,
    find_cross,
    fkey,
    fmax_f32,
    fmin_f32,
    g2r_atom_f32,
    invkey,
    ld_g_f32x4,
    ld_g_i32,
    ldg_f32,
    popc,
    scan_cross_w,
    warp_add_i32,
    warp_incl_scan_add2,
    warp_max_u32,
    warp_min_u32,
)

NB = 1024                      # kernel.cu L16 (NBH default)
STATIC_WORDS = 128             # DSL smem prelude (static-__shared__ mirror)
STATIC_BYTES = STATIC_WORDS * 4
_NEG_INF = float("-inf")
_POS_INF = float("inf")


# ---------------------------------------------------------------------------
# module-local FP spellings (probe P6)
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf(a, b, c, *, loc=None, ip=None):
    """CUDA fmaf: single fma.rn.f32 (P6 emit spelling)."""
    return cutlass.Float32(mlir_math.fma(
        a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
        c.ir_value(loc=loc, ip=ip),
        fastmath=mlir_arith.FastMathFlags.none, loc=loc, ip=ip))


@dsl_user_op
def _submul_asm(v, t, sc, *, loc=None, ip=None):
    """(v - t) * sc with two roundings, opaque to CSE/contraction (P6).

    Used at the !BRL classify site (kernel.cu L1521) so no sub-expression is
    shared with the emit's `fmaf(v - T, SC, OFF)` — the CUDA deliberately
    spells the two sites differently to stop nvcc holding all S q's live
    across the barrier (L1506-1509).
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [v.ir_value(loc=loc, ip=ip), t.ir_value(loc=loc, ip=ip),
         sc.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .f32 rtmp;\n\tsub.rn.f32 rtmp, $1, $2;\n\tmul.rn.f32 $0, rtmp, $3;\n\t}",
        "=f,f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _smem_addr_reg(addr, *, loc=None, ip=None):
    """Pin a CTA-shared 32-bit byte address in ONE register (SASS audit fix).

    Identity `mov` behind an asm boundary: without it LLVM re-folds the
    `mov.b32 %r, __dynamic_shmem__0` symbol materialisation into EVERY use
    site, and ptxas then re-derives the CGA shared window (S2UR SR_CgaCtaId
    + UMOV + ULEA, 3 instructions) inside each divergent classify block —
    measured +24 warp-instructions/warp vs the CUDA arm, which keeps the
    base in one UR. The asm result is not duplicable, so the window is
    materialised exactly once. Value-identical: a plain register copy.
    """
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [addr.ir_value(loc=loc, ip=ip)],
        "mov.u32 $0, $1;",
        "=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _red_shared_add1(addr, *, loc=None, ip=None):
    """CUDA classify `atomicAdd(&hist[bin], 1u)` with the result unused.

    `red` (not `atom`) is the result-less spelling — ptxas lowers it to the
    same ATOMS.POPC.INC.32 RZ the CUDA arm emits (kernel.cu L1521-1526).
    Same ordering contract as atomic_add_cta: .relaxed scope .cta. Takes the
    final shared byte address as a plain Int32 so the address datapath stays
    ordinary IR (ptxas fuses the shl+add into one LEA against the pinned
    `_smem_addr_reg` base).
    """
    llvm.inline_asm(
        res=None,
        operands_=[addr.ir_value(loc=loc, ip=ip)],
        asm_string="red.relaxed.cta.shared.add.u32 [$0], 1;",
        constraints="r", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


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


def _f32_smem_atom():
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)


def _sts128_f32(atom, frag, base_addr, byte_off):
    p = cute.make_ptr(cutlass.Float32, base_addr + byte_off,
                      cute.AddressSpace.smem, assumed_align=16)
    cute.copy(atom, frag, cute.make_tensor(p, cute.make_layout((4,))))


def _smem_view(dtype, sbase, word_off: int, length: int, align: int = 16):
    """Typed tensor view at a constexpr word offset into the smem window."""
    p = cute.make_ptr(dtype, sbase + cutlass.Int32(word_off * 4),
                      cute.AddressSpace.smem, assumed_align=align)
    return cute.make_tensor(p, cute.make_layout((length,)))


def _val(frags, s: int):
    """val[s] accessor over the float4[VPT] register batch (constexpr s)."""
    return frags[s // 4][s % 4]


@contextlib.contextmanager
def _no_carveout():
    """Scoped: drop the DSL's carveout hint (see module docstring)."""
    import cutlass.cutlass_dsl.cutlass as _cdsl
    orig = _cdsl._build_kernel_attrs
    _cdsl._build_kernel_attrs = lambda config: {}
    try:
        yield
    finally:
        _cdsl._build_kernel_attrs = orig


class GvrTopkRegKernel:
    """gvr_topk_reg<BLK, VPT, MINB, KPT, CUR, DEG, IMGF, NBH> (kernel.cu L1287)."""

    def __init__(self, blk: int, vpt: int, minb: int, kpt: int,
                 cur: bool, deg: bool, img: bool, nbh: int = NB,
                 pdl: bool = False):
        assert blk in (256, 512, 1024) and vpt in (1, 2, 4)
        assert nbh in (256, 512, 1024, 2048)
        assert nbh % blk == 0 or blk % nbh == 0
        self.blk = blk
        self.vpt = vpt
        self.minb = minb
        self.kpt = kpt
        self.cur = bool(cur)
        self.deg = bool(deg)
        self.img = bool(img)
        self.nbh = nbh
        self.pdl = bool(pdl)
        # derived compile-time constants (kernel.cu L1292-1294, L1359, L1389, L1485)
        self.S = vpt * 4
        self.lnbh = {256: 8, 512: 9, 2048: 11}.get(nbh, 10)
        self.use_bm = (not deg) and (not img) and kpt >= 2 and vpt == 1
        self.use_img = img and vpt == 1
        self.brl = (minb * blk <= 1024) or (vpt == 1)

    # ------------------------------------------------------------------
    @cute.kernel
    def kern(self, logits: cute.Tensor, pre_idx: cute.Tensor, out: cute.Tensor,
             n: cutlass.Int32, cmp_: cutlass.Int32, qc: cutlass.Int32,
             smem_bytes: cutlass.Int32):
        BLK = cutlass.const_expr(self.blk)
        VPT = cutlass.const_expr(self.vpt)
        KPT = cutlass.const_expr(self.kpt)
        NBH = cutlass.const_expr(self.nbh)
        S = cutlass.const_expr(self.S)
        LNBH = cutlass.const_expr(self.lnbh)
        NW = cutlass.const_expr(self.blk // 32)

        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()          # L1291 (knob default off)

        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid & cutlass.Int32(31)

        # ------------------------------------------------------------------
        # Predeclarations: the DSL AST transformer requires every scalar that
        # is (re)assigned under a dynamic if/while region to pre-exist with a
        # stable type at every enclosing region level. Constant inits are
        # sunk/dead-coded by LLVM, so this costs no registers.
        # ------------------------------------------------------------------
        i = cutlass.Int32(0)
        j = cutlass.Int32(0)
        r = cutlass.Int32(0)
        tinc = cutlass.Int32(0)
        cnt = cutlass.Int32(0)
        bit = cutlass.Int32(0)
        abv = cutlass.Int32(0)
        nA = cutlass.Int32(0)
        nT = cutlass.Int32(0)
        n1 = cutlass.Int32(0)
        n2 = cutlass.Int32(0)
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        p1e = cutlass.Int32(0)
        p2e = cutlass.Int32(0)
        lml = cutlass.Int32(0)
        aboveC = cutlass.Int32(0)
        needC = cutlass.Int32(0)
        mm = cutlass.Int32(0)
        lev = cutlass.Int32(0)
        done = cutlass.Int32(0)
        b2w = cutlass.Int32(0)
        sh2 = cutlass.Int32(0)
        it = cutlass.Int32(0)
        it2 = cutlass.Int32(0)
        idv = cutlass.Int32(0)
        q1e = cutlass.Int32(0)
        q2e = cutlass.Int32(0)
        q1f = cutlass.Int32(0)
        q2f = cutlass.Int32(0)
        b_lv = cutlass.Int32(0)
        mc = cutlass.Int32(0)
        quad = cutlass.Int32(0)
        lim1 = cutlass.Int32(0)
        p = cutlass.Int32(0)
        q2i = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        m1 = cutlass.Int32(0)
        m2 = cutlass.Int32(0)
        t1 = cutlass.Int32(0)
        t2 = cutlass.Int32(0)
        c1 = cutlass.Int32(0)
        c2 = cutlass.Int32(0)
        s1 = cutlass.Int32(0)
        s2 = cutlass.Int32(0)
        p1 = cutlass.Int32(0)
        p2 = cutlass.Int32(0)
        wm = cutlass.Int32(0)
        sdyn = cutlass.Int32(0)
        nbw = cutlass.Int32(0)
        uq = cutlass.Uint32(0)
        vq = cutlass.Uint32(0)
        kt = cutlass.Uint32(0)
        klo = cutlass.Uint32(0)
        kv = cutlass.Uint32(0)
        rlo = cutlass.Uint32(0)
        rhi = cutlass.Uint32(0)
        d2 = cutlass.Uint32(0)
        unar = cutlass.Uint32(0)
        bnn = cutlass.Uint32(0)
        nlo = cutlass.Uint32(0)
        uke = cutlass.Uint32(0)
        uk = cutlass.Uint32(0)
        bn = cutlass.Uint32(0)
        w = cutlass.Uint32(0)
        wt = cutlass.Uint32(0)
        ethr = cutlass.Int64(0)
        u64 = cutlass.Int64(0)
        LOQ = cutlass.Float32(0.0)
        HIf = cutlass.Float32(0.0)
        LOf = cutlass.Float32(0.0)
        qt2 = cutlass.Float32(0.0)
        qt3 = cutlass.Float32(0.0)

        npad = cutlass.Int32(logits.shape[1])
        k = cutlass.Int32(pre_idx.shape[1])
        out_row = out[row, None]
        x_addr = logits[row, None].iterator.toint()      # Int64 gmem byte base
        p_addr = pre_idx[row, None].iterator.toint()

        # ---- shared-memory window (map in module docstring) ----
        sptr = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
        sbase = sptr.toint()                              # Int32 shared addr

        s_res = _smem_view(cutlass.Int32, sbase, 0, 6)
        s_cnt = _smem_view(cutlass.Int32, sbase, 6, 2)    # [0]=s_o1 [1]=s_oc
        s_kmm = _smem_view(cutlass.Uint32, sbase, 8, 2)   # [0]=s_kmin [1]=s_kmax
        s_e12 = _smem_view(cutlass.Int32, sbase, 10, 2)   # [0]=s_e1 [1]=s_e2
        s_ws = _smem_view(cutlass.Int32, sbase, 16, 32)
        s_wmn = _smem_view(cutlass.Uint32, sbase, 48, 32)
        s_wmx = _smem_view(cutlass.Uint32, sbase, 80, 32)
        s_hist = _smem_view(cutlass.Int32, sbase, STATIC_WORDS, self.nbh)
        ck_base = sbase + cutlass.Int32((STATIC_WORDS + self.nbh) * 4)
        ck = cute.make_tensor(
            cute.make_ptr(cutlass.Uint32, ck_base, cute.AddressSpace.smem,
                          assumed_align=16),
            cute.make_layout((65536,)))                   # typed view, no bound
        ci = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, ck_base + cmp_ * cutlass.Int32(4),
                          cute.AddressSpace.smem, assumed_align=4),
            cute.make_layout((65536,)))
        img_f = cute.make_tensor(                         # aliases ck/ci (L1299)
            cute.make_ptr(cutlass.Float32, ck_base, cute.AddressSpace.smem,
                          assumed_align=16),
            cute.make_layout((65536,)))
        bm = cute.make_tensor(                            # aliases ck (L1409)
            cute.make_ptr(cutlass.Int32, ck_base, cute.AddressSpace.smem,
                          assumed_align=16),
            cute.make_layout((65536,)))

        n4 = n >> cutlass.Int32(2)
        ntail = n - (n4 << cutlass.Int32(2))
        tix = (n4 << cutlass.Int32(2)) + tid              # CUDA `tidx` L1351

        # ---- hint prefetch: KPT coalesced pre_idx words BEFORE any dependent
        # gather (L1314-1322); compiled out under DEG.
        pvs = []
        if cutlass.const_expr(not self.deg):
            for t in cutlass.range_constexpr(KPT):
                pv = cutlass.Int32(-1)
                j = tid + cutlass.Int32(t * self.blk)
                if j < k:
                    pv = ld_g_i32(p_addr, j)
                pvs.append(pv)

        # ---- row load: exact-fit peel + float4[VPT] register batch (L1327-1350)
        atom128 = g2r_atom_f32(128, invariant=True)
        frags = [cute.make_fragment((4,), cutlass.Float32) for _ in range(VPT)]
        if n4 >= cutlass.Int32(self.blk * self.vpt):      # block-uniform peel
            for u in cutlass.range_constexpr(VPT):
                ld_g_f32x4(atom128, x_addr, tid + cutlass.Int32(u * self.blk),
                           frags[u])
        else:                                             # predicated flat batch
            for u in cutlass.range_constexpr(VPT):
                i = tid + cutlass.Int32(u * self.blk)
                if i < n4:
                    ld_g_f32x4(atom128, x_addr, i, frags[u])
            for u in cutlass.range_constexpr(VPT):
                i = tid + cutlass.Int32(u * self.blk)
                if i >= n4:                               # -INFINITY fill L1346
                    for q in cutlass.range_constexpr(4):
                        frags[u][q] = cutlass.Float32(_NEG_INF)

        tval = cutlass.Float32(_NEG_INF)
        if tid < ntail:
            tval = ldg_f32(x_addr, tix)                   # L1352

        # ---- init (L1391-1392)
        if tid == cutlass.Int32(0):
            s_cnt[0] = cutlass.Int32(0)
            s_cnt[1] = cutlass.Int32(0)
        for z in cutlass.range_constexpr(self.nbh // self.blk):
            s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)

        # ---- bracket: 4 mutually exclusive compile-time arms (L1393-1451)
        lmin = cutlass.Uint32(0xFFFFFFFF)
        lmax = cutlass.Uint32(0)
        if cutlass.const_expr(self.use_img):
            fatom = _f32_smem_atom()
            for u in cutlass.range_constexpr(VPT):        # VPT == 1 here
                i = tid + cutlass.Int32(u * self.blk)
                if i < n4:
                    _sts128_f32(fatom, frags[u], ck_base, i * cutlass.Int32(16))
            if tid < ntail:
                img_f[tix] = tval
            cute.arch.barrier()                           # L1400
            for t in cutlass.range_constexpr(KPT):
                p = pvs[t]
                if cutlass.Uint32(p) < cutlass.Uint32(n):
                    uk = fkey(img_f[p])
                    if uk < lmin:
                        lmin = uk
                    if uk > lmax:
                        lmax = uk
            cute.arch.barrier()                           # L1406 (img dies)
        elif cutlass.const_expr(self.use_bm):
            nbw = (n + cutlass.Int32(31)) >> cutlass.Int32(5)
            i = tid
            while i < nbw:                                # bitmap clear (L1410)
                bm[i] = cutlass.Int32(0)
                i = i + cutlass.Int32(BLK)
            cute.arch.barrier()                           # L1411
            for t in cutlass.range_constexpr(KPT):
                p = pvs[t]
                if cutlass.Uint32(p) < cutlass.Uint32(n):
                    atomic_or_cta(bm.iterator + (p >> cutlass.Int32(5)),
                                  cutlass.Int32(1) << (p & cutlass.Int32(31)))
            cute.arch.barrier()                           # L1417
            lmn = cutlass.Float32(_POS_INF)
            lmx = cutlass.Float32(_NEG_INF)
            for u in cutlass.range_constexpr(VPT):
                base = (tid + cutlass.Int32(u * self.blk)) << cutlass.Int32(2)
                w = cutlass.Uint32(0)
                if cutlass.Uint32(base) < cutlass.Uint32(n):
                    w = cutlass.Uint32(bm[base >> cutlass.Int32(5)]) \
                        >> cutlass.Uint32(base & cutlass.Int32(31))
                for cbit in cutlass.range_constexpr(4):
                    if (w & cutlass.Uint32(1 << cbit)) != cutlass.Uint32(0):
                        lmn = fmin_f32(lmn, _val(frags, 4 * u + cbit))
                        lmx = fmax_f32(lmx, _val(frags, 4 * u + cbit))
            if tid < ntail:
                wt = cutlass.Uint32(bm[tix >> cutlass.Int32(5)]) \
                    >> cutlass.Uint32(tix & cutlass.Int32(31))
                if (wt & cutlass.Uint32(1)) != cutlass.Uint32(0):
                    lmn = fmin_f32(lmn, tval)
                    lmx = fmax_f32(lmx, tval)
            lmin = fkey(lmn)
            lmax = fkey(lmx)                              # monotone (L1428)
            cute.arch.barrier()                           # L1429 (bm dies)
        elif cutlass.const_expr(self.deg):
            lmn = cutlass.Float32(_POS_INF)
            lmx = cutlass.Float32(_NEG_INF)
            for s in cutlass.range_constexpr(S):          # L1436-1439
                v = _val(frags, s)
                if v > cutlass.Float32(_NEG_INF):
                    lmn = fmin_f32(lmn, v)
                    lmx = fmax_f32(lmx, v)
            if tid < ntail:
                lmn = fmin_f32(lmn, tval)
                lmx = fmax_f32(lmx, tval)
            lmin = fkey(lmn)
            lmax = fkey(lmx)
        else:
            # default: KPT scattered fkey ldg gathers, batch-then-fold (L1443-1450)
            xs = []
            for t in cutlass.range_constexpr(KPT):
                xv = cutlass.Float32(0.0)
                if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                    xv = ldg_f32(x_addr, pvs[t])
                xs.append(xv)
            for t in cutlass.range_constexpr(KPT):
                if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                    uk = fkey(xs[t])
                    if uk < lmin:
                        lmin = uk
                    if uk > lmax:
                        lmax = uk

        # ---- block min/max in ONE barrier (L1452-1462); publishes hist clear
        lmin = warp_min_u32(lmin)
        lmax = warp_max_u32(lmax)
        if lane == cutlass.Int32(0):
            s_wmn[tid >> cutlass.Int32(5)] = lmin
            s_wmx[tid >> cutlass.Int32(5)] = lmax
        cute.arch.barrier()                               # L1456
        a = cutlass.Uint32(0xFFFFFFFF)
        c = cutlass.Uint32(0)
        if lane < cutlass.Int32(NW):
            a = cutlass.Uint32(s_wmn[lane])
            c = cutlass.Uint32(s_wmx[lane])
        lmin = warp_min_u32(a)
        lmax = warp_max_u32(c)
        Tv = invkey(lmin)
        GMAX = invkey(lmax)

        # ---- collapse guard, NaN-safe (L1464-1471)
        okc = cutlass.Int32(0)
        if Tv < GMAX:
            if (GMAX - Tv) > cutlass.Float32(1e-30):
                okc = cutlass.Int32(1)
        if okc == cutlass.Int32(0):
            Tv = cutlass.Float32(SENT_LO)
            GMAX = cutlass.Float32(SENT_HI)

        # ---- bin transform constants (L1485-1511)
        BRL = cutlass.const_expr(self.brl)
        OFFf = cutlass.Float32(1.0 if self.brl else 0.0)
        recip = 1.0 / float(self.nbh - (2 if self.brl else 0))
        WD = (GMAX - Tv) * cutlass.Float32(recip)
        wsel = cutlass.Float32(1e-30)
        if WD > cutlass.Float32(0.0):
            wsel = WD
        # rcp.approx (single MUFU.RCP) — the CUDA arm's exact lowering of
        # `1.0f / wsel`; the previous `1.0 / wsel` spelling emitted the IEEE
        # div.rn Newton triple + slowpath CALL on the barrier-bounded chain
        # feeding all S classify FMULs. Output exactness is SC-invariant
        # (any SC > 0 preserves the sign/monotonicity invariants, L1485-1511)
        # and the WD > 0 arm is now bit-identical to CUDA's MUFU.RCP.
        SC = cute.arch.rcp_approx(wsel)
        QCAPf = cutlass.Float32(float(self.nbh - 1))
        CQ0 = OFFf - Tv * SC
        CQ = CQ0 + cutlass.Float32(1e-6) * (_fabsf(CQ0) + cutlass.Float32(1.0))

        # ---- histogram (L1513-1526)
        if cutlass.const_expr(self.brl):
            # fix-2 P4 (GATED, removable as one hunk): A1 ported to the BRL
            # classify arm — hist base pinned ONCE via the same
            # _smem_addr_reg identity-mov used in the !BRL arm below, and the
            # result-discarded classify atomics spelled as resultless
            # red.shared (_red_shared_add1). Value-identical: same +1 to the
            # same byte address (hb + 4*bn == &s_hist[bn]), same .relaxed.cta
            # ordering; the q/bn computations are untouched so classify/emit
            # bit-identity (BRL requirement) is preserved. Emit-path hist
            # atomics (results used, L1630+) are NOT touched.
            hb = _smem_addr_reg(sbase + cutlass.Int32(STATIC_WORDS * 4))
            for s in cutlass.range_constexpr(S):
                q = _fmaf(_val(frags, s), SC, CQ)
                bn = _umin_u32(f2u_rz(q), cutlass.Uint32(self.nbh - 1))
                _red_shared_add1(
                    hb + (cutlass.Int32(bn) << cutlass.Int32(2)))
            qt = _fmaf(tval, SC, CQ)                      # unconditional (L1517)
            bnt = _umin_u32(f2u_rz(qt), cutlass.Uint32(self.nbh - 1))
            _red_shared_add1(
                hb + (cutlass.Int32(bnt) << cutlass.Int32(2)))
        else:
            # hist base pinned ONCE (byte addr, +STATIC_BYTES = word 128 map);
            # each site below is then LEA + ATOMS exactly like the CUDA arm
            # instead of re-deriving the shared window per divergent block.
            hb = _smem_addr_reg(sbase + cutlass.Int32(STATIC_WORDS * 4))
            for s in cutlass.range_constexpr(S):
                q = _submul_asm(_val(frags, s), Tv, SC)           # anti-CSE classify
                if q >= cutlass.Float32(0.0):
                    _red_shared_add1(
                        hb + (f2s_rz(fmin_f32(q, QCAPf))
                              << cutlass.Int32(2)))
            qt = _submul_asm(tval, Tv, SC)
            if qt >= cutlass.Float32(0.0):
                _red_shared_add1(
                    hb + (f2s_rz(fmin_f32(qt, QCAPf)) << cutlass.Int32(2)))
        cute.arch.barrier()                               # L1527

        # ---- crossing-bin find (L1528-1538)
        if cutlass.const_expr(self.cur or self.nbh > 1024):
            scan_cross_w(s_hist, s_ws, k, tid, s_res, blk=self.blk, nb=self.nbh)
        else:
            find_cross(s_hist, k, tid, s_res, nb=self.nbh)
        cute.arch.barrier()                               # L1535
        above = s_res[RES_ABOVE]
        m = s_res[RES_M]
        Bv = s_res[RES_B]
        need = k - above
        whole = cutlass.Int32(0)
        if need >= m:
            whole = cutlass.Int32(1)

        # ---- prod-fix ESCAPE (L1540-1617): 32-step key-space bisection
        esc = cutlass.Int32(0)
        if whole == cutlass.Int32(0):
            if m > cmp_:
                esc = cutlass.Int32(1)
        if esc == cutlass.Int32(1):
            if tid == cutlass.Int32(0):
                s_cnt[0] = cutlass.Int32(0)
                s_cnt[1] = cutlass.Int32(0)
                # DEVIATION (race fix, see notes): the CUDA zeroes s_o1/s_oc
                # again between the nA read (L1581) and the emit (L1584) with
                # only ONE barrier pair around both — a read/write race that
                # nvcc's schedule happens to win and ptxas' does not (observed
                # ~8% row corruption under CTA co-residency). We instead emit
                # through the path-exclusive s_e1/s_e2 slots, zeroed HERE under
                # the existing L1550 barrier; the racy mid-emit rezero is
                # dropped. Barrier count unchanged.
                s_e12[0] = cutlass.Int32(0)
                s_e12[1] = cutlass.Int32(0)
            cute.arch.barrier()                           # L1550
            klo = cutlass.Uint32(0)
            bit = cutlass.Int32(31)
            while bit >= cutlass.Int32(0):
                kt = klo | (cutlass.Uint32(1) << cutlass.Uint32(bit))
                cnt = cutlass.Int32(0)
                for s in cutlass.range_constexpr(S):
                    ix = ((tid + cutlass.Int32((s // 4) * self.blk))
                          << cutlass.Int32(2)) + cutlass.Int32(s % 4)
                    if ix < n:
                        if fkey(_val(frags, s)) >= kt:
                            cnt = cnt + cutlass.Int32(1)
                if tid < ntail:
                    if fkey(tval) >= kt:
                        cnt = cnt + cutlass.Int32(1)
                cnt = cutlass.Int32(warp_add_i32(cnt))
                if lane == cutlass.Int32(0):
                    if cnt != cutlass.Int32(0):
                        atomic_add_cta(s_cnt.iterator, cnt)
                cute.arch.barrier()                       # L1563
                if s_cnt[0] >= k:
                    klo = kt
                cute.arch.barrier()                       # L1565
                if tid == cutlass.Int32(0):
                    s_cnt[0] = cutlass.Int32(0)
                cute.arch.barrier()                       # L1567
                bit = bit - cutlass.Int32(1)
            ethr = cutlass.Int64(klo)                     # k-th largest key
            abv = cutlass.Int32(0)
            for s in cutlass.range_constexpr(S):
                ix = ((tid + cutlass.Int32((s // 4) * self.blk))
                      << cutlass.Int32(2)) + cutlass.Int32(s % 4)
                if ix < n:
                    if cutlass.Int64(fkey(_val(frags, s))) > ethr:
                        abv = abv + cutlass.Int32(1)
            if tid < ntail:
                if cutlass.Int64(fkey(tval)) > ethr:
                    abv = abv + cutlass.Int32(1)
            abv = cutlass.Int32(warp_add_i32(abv))
            if lane == cutlass.Int32(0):
                if abv != cutlass.Int32(0):
                    atomic_add_cta(s_cnt.iterator + 1, abv)
            cute.arch.barrier()                           # L1580
            nA = s_cnt[1]
            nT = k - nA
            # (rezero dropped — emit counters live in s_e12, see race-fix note)
            cute.arch.barrier()                           # L1583
            lml = cutlass.Int32(cute.arch.lanemask_lt())
            for s in cutlass.range_constexpr(S):
                ixv = ((tid + cutlass.Int32((s // 4) * self.blk))
                       << cutlass.Int32(2)) + cutlass.Int32(s % 4)
                u64 = cutlass.Int64(-1)
                if ixv < n:
                    u64 = cutlass.Int64(fkey(_val(frags, s)))
                q1e = cutlass.Int32(0)
                q2e = cutlass.Int32(0)
                if u64 > ethr:
                    q1e = cutlass.Int32(1)
                if u64 == ethr:
                    q2e = cutlass.Int32(1)
                n1 = ballot(q1e == cutlass.Int32(1))
                n2 = ballot(q2e == cutlass.Int32(1))
                b1 = cutlass.Int32(0)
                b2 = cutlass.Int32(0)
                if lane == cutlass.Int32(0):
                    if n1 != cutlass.Int32(0):
                        b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                    if n2 != cutlass.Int32(0):
                        b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
                b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                p1e = b1 + popc(n1 & lml)
                p2e = b2 + popc(n2 & lml)
                if q1e == cutlass.Int32(1):
                    if p1e < nA:
                        out_row[p1e] = ixv
                if q2e == cutlass.Int32(1):
                    if p2e < nT:
                        out_row[nA + p2e] = ixv
            # tail element (L1601-1615)
            u64 = cutlass.Int64(-1)
            if tid < ntail:
                u64 = cutlass.Int64(fkey(tval))
            q1e = cutlass.Int32(0)
            q2e = cutlass.Int32(0)
            if u64 > ethr:
                q1e = cutlass.Int32(1)
            if u64 == ethr:
                q2e = cutlass.Int32(1)
            n1 = ballot(q1e == cutlass.Int32(1))
            n2 = ballot(q2e == cutlass.Int32(1))
            b1 = cutlass.Int32(0)
            b2 = cutlass.Int32(0)
            if lane == cutlass.Int32(0):
                if n1 != cutlass.Int32(0):
                    b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                if n2 != cutlass.Int32(0):
                    b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
            b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
            b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
            p1e = b1 + popc(n1 & lml)
            p2e = b2 + popc(n2 & lml)
            if q1e == cutlass.Int32(1):
                if p1e < nA:
                    out_row[p1e] = tix
            if q2e == cutlass.Int32(1):
                if p2e < nT:
                    out_row[nA + p2e] = tix
            # (CUDA returns here — everything below is the else-arm)
        else:
            # ---- emit (L1619-1701)
            if cutlass.const_expr(self.cur):
                LOQ = cutlass.Float32(Bv)                 # int->float cvt (L1626)
                lim1 = above
                if whole == cutlass.Int32(1):
                    lim1 = above + m
                for s in cutlass.range_constexpr(S):
                    if cutlass.const_expr(self.brl):
                        q = _fmaf(_val(frags, s), SC, CQ)         # bit-identical to classify
                    else:
                        q = _fmaf(_val(frags, s) - Tv, SC, OFFf)  # L1630 emit spelling
                    idx = ((tid + cutlass.Int32((s // 4) * self.blk))
                           << cutlass.Int32(2)) + cutlass.Int32(s % 4)
                    p = cutlass.Int32(0)
                    if q >= LOQ:
                        bn = _umin_u32(f2u_rz(q), cutlass.Uint32(self.nbh - 1))
                        p = atomic_add_cta(s_hist.iterator + cutlass.Int32(bn),
                                           cutlass.Int32(1))
                        if p < lim1:
                            out_row[p] = idx
                        else:
                            if whole == cutlass.Int32(0):
                                q2i = p - above
                                if q2i < cmp_:            # escape-made-safe guard
                                    ck[q2i] = fkey(_val(frags, s))
                                    ci[q2i] = idx
                # tail (L1640-1647)
                if cutlass.const_expr(self.brl):
                    qt2 = _fmaf(tval, SC, CQ)
                else:
                    qt2 = _fmaf(tval - Tv, SC, OFFf)
                p = cutlass.Int32(0)
                if qt2 >= LOQ:
                    bn = _umin_u32(f2u_rz(qt2), cutlass.Uint32(self.nbh - 1))
                    p = atomic_add_cta(s_hist.iterator + cutlass.Int32(bn),
                                       cutlass.Int32(1))
                    if p < lim1:
                        out_row[p] = tix
                    else:
                        if whole == cutlass.Int32(0):
                            q2i = p - above
                            if q2i < cmp_:
                                ck[q2i] = fkey(tval)
                                ci[q2i] = tix
            else:
                # two-mask ballot emit (L1650-1701)
                HIf = cutlass.Float32(_POS_INF)
                LOf = cutlass.Float32(_POS_INF)
                if whole == cutlass.Int32(1):
                    HIf = cutlass.Float32(Bv)
                else:
                    if Bv < cutlass.Int32(self.nbh - 1):
                        HIf = cutlass.Float32(Bv + cutlass.Int32(1))
                    LOf = cutlass.Float32(Bv)
                m1 = cutlass.Int32(0)
                m2 = cutlass.Int32(0)
                for s in cutlass.range_constexpr(S):
                    if cutlass.const_expr(self.brl):
                        q = _fmaf(_val(frags, s), SC, CQ)
                    else:
                        q = _fmaf(_val(frags, s) - Tv, SC, OFFf)
                    if q >= HIf:
                        m1 = m1 | cutlass.Int32(1 << s)
                    else:
                        if q >= LOf:
                            m2 = m2 | cutlass.Int32(1 << s)
                if cutlass.const_expr(self.brl):
                    qt3 = _fmaf(tval, SC, CQ)
                else:
                    qt3 = _fmaf(tval - Tv, SC, OFFf)
                t1 = cutlass.Int32(0)
                t2 = cutlass.Int32(0)
                if qt3 >= HIf:
                    t1 = cutlass.Int32(1)
                else:
                    if qt3 >= LOf:
                        t2 = cutlass.Int32(1)
                c1 = popc(m1) + t1
                c2 = popc(m2) + t2
                s1, s2 = warp_incl_scan_add2(c1, c2, lane)  # L1669-1673
                b1 = cutlass.Int32(0)
                b2 = cutlass.Int32(0)
                if lane == cutlass.Int32(31):
                    b1 = atomic_add_cta(s_cnt.iterator, s1)
                    b2 = atomic_add_cta(s_cnt.iterator + 1, s2)
                b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(31))
                b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(31))
                p1 = b1 + (s1 - c1)
                p2 = b2 + (s2 - c2)
                lim1 = above
                if whole == cutlass.Int32(1):
                    lim1 = k
                wm = m1                                    # sparse set-bit walk
                while wm != cutlass.Int32(0):
                    sdyn = ffs_m1(wm)
                    idx = ((tid + (sdyn >> cutlass.Int32(2))
                            * cutlass.Int32(self.blk)) << cutlass.Int32(2)) \
                        + (sdyn & cutlass.Int32(3))
                    if p1 < lim1:
                        out_row[p1] = idx
                    p1 = p1 + cutlass.Int32(1)
                    wm = wm & (wm - cutlass.Int32(1))
                if t1 == cutlass.Int32(1):
                    if p1 < lim1:
                        out_row[p1] = tix
                    p1 = p1 + cutlass.Int32(1)
                if m2 != cutlass.Int32(0):                 # static-unrolled (L1689)
                    for s in cutlass.range_constexpr(S):
                        if (m2 & cutlass.Int32(1 << s)) != cutlass.Int32(0):
                            idx = ((tid + cutlass.Int32((s // 4) * self.blk))
                                   << cutlass.Int32(2)) + cutlass.Int32(s % 4)
                            if p2 < cmp_:
                                ck[p2] = fkey(_val(frags, s))
                                ci[p2] = idx
                            p2 = p2 + cutlass.Int32(1)
                if t2 == cutlass.Int32(1):
                    if p2 < cmp_:
                        ck[p2] = fkey(tval)
                        ci[p2] = tix
                    p2 = p2 + cutlass.Int32(1)

            # ---- refine (skipped when whole — CUDA returned inside emit)
            if whole == cutlass.Int32(0):
                cute.arch.barrier()                        # L1703
                if cutlass.const_expr(self.cur):
                    mc = m
                    if mc > cmp_:
                        mc = cmp_
                else:
                    mc = s_cnt[1]
                    if mc > cmp_:
                        mc = cmp_
                quad = cutlass.Int32(0)
                if mc >= m:
                    if mc <= qc:
                        quad = cutlass.Int32(1)
                if quad == cutlass.Int32(1):
                    # O(mc^2) index-tie-broken rank (L1706-1718)
                    i = tid
                    while i < mc:
                        uq = cutlass.Uint32(ck[i])
                        r = cutlass.Int32(0)
                        j = cutlass.Int32(0)
                        while j < mc:
                            vq = cutlass.Uint32(ck[j])
                            tinc = cutlass.Int32(0)
                            if vq > uq:
                                tinc = cutlass.Int32(1)
                            if vq == uq:
                                if j < i:
                                    tinc = cutlass.Int32(1)
                            r = r + tinc
                            j = j + cutlass.Int32(1)
                        if r < need:
                            out_row[above + r] = ci[i]
                        i = i + cutlass.Int32(BLK)
                else:
                    # ---- fallback: exact key-space narrowing (L1720-1775)
                    if tid == cutlass.Int32(0):
                        s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                        s_kmm[1] = cutlass.Uint32(0)
                    cute.arch.barrier()                    # L1724
                    i = tid
                    while i < mc:
                        kv = cutlass.Uint32(ck[i])
                        atomic_min_cta(s_kmm.iterator, kv)
                        atomic_max_cta(s_kmm.iterator + 1, kv)
                        i = i + cutlass.Int32(BLK)
                    cute.arch.barrier()                    # L1726
                    rlo = cutlass.Uint32(s_kmm[0])
                    rhi = cutlass.Uint32(s_kmm[1])
                    ethr = cutlass.Int64(rlo)
                    aboveC = cutlass.Int32(0)
                    needC = need
                    mm = mc
                    lev = cutlass.Int32(0)
                    done = cutlass.Int32(0)
                    while done == cutlass.Int32(0):
                        if needC == mm:                    # L1730
                            ethr = cutlass.Int64(rlo) - cutlass.Int64(1)
                            aboveC = aboveC + mm
                            needC = cutlass.Int32(0)
                            done = cutlass.Int32(1)
                        if done == cutlass.Int32(0):
                            if rlo >= rhi:                 # L1731
                                ethr = cutlass.Int64(rlo)
                                done = cutlass.Int32(1)
                            if lev >= cutlass.Int32(6):    # L1732
                                ethr = cutlass.Int64(rlo)
                                done = cutlass.Int32(1)
                        if done == cutlass.Int32(0):
                            d2 = rhi - rlo
                            b2w = cutlass.Int32(32) - clz_i32(
                                cutlass.Int32(d2 | cutlass.Uint32(1)))
                            sh2 = cutlass.Int32(0)
                            if b2w > cutlass.Int32(LNBH):
                                sh2 = b2w - cutlass.Int32(LNBH)
                            for z in cutlass.range_constexpr(self.nbh // self.blk):
                                s_hist[tid + cutlass.Int32(z * self.blk)] = \
                                    cutlass.Int32(0)
                            cute.arch.barrier()            # L1737
                            i = tid
                            while i < mc:
                                unar = cutlass.Uint32(ck[i])
                                if unar >= rlo:
                                    if unar <= rhi:
                                        bnn = (unar - rlo) >> cutlass.Uint32(sh2)
                                        bnn = _umin_u32(
                                            bnn, cutlass.Uint32(self.nbh - 1))
                                        atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(bnn),
                                            cutlass.Int32(1))
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()            # L1743
                            if cutlass.const_expr(self.nbh > 1024):
                                scan_cross_w(s_hist, s_ws, needC, tid, s_res,
                                             blk=self.blk, nb=self.nbh)
                            else:
                                find_cross(s_hist, needC, tid, s_res,
                                           nb=self.nbh)
                            cute.arch.barrier()            # L1746
                            aboveC = aboveC + s_res[RES_ABOVE]
                            needC = needC - s_res[RES_ABOVE]
                            mm = s_res[RES_M]
                            b_lv = s_res[RES_B]
                            nlo = rlo + (cutlass.Uint32(b_lv)
                                         << cutlass.Uint32(sh2))
                            if b_lv != cutlass.Int32(self.nbh - 1):
                                rhi = nlo + ((cutlass.Uint32(1)
                                              << cutlass.Uint32(sh2))
                                             - cutlass.Uint32(1))
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    # final two-predicate ballot emit (L1752-1775)
                    if tid == cutlass.Int32(0):
                        s_e12[0] = cutlass.Int32(0)
                        s_e12[1] = cutlass.Int32(0)
                    cute.arch.barrier()                    # L1753
                    lml = cutlass.Int32(cute.arch.lanemask_lt())
                    it2 = (mc + cutlass.Int32(self.blk - 1)) \
                        // cutlass.Int32(self.blk)
                    it = cutlass.Int32(0)
                    while it < it2:
                        i = it * cutlass.Int32(BLK) + tid
                        uke = cutlass.Uint32(0)
                        idv = cutlass.Int32(0)
                        if i < mc:
                            uke = cutlass.Uint32(ck[i])
                            idv = ci[i]
                        q1f = cutlass.Int32(0)
                        q2f = cutlass.Int32(0)
                        if i < mc:
                            if cutlass.Int64(uke) > ethr:
                                q1f = cutlass.Int32(1)
                            if cutlass.Int64(uke) == ethr:
                                q2f = cutlass.Int32(1)
                        n1 = ballot(q1f == cutlass.Int32(1))
                        n2 = ballot(q2f == cutlass.Int32(1))
                        b1 = cutlass.Int32(0)
                        b2 = cutlass.Int32(0)
                        if lane == cutlass.Int32(0):
                            if n1 != cutlass.Int32(0):
                                b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                            if n2 != cutlass.Int32(0):
                                b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
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

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, logits: cute.Tensor, pre_idx: cute.Tensor,
                 out: cute.Tensor, n: cutlass.Int32, cmp_: cutlass.Int32,
                 qc: cutlass.Int32, smem_bytes: cutlass.Int32, stream):
        b = logits.shape[0]
        self.kern(logits, pre_idx, out, n, cmp_, qc, smem_bytes).launch(
            grid=(b, 1, 1), block=(self.blk, 1, 1), stream=stream,
            smem=smem_bytes, min_blocks_per_mp=self.minb,
            use_pdl=self.pdl)


# ---------------------------------------------------------------------------
# host wrapper: compile cache + route()-driven entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE: dict = {}


def get_compiled(tpl, dump_dir=None, pdl=False):
    """Compile (or fetch) the variant for constexpr tuple
    (BLK, VPT, MINB, KPT, CUR, DEG, IMG, NBH)."""
    key = (tuple(tpl), bool(pdl))
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt
        blk, vpt, minb, kpt, cur, deg, img, nbh = tpl
        kernel = GvrTopkRegKernel(blk, vpt, minb, kpt, cur, deg, img, nbh,
                                  pdl=pdl)
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
        with _no_carveout():
            compiled = cute.compile(
                kernel, lg_fake, pi_fake, out_fake, cutlass.Int32(0),
                cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0),
                stream=fake_stream, options=opts)
        _COMPILE_CACHE[key] = compiled
    return compiled


def reg_topk(logits, pre_idx, n, out, rd=None):
    """torch-facing entry for the register family.

    logits [b, npad] f32, pre_idx [b, k] i32, out [b, >=k] i32, n = valid len.
    rd: optional pre-computed ct_dispatch.route() dict (must be reg/regimg).
    """
    if rd is None:
        from ct_dispatch import route
        rd = route(logits.shape[0], int(n), logits.shape[1],
                   pre_idx.shape[1])
    assert rd['kernel'] in ('reg', 'regimg'), rd['kernel']
    tpl = rd['tpl']
    rt = rd['rt']
    assert rt['IMGOFF'] == tpl[7], (rt['IMGOFF'], tpl[7])  # IMGOFF == NBH
    compiled = get_compiled(tpl)
    smem = STATIC_BYTES + rd['smem']
    compiled(logits, pre_idx, out, int(n), rt['CMP'], rt['QC'], smem)
    return out


__all__ = ["GvrTopkRegKernel", "get_compiled", "reg_topk", "STATIC_BYTES"]
