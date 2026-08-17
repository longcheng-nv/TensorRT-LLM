# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ct_common.py — op46 shared device-helper library (CuTeDSL port).

Translated ONCE from the frozen CUDA source
`op46_selfsampling_cutedsl/src_cuda/kernel.cu` L15-376 (helpers shared by
gvr_main / gvr_topk_reg / gvr_clus / gvr_reg_clus) per TRANSLATION_SPEC.md §5
head, with every spelling pinned by probes/PROBE_RESULTS.md (P1..P15, all
BINDING) and op43 prior-art idioms (ct_tp/ct_gvr/ct_reg/ct_direct).

Conventions for kernel translators
----------------------------------
* Crossing-scan helpers write their scalar outputs into an Int32 smem tensor
  `s_res` using the slot map RES_B=0, RES_M=1, RES_ABOVE=2, RES_TOT=3,
  RES_B2=4, RES_B3=5 (mirror of the CUDA `int *s_B,*s_m,*s_above,*s_tot`
  out-params + the TWO/THREE extra pins). Slots are written ONLY on a pin,
  exactly like the CUDA.
* Histograms are Int32 smem tensors (CUDA uint32_t): adds/scans are
  bit-identical mod 2^32 and every compare the CUDA does against `target`
  is already `(int)`-cast there. Totals < 2^31 by dispatch domain.
* Warp-0-only helpers (find_cross / scan_cross0 / merge_scan0) contain NO
  barrier (probe P14); scan_cross and scan_cross_w contain EXACTLY ONE
  internal barrier (kernel.cu L199 / L306); gather_hint contains EXACTLY TWO
  (L349 / L357). Do not add or drop any (op43 lesson L5).
* All warp collectives use the full mask FULLM = 0xffffffff (kernel.cu L15).
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

# ---------------------------------------------------------------------------
# constants (kernel.cu L15-30)
# ---------------------------------------------------------------------------
FULLM = 0xFFFFFFFF
NB = 1024        # register-family base bin count (L16)
SNB = 256        # streaming-path bin count (L170-177) — MUST stay 256
MAXC = 160       # multi-CTA SPLIT row cap (L17)
GCAP = 16384     # per-row slab capacity in int2 (L18)
QUADC = 96       # O(mc^2) rank gate, streaming/reg (L21)
QUADC_CLUS = 288 # clus + gvr_main gate (L28, spec §2 conflict resolution)
IDXB = 22        # packed candidate index bits (L29)
IDXM = (1 << IDXB) - 1
GVR_WS_OFF_OFF = MAXC * 8   # workspace g_off byte offset (L42)
GVR_WS_BUF_OFF = 2048       # workspace g_buf byte offset (L43)

# degenerate-hint sentinels (kernel.cu L629, L356, L1014 exact-equality flag)
SENT_LO = -3.0e38
SENT_HI = 3.0e38

# s_res slot map (see module docstring)
RES_B = 0
RES_M = 1
RES_ABOVE = 2
RES_TOT = 3
RES_B2 = 4
RES_B3 = 5


# ---------------------------------------------------------------------------
# float <-> u32 bitcasts (op43 ct_tp.py:145-151)
# ---------------------------------------------------------------------------
def u32_of_f32(v):
    """Raw fp32 bits as Uint32 (bit-cast, no conversion)."""
    return cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, v.ir_value()))


def f32_of_u32(u):
    """Uint32 bit pattern as Float32 (bit-cast)."""
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, u.ir_value()))


def f32_of_i32(i):
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, i.ir_value()))


def i32_of_f32(v):
    return cutlass.Int32(llvm.bitcast(cutlass.Int32.mlir_type, v.ir_value()))


# ---------------------------------------------------------------------------
# fkey / invkey (kernel.cu L64-71) — order-preserving float->u32 radix key.
# fkey:   u ^ (((int32)u >> 31) | 0x80000000)  [arithmetic-shift sign trick,
#         spelled 0 - (u >> 31) on Uint32 per op43 ct_tp.py:171-175]
# invkey: (K & 0x80000000) ? K ^ 0x80000000 : ~K   [exact inverse]
# Monotone over all finite floats and +-inf; min identity 0xffffffff, max 0.
# ---------------------------------------------------------------------------
def fkey_bits(u):
    """fkey on raw fp32 bits already held as Uint32."""
    neg = cutlass.Uint32(0) - (u >> cutlass.Uint32(31))  # 0 or 0xFFFFFFFF
    return u ^ (neg | cutlass.Uint32(0x80000000))


def fkey(x):
    """CUDA fkey(float) (L68-71). x: dynamic Float32 -> Uint32 key."""
    return fkey_bits(u32_of_f32(x))


def invkey_bits(K):
    """CUDA invkey (L64-67) without the final bitcast: key -> fp32 bits."""
    s = K >> cutlass.Uint32(31)                       # 1 iff key top bit set
    m = (s - cutlass.Uint32(1)) | cutlass.Uint32(0x80000000)
    # s==1 -> m=0x80000000 (K^0x80000000); s==0 -> m=0xFFFFFFFF (~K)
    return K ^ m


def invkey(K):
    """CUDA invkey(uint32) (L64-67). K: dynamic Uint32 key -> Float32."""
    return f32_of_u32(invkey_bits(K))


# ---------------------------------------------------------------------------
# warp redux wrappers (kernel.cu L59-62; scan_cross_w L307-315; probe P1/P13).
# Values passed to the u32 forms MUST be genuine cutlass.Uint32 — an Int32
# silently lowers to redux.sync.{min,max}.s32 (probe P1 note).
# ---------------------------------------------------------------------------
def warp_min_u32(v):
    """__reduce_min_sync(FULLM, v) -> redux.sync.min.u32 (single inst)."""
    return cute.arch.warp_redux_sync(v, "min")


def warp_max_u32(v):
    """__reduce_max_sync(FULLM, v) -> redux.sync.max.u32."""
    return cute.arch.warp_redux_sync(v, "max")


def warp_add_u32(v):
    """__reduce_add_sync(FULLM, v) -> redux.sync.add.s32 (bit-identical u32)."""
    return cute.arch.warp_redux_sync(v, "add")


def warp_add_i32(v):
    """__reduce_add_sync on Int32 (scan_cross_w two-redux stage L314-315)."""
    return cute.arch.warp_redux_sync(v, "add")


def fmin_f32(a, b):
    """fminf -> native min.f32 (probe P13; op43 inline-PTX no longer needed)."""
    return cute.arch.fmin(a, b)


def fmax_f32(a, b):
    """fmaxf -> max.f32."""
    return cute.arch.fmax(a, b)


# ---------------------------------------------------------------------------
# ballot / popc / clz / ffs wrappers
# ---------------------------------------------------------------------------
def ballot(pred):
    """__ballot_sync(FULLM, pred) -> Int32 mask."""
    return cute.arch.vote_ballot_sync(pred)


def popc(x):
    return cute.arch.popc(x)


def clz_i32(x):
    """__clz as Int32."""
    return cutlass.Int32(cute.arch.clz(x))


def ffs_m1(x):
    """__ffs(x) - 1 for x != 0 (bit index of lowest set bit).

    Spelled popc((x & -x) - 1) per op43 ct_direct.py:210-221. Caller must
    guarantee x != 0 (every kernel.cu use is inside a mask-walk loop).
    """
    return cutlass.Int32(cute.arch.popc((x & (cutlass.Int32(0) - x)) - cutlass.Int32(1)))


@cute.jit
def hi_bit_or_zero(msk):
    """CUDA `msk ? (31 - __clz(msk)) : 0` (find_cross L92/L102)."""
    r = cutlass.Int32(0)
    if msk != cutlass.Int32(0):
        r = cutlass.Int32(31) - clz_i32(msk)
    return r


# ---------------------------------------------------------------------------
# warp shfl scans (op43 ct_tp.py:186-206 + the TWO-interleaved variant that
# gvr_topk_reg L1669-1673 needs)
# ---------------------------------------------------------------------------
@cute.jit
def _shfl_up_add(val, lane, offset: cutlass.Constexpr):
    """Inclusive-scan step: val += shfl_up(val, offset) gated lane >= offset.

    Native shfl.sync.up (mask_and_clamp=0, the __shfl_up_sync lowering):
    hardware clamps the source lane, deleting the VIMNMX+VIADD software
    clamp of the previous idx-kind spelling. Lanes < offset receive an
    undefined-but-discarded value (the gate keeps the result identical).
    """
    other = cute.arch.shuffle_sync_up(val, offset, mask_and_clamp=0)
    if lane >= cutlass.Int32(offset):
        val = val + other
    return val


@cute.jit
def _shfl_down_add(val, lane, offset: cutlass.Constexpr):
    """Suffix-scan step: val += shfl_down(val, offset) gated lane+offset < 32.

    Native shfl.sync.down (mask_and_clamp=31 = __shfl_down_sync lowering);
    hardware clamps, gate discards out-of-range lanes as before.
    """
    other = cute.arch.shuffle_sync_down(val, offset, mask_and_clamp=31)
    if lane + cutlass.Int32(offset) < cutlass.Int32(32):
        val = val + other
    return val


@cute.jit
def warp_incl_scan_add(val, lane):
    """5-step inclusive __shfl_up_sync add scan (e.g. L144-146, L843-849)."""
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_up_add(val, lane, o)
    return val


@cute.jit
def warp_incl_scan_add2(v1, v2, lane):
    """TWO interleaved inclusive shfl_up scans (gvr_topk_reg L1669-1673).

    Per step o: shfl(v1); gated add; shfl(v2); gated add — the two dependency
    chains interleave so the second scan hides under the first's shfl latency
    exactly as the CUDA dual-scan loop does.
    """
    for o in [1, 2, 4, 8, 16]:
        z1 = cute.arch.shuffle_sync_up(v1, o, mask_and_clamp=0)
        if lane >= cutlass.Int32(o):
            v1 = v1 + z1
        z2 = cute.arch.shuffle_sync_up(v2, o, mask_and_clamp=0)
        if lane >= cutlass.Int32(o):
            v2 = v2 + z2
    return v1, v2


@cute.jit
def warp_suffix_scan_add(val, lane):
    """5-step __shfl_down_sync suffix add scan (find_cross L86-89, L97-100)."""
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_down_add(val, lane, o)
    return val


# ---------------------------------------------------------------------------
# CTA-scope shared-memory atomics (probe P7: returns OLD value, ATOMS not RED,
# warp-aggregation preserved; never sys-scope utils.distributed.atomicAdd)
# ---------------------------------------------------------------------------
def atomic_add_cta(ptr, val):
    """shared atomicAdd returning old value. ptr: cute Pointer

    (e.g. `s_hist.iterator + bin_idx`), val: Int32.
    """
    return cutlass.Int32(cute.arch.atomic_add(ptr, val, sem="relaxed", scope="cta"))


def atomic_min_cta(ptr, val):
    """shared atomicMin (s_kmin seeds, L1094-1101). Unsigned iff val is Uint32."""
    return cute.arch.atomic_min(ptr, val, sem="relaxed", scope="cta")


def atomic_max_cta(ptr, val):
    """shared atomicMax (s_kmax seeds)."""
    return cute.arch.atomic_max(ptr, val, sem="relaxed", scope="cta")


def atomic_or_cta(ptr, val):
    """shared atomicOr (gvr_topk_reg bitmap path)."""
    return cute.arch.atomic_or(ptr, val, sem="relaxed", scope="cta")


# ---------------------------------------------------------------------------
# gpu-scope fences + global u64 atomicAdd (SPLIT slab protocol, probe P8)
# ---------------------------------------------------------------------------
def threadfence_gpu():
    """__threadfence() == fence.acq_rel.gpu — use at BOTH L959 and L968."""
    cute.arch.fence_acq_rel_gpu()


def atomic_add_u64_gpu(ptr, val):
    """atom.global.add.u64 returning the OLD value (L960 arrival RMW).

    ptr: cute Pointer to an Int64 gmem word; val: cutlass.Int64.
    Packed arrival word: `cutlass.Int64(1 << 32) + cutlass.Int64(myn)`.
    """
    return cutlass.Int64(cute.arch.atomic_add(ptr, val))


# ---------------------------------------------------------------------------
# saturating converts (probe P4: native ctors emit cvt.rzi.{u32,s32}.f32)
# ---------------------------------------------------------------------------
def f2u_rz(v):
    """__float2uint_rz: saturating (neg/-inf -> 0, huge -> 0xffffffff, NaN -> 0).

    Native ctor spelling verified on this exact toolchain (P4). Dynamic values
    only — host constants raise OverflowError on inf.
    """
    return cutlass.Uint32(v)


def f2s_rz(v):
    """__float2int_rz: saturating (-inf -> INT_MIN, huge -> INT_MAX, NaN -> 0)."""
    return cutlass.Int32(v)


# ---------------------------------------------------------------------------
# L2 prefetch escape hatch (op43 ct_gvr.py:42-54; kernel.cu sites L576/592/612)
# ---------------------------------------------------------------------------
@dsl_user_op
def _prefetch_l2(gaddr, *, loc=None, ip=None):
    """prefetch.global.L2 [gaddr]; gaddr is a byte address (Int64)."""
    llvm.inline_asm(
        res=None,
        operands_=[gaddr.ir_value(loc=loc, ip=ip)],
        asm_string="prefetch.global.L2 [$0];",
        constraints="l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# global loads: 128-bit ldg (read-only) / plain, scalar forms, and __ldcg
# (L2-direct) vector forms for the slab consume (L929/945/978/1032/1172/1194)
# ---------------------------------------------------------------------------
def g2r_atom_f32(bits: int, invariant: bool = True):
    """CopyG2ROp atom: bits=128 -> LDG.E.128[.CONSTANT], bits=32 -> scalar."""
    return cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(), cutlass.Float32, num_bits_per_copy=bits,
        invariant=invariant)


def g2r_atom_i32(bits: int, invariant: bool = False):
    return cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(), cutlass.Int32, num_bits_per_copy=bits,
        invariant=invariant)


def ld_g_f32x4(copy_atom, base_addr, v_idx, frag):
    """Load float4 #v_idx (16B units) from gmem byte base into frag[0..3].

    op43 ct_tp.py:236-245 idiom. base_addr: Int64 byte address; frag: (4,) f32
    fragment. Issue ALL batch members before consuming any (op43 lesson L1).
    """
    p = cute.make_ptr(
        cutlass.Float32,
        base_addr + cutlass.Int64(v_idx) * cutlass.Int64(16),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute.copy(copy_atom, cute.make_tensor(p, cute.make_layout((4,))), frag)


def ldg_f32(base_addr, idx):
    """__ldg(X + idx): scalar read-only 4B gather (gather_hint L343)."""
    atom = g2r_atom_f32(32, invariant=True)
    p = cute.make_ptr(
        cutlass.Float32,
        base_addr + cutlass.Int64(idx) * cutlass.Int64(4),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    frag = cute.make_fragment((1,), cutlass.Float32)
    cute.copy(atom, cute.make_tensor(p, cute.make_layout((1,))), frag)
    return frag[0]


def ld_g_i32(base_addr, idx):
    """plain P[idx] scalar int32 load (gather_hint L340)."""
    p = cute.make_ptr(
        cutlass.Int32,
        base_addr + cutlass.Int64(idx) * cutlass.Int64(4),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    return cutlass.Int32(cute.arch.load(p, cutlass.Int32))


@dsl_user_op
def _ldcg_v2_i32(gaddr, *, loc=None, ip=None):
    """__ldcg on an int2 (8B slab word): ld.global.cg.v2.u32 -> (x, y).

    x = value bits, y = index (workspace g_buf layout, kernel.cu L38-41).
    gaddr: Int64 byte address, 8B-aligned.
    """
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (cutlass.Int32(llvm.extractvalue(T.i32(), ret, [0])),
            cutlass.Int32(llvm.extractvalue(T.i32(), ret, [1])))


@dsl_user_op
def _ldcg_v4_i32(gaddr, *, loc=None, ip=None):
    """ld.global.cg.v4.b32 (16B L2-direct load), returns 4 Int32."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(cutlass.Int32(llvm.extractvalue(T.i32(), ret, [i]))
                 for i in range(4))


# ---------------------------------------------------------------------------
# 128-bit shared-memory ld/st (probe P5a copy-atom spelling) + ulonglong2 read
# ---------------------------------------------------------------------------
def smem_atom_i32_128():
    """CopyUniversalOp atom for ld/st.shared.v4.b32 on Int32 smem."""
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128)


def _smem_v4_tensor(base_addr, byte_off):
    """4-elt Int32 smem tensor at 16B-aligned base_addr+byte_off (Int32 addr)."""
    p = cute.make_ptr(cutlass.Int32, base_addr + byte_off,
                      cute.AddressSpace.smem, assumed_align=16)
    return cute.make_tensor(p, cute.make_layout((4,)))


def lds128_i32(copy_atom, base_addr, byte_off, frag):
    """ld.shared.v4.b32 -> frag(4, Int32)."""
    cute.copy(copy_atom, _smem_v4_tensor(base_addr, byte_off), frag)


def sts128_i32(copy_atom, frag, base_addr, byte_off):
    """st.shared.v4.b32 <- frag(4, Int32)."""
    cute.copy(copy_atom, frag, _smem_v4_tensor(base_addr, byte_off))


@dsl_user_op
def _lds_v2_u64(saddr, *, loc=None, ip=None):
    """ulonglong2 16B smem read (quad-rank L1080-1089, L2241): (lo, hi)."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64(), T.i64()]),
        [saddr.ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u64 {$0, $1}, [$2];",
        "=l,=l,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (cutlass.Uint64(llvm.extractvalue(T.i64(), ret, [0])),
            cutlass.Uint64(llvm.extractvalue(T.i64(), ret, [1])))


# ---------------------------------------------------------------------------
# DSMEM op set (op43 ct_tp.py:44-125 + probe P5b 128-bit remote load).
# mapa returns a byte-addressed Int32 in the PEER's shared window; offset
# arithmetic after one mapa-per-rank is the proven op43/P5b form.
# ---------------------------------------------------------------------------
@dsl_user_op
def _mapa_shared_cluster(smem_ptr, peer_rank, *, loc=None, ip=None):
    """mapa.shared::cluster of a local smem Pointer -> Int32 peer byte addr."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _mapa_shared_cluster_addr(addr_i32, peer_rank, *, loc=None, ip=None):
    """mapa of a raw Int32 shared-window byte address (already .toint()'d)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr_i32.ir_value(loc=loc, ip=ip), peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ld_shared_cluster_i32(mapped_addr, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(), [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.u32 $0, [$1];",
            "=r,r", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ld_shared_cluster_f32(mapped_addr, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(), [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ld_shared_cluster_v4_u32(mapped_addr, *, loc=None, ip=None):
    """Single-shot remote 16B DSMEM load (probe P5b; merge_scan0 L136-137)."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [mapped_addr.ir_value(loc=loc, ip=ip)],
        "ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (cutlass.Int32(llvm.extractvalue(T.i32(), ret, [0])),
            cutlass.Int32(llvm.extractvalue(T.i32(), ret, [1])),
            cutlass.Int32(llvm.extractvalue(T.i32(), ret, [2])),
            cutlass.Int32(llvm.extractvalue(T.i32(), ret, [3])))


@dsl_user_op
def _st_shared_cluster_i32(mapped_addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.u32 [$0], $1;",
        constraints="r,r", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _st_shared_cluster_f32(mapped_addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.f32 [$0], $1;",
        constraints="r,f", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _st_shared_cluster_u64(mapped_addr, val, *, loc=None, ip=None):
    """ONE packed 8B DSMEM candidate push (op43 lesson L3; kernel.cu L2185).

    val = (Uint64(key) << 32) | Uint64(idx_bits).
    """
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.u64 [$0], $1;",
        constraints="r,l", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _atom_shared_cluster_add_i32(mapped_addr, val, *, loc=None, ip=None):
    """Remote CTA smem atomicAdd (cluster scope), returns old value."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.relaxed.cluster.shared::cluster.add.u32 $0, [$1], $2;",
            "=r,r,r", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# ---------------------------------------------------------------------------
# aligned cluster barrier (op43 ct_reg.py:100-113; cg::cluster.sync() ==
# barrier.cluster.{arrive,wait}.aligned). Writers use the FULL (releasing)
# arrive — cluster_arrive_relaxed has NO release and races DSMEM (known
# lesson). Never substitute the non-aligned cute.arch forms.
# ---------------------------------------------------------------------------
@dsl_user_op
def _cluster_arrive_aligned(*, loc=None, ip=None):
    nvvm.cluster_arrive(aligned=True, loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait_aligned(*, loc=None, ip=None):
    nvvm.cluster_wait(aligned=True, loc=loc, ip=ip)


@cute.jit
def _cluster_sync_aligned():
    """cg::cluster_group::sync() (kernel.cu L2016/2135/2226/2474/2530/2648)."""
    _cluster_arrive_aligned()
    _cluster_wait_aligned()


# ===========================================================================
# find_cross<NB_=1024> (kernel.cu L73-109)
# highest bin B with sum_{j>=B} hist[j] >= target; also total, m = hist[B],
# above = sum_{j>B}. Warp-parallel (warp 0 only), bank-conflict free via the
# rotated indexing hist[lane*BPL + ((j+lane) & (BPL-1))] (L83 — DO NOT drop).
# Non-destructive. NO barrier inside.
# Writes s_res[RES_B/RES_M/RES_ABOVE] from the single pinning lane and
# s_res[RES_TOT] from lane 0.
# ===========================================================================
@cute.jit
def find_cross(s_hist, target, tidx, s_res, nb: cutlass.Constexpr):
    BPL = nb // 32                       # python int at trace time
    if tidx < cutlass.Int32(32):
        lane = tidx
        # per-lane span sum with rotated bank-skew indexing (L82-84)
        part = cutlass.Int32(0)
        for j in cutlass.range_constexpr(BPL):
            idx = lane * cutlass.Int32(BPL) + ((cutlass.Int32(j) + lane) & cutlass.Int32(BPL - 1))
            part = part + s_hist[idx]
        # 5-step suffix scan (L85-89): v = sum of part over lanes >= lane
        v = warp_suffix_scan_add(part, lane)
        if lane == cutlass.Int32(0):
            s_res[RES_TOT] = v
        # level 1: highest lane whose suffix still reaches target (L91-92)
        msk = ballot(v >= target)
        L = hi_bit_or_zero(msk)
        aboveL = cute.arch.shuffle_sync(v - part, L)      # L93
        # level 2: one bin per lane inside lane L's span (L94-100)
        h = cutlass.Int32(0)
        if lane < cutlass.Int32(BPL):
            h = s_hist[L * cutlass.Int32(BPL) + lane]
        w = warp_suffix_scan_add(h, lane)
        msk2 = ballot((aboveL + w) >= target)
        J = hi_bit_or_zero(msk2)
        if lane == J:                                     # L103-107
            s_res[RES_B] = L * cutlass.Int32(BPL) + J
            s_res[RES_M] = h
            s_res[RES_ABOVE] = aboveL + (w - h)


# ===========================================================================
# scan_cross0<NB_, ZERO, TWO, THREE, ADD> (kernel.cu L218-286)
# Warp-0-only single-barrier vectorized suffix scan (streaming workhorse,
# NB_=256 at every production call site). Contains NO barrier — the caller
# pays exactly one after it. Leaves hist[j] = per-bin OUTPUT CURSOR
# (count strictly above bin j), or ZEROS when zero=True (folds the next
# phase's histogram clear). two/three pin extra crossing bins for
# target2/target3 into RES_B2/RES_B3. addf folds the per-rank bin-offset
# vector s_addv into the cursors (L279-282).
# HOLD register guard (L236-240): NV<=2 holds the span in regs across the
# scan; wider instantiations re-READ their span (no barrier needed — each
# lane only touches its own span).
# ===========================================================================
@cute.jit
def scan_cross0(s_hist, target, tidx, s_res, target2, target3, s_addv,
                nb: cutlass.Constexpr, zero: cutlass.Constexpr,
                two: cutlass.Constexpr = False, three: cutlass.Constexpr = False,
                addf: cutlass.Constexpr = False):
    BPT = nb // 32                           # bins per lane (trace-time int)
    NV = BPT // 4                            # 16B vectors per lane
    HOLD = NV <= 2                           # register-pressure guard (L240)
    if tidx < cutlass.Int32(32):
        lane = tidx
        atom = smem_atom_i32_128()
        hbase = s_hist.iterator.toint()
        # pass 1: span sum via NV uint4 LDS.128 (L243-251)
        frags = [cute.make_fragment((4,), cutlass.Int32) for _ in range(NV)]
        sm = cutlass.Int32(0)
        for q in cutlass.range_constexpr(NV):
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            lds128_i32(atom, hbase, boff, frags[q])
            sm = sm + frags[q][0] + frags[q][1] + frags[q][2] + frags[q][3]
        # 5-step inclusive shfl_up scan (L252-254)
        w = warp_incl_scan_add(sm, lane)
        tot = cute.arch.shuffle_sync(w, cutlass.Int32(31))
        after = tot - w                       # bins strictly above my span
        if lane == cutlass.Int32(0):
            s_res[RES_TOT] = tot
        base = lane * cutlass.Int32(BPT)
        # pass 2: descending vector walk (L258-284)
        for q in cutlass.range_constexpr(NV - 1, -1, -1):
            if cutlass.const_expr(HOLD):
                vv = frags[q]
            else:
                vv = cute.make_fragment((4,), cutlass.Int32)   # re-read span
                boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                lds128_i32(atom, hbase, boff, vv)
            o4 = cute.make_fragment((4,), cutlass.Int32)
            for j in cutlass.range_constexpr(3, -1, -1):
                cq = vv[j]
                if cutlass.const_expr(zero):
                    o4[j] = cutlass.Int32(0)
                else:
                    o4[j] = after
                gb = base + cutlass.Int32(4 * q + j)
                cross = cutlass.Int32(0)
                if after < target:
                    if (after + cq) >= target:
                        cross = cutlass.Int32(1)
                    if gb == cutlass.Int32(0):
                        cross = cutlass.Int32(1)
                if cross != cutlass.Int32(0):
                    s_res[RES_B] = gb
                    s_res[RES_ABOVE] = after
                    s_res[RES_M] = cq
                if cutlass.const_expr(two):
                    cross2 = cutlass.Int32(0)
                    if after < target2:
                        if (after + cq) >= target2:
                            cross2 = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross2 = cutlass.Int32(1)
                    if cross2 != cutlass.Int32(0):
                        s_res[RES_B2] = gb
                if cutlass.const_expr(three):
                    cross3 = cutlass.Int32(0)
                    if after < target3:
                        if (after + cq) >= target3:
                            cross3 = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross3 = cutlass.Int32(1)
                    if cross3 != cutlass.Int32(0):
                        s_res[RES_B3] = gb
                after = after + cq
            if cutlass.const_expr(addf):       # fold per-rank bin offset (L279-282)
                av = cute.make_fragment((4,), cutlass.Int32)
                aoff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                lds128_i32(atom, s_addv.iterator.toint(), aoff, av)
                for j in cutlass.range_constexpr(4):
                    o4[j] = o4[j] + av[j]
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            sts128_i32(atom, o4, hbase, boff)


# ===========================================================================
# scan_cross<BLK, NB_, TWO> (kernel.cu L179-216)
# Block-parallel suffix scan over NB_ (<= BLK) bins. Leaves hist[j] = OUTPUT
# CURSOR (count in bins > j) and pins the crossing bin. Warps that hold no
# bin skip the body. EXACTLY ONE internal barrier (L199); the caller pays its
# usual publish barrier after. Used by gvr_clus whole-row degen (L2326).
# ===========================================================================
@cute.jit
def scan_cross(s_hist, s_ws, target, tidx, s_res, target2,
               blk: cutlass.Constexpr, nb: cutlass.Constexpr,
               two: cutlass.Constexpr = False):
    NWU = nb // 32
    lane = tidx & cutlass.Int32(31)
    wid = tidx >> cutlass.Int32(5)
    c = cutlass.Int32(0)
    w = cutlass.Int32(0)
    if tidx < cutlass.Int32(nb):                     # L189
        c = s_hist[tidx]
        w = warp_incl_scan_add(c, lane)              # L191-193
        if lane == cutlass.Int32(31):
            s_ws[wid] = w                            # L194
    cute.arch.barrier()                              # L199 — the ONE barrier
    if tidx < cutlass.Int32(nb):                     # L200
        v2 = cutlass.Int32(0)
        if lane < cutlass.Int32(NWU):
            v2 = s_ws[lane]
        pre = warp_incl_scan_add(v2, lane)           # L202-204
        tot = cute.arch.shuffle_sync(pre, cutlass.Int32(31))
        off = cute.arch.shuffle_sync(pre - v2, wid)
        after = tot - (off + w)
        if tidx == cutlass.Int32(0):
            s_res[RES_TOT] = tot
        s_hist[tidx] = after                         # output cursor
        cross = cutlass.Int32(0)
        if after < target:
            if (after + c) >= target:
                cross = cutlass.Int32(1)
            if tidx == cutlass.Int32(0):
                cross = cutlass.Int32(1)
        if cross != cutlass.Int32(0):
            s_res[RES_B] = tidx
            s_res[RES_ABOVE] = after
            s_res[RES_M] = c
        if cutlass.const_expr(two):                  # L212-214
            cross2 = cutlass.Int32(0)
            if after < target2:
                if (after + c) >= target2:
                    cross2 = cutlass.Int32(1)
                if tidx == cutlass.Int32(0):
                    cross2 = cutlass.Int32(1)
            if cross2 != cutlass.Int32(0):
                s_res[RES_B2] = tidx


# ===========================================================================
# scan_cross_w<BLK, NB_> (kernel.cu L288-327)
# Register-path block-parallel suffix scan for NB_ >= BLK: every thread owns
# a private contiguous BPT = NB_/BLK span, so its read->write needs no
# barrier. EXACTLY ONE internal barrier (L306). The second stage is TWO
# REDUCTIONS, not a scan (L307-315): tot = redux_add(vv), off = redux_add
# ((lane < wid) ? vv : 0) — wid is warp-uniform so the masked operand stays
# convergent.
# ===========================================================================
@cute.jit
def scan_cross_w(s_hist, s_ws, target, tidx, s_res,
                 blk: cutlass.Constexpr, nb: cutlass.Constexpr):
    BPT = nb // blk
    NW = blk // 32
    lane = tidx & cutlass.Int32(31)
    wid = tidx >> cutlass.Int32(5)
    loc = cute.make_fragment((BPT,), cutlass.Int32)
    base = tidx * cutlass.Int32(BPT)
    sm = cutlass.Int32(0)
    for i in cutlass.range_constexpr(BPT):                              # L297-300 (#pragma unroll)
        loc[i] = s_hist[base + cutlass.Int32(i)]
        sm = sm + loc[i]
    w = warp_incl_scan_add(sm, lane)                  # L301-304
    if lane == cutlass.Int32(31):
        s_ws[wid] = w                                 # L305
    cute.arch.barrier()                               # L306 — the ONE barrier
    vv = cutlass.Int32(0)
    if lane < cutlass.Int32(NW):
        vv = s_ws[lane]                               # L313
    tot = cutlass.Int32(warp_add_i32(vv))             # L314
    sel = cutlass.Int32(0)
    if lane < wid:
        sel = vv
    off = cutlass.Int32(warp_add_i32(sel))            # L315
    after = tot - (off + w)                           # L316
    if tidx == cutlass.Int32(0):
        s_res[RES_TOT] = tot
    for i in cutlass.range_constexpr(BPT - 1, -1, -1):                  # L318-326
        cq = loc[i]
        s_hist[base + cutlass.Int32(i)] = after       # per-bin OUTPUT CURSOR
        gb = base + cutlass.Int32(i)
        cross = cutlass.Int32(0)
        if after < target:
            if (after + cq) >= target:
                cross = cutlass.Int32(1)
            if gb == cutlass.Int32(0):
                cross = cutlass.Int32(1)
        if cross != cutlass.Int32(0):
            s_res[RES_B] = gb
            s_res[RES_ABOVE] = after
            s_res[RES_M] = cq
        after = after + cq


# ===========================================================================
# merge_scan0<NB_, CS> (kernel.cu L111-168)
# Warp-0-fused cluster merge + suffix scan: each lane reads its BPT-bin span
# from EVERY rank's hist via 16B DSMEM loads (probe P5b), sums the cluster
# totals (and the r<rank prefix that biases this rank's cursors) in
# registers, runs the suffix scan and writes the biased output cursors
# straight into mrg. ONE caller barrier (the post-scan publish) instead of
# two, and no hoff[] array at all. NO barrier inside.
# rank: this CTA's rank-in-cluster (dynamic Int32); cs: cluster size.
# ===========================================================================
@cute.jit
def merge_scan0(s_hist, s_mrg, rank, target, tidx, s_res,
                nb: cutlass.Constexpr, cs: cutlass.Constexpr):
    BPT = nb // 32
    NV = BPT // 4
    if tidx < cutlass.Int32(32):
        lane = tidx
        atom = smem_atom_i32_128()
        # one mapa per rank on the hist base, offsets after (op43/P5b form)
        mapped = [_mapa_shared_cluster(s_hist.iterator, cutlass.Int32(r))
                  for r in range(cs)]
        # pass 1 (L131-143): remote v4 accumulation of tot/pre per vector
        tot_r = []      # NV entries of [4 x Int32] cluster totals
        pre_r = []      # NV entries of [4 x Int32] r<rank prefixes
        sm = cutlass.Int32(0)
        for q in cutlass.range_constexpr(NV):
            boff = (lane * cutlass.Int32(BPT) + cutlass.Int32(4 * q)) * cutlass.Int32(4)
            t = [cutlass.Int32(0)] * 4
            p = [cutlass.Int32(0)] * 4
            for r in cutlass.range_constexpr(cs):
                v0, v1, v2, v3 = _ld_shared_cluster_v4_u32(mapped[r] + boff)
                t[0] = t[0] + v0
                t[1] = t[1] + v1
                t[2] = t[2] + v2
                t[3] = t[3] + v3
                if cutlass.Int32(r) < rank:            # L140 predicated adds
                    p[0] = p[0] + v0
                    p[1] = p[1] + v1
                    p[2] = p[2] + v2
                    p[3] = p[3] + v3
            tot_r.append(t)
            pre_r.append(p)
            sm = sm + t[0] + t[1] + t[2] + t[3]
        # inclusive scan + totals (L144-148)
        w = warp_incl_scan_add(sm, lane)
        tt = cute.arch.shuffle_sync(w, cutlass.Int32(31))
        after = tt - w
        if lane == cutlass.Int32(0):
            s_res[RES_TOT] = tt
        base = lane * cutlass.Int32(BPT)
        # descending walk: crossing pin + prefix-biased cursors into mrg
        for q in cutlass.range_constexpr(NV - 1, -1, -1):                # L151-165
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
                    s_res[RES_B] = gb
                    s_res[RES_ABOVE] = after
                    s_res[RES_M] = cq
                after = after + cq
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            sts128_i32(atom, o4, s_mrg.iterator.toint(), boff)


# ===========================================================================
# gather_hint == GVR_GATHER_HINT(GM_, GX_, KPTV) (kernel.cu L329-358)
# LAZY block-wide (min,max) of logits[pre_idx[j]] over all k hint slots, in
# fkey space, returned as floats. Off the hot path by design: two dependent
# memory round trips (k coalesced P[j] words, then k scattered __ldg 4B
# gathers). Contains EXACTLY 2 barriers — call sites must be block-uniform.
# Outputs are block-uniform (every thread computes them).
# NaN-safe degeneracy guard (L356): if !(GM < GX) both become sentinels.
#
# x_addr / p_addr: Int64 byte base addresses of THIS ROW of logits/pre_idx
# (pass `t.iterator.toint() + row * stride_bytes`). s_wmn/s_wmx: Uint32 smem
# tensors of >= blk//32 slots. Returns (gm, gx) Float32.
# op43 lessons L1/L2: both round trips are issued as predicated flat batches.
# ===========================================================================
@cute.jit
def gather_hint(x_addr, p_addr, k, n, tidx, s_wmn, s_wmx,
                blk: cutlass.Constexpr, kpt: cutlass.Constexpr):
    NW = blk // 32
    lane = tidx & cutlass.Int32(31)
    # batch A: KPT coalesced pre_idx loads, predicated flat (L340)
    pvs = []
    for t in cutlass.range_constexpr(kpt):
        pv = cutlass.Int32(-1)
        j = tidx + cutlass.Int32(t * blk)
        if j < k:
            pv = ld_g_i32(p_addr, j)
        pvs.append(pv)
    # batch B: KPT scattered read-only gathers, predicated flat (L341-343)
    xs = []
    for t in cutlass.range_constexpr(kpt):
        xv = cutlass.Float32(0.0)
        if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):   # (unsigned)p < (unsigned)n
            xv = ldg_f32(x_addr, pvs[t])
        xs.append(xv)
    # fold (L344-346)
    glmin = cutlass.Uint32(0xFFFFFFFF)
    glmax = cutlass.Uint32(0)
    for t in cutlass.range_constexpr(kpt):
        if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
            u2 = fkey(xs[t])
            if u2 < glmin:
                glmin = u2
            if u2 > glmax:
                glmax = u2
    # warp redux + staging (L347-348)
    glmin = warp_min_u32(glmin)
    glmax = warp_max_u32(glmax)
    if lane == cutlass.Int32(0):
        s_wmn[tidx >> cutlass.Int32(5)] = glmin
        s_wmx[tidx >> cutlass.Int32(5)] = glmax
    cute.arch.barrier()                                  # L349 (barrier 1/2)
    # cross-warp redux by EVERY thread — block-uniform outputs (L350-355)
    a2 = cutlass.Uint32(0xFFFFFFFF)
    c2 = cutlass.Uint32(0)
    if lane < cutlass.Int32(NW):
        a2 = s_wmn[lane]
        c2 = s_wmx[lane]
    gm = invkey(warp_min_u32(a2))
    gx = invkey(warp_max_u32(c2))
    # NaN-safe degeneracy guard (L356): !(GM < GX) — NaN compares false
    ok = cutlass.Int32(0)
    if gm < gx:
        ok = cutlass.Int32(1)
    if ok == cutlass.Int32(0):
        gm = cutlass.Float32(SENT_LO)
        gx = cutlass.Float32(SENT_HI)
    cute.arch.barrier()                                  # L357 (barrier 2/2)
    return gm, gx
