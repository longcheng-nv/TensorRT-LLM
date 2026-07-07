# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GVR-MS (multi-threshold sandwich) Top-K decode kernel — cuTe DSL, sm_100.

Sibling variant of ``gvr_topk_decode.py`` (opt-in via the runner; see
``CuteDSLGvrTopKDecodeMsRunner`` in ``cute_dsl_custom_ops.py``). Same
operator contract as the GVR kernel: per-row exact top-K indices of
``logits[row, :N_eff(row)]`` with ``pre_idx`` (previous-step top-K) as a
seeding hint, request-level ``seq_lens``, ``next_n`` MTP rows and
``compress_ratio`` 1 (DSv3.2) / 4 (DSv4) handled identically.

Algorithm (replaces the secant P2 / collect P3 / snap P4 pipeline):

  P1   gather the K prev-step values once; min/max/mean stats AND an smem
       stash of the gathered values (``phase1_stats_stash``).
  P1b  rank-quantile seeding: 256-bin in-smem histogram of the stashed
       values + parallel suffix scan places M=4 ladder thresholds at order
       statistics of the valid count (``qfracs`` of K_valid). Column 0 =
       min gathered value (a guaranteed count>=K_valid anchor).
       Distribution-free: no offline tables, N drops out of placement.
  P2   ONE fused M-threshold ladder scan (``block_count_collect_multi``):
       M predicated counts + speculative per-thread slot-collect of every
       v >= thr[1] during the same pass (``fuse_collect``; slot overflow
       falls back to a classic collect). 16-bit inputs use native
       ``set.ge.{bf16x2,f16x2}`` packed compares with thresholds quantized
       to the dtype grid at P1b emit (bit-equivalent to the fp32 ladder).
  P3   sandwich: the tightest column pair (thr1, thr0) with
       count(thr1) >= K > count(thr0) splits the row into direct winners
       (v >= thr0, streamed straight to the output row) and a band
       [thr1, thr0) of <= kC candidates in smem.
  P4   exact band refine (``phase4_band_rank_scatter``): coarse histogram
       -> straddling bin b* + rank_above -> fast paths (A) whole-bin
       equality emit, (B) <= 32 members: warp0 exact register ranking,
       (C) fallback = exact value-edge band snap. A fixed-depth
       sub-histogram is NOT a valid path C — a fine bin is a value
       interval, not a tie group (upstream revert ec04147502); only paths
       that terminate on a data value are exact.

``GvrMsClusterKernel`` scales the same pipeline out to ``cluster_size``
CTAs per row (row-chunked slices, replicated P1/P1b, DSMEM count merge,
distributed P3 direct-write at rank prefixes, band remote-store push into
the leader's smem via ``st.shared::cluster``, leader-only P4).

Exactness authority is unchanged from the GVR kernel: thresholds are only
seeds; counts + the exact band refine decide membership. All-invalid
``pre_idx`` rows emit identity indices (inherited GVR contract). Fail-soft
under distribution shift: no sandwich pair -> classic collect path; band
overflow -> bounded-bisection retry-shrink landing count in [K, kC].

Provenance: assembled from the op21 kernel campaign (TensorRT-LLM perf
bench, iter11), which measured nsys pure-kernel cold-L2 geomean 1.14x
(fp32) / 1.29x (bf16) / 1.27x (fp16) vs this file's sibling GVR kernel on
the B200 P0 grid (K in {512,1024,2048}, N 65K-262K, BS 1-16), HW-invariant
on B300. Kernel code is byte-identical to the validated bench composition
up to the removal of const_expr-dead bench-only branches.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.distributed import atomicAdd
from cutlass.utils.smem_allocator import SmemAllocator

from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait
from .block_scan import warp_scan
from .gvr_topk_decode import (
    GvrParams,
    _fmin_f32_inline,
    float_as_uint32,
    ld_shared_cluster_f32,
    ld_shared_cluster_i32,
    mapa_shared_cluster,
)

_INT_MAX = 0x7FFFFFFF

# ---------------------------------------------------------------------------
# iter9 native 16-bit compare primitives (inline PTX; `kind` is a trace-time
# python str "bf16"|"f16"). Microbench (probe/count16_native.cu, B200):
# set.ge.{bf16x2,f16x2} + add.rn 16x2 accumulate = 1.73x over cvt->fp32 at
# N262K single-CTA, 1.21x at the C8 slice; counts bit-match the fp32 path
# when thresholds are pre-quantized to the dtype grid.
# ---------------------------------------------------------------------------
def _k16(kind):
    return "bf16" if kind == "bf16" else "f16"


@dsl_user_op
def _quant_f32_16(f, kind, *, loc=None, ip=None):
    k = _k16(kind)
    asm = ("{.reg .b16 h; cvt.rn." + k + ".f32 h, $1; cvt.f32." + k
           + " $0, h;}")
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [f.ir_value(loc=loc, ip=ip)], asm, "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def quant_f32_16(f, kind):
    return _quant_f32_16(f, kind)


@dsl_user_op
def _pack2_16_from_f32(f, kind, *, loc=None, ip=None):
    k = _k16(kind)
    asm = "{.reg .b16 h; cvt.rn." + k + ".f32 h, $1; mov.b32 $0, {h, h};}"
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [f.ir_value(loc=loc, ip=ip)], asm, "=r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def pack2_16_from_f32(f, kind):
    return _pack2_16_from_f32(f, kind)


@dsl_user_op
def _setge_add2_16(acc, v2, t2, kind, *, loc=None, ip=None):
    k = _k16(kind) + "x2"
    asm = ("{.reg .b32 o; set.ge." + k + "." + k + " o, $2, $3; add.rn."
           + k + " $0, $1, o;}")
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [acc.ir_value(loc=loc, ip=ip), v2.ir_value(loc=loc, ip=ip),
                  t2.ir_value(loc=loc, ip=ip)], asm, "=r,r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def setge_add2_16(acc, v2, t2, kind):
    return _setge_add2_16(acc, v2, t2, kind)


@dsl_user_op
def _setge_mask2_16(v2, t2, kind, *, loc=None, ip=None):
    asm = "set.ge.u32." + _k16(kind) + "x2 $0, $1, $2;"
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [v2.ir_value(loc=loc, ip=ip), t2.ir_value(loc=loc, ip=ip)],
        asm, "=r,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def setge_mask2_16(v2, t2, kind):
    return _setge_mask2_16(v2, t2, kind)


@dsl_user_op
def _pair_half_f32_16(v2, hi, kind, *, loc=None, ip=None):
    k = _k16(kind)
    which = "hi" if hi else "lo"
    asm = ("{.reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32." + k
           + " $0, " + which + ";}")
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [v2.ir_value(loc=loc, ip=ip)], asm, "=f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def pair_half_f32_16(v2, hi, kind):
    return _pair_half_f32_16(v2, hi, kind)


@dsl_user_op
def _pair_sum_i32_16(acc2, kind, *, loc=None, ip=None):
    k = _k16(kind)
    asm = ("{.reg .b16 lo, hi; .reg .f32 a, b; mov.b32 {lo, hi}, $1; "
           "cvt.f32." + k + " a, lo; cvt.f32." + k + " b, hi; "
           "add.f32 a, a, b; cvt.rzi.s32.f32 $0, a;}")
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [acc2.ir_value(loc=loc, ip=ip)], asm, "=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.jit
def pair_sum_i32_16(acc2, kind):
    return _pair_sum_i32_16(acc2, kind)

# ---------------------------------------------------------------------------
# iter7 DSMEM remote-store primitives (st.shared::cluster counterparts of the
# vendored ld_shared_cluster_*; same mapa'd-address contract). Visibility to
# the peer's plain ld.shared is ordered by the release/acquire cluster
# barrier pair (cluster_arrive/cluster_wait), exactly as for the remote-load
# direction the gather used.
# ---------------------------------------------------------------------------
@dsl_user_op
def _st_shared_cluster_f32(mapped_addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def st_shared_cluster_f32(mapped_addr, val):
    _st_shared_cluster_f32(mapped_addr, val)


@dsl_user_op
def _st_shared_cluster_i32(mapped_addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def st_shared_cluster_i32(mapped_addr, val):
    _st_shared_cluster_i32(mapped_addr, val)

class GvrMsKernel:
    """Single-CTA-per-row GVR-MS kernel (one CTA processes one row).

    Constructor knobs (production defaults; each was an A/B lever in the
    op21 campaign — flip only for benchmarking):

    * ``M_thr`` / ``R_rounds``: ladder width / rounds. Production = (4, 1):
      one fused M=4 round; a second round only refines when the first finds
      no acceptable sandwich pair.
    * ``band_accept``: stop refining once band <= this (only relevant for
      ``R_rounds > 1``).
    * ``qfracs``: P1b rank fractions for ladder columns 1..M-1 (descending
      rank => ascending value). Column 1 doubles as the fused-collect
      predicate column.
    * ``qbins``: P1b quantile-histogram bins (<= num_threads).
    * ``fuse_collect``: collect candidates during the ladder scan. Needs
      one-CTA-wave residency headroom for the slot smem; host rule:
      ``batch_rows <= num_sms and 4 * top_k <= kC``.
    * ``p4_rank_scatter``: exact rank-scatter band refine (False = legacy
      runtime-k histogram snap, the A/B reference).
    * ``p4_smallbin``: P4 small-bin fast paths A/B (False = always snap).
    * ``p2_native``: native 16-bit ladder compares (no-op for fp32).
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        top_k: int,
        next_n: int = 1,
        num_threads: int = 512,
        compress_ratio: int = 1,
        use_256bit_load: bool = False,
        min_blocks_per_mp: int = 1,
        enable_unroll_4: bool = True,
        enable_phase3_unroll: bool = True,
        use_constant_hint: bool = False,
        enable_warp_parallel_reduce: Optional[bool] = None,
        return_output_values: bool = False,
        M_thr: int = 4,
        R_rounds: int = 1,
        band_accept: int = 64,
        mt_unroll: int = 4,
        qfracs: tuple = (0.75, 0.5, 0.25),
        qbins: int = 256,
        fuse_collect: bool = False,
        p4_rank_scatter: bool = True,
        p4_smallbin: bool = True,
        p2_native: bool = True,
    ):
        self.dtype = dtype
        self.top_k = top_k
        self.next_n = next_n
        # KV compression ratio of the indexer feeding this kernel:
        #   1 -> DSv3.2; preIdxOffset = (row % next_n) + 1 ("newest token
        #       appended" + MTP windowing).
        #   4 -> DSv4 (overlap compressor); compressed-token-index space,
        #       preIdxOffset = 0.
        assert compress_ratio in (1, 4), (
            f"compress_ratio must be 1 (V3.2) or 4 (V4); got {compress_ratio}")
        self.compress_ratio = compress_ratio

        self.WARP_SIZE = 32
        self.num_threads = num_threads
        self.num_warps = num_threads // self.WARP_SIZE
        self.min_blocks_per_mp = min_blocks_per_mp
        # Vector load width for the scan loops: 128-bit default, 256-bit
        # (LDG.E.256) needs 32-byte-aligned rows — the runner checks.
        self.use_256bit_load = use_256bit_load
        self.vec_bits = 256 if use_256bit_load else 128
        self.vec_align_bytes = self.vec_bits // 8
        self.enable_unroll_4 = enable_unroll_4
        self.enable_phase3_unroll = enable_phase3_unroll
        self.use_constant_hint = use_constant_hint
        # Warp-parallel reduce replaces tid==0 serial loops; pays only at
        # 32 warps (same policy as the GVR kernel).
        if enable_warp_parallel_reduce is None:
            enable_warp_parallel_reduce = num_threads == 1024
        self.enable_warp_parallel_reduce = enable_warp_parallel_reduce
        # The sandwich P3 defer-writes indices only (values are never
        # staged); the DSA indexer consumes indices only.
        assert not return_output_values, "GvrMsKernel is indices-only"
        self.return_output_values = return_output_values

        if dtype == cutlass.Float32:
            self._dtype_name = "float32"
        elif dtype == cutlass.BFloat16:
            self._dtype_name = "bfloat16"
        elif dtype == cutlass.Float16:
            self._dtype_name = "float16"
        else:
            raise ValueError(f"Unsupported dtype for GvrMsKernel: {dtype}")

        params = GvrParams.get(self._dtype_name, top_k, self.compress_ratio)
        self.kC = params.kC
        self.kNumBins = params.kNumBins

        self.FLT_MAX = 3.4028235e38
        self.NEG_FLT_MAX = -self.FLT_MAX

        # ---- ladder (op18 lineage) ----
        self.M_thr = int(M_thr)
        self.R_rounds = int(R_rounds)
        self.mt_unroll = int(mt_unroll)  # LSU-ILP unroll of the M-ary scan
        # ---- sandwich / rank-quantile (op21 lineage) ----
        self.band_accept = int(band_accept)
        # Native 16-bit ladder compares: thresholds are quantized to the
        # dtype grid at P1b emit (thr_q = f32(dtype(thr))), which makes
        # 16-bit-domain compares bit-equivalent to the fp32 compares every
        # other phase performs on the exactly-embedded values. The M-column
        # counts accumulate in packed 16x2 lanes (set.ge + add.rn), flushed
        # to int32 every 16 vec iters (per-half growth <= 8/iter => <= 128
        # << the 256 bf16 integer grid). The collect column uses a packed
        # mask (set.ge.u32) so the slot cursor stays exact per element.
        # fp32 binaries are untouched (const_expr).
        self.p2_native = bool(p2_native)
        self.qbins = int(qbins)
        assert self.qbins in (64, 128, 256) and self.num_threads >= self.qbins
        self.QBINS = self.qbins
        # Exact rank-scatter band refine vs legacy runtime-k snap (A/B).
        self.p4_rank_scatter = bool(p4_rank_scatter)
        # P4 small-bin fast paths (host probe on synth+real rows: cnt(b*)
        # p50=2 max=4, so path B covers ~100%); path C fallback is the
        # EXACT value-edge band snap.
        self.p4_smallbin = bool(p4_smallbin)
        # Rank fractions for the P1b quantile placement, mapped to ladder
        # columns 1..M-1 (descending rank => ascending value). Column 1
        # (qfracs[0]) doubles as the fused-collect column (pred_col).
        self.qfracs = tuple(float(f) for f in qfracs)
        assert len(self.qfracs) == self.M_thr - 1, "qfracs must be M-1 long"
        # P1 stashes the K gathered prev-step values into the (still
        # unused) P3 candidate buffer so P1b histograms from smem instead
        # of re-gathering K L2 loads per row.
        assert self.kC >= self.top_k, "P1 stash needs kC >= top_k"
        # Fused P2+P3 slot-collect during the ladder scan.
        self.fuse_collect = bool(fuse_collect)
        self.pred_col = 1 if self.M_thr >= 3 else 0
        # slot_cap needs headroom over the per-thread mean cand/threads
        # (overflow falls back to the classic collect); floor at 8.
        self.slot_cap = max(8, self.kC // self.num_threads)

    # ------------------------------------------------------------------
    # Build a vectorized copy atom for the input scan loops. With
    # use_constant_hint=True we use CopyG2ROp+invariant to get
    # xxx.E.*.CONSTANT (read-only cache, matches CUDA __ldg). Defined as
    # a plain Python method (not @cute.jit) so the if-else branches both
    # bind copy_atom in the same trace scope.
    # ------------------------------------------------------------------
    def _make_load_copy_atom(self):
        # num_bits_per_copy matches self.vec_bits (128 default; 256 when
        # use_256bit_load=True).
        if self.use_constant_hint:
            return cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(),
                self.dtype,
                num_bits_per_copy=self.vec_bits,
                invariant=True,
            )
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=self.vec_bits,
        )

    # ------------------------------------------------------------------
    # Input load helper — casts to fp32 regardless of self.dtype.
    # ------------------------------------------------------------------
    @cute.jit
    def _load_fp32(self, ptr_view, idx):
        # TODO: instructions?
        v = ptr_view[idx]
        if cutlass.const_expr(self.dtype == cutlass.Float32):
            return v
        else:
            return cutlass.Float32(v)

    # ------------------------------------------------------------------
    # Warp-level reductions
    #
    # ------------------------------------------------------------------
    @cute.jit
    def warp_reduce_sum_i32(self, val):
        # REDUX.SYNC.ADD.S32 (sm_80+)
        return cute.arch.warp_redux_sync(val, "add")

    @cute.jit
    def warp_reduce_sum_f32(self, val):
        # PTX redux.sync has no fadd.
        # will lower to SHFL.BFLY 5-step tree.
        return cute.arch.warp_reduction_sum(val)

    @cute.jit
    def warp_reduce_min_f32(self, val):
        # PTX redux.sync.fmin.f32 (sm_100).
        return cute.arch.warp_redux_sync(val, "fmin")

    @cute.jit
    def warp_reduce_max_f32(self, val):
        # PTX redux.sync.fmax.f32 (sm_100).
        return cute.arch.warp_redux_sync(val, "fmax")


    # ------------------------------------------------------------------
    # block_count_ge — Phase 2 / Phase 3 GE-count over global input
    # Per-thread accumulate (4-element strided), cache to smem_ptcnt[tid]
    # for P3 reuse, warp-reduce, block-reduce → s_iscalars[0] = cand_count.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_ge(
        self,
        input_row,  # cute.Tensor [N] fp32
        N,  # length of input_row
        threshold,  # cutlass.Float32 scalar
        smem_ptcnt,  # cute.Tensor [BLOCK_SIZE] int32 (P3 cache)
        smem_wcnt,  # cute.Tensor [NUM_WARPS] int32 (block reduce scratch)
        s_iscalars,  # cute.Tensor [5] int32 (writes [0] = cand_count)
        tidx,
        warp_id,
        lane,
    ):
        """Count input[i] >= threshold across N elements.

        Vectorized: each thread loads vec_w-bit per iter (e.g., 128 bits loading 4 fp32 / 8 bf16 / 8 fp16)
        via cute.copy + CopyUniversalOp. Falls back to scalar tail for the N-mod-vec_w remainder.
        """
        num_threads = cutlass.const_expr(self.num_threads)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        c = cutlass.Int32(0)
        copy_atom = self._make_load_copy_atom()

        step_elem = cutlass.const_expr(num_threads * vec_w)

        row_addr = input_row.iterator.toint()
        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        # Fast path: 4-way unroll for LSU-pipelining ILP.
        if self.enable_unroll_4:
            # =================================================================
            # Each loop body loads 1 vec_w chunk; LLVM unrolls 4 iters at IR
            # level. After unroll, GVN/CSE has one common base ptr in scope and
            # MAY fold the 4 derived addresses into shared base + immediate
            # offsets, e.g., loading from [base+0x2000/0x4000/0x6000]).
            # =================================================================
            rng_frag = cute.make_fragment((vec_w,), self.dtype)
            # Number of complete vec_w-aligned loads this thread can do:
            #   need: i + k*step_elem + (vec_w - 1) < N
            #   max k: floor((N - i - vec_w) / step_elem)
            #   N_iters = max_k + 1
            big_iters = cutlass.Int32(0)
            if N > i + cutlass.Int32(vec_w - 1):
                big_iters = (N - i - cutlass.Int32(vec_w)) // cutlass.Int32(
                    step_elem
                ) + cutlass.Int32(1)

            for k in cutlass.range(big_iters, unroll=4):
                i_local = i + k * cutlass.Int32(step_elem)
                src_ptr_k = cute.make_ptr(
                    self.dtype,
                    row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem,
                    assumed_align=vec_align,
                )
                src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src_k, rng_frag)
                for j in cutlass.range_constexpr(vec_w):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        vj = rng_frag[j]
                    else:
                        vj = cutlass.Float32(rng_frag[j])
                    if vj >= threshold:
                        c = c + cutlass.Int32(1)
            # Advance i past all consumed vec_w-aligned positions so the
            # medium/tail loops below correctly skip (they check i + ... < N).
            i = i + big_iters * cutlass.Int32(step_elem)

        # Tail vec loop: 1-way, handles remainder < 2*step (= remaining 1
        # full vec_w-stride or less). i is always vec_w-aligned here (it
        # advanced by multiples of step_elem = num_threads*vec_w), so the
        # same vec_align bytes hold.
        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < N:
            src_ptr = cute.make_ptr(
                self.dtype,
                row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem,
                assumed_align=vec_align,
            )
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= threshold:
                    c = c + cutlass.Int32(1)
            i = i + step

        # Tail scalar loop
        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= threshold:
                c = c + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)

        # Cache per-thread count for P3 retry-shrink reuse.
        smem_ptcnt[tidx] = c

        # Warp reduce + lane-0 write
        wc = self.warp_reduce_sum_i32(c)
        if lane == 0:
            smem_wcnt[warp_id] = wc
        cute.arch.barrier()

        # Block aggregate (sum reduce over num_warps slots). No trailing
        # barrier: caller is expected to insert its own __syncthreads after
        # its post-processing of cand_count.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # NEW: warp-parallel sum reduce in warp 0.
            if warp_id == cutlass.Int32(0):
                v = cutlass.Int32(0)
                if lane < cutlass.Int32(self.num_warps):
                    v = smem_wcnt[lane]
                total = self.warp_reduce_sum_i32(v)
                if lane == cutlass.Int32(0):
                    s_iscalars[0] = total
        else:
            # tid==0 serial sum.
            if tidx == 0:
                total = cutlass.Int32(0)
                for w in cutlass.range_constexpr(self.num_warps):
                    total = total + smem_wcnt[w]
                s_iscalars[0] = total


    # ------------------------------------------------------------------
    # Phase 3 worker: ballot-free candidate collect (stream-write).
    # Invoked by phase3_collect_candidates (the bounded-bisection
    # override below) with the threshold already landed (done=1).
    # If P2 ended with done=2 (bracket exhausted), first run a retry-shrink
    # loop (≤10 iters) to bring cand_count <= kCC.
    # Then reuse cached smem_ptcnt → warp prefix sum → block prefix sum
    # → stream-write keys[]/vals[] for v >= threshold.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_collect_stream_write(
        self,
        input_row,
        N,
        smem_keys,
        smem_vals,
        smem_ptcnt,
        smem_wcnt,
        s_thr,
        s_iscalars,
        tidx,
        warp_id,
        lane,
    ):
        """Retry-shrink (if done!=1) + warp/block prefix sum + stream-write.

        After this fn, smem_keys[0:cand_count] contains all v >= threshold
        in some order (determined by tid's per-thread index within stream-write
        loop), and smem_vals[0:cand_count] holds matching original indices.
        smem_ptcnt is reused per-thread cached counts from the LAST
        block_count_ge inside P2 (or inside the retry-shrink below).
        """
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        num_threads = cutlass.const_expr(self.num_threads)

        # ---- Retry-shrink loop (only if P2 didn't converge cleanly) ----
        if s_iscalars[1] != cutlass.Int32(1):
            # Re-count with current threshold (may already have stale cand_count)
            cur_thr = s_thr[0]
            self.block_count_ge(
                input_row,
                N,
                cur_thr,
                smem_ptcnt,
                smem_wcnt,
                s_iscalars,
                tidx,
                warp_id,
                lane,
            )
            if tidx == 0:
                if s_iscalars[0] > cutlass.Int32(kCC):
                    s_thr[1] = s_thr[0]  # val_lo = threshold
            cute.arch.barrier()

            # 10-iter retry-shrink. Runtime while with `cand_count > kCC` in the
            # loop condition.
            rs = cutlass.Int32(0)
            while rs < cutlass.Int32(10) and s_iscalars[0] > cutlass.Int32(kCC):
                if tidx == 0:
                    lo = s_thr[1]
                    hi = s_thr[2]
                    mid = (lo + hi) * cutlass.Float32(0.5)
                    if mid == lo:
                        mid = hi
                    s_thr[0] = mid
                cute.arch.barrier()
                new_thr = s_thr[0]
                self.block_count_ge(
                    input_row,
                    N,
                    new_thr,
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    tidx,
                    warp_id,
                    lane,
                )
                if tidx == 0:
                    c_rs = s_iscalars[0]
                    if c_rs > cutlass.Int32(kCC):
                        s_thr[1] = s_thr[0]
                    elif c_rs < cutlass.Int32(kK):
                        s_thr[2] = s_thr[0]
                cute.arch.barrier()
                rs = rs + cutlass.Int32(1)

        # ---- Warp prefix sum over smem_ptcnt ----
        # my_total_qual = per-thread count cached by last block_count_ge.
        my_total_qual = smem_ptcnt[tidx]
        tp = my_total_qual

        # 5-level shfl_up_sync inclusive scan within warp.
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp = tp + other

        my_excl_offset = tp - my_total_qual
        # Warp total = lane 31's tp; broadcast via shfl_sync_bfly (or
        # cross-lane read: shuffle_sync_op with lane=31).
        warp_total = cute.arch.shuffle_sync(tp, cutlass.Int32(self.WARP_SIZE - 1))

        if lane == 0:
            smem_wcnt[warp_id] = warp_total
        cute.arch.barrier()

        # Exclusive prefix sum over num_warps warp totals.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # NEW: warp-parallel via block_scan.warp_scan (Hillis-Steele
            # inclusive scan, log2(num_warps) shfl_up steps). Exclusive
            # prefix = inclusive - val. Total = inclusive at last lane.
            if warp_id == cutlass.Int32(0):
                if lane < cutlass.Int32(self.num_warps):
                    val = smem_wcnt[lane]
                    inclusive = warp_scan(val, tidx, lane, num_threads_per_warp=self.num_warps)
                    smem_wcnt[lane] = inclusive - val  # exclusive prefix
                    if lane == cutlass.Int32(self.num_warps - 1):
                        s_iscalars[0] = inclusive  # cand_count (total)
        else:
            # tid==0 serial exclusive prefix.
            if tidx == 0:
                total = cutlass.Int32(0)
                for w in cutlass.range_constexpr(self.num_warps):
                    cnt = smem_wcnt[w]
                    smem_wcnt[w] = total
                    total = total + cnt
                s_iscalars[0] = total
        cute.arch.barrier()

        # Each thread's write base = warp-prefix + intra-warp exclusive offset.
        my_base = smem_wcnt[warp_id]
        my_write_pos = my_base + my_excl_offset

        # ---- Stream-write loop ----
        thr_final = s_thr[0]
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        row_addr = input_row.iterator.toint()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        wc = my_write_pos
        ic = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        # Phase3 unrolling: master gated by self.enable_phase3_unroll.
        # When OFF, only the tail 1-way loop runs (matches the pre-unroll
        # state of phase3_collect). When ON, the inner enable_unroll_4
        # controls the 4-way fast path.
        if self.enable_phase3_unroll:
            # Fast path: 4-way unrolled vec loop (4 loading instructions in flight).
            if self.enable_unroll_4:
                # =============================================================
                # unroll: cutlass.range(unroll=4).
                # Each body loads 1 vec_w chunk; LLVM unrolls 4 iters at IR
                # level. Same intent as the Phase-2 rewrite above.
                # =============================================================
                rng_frag = cute.make_fragment((vec_w,), self.dtype)
                big_iters = cutlass.Int32(0)
                if N > ic + cutlass.Int32(vec_w - 1):
                    big_iters = (N - ic - cutlass.Int32(vec_w)) // cutlass.Int32(
                        step_elem
                    ) + cutlass.Int32(1)

                for k in cutlass.range(big_iters, unroll=4):
                    ic_local = ic + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem,
                        assumed_align=vec_align,
                    )
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                    cute.copy(copy_atom, src_k, rng_frag)
                    for j in cutlass.range_constexpr(vec_w):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj = rng_frag[j]
                        else:
                            vj = cutlass.Float32(rng_frag[j])
                        if vj >= thr_final and wc < cutlass.Int32(kCC):
                            smem_keys[wc] = vj
                            smem_vals[wc] = ic_local + cutlass.Int32(j)
                            wc = wc + cutlass.Int32(1)
                # Advance ic past all consumed vec_w-aligned positions.
                ic = ic + big_iters * cutlass.Int32(step_elem)

        # Tail vec loop: 1-way, handles remainder < 2*step. ic stays vec_w-
        # aligned across the unroll loop (steps by num_threads*vec_w).
        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while ic + cutlass.Int32(vec_w - 1) < N:
            src_ptr = cute.make_ptr(
                self.dtype,
                row_addr + cutlass.Int64(ic) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem,
                assumed_align=vec_align,
            )
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= thr_final and wc < cutlass.Int32(kCC):
                    smem_keys[wc] = vj
                    smem_vals[wc] = ic + cutlass.Int32(j)
                    wc = wc + cutlass.Int32(1)
            ic = ic + step

        # Tail scalar loop (N % vec_w)
        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= thr_final and wc < cutlass.Int32(kCC):
                smem_keys[wc] = v
                smem_vals[wc] = it
                wc = wc + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)
        cute.arch.barrier()


    # ------------------------------------------------------------------
    # block_fused_snap_iter — P4 snap convergence inner step
    # ------------------------------------------------------------------
    @cute.jit
    def block_fused_snap_iter(
        self,
        smem_keys,
        smem_wcnt,
        smem_hist,  # reused as scratch for s_up/s_down warp aggregates
        s_thr,
        s_iscalars,
        count,
        tidx,
        warp_id,
        lane,
    ):
        """One iteration of histogram snap. Updates s_iscalars[2]=cnt_lo (cge),
        s_iscalars[3]=cnt_hi (cgt), and s_thr[0]=threshold (moves toward
        the cnt-in-(kK_GT, kK_GE) bracket).
        """
        kK = cutlass.const_expr(self.top_k)
        num_threads = cutlass.const_expr(self.num_threads)
        thr = s_thr[0]

        lge = cutlass.Int32(0)
        lgt = cutlass.Int32(0)
        s_up = cutlass.Float32(self.FLT_MAX)
        s_down = cutlass.Float32(self.NEG_FLT_MAX)

        isi = tidx
        while isi < count:
            v = smem_keys[isi]
            if v >= thr:
                lge = lge + cutlass.Int32(1)
            if v > thr:
                lgt = lgt + cutlass.Int32(1)
                # s_up = min(s_up, v) — hot path in block_fused_snap_iter (~10us)
                s_up = _fmin_f32_inline(s_up, v)
            if v < thr:
                s_down = cute.arch.fmax(s_down, v)
            isi = isi + cutlass.Int32(num_threads)

        # Pack lge/lgt into a single int32 so the warp reduction below
        # sums both counts in one shuffle. Safe as long as each per-warp
        # count stays < 2^16 = 65536 — which holds because lge/lgt are
        # bounded by cand_count ≤ kC ≤ 6144 (see GvrParams). Bump kC
        # past 65536 and this packing silently corrupts.
        packed = (lge << cutlass.Int32(16)) | lgt
        packed = self.warp_reduce_sum_i32(packed)
        s_up = self.warp_reduce_min_f32(s_up)
        s_down = self.warp_reduce_max_f32(s_down)

        # Lane 0 stages results into warp slots (smem_hist[0..NW-1] = s_up,
        # smem_hist[NW..2*NW-1] = s_down stored as int32 bit-cast).
        if lane == 0:
            smem_wcnt[warp_id] = packed
            smem_hist[warp_id] = float_as_uint32(s_up)
            smem_hist[self.num_warps + warp_id] = float_as_uint32(s_down)
        cute.arch.barrier()

        # 3-way block reduce + threshold bound update.
        if cutlass.const_expr(self.enable_warp_parallel_reduce):
            # NEW: warp-parallel 3-way reduce in warp 0.
            if warp_id == cutlass.Int32(0):
                v_tp = cutlass.Int32(0)
                v_up = cutlass.Float32(self.FLT_MAX)
                v_dn = cutlass.Float32(self.NEG_FLT_MAX)
                if lane < cutlass.Int32(self.num_warps):
                    v_tp = smem_wcnt[lane]
                    vu_bits = smem_hist[lane]
                    vd_bits = smem_hist[self.num_warps + lane]
                    v_up = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vu_bits.ir_value())
                    )
                    v_dn = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, vd_bits.ir_value())
                    )
                tp = self.warp_reduce_sum_i32(v_tp)
                total_up = self.warp_reduce_min_f32(v_up)
                total_down = self.warp_reduce_max_f32(v_dn)
                if lane == cutlass.Int32(0):
                    cge = tp >> cutlass.Int32(16)
                    cgt = tp & cutlass.Int32(0xFFFF)
                    s_iscalars[2] = cge
                    s_iscalars[3] = cgt
                    if cgt >= cutlass.Int32(kK):
                        if total_up < cutlass.Float32(self.FLT_MAX):
                            s_thr[0] = total_up
                    elif cge < cutlass.Int32(kK):
                        if total_down > cutlass.Float32(self.NEG_FLT_MAX):
                            s_thr[0] = total_down
        else:
            # tid==0 serial 3-way reduce.
            if tidx == 0:
                tp = cutlass.Int32(0)
                total_up = cutlass.Float32(self.FLT_MAX)
                total_down = cutlass.Float32(self.NEG_FLT_MAX)
                for w in cutlass.range_constexpr(self.num_warps):
                    tp = tp + smem_wcnt[w]
                    vu = llvm.bitcast(cutlass.Float32.mlir_type, smem_hist[w].ir_value())
                    vd = llvm.bitcast(
                        cutlass.Float32.mlir_type, smem_hist[self.num_warps + w].ir_value()
                    )
                    vu_w = cutlass.Float32(vu)
                    vd_w = cutlass.Float32(vd)
                    total_up = _fmin_f32_inline(total_up, vu_w)
                    total_down = cute.arch.fmax(total_down, vd_w)

                cge = tp >> cutlass.Int32(16)
                cgt = tp & cutlass.Int32(0xFFFF)
                s_iscalars[2] = cge
                s_iscalars[3] = cgt
                if cgt >= cutlass.Int32(kK):
                    if total_up < cutlass.Float32(self.FLT_MAX):
                        s_thr[0] = total_up
                elif cge < cutlass.Int32(kK):
                    if total_down > cutlass.Float32(self.NEG_FLT_MAX):
                        s_thr[0] = total_down
        cute.arch.barrier()


    # ------------------------------------------------------------------
    # Phase 4: Histogram-based k-th selection + two-pass writeback
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_histogram_snap(
        self,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_wcnt,
        s_thr,
        s_iscalars,
        output_values_row,
        output_indices_row,
        cand_count,
        tidx,
        warp_id,
        lane,
    ):
        """Three branches by cand_count:
        == kK : direct emit (fast path)
        >  kK : histogram k-th bin search + snap + 2-pass writeback
        <  kK : emit cand_count + pad with -self.FLT_MAX
        """
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        # ----- Branch A: cand_count == kK (fast path) -----
        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i4] = self.dtype(smem_keys[i4])
                output_indices_row[i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        elif cand_count > cutlass.Int32(kK):
            # ----- Branch B: cand_count > kK → histogram snap -----

            # Block min/max over keys[0:cand_count]
            local_cmin = cutlass.Float32(self.FLT_MAX)
            local_cmax = cutlass.Float32(self.NEG_FLT_MAX)
            i5 = tidx
            while i5 < cand_count:
                v = smem_keys[i5]
                local_cmin = _fmin_f32_inline(local_cmin, v)
                local_cmax = cute.arch.fmax(local_cmax, v)
                i5 = i5 + cutlass.Int32(num_threads)
            cmin = self.warp_reduce_min_f32(local_cmin)
            cmax = self.warp_reduce_max_f32(local_cmax)
            # Stage warp results into smem_wcnt[w] (cmin) and smem_hist[w] (cmax)
            # as bit-cast int32. cmax stored at smem_hist[0..NW-1].
            if lane == 0:
                smem_wcnt[warp_id] = float_as_uint32(cmin)
                smem_hist[warp_id] = float_as_uint32(cmax)
            cute.arch.barrier()

            # Every thread independently recomputes block_min/block_max
            # from the warp-staged smem slots (CUDA heuristic_topk.cuh:891-898
            # pattern). No tid==0 → s_thr broadcast → saves a block barrier.
            bmin_r = cutlass.Float32(self.FLT_MAX)
            bmax_r = cutlass.Float32(self.NEG_FLT_MAX)
            # Unrolled num_warps times (16 or 32 — fixed at compile time).
            for w in cutlass.range_constexpr(self.num_warps):
                vmin_bits = smem_wcnt[w]
                vmax_bits = smem_hist[w]
                vmin = cutlass.Float32(
                    llvm.bitcast(cutlass.Float32.mlir_type, vmin_bits.ir_value())
                )
                vmax = cutlass.Float32(
                    llvm.bitcast(cutlass.Float32.mlir_type, vmax_bits.ir_value())
                )
                bmin_r = _fmin_f32_inline(bmin_r, vmin)
                bmax_r = cute.arch.fmax(bmax_r, vmax)
            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            # All threads must finish reading smem_hist[0..NW-1] (the
            # warp-staged cmax slots above) before any thread starts
            # zeroing smem_hist below — otherwise warp-0 thread-0 can
            # zero smem_hist[0] while warp-N is still reading it,
            # producing a 0 cmax → squashed bmax_r → all candidates land
            # in bin 0 → wrong K-th threshold. Hit-rate-dependent race.
            cute.arch.barrier()

            # Zero histogram (must zero ALL slots since smem_hist[0..NW-1] was
            # used as cmax scratch above).
            i6 = tidx
            while i6 < cutlass.Int32(kBins):
                smem_hist[i6] = cutlass.Int32(0)
                i6 = i6 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            range1 = bmax_r - bmin_r
            # inv1 = (kBins - 1 + 0.99) / range1  (range1 > 0 guaranteed by 1e-6 patch)
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1

            # Build histogram by atomicAdd.
            i7 = tidx
            while i7 < cand_count:
                vk = smem_keys[i7]
                bin_f = (vk - bmin_r) * inv1
                bin_i = cutlass.Int32(bin_f)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # ---- Parallel k-th bin search (3-step) ----
            # Step 1: each warp sums BINS_PER_WARP bins (high→low slice)
            warp_bin_sum = cutlass.Int32(0)
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (
                    cutlass.Int32(kBins - 1)
                    - warp_id * cutlass.Int32(bins_per_warp)
                    - cutlass.Int32(jb)
                )
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            if lane == 0:
                smem_wcnt[warp_id] = warp_bin_sum
            cute.arch.barrier()

            # Step 2: tid==0 finds target warp; stores prefix-count + warp index
            # into s_iscalars[2] (=cnt_lo: prefix before target warp)
            # and s_iscalars[3] (=cnt_hi: target warp index)
            if tidx == 0:
                cum = cutlass.Int32(0)
                tw = cutlass.Int32(num_warps - 1)
                found = cutlass.Int32(0)
                for w2 in cutlass.range_constexpr(self.num_warps):
                    cum = cum + smem_wcnt[w2]
                    if cum >= cutlass.Int32(kK) and found == cutlass.Int32(0):
                        tw = cutlass.Int32(w2)
                        found = cutlass.Int32(1)
                # Recompute prefix BEFORE target warp
                cum2 = cutlass.Int32(0)
                for w3 in cutlass.range_constexpr(self.num_warps):
                    if cutlass.Int32(w3) < tw:
                        cum2 = cum2 + smem_wcnt[w3]
                s_iscalars[2] = cum2  # prefix
                s_iscalars[3] = tw  # target warp index
            cute.arch.barrier()

            # Step 3: target warp's lane 0 scans BINS_PER_WARP bins → threshold
            # NOTE: This loop runs in a single thread (warp 0 lane 0). Tried
            # changing range_constexpr → cutlass.range(unroll=1) to mirror
            # CUDA's `for+break` (gets nvcc to keep runtime branch). SASS
            # I2FP dropped 64→1, total inst -544 for fp32, but perf was
            # WORSE (-7pp on fp32 large N, -14pp on bf16 synth). The runtime
            # branch/counter overhead in a 1-thread serial path exceeds the
            # static I2FP/FMUL/FFMA waste — those 60+ fp ops are essentially
            # free for one thread. Keeping range_constexpr.
            target_warp = s_iscalars[3]
            if warp_id == target_warp and lane == cutlass.Int32(0):
                base_cum = s_iscalars[2]
                thr_local = bmin_r
                bmin_local = bmin_r
                set_done = cutlass.Int32(0)
                for jb2 in cutlass.range_constexpr(bins_per_warp):
                    bidx2 = (
                        cutlass.Int32(kBins - 1)
                        - target_warp * cutlass.Int32(bins_per_warp)
                        - cutlass.Int32(jb2)
                    )
                    base_cum = base_cum + smem_hist[bidx2]
                    if base_cum >= cutlass.Int32(kK) and set_done == cutlass.Int32(0):
                        thr_local = bmin_local + cutlass.Float32(bidx2) * range1 / cutlass.Float32(
                            kBins
                        )
                        set_done = cutlass.Int32(1)
                s_thr[0] = thr_local
            cute.arch.barrier()

            # ---- Snap convergence loop ----
            # snap_limit = cand_count (matches CUDA heuristic_topk.cuh:985).
            # The older `cand_count / 4` bound silently accepted a non-
            # converged threshold in ~0.09% of adversarial distributions
            # (Pass 1 then picked K from cgt > kK candidates in scan order,
            # missing some true top-K members). Common path still converges
            # in 1-3 iters; the higher upper bound only affects long-tail
            # cells.
            snap_limit = cand_count

            # Loop with runtime break-via-guard. We unroll up to a safe ceiling
            # then loop the rest with a while.
            # For simplicity: while loop with explicit break-flag.
            si = cutlass.Int32(0)
            done_snap = cutlass.Int32(0)
            while si < snap_limit and done_snap == cutlass.Int32(0):
                self.block_fused_snap_iter(
                    smem_keys,
                    smem_wcnt,
                    smem_hist,
                    s_thr,
                    s_iscalars,
                    cand_count,
                    tidx,
                    warp_id,
                    lane,
                )
                # After block_fused_snap_iter, s_iscalars[2]=cge, s_iscalars[3]=cgt.
                if s_iscalars[3] < cutlass.Int32(kK) and s_iscalars[2] >= cutlass.Int32(kK):
                    done_snap = cutlass.Int32(1)
                si = si + cutlass.Int32(1)

            # ---- Two-pass output writeback (ballot+popc, CUDA-style) ----
            # Per-iter: ballot collects emit flags into a 32-bit mask, popc gives
            # within-warp count, lane 0 atomicAdds to out_count, shuffle broadcasts
            # the base offset. No per-iter barriers — only 1 barrier between passes.
            sel_thr = s_thr[0]
            if tidx == 0:
                s_iscalars[4] = cutlass.Int32(0)  # out_count
            cute.arch.barrier()

            # Pass 1: v > sel_thr — strided over (warp_id*WARP_SIZE, ...) like CUDA.
            # `if mask_gt != 0` mirrors CUDA heuristic_topk.cuh:1020 — when no
            # lane in the warp emits, skip the popc + atomicAdd + shuffle round
            # trip (the atomicAdd alone is ~10-30 cycles on a SMEM atomic unit).
            base_w = warp_id * cutlass.Int32(self.WARP_SIZE)
            while base_w < cand_count:
                ix1 = base_w + lane
                emit_gt = cutlass.Int32(0)
                v_p1 = cutlass.Float32(self.NEG_FLT_MAX)
                if ix1 < cand_count:
                    v_p1 = smem_keys[ix1]
                    if v_p1 > sel_thr:
                        emit_gt = cutlass.Int32(1)
                mask_gt = cute.arch.vote_ballot_sync(emit_gt != cutlass.Int32(0))
                if mask_gt != cutlass.Uint32(0):
                    cnt_gt = cutlass.Int32(cute.arch.popc(mask_gt))
                    lane_mask_gt = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
                    moff_gt = cutlass.Int32(cute.arch.popc(mask_gt & lane_mask_gt))
                    bp_gt = cutlass.Int32(0)
                    if lane == cutlass.Int32(0):
                        bp_gt = atomicAdd(
                            s_iscalars.iterator + cutlass.Int32(4),
                            cnt_gt,
                        )
                    bp_gt = cute.arch.shuffle_sync(bp_gt, cutlass.Int32(0))
                    wpos_p1 = bp_gt + moff_gt
                    if emit_gt != cutlass.Int32(0) and wpos_p1 < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[wpos_p1] = self.dtype(v_p1)
                        output_indices_row[wpos_p1] = smem_vals[ix1]
                base_w = base_w + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # Pass 2: v == sel_thr (same pattern as Pass 1, same `if mask` guard).
            # Empty-iter case is much more common here because only tie-values
            # at the K-th rank emit.
            base_w2 = warp_id * cutlass.Int32(self.WARP_SIZE)
            while base_w2 < cand_count:
                ix2 = base_w2 + lane
                emit_eq = cutlass.Int32(0)
                v_p2 = cutlass.Float32(self.NEG_FLT_MAX)
                if ix2 < cand_count:
                    v_p2 = smem_keys[ix2]
                    if v_p2 == sel_thr:
                        emit_eq = cutlass.Int32(1)
                mask_eq = cute.arch.vote_ballot_sync(emit_eq != cutlass.Int32(0))
                if mask_eq != cutlass.Uint32(0):
                    cnt_eq = cutlass.Int32(cute.arch.popc(mask_eq))
                    lane_mask_eq = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
                    moff_eq = cutlass.Int32(cute.arch.popc(mask_eq & lane_mask_eq))
                    bp_eq = cutlass.Int32(0)
                    if lane == cutlass.Int32(0):
                        bp_eq = atomicAdd(
                            s_iscalars.iterator + cutlass.Int32(4),
                            cnt_eq,
                        )
                    bp_eq = cute.arch.shuffle_sync(bp_eq, cutlass.Int32(0))
                    wpos_p2 = bp_eq + moff_eq
                    if emit_eq != cutlass.Int32(0) and wpos_p2 < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[wpos_p2] = self.dtype(v_p2)
                        output_indices_row[wpos_p2] = smem_vals[ix2]
                base_w2 = base_w2 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # Pad remainder with -self.FLT_MAX / -1
            filled_par = s_iscalars[4]
            if filled_par > cutlass.Int32(kK):
                filled_par = cutlass.Int32(kK)
            ipad = filled_par + tidx
            while ipad < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[ipad] = cutlass.Int32(-1)
                ipad = ipad + cutlass.Int32(num_threads)
        else:
            # ----- Branch C: cand_count < kK -----
            # Emit cand_count + pad
            i10 = tidx
            while i10 < cand_count:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i10] = self.dtype(smem_keys[i10])
                output_indices_row[i10] = smem_vals[i10]
                i10 = i10 + cutlass.Int32(num_threads)
            i11 = cand_count + tidx
            while i11 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i11] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[i11] = cutlass.Int32(-1)
                i11 = i11 + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # block_count_ge_multi<M>: ONE full-N scan, M sorted thresholds.
    # Identical memory path to block_count_ge (vec load + 4-way unroll);
    # M static register counters, M unrolled predicated adds per element.
    # Writes per-thread counts column-major smem_ptcnt_multi[m*T + tid]
    # and block totals s_mt_cnt[m]. Ends with the internal barrier of the
    # staged block reduce (ptcnt_multi visible to all threads after).
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_ge_multi(
        self, input_row, N, s_mt_thr, smem_ptcnt_multi, smem_wcnt_multi,
        s_mt_cnt, tidx, warp_id, lane,
    ):
        M = cutlass.const_expr(self.M_thr)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        thr_frag = cute.make_fragment((M,), cutlass.Float32)
        cnt_frag = cute.make_fragment((M,), cutlass.Int32)
        for m in cutlass.range_constexpr(M):
            thr_frag[m] = s_mt_thr[m]
            cnt_frag[m] = cutlass.Int32(0)

        row_addr = input_row.iterator.toint()
        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_unroll_4:
            rng_frag = cute.make_fragment((vec_w,), self.dtype)
            big_iters = cutlass.Int32(0)
            if N > i + cutlass.Int32(vec_w - 1):
                big_iters = (N - i - cutlass.Int32(vec_w)) // cutlass.Int32(step_elem) + cutlass.Int32(1)
            for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                i_local = i + k * cutlass.Int32(step_elem)
                src_ptr_k = cute.make_ptr(
                    self.dtype, row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                    cute.AddressSpace.gmem, assumed_align=vec_align)
                src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                cute.copy(copy_atom, src_k, rng_frag)
                for j in cutlass.range_constexpr(vec_w):
                    if cutlass.const_expr(self.dtype == cutlass.Float32):
                        vj = rng_frag[j]
                    else:
                        vj = cutlass.Float32(rng_frag[j])
                    for m in cutlass.range_constexpr(M):
                        # branchless predicated add (FSETP+IADD, no divergence)
                        cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < N:
            src_ptr = cute.make_ptr(
                self.dtype, row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem, assumed_align=vec_align)
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + step

        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            for m in cutlass.range_constexpr(M):
                cnt_frag[m] = cnt_frag[m] + cutlass.Int32(v >= thr_frag[m])
            it = it + cutlass.Int32(num_threads)

        # Cache all M per-thread columns (P3 seed for the winning column).
        for m in cutlass.range_constexpr(M):
            smem_ptcnt_multi[m * num_threads + tidx] = cnt_frag[m]

        # M warp reduces -> staged smem -> ONE barrier -> warp0 reduces M cols.
        for m in cutlass.range_constexpr(M):
            wc = self.warp_reduce_sum_i32(cnt_frag[m])
            if lane == 0:
                smem_wcnt_multi[m * num_warps + warp_id] = wc
        cute.arch.barrier()
        if warp_id == cutlass.Int32(0):
            for m in cutlass.range_constexpr(M):
                v = cutlass.Int32(0)
                if lane < cutlass.Int32(num_warps):
                    v = smem_wcnt_multi[m * num_warps + lane]
                total = self.warp_reduce_sum_i32(v)
                if lane == 0:
                    s_mt_cnt[m] = total

    # trace-time helpers for the iter9 native 16-bit ladder path
    def _p2n(self):
        return self.p2_native and self.dtype != cutlass.Float32

    def _kind16(self):
        return "bf16" if self.dtype == cutlass.BFloat16 else "f16"

    def _make_load_copy_atom_u32(self):
        # same vec width/caching as _make_load_copy_atom, u32-typed view
        # (16-bit pairs land packed in 32-bit registers, no repack cost)
        if self.use_constant_hint:
            return cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), cutlass.Int32,
                num_bits_per_copy=self.vec_bits, invariant=True)
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32,
            num_bits_per_copy=self.vec_bits)

    # ------------------------------------------------------------------
    # op20 iter4: fused ladder pass — block_count_ge_multi + slot-append of
    # every v >= thr[pred_col] into per-thread smem slot regions. Slot cursor
    # IS cnt_frag[pred_col] (pre-increment), so overflow == cnt > slot_cap,
    # detectable from the cached per-thread column after the pass.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_collect_multi(
        self, input_row, N, s_mt_thr, smem_ptcnt_multi, smem_wcnt_multi,
        s_mt_cnt, smem_slotk, smem_slotv, tidx, warp_id, lane,
    ):
        M = cutlass.const_expr(self.M_thr)
        PC = cutlass.const_expr(self.pred_col)
        S = cutlass.const_expr(self.slot_cap)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        thr_frag = cute.make_fragment((M,), cutlass.Float32)
        cnt_frag = cute.make_fragment((M,), cutlass.Int32)
        for m in cutlass.range_constexpr(M):
            thr_frag[m] = s_mt_thr[m]
            cnt_frag[m] = cutlass.Int32(0)
        slot_base = tidx * cutlass.Int32(S)

        row_addr = input_row.iterator.toint()
        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_unroll_4:
            big_iters = cutlass.Int32(0)
            if N > i + cutlass.Int32(vec_w - 1):
                big_iters = (N - i - cutlass.Int32(vec_w)) // cutlass.Int32(step_elem) + cutlass.Int32(1)
            if cutlass.const_expr(self._p2n()):
                # iter9: native 16-bit paired path — u32-typed loads (pairs
                # already packed), set.ge/add.rn 16x2 counts for m != PC,
                # packed mask for the collect column (exact slot cursor).
                kind = self._kind16()
                vec_w2 = cutlass.const_expr(vec_w // 2)
                copy_atom_u32 = self._make_load_copy_atom_u32()
                rng2_frag = cute.make_fragment((vec_w2,), cutlass.Int32)
                thr2_frag = cute.make_fragment((M,), cutlass.Int32)
                acc2_frag = cute.make_fragment((M,), cutlass.Int32)
                for m in cutlass.range_constexpr(M):
                    thr2_frag[m] = pack2_16_from_f32(thr_frag[m], kind)
                    acc2_frag[m] = cutlass.Int32(0)
                ccol = cutlass.Int32(0)
                for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                    i_local = i + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        cutlass.Int32, row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem, assumed_align=vec_align)
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w2,)))
                    cute.copy(copy_atom_u32, src_k, rng2_frag)
                    for p in cutlass.range_constexpr(vec_w2):
                        v2 = rng2_frag[p]
                        mpc = setge_mask2_16(v2, thr2_frag[PC], kind)
                        if mpc != cutlass.Int32(0):
                            if (mpc & cutlass.Int32(0xFFFF)) != cutlass.Int32(0):
                                if ccol < cutlass.Int32(S):
                                    smem_slotk[slot_base + ccol] = pair_half_f32_16(v2, 0, kind)
                                    smem_slotv[slot_base + ccol] = i_local + cutlass.Int32(2 * p)
                                ccol = ccol + cutlass.Int32(1)
                            if (mpc >> cutlass.Int32(16)) != cutlass.Int32(0):
                                if ccol < cutlass.Int32(S):
                                    smem_slotk[slot_base + ccol] = pair_half_f32_16(v2, 1, kind)
                                    smem_slotv[slot_base + ccol] = i_local + cutlass.Int32(2 * p + 1)
                                ccol = ccol + cutlass.Int32(1)
                        for m in cutlass.range_constexpr(M):
                            if cutlass.const_expr(m != self.pred_col):
                                acc2_frag[m] = setge_add2_16(acc2_frag[m], v2, thr2_frag[m], kind)
                    if (k & cutlass.Int32(15)) == cutlass.Int32(15):
                        for m in cutlass.range_constexpr(M):
                            if cutlass.const_expr(m != self.pred_col):
                                cnt_frag[m] = cnt_frag[m] + pair_sum_i32_16(acc2_frag[m], kind)
                                acc2_frag[m] = cutlass.Int32(0)
                for m in cutlass.range_constexpr(M):
                    if cutlass.const_expr(m != self.pred_col):
                        cnt_frag[m] = cnt_frag[m] + pair_sum_i32_16(acc2_frag[m], kind)
                cnt_frag[PC] = ccol
            else:
                rng_frag = cute.make_fragment((vec_w,), self.dtype)
                for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                    i_local = i + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        self.dtype, row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem, assumed_align=vec_align)
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                    cute.copy(copy_atom, src_k, rng_frag)
                    for j in cutlass.range_constexpr(vec_w):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj = rng_frag[j]
                        else:
                            vj = cutlass.Float32(rng_frag[j])
                        if vj >= thr_frag[PC]:
                            cpos = cnt_frag[PC]
                            if cpos < cutlass.Int32(S):
                                smem_slotk[slot_base + cpos] = vj
                                smem_slotv[slot_base + cpos] = i_local + cutlass.Int32(j)
                        for m in cutlass.range_constexpr(M):
                            cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < N:
            src_ptr = cute.make_ptr(
                self.dtype, row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem, assumed_align=vec_align)
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= thr_frag[PC]:
                    cpos = cnt_frag[PC]
                    if cpos < cutlass.Int32(S):
                        smem_slotk[slot_base + cpos] = vj
                        smem_slotv[slot_base + cpos] = i + cutlass.Int32(j)
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + step

        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= thr_frag[PC]:
                cpos = cnt_frag[PC]
                if cpos < cutlass.Int32(S):
                    smem_slotk[slot_base + cpos] = v
                    smem_slotv[slot_base + cpos] = it
            for m in cutlass.range_constexpr(M):
                cnt_frag[m] = cnt_frag[m] + cutlass.Int32(v >= thr_frag[m])
            it = it + cutlass.Int32(num_threads)

        for m in cutlass.range_constexpr(M):
            smem_ptcnt_multi[m * num_threads + tidx] = cnt_frag[m]
        for m in cutlass.range_constexpr(M):
            wc = self.warp_reduce_sum_i32(cnt_frag[m])
            if lane == 0:
                smem_wcnt_multi[m * num_warps + warp_id] = wc
        cute.arch.barrier()
        if warp_id == cutlass.Int32(0):
            for m in cutlass.range_constexpr(M):
                v = cutlass.Int32(0)
                if lane < cutlass.Int32(num_warps):
                    v = smem_wcnt_multi[m * num_warps + lane]
                total = self.warp_reduce_sum_i32(v)
                if lane == 0:
                    s_mt_cnt[m] = total

    # ------------------------------------------------------------------
    # op20 iter4: P3 from slots — identical output contract to
    # phase3_sandwich (didx flush to output[0:M0), band -> smem_keys/vals,
    # s_iscalars[4]=M0, s_iscalars[0]=band) but the source is the per-thread
    # slot regions collected during the ladder pass, NOT a full-N rescan.
    # Precondition: thr_best >= thr[pred_col] (slots are a superset) and no
    # per-thread slot overflow.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_from_slots(
        self, smem_slotk, smem_slotv, smem_keys, smem_vals, smem_ptcnt,
        smem_ptcnt_up, smem_ptcnt_multi, smem_wcnt, smem_didx, s_thr, s_swf,
        s_iscalars, output_indices_row, tidx, warp_id, lane,
    ):
        kCC = cutlass.const_expr(self.kC)
        PC = cutlass.const_expr(self.pred_col)
        S = cutlass.const_expr(self.slot_cap)
        num_threads = cutlass.const_expr(self.num_threads)

        my_up = smem_ptcnt_up[tidx]
        my_band = smem_ptcnt[tidx] - my_up
        my_pk = (my_up << cutlass.Int32(16)) | my_band
        tp0 = my_pk
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp0, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp0 = tp0 + other
        my_excl_pk = tp0 - my_pk
        warp_tot_pk = cute.arch.shuffle_sync(tp0, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_tot_pk
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                c = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + c
            s_iscalars[4] = tot >> cutlass.Int32(16)
            s_iscalars[0] = tot & cutlass.Int32(0xFFFF)
        cute.arch.barrier()
        base_pk = smem_wcnt[warp_id] + my_excl_pk
        wc0 = base_pk >> cutlass.Int32(16)
        wcb = base_pk & cutlass.Int32(0xFFFF)

        thr1 = s_thr[0]
        thr0 = s_swf[0]
        my_lc = smem_ptcnt_multi[cutlass.Int32(PC) * cutlass.Int32(num_threads) + tidx]
        if my_lc > cutlass.Int32(S):
            my_lc = cutlass.Int32(S)
        slot_base = tidx * cutlass.Int32(S)
        sw = cutlass.Int32(0)
        while sw < my_lc:
            v = smem_slotk[slot_base + sw]
            if v >= thr1:
                if v >= thr0:
                    if wc0 < cutlass.Int32(self.top_k):
                        smem_didx[wc0] = smem_slotv[slot_base + sw]
                        wc0 = wc0 + cutlass.Int32(1)
                elif wcb < cutlass.Int32(kCC):
                    smem_keys[wcb] = v
                    smem_vals[wcb] = smem_slotv[slot_base + sw]
                    wcb = wcb + cutlass.Int32(1)
            sw = sw + cutlass.Int32(1)
        cute.arch.barrier()

        m0t = s_iscalars[4]
        iF = tidx
        while iF < m0t:
            output_indices_row[iF] = smem_didx[iF]
            iF = iF + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Phase-3 sandwich: dual-predicate scan, direct-write + band-collect.
    # smem_ptcnt      = per-thread counts at thr1 (winning column, cached)
    # smem_ptcnt_up   = per-thread counts at thr0 (sandwich column, cached)
    # After: output_indices_row[0:M0] = direct top-K members;
    #        smem_keys/vals[0:band]   = band candidates;
    #        s_iscalars[0]            = band count.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_sandwich(
        self, input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_ptcnt_up,
        smem_wcnt, smem_didx, s_thr, s_swf, s_iscalars,
        output_values_row, output_indices_row, tidx, warp_id, lane,
    ):
        kCC = cutlass.const_expr(self.kC)
        num_threads = cutlass.const_expr(self.num_threads)

        # ---- ONE packed prefix sum: (direct << 16) | band positions ----
        # Safe: per-thread counts < 2^16, block totals M0 <= K <= 2048 and
        # band <= kC <= 6144 (same bound as block_fused_snap_iter packing).
        my_up = smem_ptcnt_up[tidx]
        my_band = smem_ptcnt[tidx] - my_up
        my_pk = (my_up << cutlass.Int32(16)) | my_band
        tp0 = my_pk
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp0, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp0 = tp0 + other
        my_excl_pk = tp0 - my_pk
        warp_tot_pk = cute.arch.shuffle_sync(tp0, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_tot_pk
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                c = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + c
            s_iscalars[4] = tot >> cutlass.Int32(16)          # M0 total
            s_iscalars[0] = tot & cutlass.Int32(0xFFFF)       # band count
        cute.arch.barrier()
        base_pk = smem_wcnt[warp_id] + my_excl_pk
        my_pos0 = base_pk >> cutlass.Int32(16)
        my_posb = base_pk & cutlass.Int32(0xFFFF)

        # ---- fused stream-write scan ----
        thr1 = s_thr[0]
        thr0 = s_swf[0]
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        row_addr = input_row.iterator.toint()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        n_aligned = (N // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        wc0 = my_pos0
        wcb = my_posb
        ic = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_phase3_unroll:
            if self.enable_unroll_4:
                rng_frag = cute.make_fragment((vec_w,), self.dtype)
                big_iters = cutlass.Int32(0)
                if N > ic + cutlass.Int32(vec_w - 1):
                    big_iters = (N - ic - cutlass.Int32(vec_w)) // cutlass.Int32(
                        step_elem) + cutlass.Int32(1)
                for k in cutlass.range(big_iters, unroll=4):
                    ic_local = ic + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        self.dtype,
                        row_addr + cutlass.Int64(ic_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem, assumed_align=vec_align)
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                    cute.copy(copy_atom, src_k, rng_frag)
                    for j in cutlass.range_constexpr(vec_w):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj = rng_frag[j]
                        else:
                            vj = cutlass.Float32(rng_frag[j])
                        # single rare outer branch (op18-P3 shape): keeps the
                        # 4-way LSU pipeline intact on the common miss path
                        if vj >= thr1:
                            if vj >= thr0:
                                if wc0 < cutlass.Int32(self.top_k):
                                    smem_didx[wc0] = ic_local + cutlass.Int32(j)
                                    wc0 = wc0 + cutlass.Int32(1)
                            elif wcb < cutlass.Int32(kCC):
                                smem_keys[wcb] = vj
                                smem_vals[wcb] = ic_local + cutlass.Int32(j)
                                wcb = wcb + cutlass.Int32(1)
                ic = ic + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while ic + cutlass.Int32(vec_w - 1) < N:
            src_ptr = cute.make_ptr(
                self.dtype, row_addr + cutlass.Int64(ic) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem, assumed_align=vec_align)
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= thr1:
                    if vj >= thr0:
                        if wc0 < cutlass.Int32(self.top_k):
                            smem_didx[wc0] = ic + cutlass.Int32(j)
                            wc0 = wc0 + cutlass.Int32(1)
                    elif wcb < cutlass.Int32(kCC):
                        smem_keys[wcb] = vj
                        smem_vals[wcb] = ic + cutlass.Int32(j)
                        wcb = wcb + cutlass.Int32(1)
            ic = ic + step

        it = n_aligned + tidx
        while it < N:
            v = self._load_fp32(input_row, it)
            if v >= thr1:
                if v >= thr0:
                    if wc0 < cutlass.Int32(self.top_k):
                        smem_didx[wc0] = it
                        wc0 = wc0 + cutlass.Int32(1)
                elif wcb < cutlass.Int32(kCC):
                    smem_keys[wcb] = v
                    smem_vals[wcb] = it
                    wcb = wcb + cutlass.Int32(1)
            it = it + cutlass.Int32(num_threads)
        cute.arch.barrier()

        # coalesced flush of the deferred direct-write indices -> output[0:M0)
        m0t = s_iscalars[4]
        iF = tidx
        while iF < m0t:
            output_indices_row[iF] = smem_didx[iF]
            iF = iF + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # Runtime-k snap iteration (block_fused_snap_iter with k_target arg).
    # ------------------------------------------------------------------
    @cute.jit
    def block_band_snap_iter(
        self, smem_keys, smem_wcnt, smem_hist, s_thr, s_iscalars, count,
        k_target, tidx, warp_id, lane,
    ):
        num_threads = cutlass.const_expr(self.num_threads)
        thr = s_thr[0]

        lge = cutlass.Int32(0)
        lgt = cutlass.Int32(0)
        s_up = cutlass.Float32(self.FLT_MAX)
        s_down = cutlass.Float32(self.NEG_FLT_MAX)
        isi = tidx
        while isi < count:
            v = smem_keys[isi]
            if v >= thr:
                lge = lge + cutlass.Int32(1)
            if v > thr:
                lgt = lgt + cutlass.Int32(1)
                s_up = _fmin_f32_inline(s_up, v)
            if v < thr:
                s_down = cute.arch.fmax(s_down, v)
            isi = isi + cutlass.Int32(num_threads)

        packed = (lge << cutlass.Int32(16)) | lgt
        packed = self.warp_reduce_sum_i32(packed)
        s_up = self.warp_reduce_min_f32(s_up)
        s_down = self.warp_reduce_max_f32(s_down)
        if lane == 0:
            smem_wcnt[warp_id] = packed
            smem_hist[warp_id] = float_as_uint32(s_up)
            smem_hist[self.num_warps + warp_id] = float_as_uint32(s_down)
        cute.arch.barrier()

        if tidx == 0:
            tp = cutlass.Int32(0)
            total_up = cutlass.Float32(self.FLT_MAX)
            total_down = cutlass.Float32(self.NEG_FLT_MAX)
            for w in cutlass.range_constexpr(self.num_warps):
                tp = tp + smem_wcnt[w]
                vu = llvm.bitcast(cutlass.Float32.mlir_type, smem_hist[w].ir_value())
                vd = llvm.bitcast(cutlass.Float32.mlir_type,
                                  smem_hist[self.num_warps + w].ir_value())
                total_up = _fmin_f32_inline(total_up, cutlass.Float32(vu))
                total_down = cute.arch.fmax(total_down, cutlass.Float32(vd))
            cge = tp >> cutlass.Int32(16)
            cgt = tp & cutlass.Int32(0xFFFF)
            s_iscalars[2] = cge
            s_iscalars[3] = cgt
            if cgt >= k_target:
                if total_up < cutlass.Float32(self.FLT_MAX):
                    s_thr[0] = total_up
            elif cge < k_target:
                if total_down > cutlass.Float32(self.NEG_FLT_MAX):
                    s_thr[0] = total_down
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # Phase-4 band refine dispatcher: pick k_rem winners from band candidates
    # in smem_keys/vals[0:band] (all in [thr1, thr0)), writing output
    # positions m0..K-1. Compile-time switch between the exact rank-scatter
    # (default) and the legacy runtime-k histogram snap (A/B reference).
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_band_snap(
        self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
        s_iscalars, output_values_row, output_indices_row, band, k_rem, m0,
        tidx, warp_id, lane,
    ):
        if cutlass.const_expr(self.p4_rank_scatter):
            self.phase4_band_rank_scatter(
                smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
                s_iscalars, output_values_row, output_indices_row, band,
                k_rem, m0, tidx, warp_id, lane)
        else:
            self.phase4_band_snap_hist(
                smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
                s_iscalars, output_values_row, output_indices_row, band,
                k_rem, m0, tidx, warp_id, lane)

    # ------------------------------------------------------------------
    # op21 iter5/6/11: rank-scatter band refine (op8 phase4_rank_scatter
    # lineage). Differences vs op8: histogram range is the KNOWN
    # [thr1, thr0) band bracket (no min/max pass), rank target is the
    # runtime k_rem (not const K), all output positions offset by m0.
    # Chain: coarse kBins hist -> straddling bin b* + rank_above -> iter6
    # fast paths on b*: (A) whole-bin equality emit, (B) <=32 members ->
    # warp0 exact register ranking; (C) fallback = the EXACT value-edge
    # band snap (iter11). NOTE (iter11 falsification): a fixed-depth
    # sub-histogram of b* is NOT a valid path C — a fine bin is a value
    # INTERVAL, not a tie group, so cutting it at k_rem in stash order can
    # emit below the true K-th rank (upstream revert ec04147502; local
    # proof smoke_adversarial_band.py 72/72). Only paths that terminate on
    # a DATA value (register ranking, value-edge snap) are exact.
    # Scratch: s_iscalars[0..4] (all dead at P4 entry — P4 is terminal at
    # every call site); path B stashes b* members in smem_hist[8..39]
    # (coarse hist dead there).
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_band_rank_scatter(
        self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
        s_iscalars, output_values_row, output_indices_row, band, k_rem, m0,
        tidx, warp_id, lane,
    ):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        if band == k_rem:
            # every band candidate is a winner: emit at offset m0
            i4 = tidx
            while i4 < band:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[m0 + i4] = self.dtype(smem_keys[i4])
                output_indices_row[m0 + i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        else:
            # ---- coarse histogram over the known band range ----
            bmin_r = s_thr[0]
            bmax_r = s_swf[0]
            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            i6 = tidx
            while i6 < cutlass.Int32(kBins):
                smem_hist[i6] = cutlass.Int32(0)
                i6 = i6 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            range1 = bmax_r - bmin_r
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            i7 = tidx
            while i7 < band:
                vk = smem_keys[i7]
                bin_f = (vk - bmin_r) * inv1
                bin_i = cutlass.Int32(bin_f)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # ---- 3-step high->low search: straddling bin b* + rank_above ----
            warp_bin_sum = cutlass.Int32(0)
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (cutlass.Int32(kBins - 1)
                          - warp_id * cutlass.Int32(bins_per_warp)
                          - cutlass.Int32(jb))
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            if lane == 0:
                smem_wcnt[warp_id] = warp_bin_sum
            cute.arch.barrier()

            if tidx == 0:
                cum = cutlass.Int32(0)
                tw = cutlass.Int32(num_warps - 1)
                found = cutlass.Int32(0)
                for w2 in cutlass.range_constexpr(self.num_warps):
                    cum = cum + smem_wcnt[w2]
                    if cum >= k_rem and found == cutlass.Int32(0):
                        tw = cutlass.Int32(w2)
                        found = cutlass.Int32(1)
                cum2 = cutlass.Int32(0)
                for w3 in cutlass.range_constexpr(self.num_warps):
                    if cutlass.Int32(w3) < tw:
                        cum2 = cum2 + smem_wcnt[w3]
                s_iscalars[2] = cum2
                s_iscalars[3] = tw
            cute.arch.barrier()

            target_warp = s_iscalars[3]
            if warp_id == target_warp and lane == cutlass.Int32(0):
                base_cum = s_iscalars[2]
                b_star_l = cutlass.Int32(kBins - 1)
                rank_above_l = base_cum
                set_d = cutlass.Int32(0)
                for jb2 in cutlass.range_constexpr(bins_per_warp):
                    bidx2 = (cutlass.Int32(kBins - 1)
                             - target_warp * cutlass.Int32(bins_per_warp)
                             - cutlass.Int32(jb2))
                    ra_b = base_cum
                    base_cum = base_cum + smem_hist[bidx2]
                    if base_cum >= k_rem and set_d == cutlass.Int32(0):
                        b_star_l = bidx2
                        rank_above_l = ra_b  # count in bins strictly above b*
                        set_d = cutlass.Int32(1)
                s_iscalars[2] = rank_above_l
                s_iscalars[3] = b_star_l
            cute.arch.barrier()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]

            # ---- iter6 small-bin fast paths (host probe: cnt(b*) p50=2
            # p90=3 max=4 on 68 synth+real rows — the snap fallback is
            # cold in practice) ----
            if cutlass.const_expr(self.p4_smallbin):
                cnt_bstar = smem_hist[b_star]
                r_need = k_rem - rank_above
                if cnt_bstar <= cutlass.Int32(32):
                    # path B: stash the <=32 b* members, warp0 exact
                    # register ranking, direct positional emit (no atomics
                    # on the b* group, no fine hist, 2 barriers total)
                    if tidx == 0:
                        s_iscalars[4] = cutlass.Int32(0)  # cnt_above cursor
                        s_iscalars[1] = cutlass.Int32(0)  # b* stash cursor
                    cute.arch.barrier()
                    isc = tidx
                    while isc < band:
                        v = smem_keys[isc]
                        bin_i = cutlass.Int32((v - bmin_r) * inv1)
                        if bin_i < cutlass.Int32(0):
                            bin_i = cutlass.Int32(0)
                        if bin_i > cutlass.Int32(kBins - 1):
                            bin_i = cutlass.Int32(kBins - 1)
                        if bin_i > b_star:
                            o_a = atomicAdd(s_iscalars.iterator + cutlass.Int32(4),
                                            cutlass.Int32(1))
                            pos = m0 + o_a
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif bin_i == b_star:
                            js = atomicAdd(s_iscalars.iterator + cutlass.Int32(1),
                                           cutlass.Int32(1))
                            if js < cutlass.Int32(32):
                                # stash band index; safe even if b_star lands
                                # in [8,40): cnt_bstar is already in a
                                # register and the coarse hist is dead here
                                smem_hist[cutlass.Int32(8) + js] = isc
                        isc = isc + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    if warp_id == cutlass.Int32(0):
                        v_i = cutlass.Float32(self.NEG_FLT_MAX)
                        bidx_i = cutlass.Int32(0)
                        if lane < cnt_bstar:
                            bidx_i = smem_hist[cutlass.Int32(8) + lane]
                            v_i = smem_keys[bidx_i]
                        rankp = cutlass.Int32(0)
                        for src_c in cutlass.range_constexpr(32):
                            v_o = cute.arch.shuffle_sync(
                                v_i, cutlass.Int32(src_c))
                            if cutlass.Int32(src_c) < cnt_bstar:
                                if v_o > v_i:
                                    rankp = rankp + cutlass.Int32(1)
                                elif v_o == v_i and cutlass.Int32(src_c) < lane:
                                    rankp = rankp + cutlass.Int32(1)
                        if lane < cnt_bstar and rankp < r_need:
                            pos = m0 + rank_above + rankp
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v_i)
                                output_indices_row[pos] = smem_vals[bidx_i]
                    cute.arch.barrier()
                elif cnt_bstar == r_need:
                    # path A: every b* member is a winner — whole-bin emit,
                    # skip the fallback entirely
                    if tidx == 0:
                        s_iscalars[4] = cutlass.Int32(0)  # cnt_above cursor
                        s_iscalars[1] = cutlass.Int32(0)  # b* cursor
                    cute.arch.barrier()
                    isc = tidx
                    while isc < band:
                        v = smem_keys[isc]
                        bin_i = cutlass.Int32((v - bmin_r) * inv1)
                        if bin_i < cutlass.Int32(0):
                            bin_i = cutlass.Int32(0)
                        if bin_i > cutlass.Int32(kBins - 1):
                            bin_i = cutlass.Int32(kBins - 1)
                        if bin_i > b_star:
                            o_a = atomicAdd(s_iscalars.iterator + cutlass.Int32(4),
                                            cutlass.Int32(1))
                            pos = m0 + o_a
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        elif bin_i == b_star:
                            o_s = atomicAdd(s_iscalars.iterator + cutlass.Int32(1),
                                            cutlass.Int32(1))
                            pos = m0 + rank_above + o_s
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(v)
                                output_indices_row[pos] = smem_vals[isc]
                        isc = isc + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                else:
                    # path C (distribution shift, never fires on real/synth
                    # probes): fall back to the EXACT value-edge snap on the
                    # whole band. iter11: the previous fixed-depth fine
                    # scatter here was NOT exact on arbitrary continuous
                    # logits (same failure mode upstream reverted in
                    # ec04147502; falsified by smoke_adversarial_band.py,
                    # 72/72) — a fixed-depth histogram cannot separate two
                    # distinct values in one (sub-)bin, so the straddling
                    # bin's stash-order emit can pick a value below the true
                    # K-th rank. The snap converges sel_thr onto an actual
                    # DATA value (block_band_snap_iter steps to value
                    # edges), so its ==sel_thr group is a true tie group and
                    # any k_rem-cut of it is exact.
                    self.phase4_band_snap_hist(
                        smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr,
                        s_swf, s_iscalars, output_values_row,
                        output_indices_row, band, k_rem, m0, tidx, warp_id,
                        lane)
            else:
                # p4_smallbin=False (OP21_P4_FAST=0): fast paths disabled —
                # everything falls to the exact snap (iter11; the old
                # always-fine reference was retired with the inexact fine
                # scatter).
                self.phase4_band_snap_hist(
                    smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
                    s_iscalars, output_values_row, output_indices_row, band,
                    k_rem, m0, tidx, warp_id, lane)

    # ------------------------------------------------------------------
    # Legacy phase-4 band snap (A/B reference): runtime-k histogram snap.
    # Histogram range seeded [thr1, thr0). out_count starts at M0 so the
    # writeback fills output positions M0..K-1.
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_band_snap_hist(
        self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf,
        s_iscalars, output_values_row, output_indices_row, band, k_rem, m0,
        tidx, warp_id, lane,
    ):
        kK = cutlass.const_expr(self.top_k)
        kBins = cutlass.const_expr(self.kNumBins)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        bins_per_warp = cutlass.const_expr(kBins // self.num_warps)

        if band == k_rem:
            # every band candidate is a winner: emit at offset m0
            i4 = tidx
            while i4 < band:
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[m0 + i4] = self.dtype(smem_keys[i4])
                output_indices_row[m0 + i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
        else:
            # band > k_rem: histogram + runtime-k snap. Range = [thr1, thr0)
            # by construction — no min/max pass needed.
            bmin_r = s_thr[0]
            bmax_r = s_swf[0]
            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            i6 = tidx
            while i6 < cutlass.Int32(kBins):
                smem_hist[i6] = cutlass.Int32(0)
                i6 = i6 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            range1 = bmax_r - bmin_r
            inv1 = (cutlass.Float32(kBins - 1) + cutlass.Float32(0.99)) / range1
            i7 = tidx
            while i7 < band:
                vk = smem_keys[i7]
                bin_f = (vk - bmin_r) * inv1
                bin_i = cutlass.Int32(bin_f)
                if bin_i < cutlass.Int32(0):
                    bin_i = cutlass.Int32(0)
                if bin_i > cutlass.Int32(kBins - 1):
                    bin_i = cutlass.Int32(kBins - 1)
                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            # k_rem-th bin search (runtime k)
            warp_bin_sum = cutlass.Int32(0)
            for jb in cutlass.range_constexpr(bins_per_warp):
                bidx_s = (cutlass.Int32(kBins - 1)
                          - warp_id * cutlass.Int32(bins_per_warp)
                          - cutlass.Int32(jb))
                warp_bin_sum = warp_bin_sum + smem_hist[bidx_s]
            if lane == 0:
                smem_wcnt[warp_id] = warp_bin_sum
            cute.arch.barrier()

            if tidx == 0:
                cum = cutlass.Int32(0)
                tw = cutlass.Int32(num_warps - 1)
                found = cutlass.Int32(0)
                for w2 in cutlass.range_constexpr(self.num_warps):
                    cum = cum + smem_wcnt[w2]
                    if cum >= k_rem and found == cutlass.Int32(0):
                        tw = cutlass.Int32(w2)
                        found = cutlass.Int32(1)
                cum2 = cutlass.Int32(0)
                for w3 in cutlass.range_constexpr(self.num_warps):
                    if cutlass.Int32(w3) < tw:
                        cum2 = cum2 + smem_wcnt[w3]
                s_iscalars[2] = cum2
                s_iscalars[3] = tw
            cute.arch.barrier()

            target_warp = s_iscalars[3]
            if warp_id == target_warp and lane == cutlass.Int32(0):
                base_cum = s_iscalars[2]
                thr_local = bmin_r
                set_done = cutlass.Int32(0)
                for jb2 in cutlass.range_constexpr(bins_per_warp):
                    bidx2 = (cutlass.Int32(kBins - 1)
                             - target_warp * cutlass.Int32(bins_per_warp)
                             - cutlass.Int32(jb2))
                    base_cum = base_cum + smem_hist[bidx2]
                    if base_cum >= k_rem and set_done == cutlass.Int32(0):
                        thr_local = bmin_r + cutlass.Float32(bidx2) * range1 / cutlass.Float32(kBins)
                        set_done = cutlass.Int32(1)
                s_thr[0] = thr_local
            cute.arch.barrier()

            # snap convergence: cgt < k_rem <= cge
            si = cutlass.Int32(0)
            done_snap = cutlass.Int32(0)
            while si < band and done_snap == cutlass.Int32(0):
                self.block_band_snap_iter(smem_keys, smem_wcnt, smem_hist,
                                          s_thr, s_iscalars, band, k_rem,
                                          tidx, warp_id, lane)
                if s_iscalars[3] < k_rem and s_iscalars[2] >= k_rem:
                    done_snap = cutlass.Int32(1)
                si = si + cutlass.Int32(1)

            # two-pass writeback at offset m0 (out_count starts at m0)
            sel_thr = s_thr[0]
            if tidx == 0:
                s_iscalars[4] = m0
            cute.arch.barrier()

            base_w = warp_id * cutlass.Int32(self.WARP_SIZE)
            while base_w < band:
                ix1 = base_w + lane
                emit_gt = cutlass.Int32(0)
                v_p1 = cutlass.Float32(self.NEG_FLT_MAX)
                if ix1 < band:
                    v_p1 = smem_keys[ix1]
                    if v_p1 > sel_thr:
                        emit_gt = cutlass.Int32(1)
                mask_gt = cute.arch.vote_ballot_sync(emit_gt != cutlass.Int32(0))
                if mask_gt != cutlass.Uint32(0):
                    cnt_gt = cutlass.Int32(cute.arch.popc(mask_gt))
                    lane_mask_gt = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
                    moff_gt = cutlass.Int32(cute.arch.popc(mask_gt & lane_mask_gt))
                    bp_gt = cutlass.Int32(0)
                    if lane == cutlass.Int32(0):
                        bp_gt = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cnt_gt)
                    bp_gt = cute.arch.shuffle_sync(bp_gt, cutlass.Int32(0))
                    wpos_p1 = bp_gt + moff_gt
                    if emit_gt != cutlass.Int32(0) and wpos_p1 < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[wpos_p1] = self.dtype(v_p1)
                        output_indices_row[wpos_p1] = smem_vals[ix1]
                base_w = base_w + cutlass.Int32(num_threads)
            cute.arch.barrier()

            base_w2 = warp_id * cutlass.Int32(self.WARP_SIZE)
            while base_w2 < band:
                ix2 = base_w2 + lane
                emit_eq = cutlass.Int32(0)
                v_p2 = cutlass.Float32(self.NEG_FLT_MAX)
                if ix2 < band:
                    v_p2 = smem_keys[ix2]
                    if v_p2 == sel_thr:
                        emit_eq = cutlass.Int32(1)
                mask_eq = cute.arch.vote_ballot_sync(emit_eq != cutlass.Int32(0))
                if mask_eq != cutlass.Uint32(0):
                    cnt_eq = cutlass.Int32(cute.arch.popc(mask_eq))
                    lane_mask_eq = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
                    moff_eq = cutlass.Int32(cute.arch.popc(mask_eq & lane_mask_eq))
                    bp_eq = cutlass.Int32(0)
                    if lane == cutlass.Int32(0):
                        bp_eq = atomicAdd(s_iscalars.iterator + cutlass.Int32(4), cnt_eq)
                    bp_eq = cute.arch.shuffle_sync(bp_eq, cutlass.Int32(0))
                    wpos_p2 = bp_eq + moff_eq
                    if emit_eq != cutlass.Int32(0) and wpos_p2 < cutlass.Int32(kK):
                        if cutlass.const_expr(self.return_output_values):
                            output_values_row[wpos_p2] = self.dtype(v_p2)
                        output_indices_row[wpos_p2] = smem_vals[ix2]
                base_w2 = base_w2 + cutlass.Int32(num_threads)
            cute.arch.barrier()

            filled_par = s_iscalars[4]
            if filled_par > cutlass.Int32(kK):
                filled_par = cutlass.Int32(kK)
            ipad = filled_par + tidx
            while ipad < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[ipad] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[ipad] = cutlass.Int32(-1)
                ipad = ipad + cutlass.Int32(num_threads)

    # ------------------------------------------------------------------
    # op21 FIX (root cause of the op18/19 real-data red-card): the vendored
    # done!=1 retry-shrink exits on the FIRST count <= kCC — INCLUDING
    # count < K (undershoot: -1 slots) — and its bracket hi end carries no
    # count < K guarantee (hit~0 rows can have >= K elements above v_hi).
    # This override runs a CORRECT bounded bisection to land count in
    # [kK, kCC] (expanding hi to +LARGE first if count(hi) >= kK), marks
    # done=1, then delegates to the stream-write worker
    # (contract: smem_ptcnt caches per-thread counts of the LAST
    # block_count_ge at the final s_thr[0]). If 30 iters exhaust (tie-block,
    # ~16-bit only), it lands on the undershoot side = production fail-soft
    # semantics rather than an arbitrary truncated subset.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_collect_candidates(self, input_row, N, smem_keys, smem_vals,
                                  smem_ptcnt, smem_wcnt, s_thr, s_iscalars,
                                  tidx, warp_id, lane):
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        if s_iscalars[1] != cutlass.Int32(1):
            # entry count at the current threshold (also warms smem_ptcnt)
            self.block_count_ge(input_row, N, s_thr[0], smem_ptcnt, smem_wcnt,
                                s_iscalars, tidx, warp_id, lane)
            cute.arch.barrier()  # block_count_ge has NO trailing barrier
            need = cutlass.Int32(0)
            if s_iscalars[0] > cutlass.Int32(kCC) or s_iscalars[0] < cutlass.Int32(kK):
                need = cutlass.Int32(1)
            if need == cutlass.Int32(1):
                if tidx == 0:
                    if s_iscalars[0] > cutlass.Int32(kCC):
                        s_thr[1] = s_thr[0]
                    else:
                        s_thr[2] = s_thr[0]
                cute.arch.barrier()
                # hi-end guarantee: count(hi) must be < kK, else expand
                self.block_count_ge(input_row, N, s_thr[2], smem_ptcnt,
                                    smem_wcnt, s_iscalars, tidx, warp_id, lane)
                cute.arch.barrier()
                if s_iscalars[0] >= cutlass.Int32(kK) and s_iscalars[0] <= cutlass.Int32(kCC):
                    # hi itself already valid: adopt it
                    if tidx == 0:
                        s_thr[0] = s_thr[2]
                    cute.arch.barrier()
                else:
                    # geometric expansion: double the bracket upward until
                    # count(hi) < kK (hit~0 rows can have >= K elements above
                    # the max gathered value; a one-shot huge hi would need
                    # ~96 bisection steps — doubling keeps the bracket tight)
                    ex = cutlass.Int32(0)
                    while ex < cutlass.Int32(12) and s_iscalars[0] >= cutlass.Int32(kK):
                        if tidx == 0:
                            rngx = s_thr[2] - s_thr[1]
                            if rngx <= cutlass.Float32(0.0):
                                rngx = cutlass.Float32(1e-3)
                            s_thr[2] = s_thr[2] + rngx
                        cute.arch.barrier()
                        self.block_count_ge(input_row, N, s_thr[2], smem_ptcnt,
                                            smem_wcnt, s_iscalars, tidx,
                                            warp_id, lane)
                        cute.arch.barrier()
                        ex = ex + cutlass.Int32(1)
                    if tidx == 0:
                        s_thr[0] = s_thr[2]  # provisional; loop re-places
                    cute.arch.barrier()
                    rs = cutlass.Int32(0)
                    ok3 = cutlass.Int32(0)
                    while rs < cutlass.Int32(30) and ok3 == cutlass.Int32(0):
                        if tidx == 0:
                            lo3 = s_thr[1]
                            hi3 = s_thr[2]
                            mid3 = (lo3 + hi3) * cutlass.Float32(0.5)
                            if mid3 <= lo3:
                                mid3 = hi3
                            s_thr[0] = mid3
                        cute.arch.barrier()
                        self.block_count_ge(input_row, N, s_thr[0], smem_ptcnt,
                                            smem_wcnt, s_iscalars, tidx,
                                            warp_id, lane)
                        cute.arch.barrier()
                        c3 = s_iscalars[0]
                        if c3 >= cutlass.Int32(kK) and c3 <= cutlass.Int32(kCC):
                            ok3 = cutlass.Int32(1)
                        if tidx == 0:
                            if c3 > cutlass.Int32(kCC):
                                s_thr[1] = s_thr[0]
                            elif c3 < cutlass.Int32(kK):
                                s_thr[2] = s_thr[0]
                        cute.arch.barrier()
                        rs = rs + cutlass.Int32(1)
                    if ok3 == cutlass.Int32(0):
                        # tie-block: land on the undershoot (fail-soft) side
                        self.block_count_ge(input_row, N, s_thr[2], smem_ptcnt,
                                            smem_wcnt, s_iscalars, tidx,
                                            warp_id, lane)
                        cute.arch.barrier()
                        if tidx == 0:
                            s_thr[0] = s_thr[2]
                        cute.arch.barrier()
            if tidx == 0:
                s_iscalars[1] = cutlass.Int32(1)  # converged: skip vendored retry
            cute.arch.barrier()
        self.phase3_collect_stream_write(
            input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_wcnt,
            s_thr, s_iscalars, tidx, warp_id, lane)

    # ------------------------------------------------------------------
    # op21: phase1 variant that gathers the prev-K values ONCE — computing
    # the min/max/mean stats of the valid gathered values AND
    # stashing every gathered value into smem_stash (the P3 slot buffer,
    # free until the ladder pass). phase1b then histograms from smem, so
    # the K-load L2 re-gather disappears from the critical path.
    # Invalid preIdx entries stash NEG_FLT_MAX (always < v_lo == min of the
    # valid values), which phase1b skips with a single compare.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1_stats_stash(
        self, input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
        smem_stash, smem_wmin_f32, smem_wmax_f32, smem_wsum_f32,
        smem_wcnt_i32, s_thr, s_iscalars, tidx, warp_id, lane,
    ):
        local_min = cutlass.Float32(self.FLT_MAX)
        local_max = cutlass.Float32(self.NEG_FLT_MAX)
        local_sum = cutlass.Float32(0.0)
        local_cnt = cutlass.Int32(0)
        if cutlass.const_expr(pre_idx_count >= self.num_threads):
            n_iters = cutlass.const_expr(pre_idx_count // self.num_threads)
            for u in cutlass.range_constexpr(n_iters):
                i = tidx + cutlass.Int32(u * self.num_threads)
                raw = pre_idx_row[i]
                idx = raw + pre_idx_offset
                v = cutlass.Float32(self.NEG_FLT_MAX)  # stash sentinel
                if idx >= 0 and idx < N:
                    v = self._load_fp32(input_row, idx)
                    local_max = cute.arch.fmax(local_max, v)
                    local_min = _fmin_f32_inline(local_min, v)
                    local_sum = local_sum + v
                    local_cnt = local_cnt + 1
                smem_stash[i] = v
        else:
            idx = cutlass.Int32(-1)
            if tidx < cutlass.Int32(pre_idx_count):
                idx = pre_idx_row[tidx] + pre_idx_offset
            v = cutlass.Float32(self.NEG_FLT_MAX)
            if idx >= 0 and idx < N:
                v = self._load_fp32(input_row, idx)
                local_max = cute.arch.fmax(local_max, v)
                local_min = _fmin_f32_inline(local_min, v)
                local_sum = local_sum + v
                local_cnt = local_cnt + 1
            if tidx < cutlass.Int32(pre_idx_count):
                smem_stash[tidx] = v
        active_preidx_warps = cutlass.const_expr(
            min(pre_idx_count // self.WARP_SIZE, self.num_warps)
        )
        if cutlass.const_expr(active_preidx_warps < self.num_warps):
            if warp_id < cutlass.Int32(active_preidx_warps):
                wmin = self.warp_reduce_min_f32(local_min)
                wmax = self.warp_reduce_max_f32(local_max)
                wsum = self.warp_reduce_sum_f32(local_sum)
                wcnt = self.warp_reduce_sum_i32(local_cnt)
                if lane == 0:
                    smem_wmin_f32[warp_id] = wmin
                    smem_wmax_f32[warp_id] = wmax
                    smem_wsum_f32[warp_id] = wsum
                    smem_wcnt_i32[warp_id] = wcnt
        else:
            wmin = self.warp_reduce_min_f32(local_min)
            wmax = self.warp_reduce_max_f32(local_max)
            wsum = self.warp_reduce_sum_f32(local_sum)
            wcnt = self.warp_reduce_sum_i32(local_cnt)
            if lane == 0:
                smem_wmin_f32[warp_id] = wmin
                smem_wmax_f32[warp_id] = wmax
                smem_wsum_f32[warp_id] = wsum
                smem_wcnt_i32[warp_id] = wcnt
        cute.arch.barrier()
        if tidx == 0:
            pmin = cutlass.Float32(self.FLT_MAX)
            pmax = cutlass.Float32(self.NEG_FLT_MAX)
            psum = cutlass.Float32(0.0)
            pcnt = cutlass.Int32(0)
            for w in cutlass.range_constexpr(active_preidx_warps):
                v_min = smem_wmin_f32[w]
                v_max = smem_wmax_f32[w]
                v_sum = smem_wsum_f32[w]
                v_cnt = smem_wcnt_i32[w]
                pmax = cute.arch.fmax(pmax, v_max)
                pmin = _fmin_f32_inline(pmin, v_min)
                psum = psum + v_sum
                pcnt = pcnt + v_cnt
            pmean = cutlass.Float32(0.0)
            if pcnt > 0:
                pmean = psum / cutlass.Float32(pcnt)
            else:
                pmean = (pmin + pmax) * cutlass.Float32(0.5)
            cnt_lo_seed = pre_idx_count + (pre_idx_count >> 2)
            s_thr[0] = pmean
            s_thr[1] = pmin
            s_thr[2] = pmax
            s_iscalars[0] = cutlass.Int32(0)  # cand_count
            s_iscalars[1] = cutlass.Int32(0)  # done
            s_iscalars[2] = cutlass.Int32(cnt_lo_seed)  # cnt_lo
            s_iscalars[3] = cutlass.Int32(1)  # cnt_hi
            s_iscalars[4] = cutlass.Int32(0)  # out_count
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # op21 P1b: rank-quantile round-0 placement. Histogram the K
    # prev-step values (stashed in smem by phase1_stats_stash) over
    # [v_lo, v_hi] with QBINS fixed bins (smem_hist is free until P4, which
    # re-zeroes it); a parallel suffix-scan then drops ladder column m+1 at
    # the bin left-edge where the ge-count crosses qfracs[m] * K_valid.
    # Thresholds are SEEDS only; exactness is enforced by the unchanged
    # downstream count/snap machinery.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_rank_quantile(
        self, smem_stash, pre_idx_count,
        smem_hist, s_thr, s_mt_thr, s_mt_cnt, tidx,
    ):
        QBINS = cutlass.const_expr(self.QBINS)
        M = cutlass.const_expr(self.M_thr)
        num_threads = cutlass.const_expr(self.num_threads)
        lo = s_thr[1]
        hi = s_thr[2]
        iz = cutlass.Int32(tidx)
        while iz < cutlass.Int32(QBINS):
            smem_hist[iz] = cutlass.Int32(0)
            iz = iz + cutlass.Int32(num_threads)
        cute.arch.barrier()
        rng = hi - lo
        inv = (cutlass.Float32(QBINS - 1) + cutlass.Float32(0.99)) / rng
        # histogram the phase1-stashed values from smem (no L2 re-gather);
        # NEG_FLT_MAX sentinel entries (invalid preIdx) are < lo and skipped
        if cutlass.const_expr(pre_idx_count >= self.num_threads):
            n_iters = cutlass.const_expr(pre_idx_count // self.num_threads)
            for u in cutlass.range_constexpr(n_iters):
                i = tidx + cutlass.Int32(u * self.num_threads)
                v = smem_stash[i]
                if v >= lo:
                    b = cutlass.Int32((v - lo) * inv)
                    if b > cutlass.Int32(QBINS - 1):
                        b = cutlass.Int32(QBINS - 1)
                    atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
        else:
            v = cutlass.Float32(self.NEG_FLT_MAX)
            if tidx < cutlass.Int32(pre_idx_count):
                v = smem_stash[tidx]
            if v >= lo:
                b = cutlass.Int32((v - lo) * inv)
                if b > cutlass.Int32(QBINS - 1):
                    b = cutlass.Int32(QBINS - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
        cute.arch.barrier()
        # Parallel suffix-scan + crossing detect (v1 ran this cum scan serially
        # on tid0: ~1800 dependent smem ops ~= +30us fixed, N-independent).
        binw = rng / cutlass.Float32(QBINS)
        if tidx == 0:
            s_mt_thr[0] = lo  # g_min anchor: count >= K_valid guaranteed
            for md in cutlass.range_constexpr(M - 1):
                s_mt_thr[md + 1] = lo + rng * (cutlass.Float32(md + 1) / cutlass.Float32(M))
        # in-place Hillis-Steele suffix sum over the QBINS bins (thread b owns
        # bin b; num_threads >= QBINS guaranteed by _config). The tid0 default
        # writes above overlap step 0's reads (disjoint smem).
        for e in cutlass.range_constexpr(self.QBINS.bit_length() - 1):
            step = cutlass.const_expr(1 << e)
            v = cutlass.Int32(0)
            if tidx < cutlass.Int32(QBINS):
                v = smem_hist[tidx]
                if tidx + cutlass.Int32(step) < cutlass.Int32(QBINS):
                    v = v + smem_hist[tidx + cutlass.Int32(step)]
            cute.arch.barrier()
            if tidx < cutlass.Int32(QBINS):
                smem_hist[tidx] = v
            cute.arch.barrier()
        # crossing: the unique largest bin b with suffix(b) >= tgt (== first
        # crossing of the old top-down cum scan) writes column m+1's seed
        if tidx < cutlass.Int32(QBINS):
            sfx = smem_hist[tidx]
            nxt = cutlass.Int32(0)
            if tidx < cutlass.Int32(QBINS - 1):
                nxt = smem_hist[tidx + 1]
            total = smem_hist[0]
            for m in cutlass.range_constexpr(M - 1):
                tgt = cutlass.Int32(cutlass.Float32(total) * cutlass.Float32(self.qfracs[m]))
                if tgt < cutlass.Int32(1):
                    tgt = cutlass.Int32(1)
                if sfx >= tgt and (tidx == cutlass.Int32(QBINS - 1) or nxt < tgt):
                    s_mt_thr[m + 1] = lo + cutlass.Float32(tidx) * binw
        cute.arch.barrier()
        if tidx == 0:
            # enforce non-descending columns (equal columns tolerated: counts
            # equal => straddle logic still picks a valid pair)
            for mm in cutlass.range_constexpr(M - 1):
                pv = s_mt_thr[mm]
                if s_mt_thr[mm + 1] < pv:
                    s_mt_thr[mm + 1] = pv
            if cutlass.const_expr(self._p2n()):
                # iter9: quantize every column to the dtype grid so the
                # native 16-bit ladder compares agree bit-for-bit with the
                # fp32 compares in P3/P4/fallback (cvt.rn is monotonic =>
                # the non-descending property survives; column 0 = g_min is
                # a data value, already on the grid).
                for mq in cutlass.range_constexpr(M):
                    s_mt_thr[mq] = quant_f32_16(s_mt_thr[mq],
                                                self._kind16())
        # visibility: callers ladder-init barrier follows before any read

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
        M = cutlass.const_expr(self.M_thr)
        R = cutlass.const_expr(self.R_rounds)
        bAcc = cutlass.const_expr(self.band_accept)
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
        # NOTE: keep hist and didx SEPARATE. A union buffer (8KB less smem at
        # K2048) raises residency 2->3 CTAs/SM at high BS and measures 0.745x
        # vs separate's 0.893x at K2048/16K/BS2048 — lower occupancy wins in
        # that regime. (measured 2026-07-02)
        smem_hist = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((kNumBins,), order=(0,)), byte_alignment=128)
        smem_ptcnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=128)
        smem_wmin = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wmax = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wsum = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        smem_wcnt_p1 = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_warps,), order=(0,)), byte_alignment=64)
        s_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        s_iscalars = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((5,), order=(0,)), byte_alignment=16)
        smem_ptcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_warps,), order=(0,)), byte_alignment=64)
        s_mt_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        s_mt_cnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        # [0]=br_lo, [1]=br_hi, [2]=best_thr
        s_mstf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        # [0]=best_cnt, [1]=best_col_this_round(-1 none), [2]=continue_flag
        s_msti = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)
        # op19 additions: sandwich upper threshold snapshot
        smem_ptcnt_up = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_didx = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((top_k,), order=(0,)), byte_alignment=128)
        # [0]=thr0
        s_swf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((1,), order=(0,)), byte_alignment=16)
        # [0]=M0, [1]=up_col_this_round (-1 none)
        s_swi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((2,), order=(0,)), byte_alignment=16)
        # op20 iter4: per-thread collect slots for the fused ladder pass
        smem_slotk = None
        smem_slotv = None
        if cutlass.const_expr(self.fuse_collect):
            slot_elems = cutlass.const_expr(self.num_threads * self.slot_cap)
            smem_slotk = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((slot_elems,), order=(0,)), byte_alignment=128)
            smem_slotv = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((slot_elems,), order=(0,)), byte_alignment=128)

        if N <= cutlass.Int32(top_k):
            jd = tidx
            while jd < N:
                output_indices_row[jd] = cutlass.Int32(jd)
                jd = jd + cutlass.Int32(num_threads)
            jp = N + cutlass.Int32(tidx)
            while jp < cutlass.Int32(top_k):
                output_indices_row[jp] = cutlass.Int32(-1)
                jp = jp + cutlass.Int32(num_threads)
        else:
            # P1 gathers once: stats + smem stash (smem_keys is free
            # until P3)
            self.phase1_stats_stash(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                    smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if tidx == 0:
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        je = je + cutlass.Int32(1)
            else:
                # ---- P2: adaptive M-ary ladder + sandwich pair tracking ----
                self.phase1b_rank_quantile(smem_keys, pre_idx_count,
                                           smem_hist, s_thr, s_mt_thr,
                                           s_mt_cnt, tidx)
                smem_ptcnt_up[tidx] = cutlass.Int32(0)  # M0=0 default column
                if tidx == 0:
                    s_mstf[0] = v_lo
                    s_mstf[1] = v_hi
                    s_mstf[2] = v_lo
                    s_msti[0] = cutlass.Int32(_INT_MAX)  # best_cnt (M1)
                    s_msti[1] = cutlass.Int32(-1)
                    s_msti[2] = cutlass.Int32(1)
                    s_swf[0] = cutlass.Float32(self.FLT_MAX)  # thr0
                    s_swi[0] = cutlass.Int32(0)   # M0
                    s_swi[1] = cutlass.Int32(-1)  # up col this round
                cute.arch.barrier()

                rr = cutlass.Int32(0)
                while rr < cutlass.Int32(R) and s_msti[2] == cutlass.Int32(1):
                    if tidx == 0:
                        lo = s_mstf[0]
                        hi = s_mstf[1]
                        d = hi - lo
                        if rr == cutlass.Int32(0):
                            pass  # round-0 columns placed by phase1b
                        else:
                            for m in cutlass.range_constexpr(M):
                                s_mt_thr[m] = lo + d * (cutlass.Float32(m + 1) / cutlass.Float32(M + 1))
                    cute.arch.barrier()

                    if cutlass.const_expr(self.fuse_collect):
                        # R==1 gated: all thresholds known up-front, collect at
                        # the pred_col column during the same pass
                        self.block_count_collect_multi(
                            input_row, N, s_mt_thr, smem_ptcnt_multi,
                            smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv,
                            tidx, warp_id, lane)
                    else:
                        self.block_count_ge_multi(input_row, N, s_mt_thr, smem_ptcnt_multi,
                                                  smem_wcnt_multi, s_mt_cnt, tidx, warp_id, lane)
                    cute.arch.barrier()

                    if tidx == 0:
                        best_m = cutlass.Int32(-1)
                        for m in cutlass.range_constexpr(M):
                            if s_mt_cnt[m] >= cutlass.Int32(top_k):
                                best_m = cutlass.Int32(m)
                        # sandwich upper: first column with count < K is
                        # best_m+1 (thresholds ascending => counts descending)
                        up_m = best_m + cutlass.Int32(1)
                        s_swi[1] = cutlass.Int32(-1)
                        if up_m < cutlass.Int32(M):
                            c_up = s_mt_cnt[up_m]
                            if c_up < cutlass.Int32(top_k) and c_up > s_swi[0]:
                                s_swi[0] = c_up
                                s_swf[0] = s_mt_thr[up_m]
                                s_swi[1] = up_m
                        if best_m >= cutlass.Int32(0):
                            c_new = s_mt_cnt[best_m]
                            t_new = s_mt_thr[best_m]
                            if c_new <= s_msti[0]:
                                s_msti[0] = c_new
                                s_mstf[2] = t_new
                                s_msti[1] = best_m
                            else:
                                s_msti[1] = cutlass.Int32(-1)
                            s_mstf[0] = t_new
                            if best_m < cutlass.Int32(M - 1):
                                s_mstf[1] = s_mt_thr[best_m + cutlass.Int32(1)]
                        else:
                            s_msti[1] = cutlass.Int32(-1)
                            s_mstf[1] = s_mt_thr[0]
                        # continue while band > bAcc and bracket nonempty
                        cont = cutlass.Int32(0)
                        if s_msti[0] == cutlass.Int32(_INT_MAX):
                            cont = cutlass.Int32(1)
                        elif (s_msti[0] - s_swi[0]) > cutlass.Int32(bAcc) and s_mstf[1] > s_mstf[0]:
                            cont = cutlass.Int32(1)
                        s_msti[2] = cont
                    cute.arch.barrier()

                    bc = s_msti[1]
                    if bc >= cutlass.Int32(0):
                        smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                    uc = s_swi[1]
                    if uc >= cutlass.Int32(0):
                        smem_ptcnt_up[tidx] = smem_ptcnt_multi[uc * cutlass.Int32(num_threads) + tidx]
                    cute.arch.barrier()
                    rr = rr + cutlass.Int32(1)

                # op20 iter4: slot-overflow reduce (any thread whose l1-column
                # count exceeded slot_cap => slots are incomplete => fall back
                # to the classic P3 rescan). s_iscalars[2] = overflow count.
                if cutlass.const_expr(self.fuse_collect):
                    ofv = cutlass.Int32(0)
                    if smem_ptcnt_multi[cutlass.Int32(self.pred_col) * cutlass.Int32(num_threads) + tidx] > cutlass.Int32(self.slot_cap):
                        ofv = cutlass.Int32(1)
                    ofv = self.warp_reduce_sum_i32(ofv)
                    if lane == 0:
                        smem_wcnt[warp_id] = ofv
                    cute.arch.barrier()
                    if tidx == 0:
                        oft = cutlass.Int32(0)
                        for w7 in cutlass.range_constexpr(num_warps):
                            oft = oft + smem_wcnt[w7]
                        s_iscalars[2] = oft
                # ---- finalize ----
                # done=1 sandwich: pair found and band fits smem
                # done=1 no-pair:  op18 rule (M1 <= kC), M0=0 column = zeros
                # done=2:          band > kC -> baseline retry-shrink
                if tidx == 0:
                    s_thr[0] = s_mstf[2]
                    band_f = s_msti[0] - s_swi[0]
                    if s_msti[0] != cutlass.Int32(_INT_MAX) and band_f <= cutlass.Int32(kC):
                        s_iscalars[0] = band_f
                        s_iscalars[1] = cutlass.Int32(1)
                    else:
                        s_iscalars[1] = cutlass.Int32(2)
                        s_thr[1] = s_mstf[2]
                        s_thr[2] = s_mstf[1]
                        s_swi[0] = cutlass.Int32(0)  # no sandwich on fallback
                cute.arch.barrier()

                if s_iscalars[1] == cutlass.Int32(1) and s_swi[0] > cutlass.Int32(0):
                    # ---- sandwich path ----
                    use_slots = cutlass.Int32(0)
                    if cutlass.const_expr(self.fuse_collect):
                        # usable iff best column >= l1 collect column (slots
                        # are a superset of the candidates) and no overflow
                        if s_msti[1] >= cutlass.Int32(self.pred_col) and s_iscalars[2] == cutlass.Int32(0):
                            use_slots = cutlass.Int32(1)
                    if cutlass.const_expr(self.fuse_collect):
                        if use_slots == cutlass.Int32(1):
                            self.phase3_from_slots(
                                smem_slotk, smem_slotv, smem_keys, smem_vals,
                                smem_ptcnt, smem_ptcnt_up, smem_ptcnt_multi,
                                smem_wcnt, smem_didx, s_thr, s_swf, s_iscalars,
                                output_indices_row, tidx, warp_id, lane)
                        else:
                            self.phase3_sandwich(input_row, N, smem_keys, smem_vals,
                                                 smem_ptcnt, smem_ptcnt_up, smem_wcnt,
                                                 smem_didx, s_thr, s_swf, s_iscalars,
                                                 output_values_row, output_indices_row,
                                                 tidx, warp_id, lane)
                    else:
                        self.phase3_sandwich(input_row, N, smem_keys, smem_vals,
                                             smem_ptcnt, smem_ptcnt_up, smem_wcnt,
                                             smem_didx, s_thr, s_swf, s_iscalars,
                                             output_values_row, output_indices_row,
                                             tidx, warp_id, lane)
                    band = s_iscalars[0]
                    if band > cutlass.Int32(kC):
                        band = cutlass.Int32(kC)
                    m0 = s_swi[0]
                    k_rem = cutlass.Int32(top_k) - m0
                    self.phase4_band_snap(smem_keys, smem_vals, smem_hist,
                                          smem_wcnt, s_thr, s_swf, s_iscalars,
                                          output_values_row, output_indices_row,
                                          band, k_rem, m0, tidx, warp_id, lane)
                else:
                    # ---- op18/baseline path (M0 == 0 or done=2) ----
                    self.phase3_collect_candidates(input_row, N, smem_keys, smem_vals, smem_ptcnt,
                                                   smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane)
                    cand_count_p4 = s_iscalars[0]
                    if cand_count_p4 > cutlass.Int32(self.kC):
                        cand_count_p4 = cutlass.Int32(self.kC)
                    self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_iscalars,
                                               output_values_row, output_indices_row, cand_count_p4, tidx, warp_id, lane)
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values, output_indices, stream):
        num_rows = input_data.shape[0]
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values, output_indices).launch(
            grid=(num_rows, 1, 1), block=(self.num_threads, 1, 1),
            stream=stream, use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp)

class GvrMsClusterKernel(GvrMsKernel):
    """Row-chunked ``cluster_size``-CTA cluster around the MS pipeline.

    C CTAs cooperate on ONE row (aggregate L2 bandwidth for the
    small-batch / large-N regime):

    - P1 stash + P1b rank-quantile seeding run REPLICATED on every CTA
      (same inputs -> bit-identical thresholds; zero cross-CTA traffic).
    - ONE fused ladder pass per CTA over its 64-elt-aligned slice (slot
      indices stored GLOBAL via the slice base offset).
    - DSMEM count merge (M ints per CTA, one cluster barrier) -> every CTA
      picks the same sandwich pair.
    - P3 distributed: per-CTA direct-write of >= thr0 winners straight to
      the output row at a rank-prefix offset; band entries are pushed
      straight into the LEADER's smem at the pre-known global band prefix
      via ``st.shared::cluster`` (``p3_push``; False restores the leader
      DSMEM gather, the A/B reference).
    - Leader (rank 0) runs the unchanged exact P4 for the last K-m0 slots.
    - Fallback (no pair / band > kC / slot overflow): leader re-runs the
      exact classic collect over the FULL row; peers idle. Rare and exact.

    Requires ``R_rounds == 1`` and ``fuse_collect=True`` (all thresholds
    known before the scan).
    """

    def __init__(self, *a, cluster_size: int = 4, p3_push: bool = True, **kw):
        kw.setdefault("fuse_collect", True)
        super().__init__(*a, **kw)
        self.C_cta = int(cluster_size)
        # P3 band remote-store push: during the slot walk each CTA writes
        # its band entries into the LEADER's smem at its global band prefix
        # (known pre-walk from the ladder counts) via st.shared::cluster —
        # deletes the leader DSMEM gather pass and one cluster barrier
        # pair.
        self.p3_push = bool(p3_push)
        assert self.R_rounds == 1 and self.fuse_collect
        assert self.C_cta >= 2, "use GvrMsKernel for cluster_size == 1"


    # ------------------------------------------------------------------
    # slice-aware fused M-count + slot-collect: identical to the parent's
    # block_count_collect_multi but scans input_row[base : base+Ns) and
    # records GLOBAL indices (local + base). Counts returned are slice-local.
    # ------------------------------------------------------------------
    @cute.jit
    def block_count_collect_multi_base(
        self, input_row, base, Ns, s_mt_thr, smem_ptcnt_multi,
        smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv, tidx, warp_id,
        lane,
    ):
        M = cutlass.const_expr(self.M_thr)
        PC = cutlass.const_expr(self.pred_col)
        S = cutlass.const_expr(self.slot_cap)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        vec_w = cutlass.const_expr(self.vec_bits // self.dtype.width)
        elem_bytes = cutlass.const_expr(self.dtype.width // 8)
        vec_align = cutlass.const_expr(self.vec_align_bytes)
        copy_atom = self._make_load_copy_atom()
        step_elem = cutlass.const_expr(num_threads * vec_w)

        thr_frag = cute.make_fragment((M,), cutlass.Float32)
        cnt_frag = cute.make_fragment((M,), cutlass.Int32)
        for m in cutlass.range_constexpr(M):
            thr_frag[m] = s_mt_thr[m]
            cnt_frag[m] = cutlass.Int32(0)
        slot_base = tidx * cutlass.Int32(S)

        # base is 64-elt aligned => vector alignment preserved
        row_addr = input_row.iterator.toint() + cutlass.Int64(base) * cutlass.Int64(elem_bytes)
        n_aligned = (Ns // cutlass.Int32(vec_w)) * cutlass.Int32(vec_w)
        i = tidx * cutlass.Int32(vec_w)
        step = cutlass.Int32(step_elem)

        if self.enable_unroll_4:
            big_iters = cutlass.Int32(0)
            if Ns > i + cutlass.Int32(vec_w - 1):
                big_iters = (Ns - i - cutlass.Int32(vec_w)) // cutlass.Int32(step_elem) + cutlass.Int32(1)
            if cutlass.const_expr(self._p2n()):
                # iter9 native 16-bit paired path (see gvr_ms_op ladder)
                kind = self._kind16()
                vec_w2 = cutlass.const_expr(vec_w // 2)
                copy_atom_u32 = self._make_load_copy_atom_u32()
                rng2_frag = cute.make_fragment((vec_w2,), cutlass.Int32)
                thr2_frag = cute.make_fragment((M,), cutlass.Int32)
                acc2_frag = cute.make_fragment((M,), cutlass.Int32)
                for m in cutlass.range_constexpr(M):
                    thr2_frag[m] = pack2_16_from_f32(thr_frag[m], kind)
                    acc2_frag[m] = cutlass.Int32(0)
                ccol = cutlass.Int32(0)
                for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                    i_local = i + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        cutlass.Int32, row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem, assumed_align=vec_align)
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w2,)))
                    cute.copy(copy_atom_u32, src_k, rng2_frag)
                    for p in cutlass.range_constexpr(vec_w2):
                        v2 = rng2_frag[p]
                        mpc = setge_mask2_16(v2, thr2_frag[PC], kind)
                        if mpc != cutlass.Int32(0):
                            if (mpc & cutlass.Int32(0xFFFF)) != cutlass.Int32(0):
                                if ccol < cutlass.Int32(S):
                                    smem_slotk[slot_base + ccol] = pair_half_f32_16(v2, 0, kind)
                                    smem_slotv[slot_base + ccol] = base + i_local + cutlass.Int32(2 * p)
                                ccol = ccol + cutlass.Int32(1)
                            if (mpc >> cutlass.Int32(16)) != cutlass.Int32(0):
                                if ccol < cutlass.Int32(S):
                                    smem_slotk[slot_base + ccol] = pair_half_f32_16(v2, 1, kind)
                                    smem_slotv[slot_base + ccol] = base + i_local + cutlass.Int32(2 * p + 1)
                                ccol = ccol + cutlass.Int32(1)
                        for m in cutlass.range_constexpr(M):
                            if cutlass.const_expr(m != self.pred_col):
                                acc2_frag[m] = setge_add2_16(acc2_frag[m], v2, thr2_frag[m], kind)
                    if (k & cutlass.Int32(15)) == cutlass.Int32(15):
                        for m in cutlass.range_constexpr(M):
                            if cutlass.const_expr(m != self.pred_col):
                                cnt_frag[m] = cnt_frag[m] + pair_sum_i32_16(acc2_frag[m], kind)
                                acc2_frag[m] = cutlass.Int32(0)
                for m in cutlass.range_constexpr(M):
                    if cutlass.const_expr(m != self.pred_col):
                        cnt_frag[m] = cnt_frag[m] + pair_sum_i32_16(acc2_frag[m], kind)
                cnt_frag[PC] = ccol
            else:
                rng_frag = cute.make_fragment((vec_w,), self.dtype)
                for k in cutlass.range(big_iters, unroll=self.mt_unroll):
                    i_local = i + k * cutlass.Int32(step_elem)
                    src_ptr_k = cute.make_ptr(
                        self.dtype, row_addr + cutlass.Int64(i_local) * cutlass.Int64(elem_bytes),
                        cute.AddressSpace.gmem, assumed_align=vec_align)
                    src_k = cute.make_tensor(src_ptr_k, cute.make_layout((vec_w,)))
                    cute.copy(copy_atom, src_k, rng_frag)
                    for j in cutlass.range_constexpr(vec_w):
                        if cutlass.const_expr(self.dtype == cutlass.Float32):
                            vj = rng_frag[j]
                        else:
                            vj = cutlass.Float32(rng_frag[j])
                        if vj >= thr_frag[PC]:
                            cpos = cnt_frag[PC]
                            if cpos < cutlass.Int32(S):
                                smem_slotk[slot_base + cpos] = vj
                                smem_slotv[slot_base + cpos] = base + i_local + cutlass.Int32(j)
                        for m in cutlass.range_constexpr(M):
                            cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + big_iters * cutlass.Int32(step_elem)

        tail_frag = cute.make_fragment((vec_w,), self.dtype)
        while i + cutlass.Int32(vec_w - 1) < Ns:
            src_ptr = cute.make_ptr(
                self.dtype, row_addr + cutlass.Int64(i) * cutlass.Int64(elem_bytes),
                cute.AddressSpace.gmem, assumed_align=vec_align)
            src = cute.make_tensor(src_ptr, cute.make_layout((vec_w,)))
            cute.copy(copy_atom, src, tail_frag)
            for j in cutlass.range_constexpr(vec_w):
                if cutlass.const_expr(self.dtype == cutlass.Float32):
                    vj = tail_frag[j]
                else:
                    vj = cutlass.Float32(tail_frag[j])
                if vj >= thr_frag[PC]:
                    cpos = cnt_frag[PC]
                    if cpos < cutlass.Int32(S):
                        smem_slotk[slot_base + cpos] = vj
                        smem_slotv[slot_base + cpos] = base + i + cutlass.Int32(j)
                for m in cutlass.range_constexpr(M):
                    cnt_frag[m] = cnt_frag[m] + cutlass.Int32(vj >= thr_frag[m])
            i = i + step

        it = n_aligned + tidx
        while it < Ns:
            v = self._load_fp32_base(input_row, base, it)
            if v >= thr_frag[PC]:
                cpos = cnt_frag[PC]
                if cpos < cutlass.Int32(S):
                    smem_slotk[slot_base + cpos] = v
                    smem_slotv[slot_base + cpos] = base + it
            for m in cutlass.range_constexpr(M):
                cnt_frag[m] = cnt_frag[m] + cutlass.Int32(v >= thr_frag[m])
            it = it + cutlass.Int32(num_threads)

        for m in cutlass.range_constexpr(M):
            smem_ptcnt_multi[m * num_threads + tidx] = cnt_frag[m]
        for m in cutlass.range_constexpr(M):
            wc = self.warp_reduce_sum_i32(cnt_frag[m])
            if lane == 0:
                smem_wcnt_multi[m * num_warps + warp_id] = wc
        cute.arch.barrier()
        if tidx == 0:
            for m in cutlass.range_constexpr(M):
                v = cutlass.Int32(0)
                for w in cutlass.range_constexpr(num_warps):
                    v = v + smem_wcnt_multi[m * num_warps + w]
                s_mt_cnt[m] = v
        cute.arch.barrier()

    @cute.jit
    def _load_fp32_base(self, input_row, base, i):
        return self._load_fp32(input_row, base + i)

    # ------------------------------------------------------------------
    # distributed P3 from slots: direct-write >=thr0 winners straight to
    # output_indices_row[d_off + ...]; band [thr1, thr0) compacts into the
    # LOCAL smem_keys/vals starting at 0. Assumes per-thread thr1/thr0 slot
    # columns already copied into smem_ptcnt / smem_ptcnt_up.
    # After: s_iscalars[4] = local direct count, s_iscalars[0] = local band.
    # ------------------------------------------------------------------
    @cute.jit
    def phase3_from_slots_mc(
        self, smem_slotk, smem_slotv, smem_keys, smem_vals, smem_ptcnt,
        smem_ptcnt_up, smem_ptcnt_multi, smem_wcnt, s_thr, s_swf,
        s_iscalars, output_indices_row, d_off, b_off, rank, tidx, warp_id,
        lane,
    ):
        kCC = cutlass.const_expr(self.kC)
        PC = cutlass.const_expr(self.pred_col)
        S = cutlass.const_expr(self.slot_cap)
        num_threads = cutlass.const_expr(self.num_threads)

        my_up = smem_ptcnt_up[tidx]
        my_band = smem_ptcnt[tidx] - my_up
        my_pk = (my_up << cutlass.Int32(16)) | my_band
        tp0 = my_pk
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp0, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp0 = tp0 + other
        my_excl_pk = tp0 - my_pk
        warp_tot_pk = cute.arch.shuffle_sync(tp0, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_tot_pk
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(self.num_warps):
                c = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + c
            s_iscalars[4] = tot >> cutlass.Int32(16)
            s_iscalars[0] = tot & cutlass.Int32(0xFFFF)
        cute.arch.barrier()
        base_pk = smem_wcnt[warp_id] + my_excl_pk
        wc0 = d_off + (base_pk >> cutlass.Int32(16))
        wcb = base_pk & cutlass.Int32(0xFFFF)

        thr1 = s_thr[0]
        thr0 = s_swf[0]
        my_lc = smem_ptcnt_multi[cutlass.Int32(PC) * cutlass.Int32(num_threads) + tidx]
        if my_lc > cutlass.Int32(S):
            my_lc = cutlass.Int32(S)
        slot_base = tidx * cutlass.Int32(S)
        if cutlass.const_expr(self.p3_push):
            # iter7 push: band entries land at the GLOBAL prefix position in
            # the LEADER's smem (b_off local-CTA global band offset + wcb
            # local prefix). Leader stores locally (b_off == 0 for rank 0);
            # peers fire st.shared::cluster. Visibility to the leader's P4 is
            # ordered by the caller's cluster_arrive/wait.
            wcg = b_off + wcb
            sw = cutlass.Int32(0)
            while sw < my_lc:
                v = smem_slotk[slot_base + sw]
                if v >= thr1:
                    if v >= thr0:
                        if wc0 < cutlass.Int32(self.top_k):
                            output_indices_row[wc0] = smem_slotv[slot_base + sw]
                            wc0 = wc0 + cutlass.Int32(1)
                    elif wcg < cutlass.Int32(kCC):
                        if rank == cutlass.Int32(0):
                            smem_keys[wcg] = v
                            smem_vals[wcg] = smem_slotv[slot_base + sw]
                        else:
                            ka = mapa_shared_cluster(smem_keys.iterator + wcg, cutlass.Int32(0))
                            va = mapa_shared_cluster(smem_vals.iterator + wcg, cutlass.Int32(0))
                            st_shared_cluster_f32(ka, v)
                            st_shared_cluster_i32(va, smem_slotv[slot_base + sw])
                        wcg = wcg + cutlass.Int32(1)
                sw = sw + cutlass.Int32(1)
        else:
            sw = cutlass.Int32(0)
            while sw < my_lc:
                v = smem_slotk[slot_base + sw]
                if v >= thr1:
                    if v >= thr0:
                        if wc0 < cutlass.Int32(self.top_k):
                            output_indices_row[wc0] = smem_slotv[slot_base + sw]
                            wc0 = wc0 + cutlass.Int32(1)
                    elif wcb < cutlass.Int32(kCC):
                        smem_keys[wcb] = v
                        smem_vals[wcb] = smem_slotv[slot_base + sw]
                        wcb = wcb + cutlass.Int32(1)
                sw = sw + cutlass.Int32(1)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # iter7: leader DSMEM band gather (op8 Shift-D pattern), extracted from
    # the kernel body so phase ablation can no-op it independently of the
    # P3 slot walk. Copies each peer's compacted band entries
    # (smem_keys/vals[0..p_cnt)) into the leader's smem at the peer's global
    # band prefix p_off (both packed in s_cluster[M] as off<<16|cnt).
    # Caller owns the surrounding cluster barriers.
    # ------------------------------------------------------------------
    @cute.jit
    def _p3_leader_band_gather(self, rank, smem_keys, smem_vals, s_cluster,
                               tidx):
        M = cutlass.const_expr(self.M_thr)
        C = cutlass.const_expr(self.C_cta)
        num_threads = cutlass.const_expr(self.num_threads)
        if rank == cutlass.Int32(0):
            for peer in cutlass.range_constexpr(C):
                if cutlass.const_expr(peer > 0):
                    pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M), cutlass.Int32(peer))
                    pk = ld_shared_cluster_i32(pa)
                    p_off = pk >> cutlass.Int32(16)
                    p_cnt = pk & cutlass.Int32(0xFFFF)
                    ig = tidx
                    while ig < p_cnt:
                        ka = mapa_shared_cluster(smem_keys.iterator + ig, cutlass.Int32(peer))
                        va = mapa_shared_cluster(smem_vals.iterator + ig, cutlass.Int32(peer))
                        smem_keys[p_off + ig] = ld_shared_cluster_f32(ka)
                        smem_vals[p_off + ig] = ld_shared_cluster_i32(va)
                        ig = ig + cutlass.Int32(num_threads)

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
        M = cutlass.const_expr(self.M_thr)
        C = cutlass.const_expr(self.C_cta)
        PCc = cutlass.const_expr(self.pred_col)
        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        rank = bidx % cutlass.Int32(C)
        row_idx = bidx // cutlass.Int32(C)
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
        smem_ptcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_threads,), order=(0,)), byte_alignment=128)
        smem_wcnt_multi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M * num_warps,), order=(0,)), byte_alignment=64)
        s_mt_thr = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        s_mt_cnt = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M,), order=(0,)), byte_alignment=16)
        smem_ptcnt_up = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_didx = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((top_k,), order=(0,)), byte_alignment=128)
        s_swf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((1,), order=(0,)), byte_alignment=16)
        s_swi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((2,), order=(0,)), byte_alignment=16)
        slot_elems = cutlass.const_expr(self.slot_cap * self.num_threads)
        smem_slotk = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((slot_elems,), order=(0,)), byte_alignment=128)
        smem_slotv = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((slot_elems,), order=(0,)), byte_alignment=128)
        # cluster exchange: [0..M) local ladder counts; [M] pk(direct<<16|band);
        # [M+1] overflow flag; [M+2] ok flag; [M+3] d_off; [M+4] b_off;
        # [M+5] merged m1g. (M+3.. are CTA-local scratch — NEVER stash these
        # in s_iscalars: the vendored P3/P4 own its done/cnt_lo/cnt_hi slots.)
        s_cluster = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((M + 6,), order=(0,)), byte_alignment=16)

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
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            # ---- P1: replicated stash + stats (every CTA gathers the
            # same K addresses; after the first CTA misses, the rest
            # hit L2 — replication beats a distributed gather + DSMEM
            # merges) ----
            self.phase1_stats_stash(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                    smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)
            v_lo = s_thr[1]
            v_hi = s_thr[2]
            if v_hi <= cutlass.Float32(self.NEG_FLT_MAX) or v_lo >= v_hi:
                if rank == cutlass.Int32(0) and tidx == 0:
                    emit_count = cutlass.Int32(top_k) if cutlass.Int32(top_k) < N else N
                    je = cutlass.Int32(0)
                    while je < emit_count:
                        output_indices_row[je] = je
                        je = je + cutlass.Int32(1)
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            else:
                self.phase1b_rank_quantile(smem_keys, pre_idx_count,
                                           smem_hist, s_thr, s_mt_thr,
                                           s_mt_cnt, tidx)
                cute.arch.barrier()

                # ---- fused ladder over my 64-elt-aligned slice ----
                chunk = ((N + cutlass.Int32(C * 64 - 1)) // cutlass.Int32(C * 64)) * cutlass.Int32(64)
                sl_start = rank * chunk
                if sl_start > N:
                    sl_start = N
                sl_end = sl_start + chunk
                if sl_end > N:
                    sl_end = N
                Ns = sl_end - sl_start
                self.block_count_collect_multi_base(
                    input_row, sl_start, Ns, s_mt_thr, smem_ptcnt_multi,
                    smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv,
                    tidx, warp_id, lane)

                # local overflow check on the collect column
                ofv = cutlass.Int32(0)
                if smem_ptcnt_multi[cutlass.Int32(PCc) * cutlass.Int32(num_threads) + tidx] > cutlass.Int32(self.slot_cap):
                    ofv = cutlass.Int32(1)
                ofv = self.warp_reduce_sum_i32(ofv)
                if lane == 0:
                    smem_wcnt[warp_id] = ofv
                cute.arch.barrier()
                if tidx == 0:
                    oft = cutlass.Int32(0)
                    for w7 in cutlass.range_constexpr(num_warps):
                        oft = oft + smem_wcnt[w7]
                    for m in cutlass.range_constexpr(M):
                        s_cluster[m] = s_mt_cnt[m]
                    s_cluster[M + 1] = oft
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

                # ---- merge counts + pick pair (replicated, deterministic) ----
                if tidx == 0:
                    of_g = cutlass.Int32(0)
                    for m in cutlass.range_constexpr(M):
                        tot_m = cutlass.Int32(0)
                        for peer in cutlass.range_constexpr(C):
                            pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(m), cutlass.Int32(peer))
                            tot_m = tot_m + ld_shared_cluster_i32(pa)
                        s_mt_cnt[m] = tot_m
                    for peer in cutlass.range_constexpr(C):
                        pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M + 1), cutlass.Int32(peer))
                        of_g = of_g + ld_shared_cluster_i32(pa)
                    best_m = cutlass.Int32(-1)
                    for m in cutlass.range_constexpr(M):
                        if s_mt_cnt[m] >= cutlass.Int32(top_k):
                            best_m = cutlass.Int32(m)
                    up_m = best_m + cutlass.Int32(1)
                    m1g = cutlass.Int32(_INT_MAX)
                    m0g = cutlass.Int32(0)
                    if best_m >= cutlass.Int32(0):
                        m1g = s_mt_cnt[best_m]
                    if up_m >= cutlass.Int32(1) and up_m < cutlass.Int32(M):
                        m0g = s_mt_cnt[up_m]
                    ok = cutlass.Int32(0)
                    if (best_m >= cutlass.Int32(PCc) and up_m < cutlass.Int32(M)
                            and m0g > cutlass.Int32(0) and m0g < cutlass.Int32(top_k)
                            and (m1g - m0g) <= cutlass.Int32(kC)
                            and of_g == cutlass.Int32(0)):
                        ok = cutlass.Int32(1)
                    s_swi[0] = m0g
                    s_swi[1] = best_m
                    s_cluster[M + 5] = m1g
                    s_cluster[M + 2] = ok
                    if best_m >= cutlass.Int32(0):
                        s_thr[0] = s_mt_thr[best_m]
                    if up_m < cutlass.Int32(M):
                        s_swf[0] = s_mt_thr[up_m]
                    else:
                        s_swf[0] = cutlass.Float32(self.FLT_MAX)
                cute.arch.barrier()
                ok = s_cluster[M + 2]
                best_m = s_swi[1]
                m0g = s_swi[0]
                m1g = s_cluster[M + 5]

                if ok == cutlass.Int32(1):
                    # ---- distributed P3: rank prefix over peers' pk ----
                    bc = best_m
                    up_c = bc + cutlass.Int32(1)
                    smem_ptcnt[tidx] = smem_ptcnt_multi[bc * cutlass.Int32(num_threads) + tidx]
                    smem_ptcnt_up[tidx] = smem_ptcnt_multi[up_c * cutlass.Int32(num_threads) + tidx]
                    cute.arch.barrier()
                    # local direct/band totals -> s_cluster[M] for peers
                    if tidx == 0:
                        ld = cutlass.Int32(0)
                        lb = cutlass.Int32(0)
                        # cheap: local col totals already in ptcnt copies? need
                        # block totals: reuse s_cluster local ladder counts
                        ld = s_cluster[up_c]
                        lb = s_cluster[bc] - s_cluster[up_c]
                        s_cluster[M] = (ld << cutlass.Int32(16)) | lb
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    d_off = cutlass.Int32(0)
                    b_off = cutlass.Int32(0)
                    if tidx == 0:
                        for peer in cutlass.range_constexpr(C):
                            if cutlass.Int32(peer) < rank:
                                pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M), cutlass.Int32(peer))
                                pk = ld_shared_cluster_i32(pa)
                                d_off = d_off + (pk >> cutlass.Int32(16))
                                b_off = b_off + (pk & cutlass.Int32(0xFFFF))
                        s_cluster[M + 3] = d_off
                        s_cluster[M + 4] = b_off
                    cute.arch.barrier()
                    d_off = s_cluster[M + 3]
                    b_off2 = s_cluster[M + 4]
                    self.phase3_from_slots_mc(
                        smem_slotk, smem_slotv, smem_keys, smem_vals,
                        smem_ptcnt, smem_ptcnt_up, smem_ptcnt_multi,
                        smem_wcnt, s_thr, s_swf, s_iscalars,
                        output_indices_row, d_off, b_off2, rank, tidx,
                        warp_id, lane)
                    if cutlass.const_expr(self.p3_push):
                        # iter7: band already pushed into the leader's
                        # smem during the walk — ONE release/acquire
                        # cluster barrier makes the remote stores
                        # visible; no publish, no gather pass.
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                    else:
                        # publish local band count for the leader gather
                        if tidx == 0:
                            s_cluster[M] = (s_cluster[M + 4] << cutlass.Int32(16)) | s_iscalars[0]
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()

                        # ---- leader: gather peers' band entries via DSMEM ----
                        self._p3_leader_band_gather(rank, smem_keys,
                                                    smem_vals, s_cluster,
                                                    tidx)
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                    if rank == cutlass.Int32(0):
                        band_g = m1g - m0g
                        if band_g > cutlass.Int32(kC):
                            band_g = cutlass.Int32(kC)
                        k_rem = cutlass.Int32(top_k) - m0g
                        if tidx == 0:
                            s_iscalars[0] = band_g
                        cute.arch.barrier()
                        self.phase4_band_snap(smem_keys, smem_vals, smem_hist,
                                              smem_wcnt, s_thr, s_swf, s_iscalars,
                                              output_values_row, output_indices_row,
                                              band_g, k_rem, m0g, tidx, warp_id, lane)
                else:
                    # ---- fallback: leader-only exact classic path, full row.
                    # phase3_collect_candidates PREFIXES over smem_ptcnt (per-
                    # thread counts at s_thr[0] over the SAME [0,N) striding),
                    # so the leader must recount the full row first — its
                    # ladder counts cover only its slice (found the hard way:
                    # 646/1024 holes on pro L4). Vendored done/cnt seeds in
                    # s_iscalars are restored exactly as phase1 leaves them.
                    if rank == cutlass.Int32(0):
                        thr_c = s_mt_thr[0]
                        if best_m >= cutlass.Int32(0):
                            thr_c = s_mt_thr[best_m]
                        if tidx == cutlass.Int32(0):
                            s_thr[0] = thr_c
                            if best_m >= cutlass.Int32(0) and m1g <= cutlass.Int32(kC):
                                s_iscalars[1] = cutlass.Int32(1)
                            else:
                                s_iscalars[1] = cutlass.Int32(2)
                                if best_m >= cutlass.Int32(0):
                                    s_thr[1] = s_mt_thr[best_m]
                                    if best_m < cutlass.Int32(M - 1):
                                        s_thr[2] = s_mt_thr[best_m + cutlass.Int32(1)]
                                    else:
                                        s_thr[2] = v_hi
                                else:
                                    s_thr[1] = v_lo
                                    s_thr[2] = s_mt_thr[0]
                            s_iscalars[2] = pre_idx_count + (pre_idx_count >> 2)
                            s_iscalars[3] = cutlass.Int32(1)
                            s_iscalars[4] = cutlass.Int32(0)
                            s_swi[0] = cutlass.Int32(0)  # no sandwich
                        smem_ptcnt_up[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()
                        # full-row recount at thr_c -> smem_ptcnt + cand_count
                        self.block_count_ge(input_row, N, thr_c, smem_ptcnt,
                                            smem_wcnt, s_iscalars, tidx,
                                            warp_id, lane)
                        cute.arch.barrier()
                        self.phase3_collect_candidates(input_row, N, smem_keys, smem_vals, smem_ptcnt,
                                                       smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane)
                        cand_count_p4 = s_iscalars[0]
                        if cand_count_p4 > cutlass.Int32(self.kC):
                            cand_count_p4 = cutlass.Int32(self.kC)
                        self.phase4_histogram_snap(smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_iscalars,
                                                   output_values_row, output_indices_row, cand_count_p4, tidx, warp_id, lane)
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()
        griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(self, input_data, pre_idx, seq_lens, output_values, output_indices, stream):
        num_rows = input_data.shape[0]
        C = cutlass.const_expr(self.C_cta)
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values, output_indices).launch(
            grid=(num_rows * C, 1, 1), block=(self.num_threads, 1, 1),
            cluster=(C, 1, 1), stream=stream, use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp)
