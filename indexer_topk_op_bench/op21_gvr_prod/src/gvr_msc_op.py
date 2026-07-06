# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op21 iter2: row-chunked multi-CTA (cluster) rank-quantile sandwich.

C CTAs cooperate on ONE row (aggregate L2 bandwidth — the P0 lever proven
by op8's 17ns/Kelt N-slope vs single-CTA's 87):
- P1 stash + P1b rank-quantile seeding run REPLICATED on every CTA (same
  inputs -> bit-identical thresholds; zero cross-CTA traffic). Cost: C*K
  extra gather loads, negligible vs N at the C>1 dispatch sizes.
- ONE fused M-threshold ladder pass per CTA over its 64-elt-aligned slice
  (slice-aware count+collect; slot indices stored GLOBAL via base offset).
- DSMEM count merge (M ints per CTA, one cluster barrier) -> every CTA
  picks the same sandwich pair.
- P3 distributed: per-CTA direct-write of >=thr0 winners STRAIGHT to the
  output row at a rank-prefix offset (output GMEM is shared; no staging);
  band candidates compact into each CTA's local smem_keys.
- Leader (rank 0) DSMEM-gathers the <=kC band entries from peers, runs the
  unchanged exact phase4_band_snap for the remaining K-m0 slots.
- Fallback (no pair / band > kC / slot overflow / non-slot column): leader
  re-runs the exact classic P3+P4 over the FULL row; peers idle. Rare
  (iter0.5: all_ge 5.5% worst-case model) and exact.

Exactness authority unchanged: counts + band refine, tie handling in P4.
Red lines respected: no gmem scratch in the hot path (band moves via
DSMEM), counts merged once (single cluster barrier chain), no warp
collectives in the streaming loop.
"""
import os
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "ops"))
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_HERE))
from gvr_ms_op import (  # noqa: E402
    GvrSandwichKernel, gvr_ms, NUM_SMS, _DT, _INT_MAX, pack2_16_from_f32,
    setge_add2_16, setge_mask2_16, pair_half_f32_16, pair_sum_i32_16,
)
from cute_vendored.blackwell.top_k.gvr_topk_decode import (  # noqa: E402
    _fmin_f32_inline, atomicAdd,
)
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: E402
    ld_shared_cluster_f32, ld_shared_cluster_i32, mapa_shared_cluster,
)
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
from cutlass._mlir.dialects import llvm  # noqa: E402
from cutlass.cutlass_dsl import T, dsl_user_op  # noqa: E402
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402


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


class GvrMsClusterKernel(GvrSandwichKernel):
    """C-CTA row-chunked cluster around the mode-5 sandwich. Requires
    place_mode=5, R=1, fuse_collect=True (thresholds known pre-scan)."""

    def __init__(self, *a, C_cta=4, dist_p1=False, dist_p4=False,
                 p3_push=True, **kw):
        super().__init__(*a, **kw)
        self.C_cta = int(C_cta)
        # iter7: P3 band remote-store push — during the slot walk each CTA
        # writes its band entries straight into the LEADER's smem at its
        # global band prefix (b_off known pre-walk from the ladder counts),
        # via st.shared::cluster. Replaces the leader DSMEM gather pass AND
        # one cluster barrier pair (ablation pinned the gather at 1.7/2.4us
        # on the K1024/K2048 262K hole cells). dist_p4 needs the LOCAL band
        # copy, so push is forced off there.
        self.p3_push = bool(p3_push) and not bool(dist_p4)
        # iter4: distributed P4 — the leader-only band snap measured as THE
        # dominant fixed cost (ablation 2026-07-05: 3.9us @K1024, 7.0us
        # @K2048 C8). Bulk (bins > cut) emitted by every CTA; only the
        # boundary bin is gathered and exact-snapped by the leader.
        self.dist_p4 = bool(dist_p4)
        # iter3 FALSIFIED lever (kept as A/B reference, default OFF):
        # distributing P1 across the cluster (each CTA gathers K/C preIdx +
        # two DSMEM merges) measured +0.6-1.7us at every P0 cell vs the
        # replicated gather. At BS1-16 all C CTAs gather the SAME addresses
        # — after the first CTA misses, the rest hit L2, so replication is
        # nearly free; the 3 extra cluster barriers cost more than the
        # saved loads. (event A/B 2026-07-05, 8 cells)
        self.dist_p1 = bool(dist_p1)
        # iter9: phase1b_dist (the dist_p1 reference) does not quantize its
        # thresholds to the dtype grid — the native-compare ladder would be
        # inconsistent with the fp32 phases there, so force it off.
        if self.dist_p1:
            self.p2_native = False
        assert self.place_mode == 5 and self.R_rounds == 1 and self.fuse_collect
        assert self.C_cta >= 2
        assert self.top_k % self.C_cta == 0, "dist P1 needs C | K"

    # ------------------------------------------------------------------
    # iter3 distributed P1 (stats half): CTA r gathers preIdx slice
    # [r*Kc, (r+1)*Kc), stashes values (sentinel NEG_FLT_MAX), reduces its
    # LOCAL min/max/sum/cnt, then one DSMEM merge rebuilds the GLOBAL stats
    # identically on every CTA. Publishes local stats via s_p1f (floats) and
    # s_cluster[M+5] (cnt — free until pair-pick writes m1g there).
    # ------------------------------------------------------------------
    @cute.jit
    def phase1_dist_stats(
        self, input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
        rank, smem_stash, smem_wmin_f32, smem_wmax_f32, smem_wsum_f32,
        smem_wcnt_i32, s_p1f, s_cluster, s_thr, s_iscalars, tidx, warp_id,
        lane,
    ):
        C = cutlass.const_expr(self.C_cta)
        M = cutlass.const_expr(self.M_thr)
        Kc = cutlass.const_expr(self.top_k // self.C_cta)
        num_threads = cutlass.const_expr(self.num_threads)
        j0 = rank * cutlass.Int32(Kc)
        local_min = cutlass.Float32(self.FLT_MAX)
        local_max = cutlass.Float32(self.NEG_FLT_MAX)
        local_sum = cutlass.Float32(0.0)
        local_cnt = cutlass.Int32(0)
        if cutlass.const_expr(Kc >= self.num_threads):
            n_iters = cutlass.const_expr(Kc // self.num_threads)
            for u in cutlass.range_constexpr(n_iters):
                i = tidx + cutlass.Int32(u * self.num_threads)
                idx = pre_idx_row[j0 + i] + pre_idx_offset
                v = cutlass.Float32(self.NEG_FLT_MAX)
                if idx >= 0 and idx < N:
                    v = self._load_fp32(input_row, idx)
                    local_max = cute.arch.fmax(local_max, v)
                    local_min = _fmin_f32_inline(local_min, v)
                    local_sum = local_sum + v
                    local_cnt = local_cnt + 1
                smem_stash[i] = v
        else:
            idx = cutlass.Int32(-1)
            if tidx < cutlass.Int32(Kc):
                idx = pre_idx_row[j0 + tidx] + pre_idx_offset
            v = cutlass.Float32(self.NEG_FLT_MAX)
            if idx >= 0 and idx < N:
                v = self._load_fp32(input_row, idx)
                local_max = cute.arch.fmax(local_max, v)
                local_min = _fmin_f32_inline(local_min, v)
                local_sum = local_sum + v
                local_cnt = local_cnt + 1
            if tidx < cutlass.Int32(Kc):
                smem_stash[tidx] = v
        active_warps = cutlass.const_expr(
            max(1, min(Kc // self.WARP_SIZE, self.num_warps)))
        if warp_id < cutlass.Int32(active_warps):
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
            for w in cutlass.range_constexpr(active_warps):
                pmax = cute.arch.fmax(pmax, smem_wmax_f32[w])
                pmin = _fmin_f32_inline(pmin, smem_wmin_f32[w])
                psum = psum + smem_wsum_f32[w]
                pcnt = pcnt + smem_wcnt_i32[w]
            s_p1f[0] = pmin
            s_p1f[1] = pmax
            s_p1f[2] = psum
            s_cluster[M + 5] = pcnt
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if tidx == 0:
            gmin = cutlass.Float32(self.FLT_MAX)
            gmax = cutlass.Float32(self.NEG_FLT_MAX)
            gsum = cutlass.Float32(0.0)
            gcnt = cutlass.Int32(0)
            for peer in cutlass.range_constexpr(C):
                pf0 = mapa_shared_cluster(s_p1f.iterator + cutlass.Int32(0), cutlass.Int32(peer))
                pf1 = mapa_shared_cluster(s_p1f.iterator + cutlass.Int32(1), cutlass.Int32(peer))
                pf2 = mapa_shared_cluster(s_p1f.iterator + cutlass.Int32(2), cutlass.Int32(peer))
                pi0 = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M + 5), cutlass.Int32(peer))
                gmin = _fmin_f32_inline(gmin, ld_shared_cluster_f32(pf0))
                gmax = cute.arch.fmax(gmax, ld_shared_cluster_f32(pf1))
                gsum = gsum + ld_shared_cluster_f32(pf2)
                gcnt = gcnt + ld_shared_cluster_i32(pi0)
            gmean = cutlass.Float32(0.0)
            if gcnt > 0:
                gmean = gsum / cutlass.Float32(gcnt)
            else:
                gmean = (gmin + gmax) * cutlass.Float32(0.5)
            s_thr[0] = gmean
            s_thr[1] = gmin
            s_thr[2] = gmax
            s_iscalars[0] = cutlass.Int32(0)
            s_iscalars[1] = cutlass.Int32(0)
            s_iscalars[2] = pre_idx_count + (pre_idx_count >> 2)
            s_iscalars[3] = cutlass.Int32(1)
            s_iscalars[4] = cutlass.Int32(0)
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # iter3 distributed P1b: local QBINS histogram of the Kc stashed values
    # (global [lo,hi] range), DSMEM histogram merge (1 register/thread, two
    # cluster barriers), then the parent's parallel suffix-scan + crossing
    # on the MERGED histogram — every CTA lands identical seed thresholds.
    # ------------------------------------------------------------------
    @cute.jit
    def phase1b_dist(
        self, smem_stash, smem_hist, s_thr, s_mt_thr, tidx,
    ):
        QBINS = cutlass.const_expr(self.QBINS)
        M = cutlass.const_expr(self.M_thr)
        C = cutlass.const_expr(self.C_cta)
        Kc = cutlass.const_expr(self.top_k // self.C_cta)
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
        if cutlass.const_expr(Kc >= self.num_threads):
            n_iters = cutlass.const_expr(Kc // self.num_threads)
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
            if tidx < cutlass.Int32(Kc):
                v = smem_stash[tidx]
            if v >= lo:
                b = cutlass.Int32((v - lo) * inv)
                if b > cutlass.Int32(QBINS - 1):
                    b = cutlass.Int32(QBINS - 1)
                atomicAdd(smem_hist.iterator + b, cutlass.Int32(1))
        # DSMEM histogram merge: hist ready -> read C peers -> write merged
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        mg = cutlass.Int32(0)
        if tidx < cutlass.Int32(QBINS):
            for peer in cutlass.range_constexpr(C):
                ha = mapa_shared_cluster(smem_hist.iterator + tidx, cutlass.Int32(peer))
                mg = mg + ld_shared_cluster_i32(ha)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if tidx < cutlass.Int32(QBINS):
            smem_hist[tidx] = mg
        cute.arch.barrier()
        # ---- identical to parent phase1b from here: defaults, suffix scan,
        # crossing, non-descend fixup ----
        binw = rng / cutlass.Float32(QBINS)
        if tidx == 0:
            s_mt_thr[0] = lo
            for md in cutlass.range_constexpr(M - 1):
                s_mt_thr[md + 1] = lo + rng * (cutlass.Float32(md + 1) / cutlass.Float32(M))
        for e in cutlass.range_constexpr(self.QBINS.bit_length() - 1):
            step = cutlass.const_expr(1 << e)
            v2 = cutlass.Int32(0)
            if tidx < cutlass.Int32(QBINS):
                v2 = smem_hist[tidx]
                if tidx + cutlass.Int32(step) < cutlass.Int32(QBINS):
                    v2 = v2 + smem_hist[tidx + cutlass.Int32(step)]
            cute.arch.barrier()
            if tidx < cutlass.Int32(QBINS):
                smem_hist[tidx] = v2
            cute.arch.barrier()
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
            for mm in cutlass.range_constexpr(M - 1):
                pv = s_mt_thr[mm]
                if s_mt_thr[mm + 1] < pv:
                    s_mt_thr[mm + 1] = pv
        # callers ladder-init barrier publishes before any read

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
    # iter4 distributed P4: each CTA histograms its LOCAL band members
    # (smem_keys/vals[0..b_r), all in [thr1, thr0)) into QBINS bins, DSMEM
    # merge -> every CTA knows the cut bin c* (largest b with suffix >=
    # k_rem). Members in bins > c* are top-K for sure: emitted distributed
    # at prefix offsets (the ~99% bulk). ONLY the boundary-bin members
    # (expected ~band/QBINS) are compacted + gathered to the leader, which
    # runs the unchanged exact phase4_band_snap on them for the last
    # r = k_rem - above slots. Exactness: bin index is recomputed with the
    # SAME formula in histogram and walk (bit-identical binning); boundary
    # bin resolved by the existing exact snap.
    # ------------------------------------------------------------------
    @cute.jit
    def phase4_dist(
        self, rank, m0g, k_rem, smem_keys, smem_vals, smem_hist,
        smem_merged, smem_slotk, smem_slotv, smem_wcnt, s_cluster, s_thr,
        s_swf, s_iscalars, output_indices_row, tidx, warp_id, lane,
    ):
        QBINS = cutlass.const_expr(self.QBINS)
        M = cutlass.const_expr(self.M_thr)
        C = cutlass.const_expr(self.C_cta)
        num_threads = cutlass.const_expr(self.num_threads)
        num_warps = cutlass.const_expr(self.num_warps)
        b_r = s_iscalars[0]          # my local band count (phase3 output)
        thr1 = s_thr[0]
        thr0 = s_swf[0]
        iz = cutlass.Int32(tidx)
        while iz < cutlass.Int32(QBINS):
            smem_hist[iz] = cutlass.Int32(0)
            iz = iz + cutlass.Int32(num_threads)
        cute.arch.barrier()
        rng = thr0 - thr1
        inv = (cutlass.Float32(QBINS - 1) + cutlass.Float32(0.99)) / rng
        ih = cutlass.Int32(tidx)
        while ih < b_r:
            v = smem_keys[ih]
            bb = cutlass.Int32((v - thr1) * inv)
            if bb < cutlass.Int32(0):
                bb = cutlass.Int32(0)
            if bb > cutlass.Int32(QBINS - 1):
                bb = cutlass.Int32(QBINS - 1)
            atomicAdd(smem_hist.iterator + bb, cutlass.Int32(1))
            ih = ih + cutlass.Int32(num_threads)
        # DSMEM merge into smem_merged (local hist preserved in smem_hist)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        mg = cutlass.Int32(0)
        if tidx < cutlass.Int32(QBINS):
            for peer in cutlass.range_constexpr(C):
                ha = mapa_shared_cluster(smem_hist.iterator + tidx, cutlass.Int32(peer))
                mg = mg + ld_shared_cluster_i32(ha)
            smem_merged[tidx] = mg
        cute.arch.barrier()
        # replicated suffix scan of merged hist (in smem_merged)
        for e in cutlass.range_constexpr(self.QBINS.bit_length() - 1):
            step = cutlass.const_expr(1 << e)
            v2 = cutlass.Int32(0)
            if tidx < cutlass.Int32(QBINS):
                v2 = smem_merged[tidx]
                if tidx + cutlass.Int32(step) < cutlass.Int32(QBINS):
                    v2 = v2 + smem_merged[tidx + cutlass.Int32(step)]
            cute.arch.barrier()
            if tidx < cutlass.Int32(QBINS):
                smem_merged[tidx] = v2
            cute.arch.barrier()
        # cut bin c* = largest b with suffix(b) >= k_rem; above = suffix(c*+1)
        if tidx < cutlass.Int32(QBINS):
            sfx = smem_merged[tidx]
            nxt = cutlass.Int32(0)
            if tidx < cutlass.Int32(QBINS - 1):
                nxt = smem_merged[tidx + 1]
            if sfx >= k_rem and (tidx == cutlass.Int32(QBINS - 1) or nxt < k_rem):
                s_iscalars[3] = tidx        # c*
                s_iscalars[4] = nxt         # above total (global)
        cute.arch.barrier()
        cstar = s_iscalars[3]
        above_g = s_iscalars[4]
        # pass A: per-thread counts of (bin > c*) and (bin == c*) members
        my_ab = cutlass.Int32(0)
        my_cb = cutlass.Int32(0)
        ia = cutlass.Int32(tidx)
        while ia < b_r:
            v = smem_keys[ia]
            bb = cutlass.Int32((v - thr1) * inv)
            if bb < cutlass.Int32(0):
                bb = cutlass.Int32(0)
            if bb > cutlass.Int32(QBINS - 1):
                bb = cutlass.Int32(QBINS - 1)
            if bb > cstar:
                my_ab = my_ab + cutlass.Int32(1)
            if bb == cstar:
                my_cb = my_cb + cutlass.Int32(1)
            ia = ia + cutlass.Int32(num_threads)
        # block prefix over packed (above<<16 | cut) — same trick as P3
        my_pk = (my_ab << cutlass.Int32(16)) | my_cb
        tp0 = my_pk
        for off_i in cutlass.range_constexpr(5):
            off_v = cutlass.const_expr(1 << off_i)
            other = cute.arch.shuffle_sync_up(tp0, off_v, mask_and_clamp=0)
            if lane >= cutlass.Int32(off_v):
                tp0 = tp0 + other
        my_excl = tp0 - my_pk
        warp_tot = cute.arch.shuffle_sync(tp0, cutlass.Int32(self.WARP_SIZE - 1))
        if lane == 0:
            smem_wcnt[warp_id] = warp_tot
        cute.arch.barrier()
        if tidx == 0:
            tot = cutlass.Int32(0)
            for w in cutlass.range_constexpr(num_warps):
                cw = smem_wcnt[w]
                smem_wcnt[w] = tot
                tot = tot + cw
            # publish my CTA's (above, cut) totals for the cluster prefix
            s_cluster[M] = tot
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # rank prefix of (above, cut) over peers < me; leader also needs
        # every peer's cut-bin count for the gather
        if tidx == 0:
            ab_off = cutlass.Int32(0)
            cb_off = cutlass.Int32(0)
            for peer in cutlass.range_constexpr(C):
                if cutlass.Int32(peer) < rank:
                    pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M), cutlass.Int32(peer))
                    pk = ld_shared_cluster_i32(pa)
                    ab_off = ab_off + (pk >> cutlass.Int32(16))
                    cb_off = cb_off + (pk & cutlass.Int32(0xFFFF))
            s_cluster[M + 3] = ab_off
            s_cluster[M + 4] = cb_off
        cute.arch.barrier()
        ab_off = s_cluster[M + 3]
        base_pk = smem_wcnt[warp_id] + my_excl
        wca = m0g + ab_off + (base_pk >> cutlass.Int32(16))
        wcc = base_pk & cutlass.Int32(0xFFFF)
        # pass B: emit above-bin members to the output row; compact cut-bin
        # members into smem_slotk/v (free after phase3)
        ib = cutlass.Int32(tidx)
        while ib < b_r:
            v = smem_keys[ib]
            bb = cutlass.Int32((v - thr1) * inv)
            if bb < cutlass.Int32(0):
                bb = cutlass.Int32(0)
            if bb > cutlass.Int32(QBINS - 1):
                bb = cutlass.Int32(QBINS - 1)
            if bb > cstar:
                output_indices_row[wca] = smem_vals[ib]
                wca = wca + cutlass.Int32(1)
            if bb == cstar:
                smem_slotk[wcc] = v
                smem_slotv[wcc] = smem_vals[ib]
                wcc = wcc + cutlass.Int32(1)
            ib = ib + cutlass.Int32(num_threads)
        cute.arch.barrier()
        # local cut-bin members -> smem_keys[0..my_cut) (walk done, keys free)
        my_cut_total = s_cluster[M] & cutlass.Int32(0xFFFF)
        ic = cutlass.Int32(tidx)
        while ic < my_cut_total:
            smem_keys[ic] = smem_slotk[ic]
            smem_vals[ic] = smem_slotv[ic]
            ic = ic + cutlass.Int32(num_threads)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        # leader: gather peers' cut-bin members, exact-snap the last r slots
        if rank == cutlass.Int32(0):
            for peer in cutlass.range_constexpr(C):
                if cutlass.const_expr(peer > 0):
                    pa = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M), cutlass.Int32(peer))
                    pk = ld_shared_cluster_i32(pa)
                    p_cnt = pk & cutlass.Int32(0xFFFF)
                    po = mapa_shared_cluster(s_cluster.iterator + cutlass.Int32(M + 4), cutlass.Int32(peer))
                    p_off = ld_shared_cluster_i32(po)
                    ig = tidx
                    while ig < p_cnt:
                        ka = mapa_shared_cluster(smem_keys.iterator + ig, cutlass.Int32(peer))
                        va = mapa_shared_cluster(smem_vals.iterator + ig, cutlass.Int32(peer))
                        smem_keys[p_off + ig] = ld_shared_cluster_f32(ka)
                        smem_vals[p_off + ig] = ld_shared_cluster_i32(va)
                        ig = ig + cutlass.Int32(num_threads)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if rank == cutlass.Int32(0):
            # smem_merged holds SUFFIX sums post-scan: cut-bin population =
            # suffix(c*) - suffix(c*+1)
            cutbin_total = smem_merged[cstar]
            if cstar < cutlass.Int32(QBINS - 1):
                cutbin_total = smem_merged[cstar] - smem_merged[cstar + 1]
            r_rem = k_rem - above_g
            binw = rng / cutlass.Float32(QBINS)
            if tidx == 0:
                s_iscalars[0] = cutbin_total
                s_thr[0] = thr1 + cutlass.Float32(cstar) * binw
                s_swf[0] = thr1 + cutlass.Float32(cstar + 1) * binw
            cute.arch.barrier()
            self.phase4_band_snap(smem_keys, smem_vals, smem_hist,
                                  smem_wcnt, s_thr, s_swf, s_iscalars,
                                  None, output_indices_row,
                                  cutbin_total, r_rem, m0g + above_g,
                                  tidx, warp_id, lane)

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
        # iter3 dist-P1 local stats publish (min/max/sum) for the DSMEM merge
        s_p1f = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((3,), order=(0,)), byte_alignment=16)

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
            # ---- P1: distributed (K/C gather + DSMEM stats merge) or the
            # iter2 replicated reference ----
            if cutlass.const_expr(self.dist_p1):
                self.phase1_dist_stats(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                       rank, smem_keys, smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1,
                                       s_p1f, s_cluster, s_thr, s_iscalars, tidx, warp_id, lane)
            else:
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
                if cutlass.const_expr(self.dist_p1):
                    self.phase1b_dist(smem_keys, smem_hist, s_thr, s_mt_thr,
                                      tidx)
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
                    if cutlass.const_expr(self.dist_p4):
                        # ---- iter4: distributed P4 (bulk emitted by every
                        # CTA; only the boundary bin goes to the leader) ----
                        k_rem4 = cutlass.Int32(top_k) - m0g
                        self.phase4_dist(rank, m0g, k_rem4, smem_keys,
                                         smem_vals, smem_hist, smem_ptcnt_up,
                                         smem_slotk, smem_slotv, smem_wcnt,
                                         s_cluster, s_thr, s_swf, s_iscalars,
                                         output_indices_row, tidx, warp_id,
                                         lane)
                    else:
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


_compiled = {}


def _compile(dtype, n, K, cr_val, C, threads, dist_p1=False, dist_p4=False):
    # iter5 A/B: OP21_P4_RS=0 falls back to the legacy runtime-k band snap
    p4_rs = os.environ.get("OP21_P4_RS", "1") == "1"
    # iter6 A/B: OP21_P4_FAST=0 forces the fine-recursion path (no fast paths)
    p4_fast = os.environ.get("OP21_P4_FAST", "1") == "1"
    # iter7 A/B: OP21_P3_PUSH=0 restores the leader DSMEM gather (2 barriers)
    p3_push = os.environ.get("OP21_P3_PUSH", "1") == "1"
    # iter9 A/B: OP21_P2_NATIVE=0 restores the cvt->fp32 ladder (16-bit only)
    p2_nat = os.environ.get("OP21_P2_NATIVE", "1") == "1"
    key = (dtype, n, K, cr_val, C, threads, dist_p1, dist_p4, p4_rs, p4_fast,
           p3_push, p2_nat)
    if key in _compiled:
        return _compiled[key]
    use256 = (n >= 16384)
    kobj = GvrMsClusterKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=threads,
                              compress_ratio=cr_val, use_256bit_load=use256,
                              enable_unroll_4=True, enable_phase3_unroll=True,
                              min_blocks_per_mp=1, return_output_values=False,
                              M_thr=4, R_rounds=1, band_accept=64, place_mode=5,
                              fuse_collect=True, C_cta=C, dist_p1=dist_p1,
                              dist_p4=dist_p4, p4_rank_scatter=p4_rs,
                              p4_smallbin=p4_fast, p3_push=p3_push,
                              p2_native=p2_nat)
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


def gvr_msc(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None,
            C=4, threads=1024, dist_p1=False, dist_p4=False):
    """Row-chunked C-CTA cluster mode-5 sandwich. C must be >= 2; use gvr_ms
    for C == 1."""
    bs, n = logits.shape
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled = _compile(logits.dtype, n, index_topk, compress_ratio, int(C),
                        int(threads), bool(dist_p1), bool(dist_p4))
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


# production entry: THREE extra dispatch rules on (dtype, K, BS, max-N
# buffer) only.
# C=4 measured best or tied-best on 15/17 fp32 P0 cells (event 2026-07-05).
# C=8 fp32 is a consistent ~5% win ONLY at K2048 huge-N BS<=4; at K1024 it
# is noise-level (iter6 re-probe +0.6-1.3%) and collapses at BS16 — tight
# gate. At 16-bit the calculus FLIPS (iter8 probe 2026-07-06): the cheaper
# scan makes 8-way chunking a 1.08-1.14x win across N>=131K BS<=4 AND 262K
# BS8, still collapsing at 262K BS16 (0.71) — the single rule
# `N >= 32768*BS` covers exactly the measured win region (65K BS1 neutral
# 1.007 included; 131K BS8 marginal 1.019 excluded).
def gvr_ms_auto(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                out=None):
    bs, n = logits.shape
    dt16 = logits.dtype in (torch.bfloat16, torch.float16)
    if dt16 and n >= 65536 and n >= 32768 * bs:
        return gvr_msc(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                       out=out, C=8)
    if index_topk >= 2048 and n >= 196608 and bs <= 4:
        return gvr_msc(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                       out=out, C=8)
    if n >= 65536 and bs * 4 <= NUM_SMS:
        return gvr_msc(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                       out=out, C=4)
    return gvr_ms(logits, pre_idx, seq_lens, index_topk, compress_ratio,
                  out=out)


if __name__ == "__main__":
    import synth_data
    C = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"op21 iter2 cluster smoke (fp32, C={C})")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (65536, 131072, 262144):
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
            Npad = b["Npad"]
            seq_lens = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
            out = gvr_msc(logits, pre, seq_lens, K, crv, C=C)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nu = len(set(out[0].tolist()))
            tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
            print(f"  K={K:4d} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")
