# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""op19 Strategy-B: cooperative-cluster SANDWICH portfolio (multi-CTA).

op17's cluster portfolio + the op19 sandwich, with straddle-aware slot
placement: G CTAs count the full row at thresholds
  thr_0 = pmin (exactness anchor), thr_r = pmin + frac_r*(pmax-pmin),
  frac_r in linspace(l1, l0, G-1)  (l1/l0 from results/straddle_fracs.json)
DSMEM-share the G counts; all ranks compute the sandwich pair:
  r1 = highest rank with count >= K (thr1, M1);  r0 = r1+1 (thr0, M0 < K).
Winner (rank r1) DSM-copies r0's per-thread count column into its own
smem_ptcnt_up (barrier #2 keeps r0's smem alive), then runs the EXACT
Strategy-A sandwich P3 (direct-write M0 + band collect) + runtime-k band
snap P4. Everyone else exits after barrier #2.

Also fixes op17's D0 exactness edge: when the winning count > kC (no pair),
done=2 -> vendored P3 retry-shrink instead of silent collect-and-cap.
"""
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
from gvr_sw_op import GvrSandwichKernel, _load_straddle  # noqa: E402
from cute_vendored.blackwell.utils import (  # noqa: E402
    TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait,
)
from cute_vendored.blackwell.top_k.gvr_topk_decode_cluster import (  # noqa: E402
    mapa_shared_cluster, ld_shared_cluster_i32,
)
from cutlass._mlir.dialects import llvm  # noqa: E402
from cutlass.cutlass_dsl import T, dsl_user_op  # noqa: E402
from cutlass.utils.smem_allocator import SmemAllocator  # noqa: E402


@dsl_user_op
def _st_shared_cluster_i32(mapped_addr, val, *, loc=None, ip=None):
    """Store an int32 to a peer CTA's SMEM via cluster mapped address."""
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

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16,
       torch.float16: cutlass.Float16}


class GvrClusterSandwichKernel(GvrSandwichKernel):
    """G-CTA cluster; slot fracs are a compile-time tuple of length G
    (fracs[0] MUST be 0.0 — the count(pmin)>=K exactness anchor)."""

    def __init__(self, *a, G_thr=8, slot_fracs=None, sw_enable=True,
                 use_push=True, **kw):
        super().__init__(*a, **kw)
        self.G_thr = int(G_thr)
        self.sw_enable = bool(sw_enable)  # False -> op17 path (bisect flag)
        self.use_push = bool(use_push)    # False -> ld-copy + barrier #2
        self.slot_fracs = tuple(float(f) for f in slot_fracs)
        assert len(self.slot_fracs) == self.G_thr and self.slot_fracs[0] == 0.0

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
        G = cutlass.const_expr(self.G_thr)
        warp_id = tidx // self.WARP_SIZE
        lane = tidx & (self.WARP_SIZE - 1)

        rank = bidx % cutlass.Int32(G)
        row_idx = bidx // cutlass.Int32(G)
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
        # op19 sandwich scratch
        smem_ptcnt_up = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((num_threads,), order=(0,)), byte_alignment=128)
        smem_didx = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((top_k,), order=(0,)), byte_alignment=128)
        s_swf = smem.allocate_tensor(element_type=cutlass.Float32, layout=cute.make_ordered_layout((1,), order=(0,)), byte_alignment=16)
        s_swi = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((2,), order=(0,)), byte_alignment=16)
        # cluster scratch: [0]=this CTA's count (DSMEM-shared), [1]=r1, [2]=M1, [3]=M0
        s_cluster = smem.allocate_tensor(element_type=cutlass.Int32, layout=cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16)

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
            self.phase1_preidx_stats(input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset,
                                     smem_wmin, smem_wmax, smem_wsum, smem_wcnt_p1, s_thr, s_iscalars, tidx, warp_id, lane)
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
                # ---- slot threshold: compile-time straddle-aware fracs ----
                band_lo = s_thr[1]
                band_hi = s_thr[2]
                d = band_hi - band_lo
                thr_r = band_lo
                for g in cutlass.range_constexpr(G):
                    if rank == cutlass.Int32(g):
                        thr_r = band_lo + d * cutlass.Float32(self.slot_fracs[g])
                smem_ptcnt_up[tidx] = cutlass.Int32(0)
                self.block_count_ge(input_row, N, thr_r, smem_ptcnt, smem_wcnt, s_iscalars, tidx, warp_id, lane)
                cute.arch.barrier()
                # proactive push: rank r stores its column into rank r-1's
                # smem_ptcnt_up BEFORE the (single) cluster barrier — the
                # eventual winner r1 then already holds r0=r1+1's column
                # locally, killing op17-style barrier #2 entirely.
                if cutlass.const_expr(self.use_push):
                    if rank > cutlass.Int32(0):
                        up_slot = smem_ptcnt_up.iterator + tidx
                        peer_addr = mapa_shared_cluster(up_slot, rank - cutlass.Int32(1))
                        st_shared_cluster_i32(peer_addr, smem_ptcnt[tidx])
                if tidx == cutlass.Int32(0):
                    s_cluster[0] = s_iscalars[0]
                if cutlass.const_expr(self.use_push):
                    cute.arch.cluster_arrive()
                else:
                    cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

                # all ranks: find sandwich pair (r1 tightest >=K; r0 = r1+1)
                if tidx == cutlass.Int32(0):
                    r1 = cutlass.Int32(0)
                    m1 = cutlass.Int32(0x7FFFFFFF)
                    m0 = cutlass.Int32(0)
                    local_ptr = s_cluster.iterator + cutlass.Int32(0)
                    for peer in cutlass.range_constexpr(G):
                        peer_addr = mapa_shared_cluster(local_ptr, cutlass.Int32(peer))
                        peer_cnt = ld_shared_cluster_i32(peer_addr)
                        if peer_cnt >= cutlass.Int32(top_k):
                            r1 = cutlass.Int32(peer)
                            m1 = peer_cnt
                        else:
                            if m0 == cutlass.Int32(0):
                                m0 = peer_cnt  # first count<K == r1+1 (sorted)
                    s_cluster[1] = r1
                    s_cluster[2] = m1
                    s_cluster[3] = m0
                cute.arch.barrier()
                r1 = s_cluster[1]
                m1 = s_cluster[2]
                m0 = s_cluster[3]
                band_cnt = m1 - m0

                # winner setup (r0's column already pushed into ptcnt_up)
                is_winner = rank == r1
                use_sw = (m0 > cutlass.Int32(0)) and (band_cnt <= cutlass.Int32(kC))
                if cutlass.const_expr(not self.sw_enable):
                    use_sw = False
                if is_winner:
                    if cutlass.const_expr(not self.use_push):
                        if use_sw:
                            my_slot = smem_ptcnt.iterator + tidx
                            peer_addr = mapa_shared_cluster(my_slot, r1 + cutlass.Int32(1))
                            smem_ptcnt_up[tidx] = ld_shared_cluster_i32(peer_addr)
                    if tidx == cutlass.Int32(0):
                        s_swi[0] = m0 if use_sw else cutlass.Int32(0)
                        # thr0 value: recompute from r0's frac
                        t0 = cutlass.Float32(self.FLT_MAX)
                        for g in cutlass.range_constexpr(G):
                            if r1 + cutlass.Int32(1) == cutlass.Int32(g):
                                t0 = band_lo + d * cutlass.Float32(self.slot_fracs[g])
                        s_swf[0] = t0
                        s_thr[0] = thr_r  # == thr1 on the winner
                    if not use_sw:
                        smem_ptcnt_up[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()
                if cutlass.const_expr(not self.use_push):
                    # barrier #2: r0's smem must stay alive through the ld-copy
                    cute.arch.cluster_arrive_relaxed()
                    cute.arch.cluster_wait()

                if is_winner:
                    if use_sw:
                        self.phase3_sandwich(input_row, N, smem_keys, smem_vals,
                                             smem_ptcnt, smem_ptcnt_up, smem_wcnt,
                                             smem_didx, s_thr, s_swf, s_iscalars,
                                             output_values_row, output_indices_row,
                                             tidx, warp_id, lane)
                        band = s_iscalars[0]
                        if band > cutlass.Int32(kC):
                            band = cutlass.Int32(kC)
                        k_rem = cutlass.Int32(top_k) - m0
                        self.phase4_band_snap(smem_keys, smem_vals, smem_hist,
                                              smem_wcnt, s_thr, s_swf, s_iscalars,
                                              output_values_row, output_indices_row,
                                              band, k_rem, m0, tidx, warp_id, lane)
                    else:
                        # op17 path + D0 fix: done=2 retry-shrink when M1 > kC
                        if tidx == cutlass.Int32(0):
                            if m1 <= cutlass.Int32(kC):
                                s_iscalars[0] = m1
                                s_iscalars[1] = cutlass.Int32(1)
                            else:
                                s_iscalars[1] = cutlass.Int32(2)
                                s_thr[1] = thr_r
                                s_thr[2] = band_hi
                        cute.arch.barrier()
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
        G = cutlass.const_expr(self.G_thr)
        self.gvr_topk_kernel(input_data, pre_idx, seq_lens, output_values, output_indices).launch(
            grid=(num_rows * G, 1, 1), block=(self.num_threads, 1, 1),
            cluster=(G, 1, 1), stream=stream, use_pdl=TRTLLM_ENABLE_PDL,
            min_blocks_per_mp=self.min_blocks_per_mp)


_compiled = {}


def _slot_fracs(K, n, G, dtype_name="fp32"):
    """G slot fracs: {0 anchor} + linspace(l1, l0, G-1) from the straddle
    table (l1 = fracs[1], l0 = fracs[-1] of the M=4 entry)."""
    base = _load_straddle(K, n, 4, dtype_name)
    l1, l0 = base[1], base[-1]
    inner = [l1 + (l0 - l1) * j / (G - 2) for j in range(G - 1)]
    return tuple([0.0] + inner)


def _config(bs, n):
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def _compile(dtype, bs, n, K, cr_val, G, kC, sw_enable=True, use_push=True):
    key = (dtype, bs, n, K, cr_val, G, kC, sw_enable, use_push)
    if key in _compiled:
        return _compiled[key]
    dtn = {torch.float32: "fp32", torch.bfloat16: "bf16",
           torch.float16: "fp16"}[dtype]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrClusterSandwichKernel(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                                    use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                                    min_blocks_per_mp=min_bpm, return_output_values=False,
                                    G_thr=G, slot_fracs=_slot_fracs(K, n, G, dtn), sw_enable=sw_enable, use_push=use_push,
                                    kC_override=kC, M_thr=2, R_rounds=1)
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


def pick_G(bs, G_max=16):
    budget = NUM_SMS // 2
    for g in (16, 8, 4):
        if bs * g <= budget:
            return g
    return 1


def gvr_swc(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None, G=16, kC=None, sw_enable=True, use_push=True):
    from gvr_cutedsl_op import gvr_cutedsl
    bs, n = logits.shape
    if G == "auto":
        G = pick_G(bs)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    if G < 2:
        return gvr_cutedsl(logits, pre_idx, seq_lens, index_topk, compress_ratio, out=out)
    compiled = _compile(logits.dtype, bs, n, index_topk, compress_ratio, G, kC, sw_enable, use_push)
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    import synth_data
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"op19 cluster sandwich smoke (fp32, G={G})")
    for K, crv in ((512, 4), (1024, 4), (2048, 1)):
        for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            if N <= 2 * K:
                continue
            b = synth_data.get_bundle(K, torch.float32, N)
            logits, pre = b["logits"].cuda(), b["preIdx"].cuda()
            Npad = b["Npad"]
            seq_lens = torch.full((1,), Npad * crv, dtype=torch.int32, device="cuda")
            out = gvr_swc(logits, pre, seq_lens, K, crv, G=G)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nu = len(set(out[0].tolist()))
            tag = "OK" if (d == 0.0 and nu == K) else "**FAIL**"
            print(f"  K={K:4d} N={N:6d}: uniq={nu}/{K} valdiff={d:.2e} {tag}")
