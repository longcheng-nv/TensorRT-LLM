# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[d1a] Splice the peer-push gather inversion into variant/gvrpkg37
(apply after splice_d2a/splice_d2b; anchors are independent of both).

Current head: after cluster handoff #2 the leader serially pulls every
peer's (keys, vals) with remote loads (§9f: med 3.11 µs = 33% of P4 at
cs=8) while peers idle. Inversion: each non-leader CTA pushes its own
chunk into the leader's SMEM at its cluster-rank prefix offset with
remote STORES (parallel across CTAs, fire-and-forget), then one extra
cluster arrive(release)+wait publishes the data — the leader only sums
counts. Offsets replicate the pull path's accounting exactly
(leader-own raw + per-peer min(cnt, kC), dst < kC cap).

Memory-model discipline (see cluster_arrive_relaxed DSMEM race lesson):
the pushing CTAs synchronize via cute.arch.cluster_arrive() — the
RELEASE variant — and the leader's subsequent local reads are ordered by
cluster_wait() (acquire). Counts read for prefixes are made visible by
the UNCHANGED handoff-#2 barrier above.

Flag: `p4_peer_push` (ctor, default False for A/B; cs>1 + do_cluster_sync
paths only — degrade/cs=1 unchanged).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "src" / "gvrpkg40v1" / "top_k" / "gvr_topk_decode.py"
src = TARGET.read_text()

if "[d1a]" in src:
    raise SystemExit("[d1a] already spliced")


def indent(block, n=4):
    pad = " " * n
    return "".join(pad + l if l.strip() else l
                   for l in block.splitlines(keepends=True))


# ------------------------------------------------------- 1 st primitives
ANCH = """@cute.jit
def ld_shared_cluster_f32(mapped_addr):
    return _ld_shared_cluster_f32(mapped_addr)
"""
NEW = ANCH + '''

@dsl_user_op
def _st_shared_cluster_i32(mapped_addr, val, *, loc=None, ip=None):
    """[d1a] Store an int32 to a peer CTA's SMEM via cluster mapped addr."""
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip),
                   val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.u32 [$0], $1;",
        constraints="r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def st_shared_cluster_i32(mapped_addr, val):
    _st_shared_cluster_i32(mapped_addr, val)


@dsl_user_op
def _st_shared_cluster_f32(mapped_addr, val, *, loc=None, ip=None):
    """[d1a] Store an fp32 to a peer CTA's SMEM via cluster mapped addr."""
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip),
                   val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.f32 [$0], $1;",
        constraints="r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def st_shared_cluster_f32(mapped_addr, val):
    _st_shared_cluster_f32(mapped_addr, val)
'''
assert src.count(ANCH) == 1
src = src.replace(ANCH, NEW)

# ---------------------------------------------------------------- 2 ctor
OLD = "        p4_fine_skip: Optional[bool] = None,  # [d2b]\n"
NEW = OLD + "        p4_peer_push: Optional[bool] = None,  # [d1a]\n"
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)

OLD = """        self.p4_fine_skip = (
            bool(p4_fine_skip) and self.enable_p4_rank_scatter_exact
        )
"""
NEW = OLD + (
    "        # [d1a] peer-push gather inversion (cs>1 clustered paths only).\n"
    "        if p4_peer_push is None:\n"
    "            p4_peer_push = False\n"
    "        self.p4_peer_push = bool(p4_peer_push)\n")
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)

# ------------------------------------------------ 3 peer push after h#2
OLD = """            # Cluster handoff #2: leader's DSMEM gather of peer
            # smem_keys/smem_vals. Skipped at do_cluster_sync=False.
            if cutlass.const_expr(cluster_size > 1):
                if do_cluster_sync:
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
"""
NEW = OLD + """
            # [d1a] peer-push: each non-leader CTA writes its collected
            # (keys, vals) chunk into the LEADER's SMEM at its cluster-rank
            # prefix offset (remote STs, parallel across CTAs), replacing
            # the leader's serial remote-LD gather below. One extra cluster
            # barrier (arrive = RELEASE) publishes the pushed data; the
            # counts consumed for prefixes were published by handoff #2.
            if cutlass.const_expr(cluster_size > 1 and self.p4_peer_push):
                if do_cluster_sync:
                    if cta_in_cluster != cutlass.Int32(0):
                        pp_isc_ptr = s_iscalars.iterator + cutlass.Int32(5)
                        # exclusive prefix over ranks < mine; rank-0 term is
                        # the leader's RAW own count (its chunk stays in
                        # place), peers are kC-capped — mirrors the pull
                        # path's base_offset accounting exactly.
                        pp_addr0 = mapa_shared_cluster(pp_isc_ptr, cutlass.Int32(0))
                        base_pp = ld_shared_cluster_i32(pp_addr0)
                        for peer_pp in cutlass.range_constexpr(1, cluster_size):
                            if cutlass.Int32(peer_pp) < cta_in_cluster:
                                pp_addr = mapa_shared_cluster(
                                    pp_isc_ptr, cutlass.Int32(peer_pp)
                                )
                                pp_cnt = ld_shared_cluster_i32(pp_addr)
                                base_pp = base_pp + min(
                                    pp_cnt, cutlass.Int32(self.kC)
                                )
                        my_cnt_pp = min(s_iscalars[5], cutlass.Int32(self.kC))
                        pp_keys_it = smem_keys.iterator
                        pp_vals_it = smem_vals.iterator
                        ipp = tidx
                        while ipp < my_cnt_pp:
                            dst_pp = base_pp + ipp
                            if dst_pp < cutlass.Int32(self.kC):
                                pp_kaddr = mapa_shared_cluster(
                                    pp_keys_it + dst_pp, cutlass.Int32(0)
                                )
                                pp_vaddr = mapa_shared_cluster(
                                    pp_vals_it + dst_pp, cutlass.Int32(0)
                                )
                                st_shared_cluster_f32(pp_kaddr, smem_keys[ipp])
                                st_shared_cluster_i32(pp_vaddr, smem_vals[ipp])
                            ipp = ipp + cutlass.Int32(num_threads)
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
"""
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)

# --------------------------------------- 4 leader side: skip data loop
START = """                    if do_cluster_sync:
                        # DSMEM-gather peer candidates into the leader's
"""
i = src.index(START)
END = """                        # Reset s_iscalars[0] to cluster-wide cand_count.
                        if tidx == cutlass.Int32(0):
                            s_iscalars[0] = base_offset
                        cute.arch.barrier()
"""
j = src.index(END, i) + len(END)
gather_old = src[i:j]
gather_body = gather_old[len("                    if do_cluster_sync:\n"):]

gather_new = (
    "                    if do_cluster_sync:\n"
    "                        if cutlass.const_expr(self.p4_peer_push):\n"
    "                            # [d1a] data already pushed by the peers;\n"
    "                            # only sum counts for the cluster-wide\n"
    "                            # cand_count (same accounting as the pull).\n"
    "                            pp_l_ptr = s_iscalars.iterator + cutlass.Int32(5)\n"
    "                            base_offset_pp = s_iscalars[5]\n"
    "                            for peer_l in cutlass.range_constexpr(1, cluster_size):\n"
    "                                pp_l_addr = mapa_shared_cluster(\n"
    "                                    pp_l_ptr, cutlass.Int32(peer_l)\n"
    "                                )\n"
    "                                pp_l_cnt = ld_shared_cluster_i32(pp_l_addr)\n"
    "                                base_offset_pp = base_offset_pp + min(\n"
    "                                    pp_l_cnt, cutlass.Int32(self.kC)\n"
    "                                )\n"
    "                            if tidx == cutlass.Int32(0):\n"
    "                                s_iscalars[0] = base_offset_pp\n"
    "                            cute.arch.barrier()\n"
    "                        else:\n"
    + indent(gather_body)
)
src = src[:i] + gather_new + src[j:]

TARGET.write_text(src)
print("[d1a] spliced: st prims + ctor flag + peer-push + leader count-only")
