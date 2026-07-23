# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[d2a] Splice the redundant-warp coarse/fine rank search into
phase4_rank_scatter of variant/gvrpkg37 (copy of the PR#16457 head).

Design: port the silicon-proven `_kth_bin_search_rw` idiom (p4_warp_redundant,
already default-ON for the snap path) to the rank-scatter path's two serial
3-step searches. One generic `_p4_rw_rank_search(nbins, target)` returns
(bin_star, rank_above) in registers on EVERY warp — no leader, no
s_iscalars/smem staging, ONE barrier instead of 3 (coarse) / 4 (fine incl.
counter-reset staging).

Flag: `p4_rs_rw_search` (ctor, default False for A/B; gated on
enable_p4_rank_scatter_exact — the approx branch depends on the coarse-stage
s_iscalars resets that the rw path skips).

Idempotent-ish: refuses if '[d2a]' present. Block replacements extract the
original code between exact anchors and re-indent it under `else:` so the
baseline path stays byte-identical.
"""
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "src" / "gvrpkg40v1" / "top_k" / "gvr_topk_decode.py"
src = TARGET.read_text()

if "[d2a]" in src:
    raise SystemExit("[d2a] already spliced")


def find_block(start_anchor, end_anchor):
    i = src.index(start_anchor)
    j = src.index(end_anchor, i) + len(end_anchor)
    return src[i:j]


def indent(block, n=4):
    pad = " " * n
    return "".join(pad + l if l.strip() else l
                   for l in block.splitlines(keepends=True))


# ---------------------------------------------------------------- 1 ctor arg
OLD = "        p4_tail_fast: Optional[bool] = None,  # [p4tt]\n"
NEW = ("        p4_tail_fast: Optional[bool] = None,  # [p4tt]\n"
       "        p4_rs_rw_search: Optional[bool] = None,  # [d2a]\n")
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)

# ---------------------------------------------------------------- 2 resolve
OLD = "        self.p4_tail_fast = bool(p4_tail_fast) and self.p4_exact_tail  # [p4tt]\n"
NEW = OLD + (
    "        # [d2a] redundant-warp coarse/fine rank search inside\n"
    "        # phase4_rank_scatter (ports the _kth_bin_search_rw idiom).\n"
    "        # Exact path only: the approx branch consumes the coarse-stage\n"
    "        # s_iscalars[4]/[1] resets that the rw path skips.\n"
    "        if p4_rs_rw_search is None:\n"
    "            p4_rs_rw_search = False\n"
    "        self.p4_rs_rw_search = (\n"
    "            bool(p4_rs_rw_search) and self.enable_p4_rank_scatter_exact\n"
    "        )\n")
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)

# ---------------------------------------------------------------- 3 method
METHOD = '''
    # ------------------------------------------------------------------
    # [d2a] _p4_rw_rank_search — redundant-warp descending rank search
    # for phase4_rank_scatter. Mirrors _kth_bin_search_rw (silicon-proven
    # on the snap path) but returns (bin_star, rank_above_local) instead
    # of (threshold, count), parameterized over the bin count and rank
    # target so one body serves the coarse (kNumBins, target=K) and fine
    # (256, target=K-rank_above) searches. ONE barrier (Step-1 staging);
    # every warp computes bit-identical results in registers — no leader,
    # no publish barrier, no s_iscalars staging.
    # ------------------------------------------------------------------
    @cute.jit
    def _p4_rw_rank_search(
        self, smem_hist, smem_wcnt, target, warp_id, lane,
        nbins: cutlass.Constexpr,
    ):
        bpw = cutlass.const_expr(nbins // self.num_warps)

        # Step 1: per-warp descending bin-slice sums -> smem_wcnt.
        wsum = cutlass.Int32(0)
        if cutlass.const_expr(bpw % self.WARP_SIZE == 0):
            for jm in cutlass.range_constexpr(bpw // self.WARP_SIZE):
                bidx_s = (
                    cutlass.Int32(nbins - 1)
                    - warp_id * cutlass.Int32(bpw)
                    - (lane + cutlass.Int32(jm * self.WARP_SIZE))
                )
                wsum = wsum + smem_hist[bidx_s]
            wsum = self.warp_reduce_sum_i32(wsum)
        else:
            for jb in cutlass.range_constexpr(bpw):
                bidx_s = (
                    cutlass.Int32(nbins - 1)
                    - warp_id * cutlass.Int32(bpw)
                    - cutlass.Int32(jb)
                )
                wsum = wsum + smem_hist[bidx_s]
        if lane == cutlass.Int32(0):
            smem_wcnt[warp_id] = wsum
        cute.arch.barrier()

        # Step 2 (every warp, lane-parallel): lane w holds slot w; an
        # inclusive idx-shuffle scan + ballot locate the target warp and
        # its exclusive prefix. (shuffle_sync with a computed source lane
        # is the working shfl idiom; shuffle_sync_up ignores its offset —
        # probed, see _kth_bin_search_rw.)
        v_s = cutlass.Int32(0)
        if lane < cutlass.Int32(self.num_warps):
            v_s = smem_wcnt[lane]
        run2 = v_s
        for d2 in cutlass.range_constexpr(5):
            off2 = cutlass.const_expr(1 << d2)
            src2 = lane - cutlass.Int32(off2)
            if src2 < cutlass.Int32(0):
                src2 = cutlass.Int32(0)
            up2 = cute.arch.shuffle_sync(run2, src2)
            if lane >= cutlass.Int32(off2):
                run2 = run2 + up2
        m2 = cute.arch.vote_ballot_sync(run2 >= target)
        tw = cutlass.Int32(self.num_warps - 1)
        if m2 != cutlass.Uint32(0):
            low2 = m2 & (cutlass.Uint32(0) - m2)
            tw = cutlass.Int32(cute.arch.popc(low2 - cutlass.Uint32(1)))
        incl_tw = cute.arch.shuffle_sync(run2, tw)
        slot_tw = cute.arch.shuffle_sync(v_s, tw)
        prefix = incl_tw - slot_tw

        # Step 3 (every warp, lane-parallel): lane l owns the contiguous
        # descending positions [l*ppl, (l+1)*ppl) of the target slice;
        # exclusive cross-lane prefix; the unique crossing lane yields
        # (bin_star, rank strictly above it).
        ppl = cutlass.const_expr((bpw + self.WARP_SIZE - 1) // self.WARP_SIZE)
        cnt_frag = cute.make_fragment((ppl,), cutlass.Int32)
        my_sum = cutlass.Int32(0)
        for j3 in cutlass.range_constexpr(ppl):
            pos = lane * cutlass.Int32(ppl) + cutlass.Int32(j3)
            cnt_j = cutlass.Int32(0)
            if pos < cutlass.Int32(bpw):
                bidx3 = cutlass.Int32(nbins - 1) - tw * cutlass.Int32(bpw) - pos
                cnt_j = smem_hist[bidx3]
            cnt_frag[j3] = cnt_j
            my_sum = my_sum + cnt_j
        run3 = my_sum
        for d3 in cutlass.range_constexpr(5):
            off3 = cutlass.const_expr(1 << d3)
            src3 = lane - cutlass.Int32(off3)
            if src3 < cutlass.Int32(0):
                src3 = cutlass.Int32(0)
            up3 = cute.arch.shuffle_sync(run3, src3)
            if lane >= cutlass.Int32(off3):
                run3 = run3 + up3
        base3 = prefix + (run3 - my_sum)

        bin_loc = cutlass.Int32(0)
        above_loc = cutlass.Int32(0)
        hit = cutlass.Int32(0)
        r3 = base3
        for j4 in cutlass.range_constexpr(ppl):
            pos4 = lane * cutlass.Int32(ppl) + cutlass.Int32(j4)
            cnt4 = cnt_frag[j4]
            if (
                pos4 < cutlass.Int32(bpw)
                and r3 < target
                and r3 + cnt4 >= target
                and hit == cutlass.Int32(0)
            ):
                bin_loc = cutlass.Int32(nbins - 1) - tw * cutlass.Int32(bpw) - pos4
                above_loc = r3
                hit = cutlass.Int32(1)
            r3 = r3 + cnt4
        # Broadcast from the (at most one) hitting lane; the no-hit
        # fallback mirrors the serial walk's init (top bin, 0 above) and
        # is unreachable when sum(hist) >= target (guaranteed by both
        # call sites: cand_count >= K coarse, cnt[b*] >= K-rank_above fine).
        mask3 = cute.arch.vote_ballot_sync(hit != cutlass.Int32(0))
        bin_out = cutlass.Int32(nbins - 1)
        above_out = cutlass.Int32(0)
        if mask3 != cutlass.Uint32(0):
            low = mask3 & (cutlass.Uint32(0) - mask3)
            srcl = cutlass.Int32(cute.arch.popc(low - cutlass.Uint32(1)))
            bin_out = cute.arch.shuffle_sync(bin_loc, srcl)
            above_out = cute.arch.shuffle_sync(above_loc, srcl)
        return bin_out, above_out

'''
ANCH = "    # ------------------------------------------------------------------\n    # Phase 4 (alt): op#7 fused rank-and-scatter (enable_p4_rank_scatter).\n"
assert src.count(ANCH) == 1
src = src.replace(ANCH, METHOD + ANCH)

# ------------------------------------------------------- 4 coarse search
COARSE_START = "            # ---- 3-step high→low bin search → straddling bin b* + rank_above ----\n"
COARSE_END = """            cute.arch.barrier()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]
"""
coarse_old = find_block(COARSE_START, COARSE_END)
assert src.count(coarse_old) == 1
coarse_body = coarse_old[len(COARSE_START):]
coarse_new = (
    COARSE_START
    + "            b_star = cutlass.Int32(kBins - 1)\n"
    + "            rank_above = cutlass.Int32(0)\n"
    + "            if cutlass.const_expr(self.p4_rs_rw_search):\n"
    + "                # [d2a] redundant-warp search: 1 barrier, results in\n"
    + "                # registers on every warp; no s_iscalars staging.\n"
    + "                b_star, rank_above = self._p4_rw_rank_search(\n"
    + "                    smem_hist, smem_wcnt, cutlass.Int32(kK),\n"
    + "                    warp_id, lane, kBins,\n"
    + "                )\n"
    + "            else:\n"
    + indent(coarse_body)
)
src = src.replace(coarse_old, coarse_new)

# --------------------------------------------------------- 5 fine search
FINE_START = "                # fine 3-step search seeded at rank_above (over fbins bins)\n"
FINE_END = """                cute.arch.barrier()
                sb_star = smem_hist[2]
                rank_above_fine = smem_hist[3]
"""
fine_old = find_block(FINE_START, FINE_END)
assert src.count(fine_old) == 1
fine_body = fine_old[len(FINE_START):]
fine_new = (
    FINE_START
    + "                sb_star = cutlass.Int32(fbins - 1)\n"
    + "                rank_above_fine = rank_above\n"
    + "                if cutlass.const_expr(self.p4_rs_rw_search):\n"
    + "                    # [d2a] rw fine search (target = K - rank_above in\n"
    + "                    # the local fine hist) + counter reset: 2 barriers\n"
    + "                    # replace the serial path's 4.\n"
    + "                    sb_star, above_f = self._p4_rw_rank_search(\n"
    + "                        smem_hist, smem_wcnt,\n"
    + "                        cutlass.Int32(kK) - rank_above,\n"
    + "                        warp_id, lane, 256,\n"
    + "                    )\n"
    + "                    rank_above_fine = rank_above + above_f\n"
    + "                    if tidx == cutlass.Int32(0):\n"
    + "                        s_iscalars[4] = cutlass.Int32(0)  # cnt_above\n"
    + "                        s_iscalars[0] = cutlass.Int32(0)  # cnt_mid\n"
    + "                        s_iscalars[1] = cutlass.Int32(0)  # cnt_strad\n"
    + "                    cute.arch.barrier()\n"
    + "                else:\n"
    + indent(fine_body)
)
src = src.replace(fine_old, fine_new)

TARGET.write_text(src)
print("[d2a] spliced: ctor flag + _p4_rw_rank_search + coarse/fine rw branches")
