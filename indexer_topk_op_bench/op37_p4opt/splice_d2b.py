# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""[d2b] Splice the tiny-bin fine-skip into phase4_rank_scatter of
variant/gvrpkg37 (apply AFTER splice_d2a.py).

Probe grounding (probe_d2b, 865 real cells): cnt[b*] <= 128 fires on
862/862 non-degenerate cells (med 23, p90 63). When it fires, the whole
fine 256-bin recursion (re-zero + full-candidate rescan + search) AND the
exact-tail/p4tt repair are replaced by:
  scatter pass:  bin > b*  -> positional write (unchanged semantics)
                 bin == b* -> collect (value_bits, idx) pairs into
                              smem_hist[2o]/[2o+1]  (<=128 pairs)
  warp0 select:  repeated warp-max over the pairs writes the top
                 need = K - rank_above exactly to [rank_above, K)
Value-set exact by construction (bit-equal ties pick arbitrarily — same
contract as the existing p4tt/exact-tail). The cnt[b*] > 128 fallback is
the UNCHANGED original fine path (uniform dynamic branch; barriers legal).

Flag: `p4_fine_skip` (ctor, default False for A/B; requires
enable_p4_rank_scatter_exact).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "variant" / "gvrpkg37" / "top_k" / "gvr_topk_decode.py"
src = TARGET.read_text()

if "[d2b]" in src:
    raise SystemExit("[d2b] already spliced")
assert "[d2a]" in src, "apply splice_d2a.py first (anchors assume it)"


def indent(block, n=4):
    pad = " " * n
    return "".join(pad + l if l.strip() else l
                   for l in block.splitlines(keepends=True))


# ---------------------------------------------------------------- 1 ctor arg
OLD = "        p4_rs_rw_search: Optional[bool] = None,  # [d2a]\n"
NEW = OLD + "        p4_fine_skip: Optional[bool] = None,  # [d2b]\n"
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)

# ---------------------------------------------------------------- 2 resolve
OLD = """        self.p4_rs_rw_search = (
            bool(p4_rs_rw_search) and self.enable_p4_rank_scatter_exact
        )
"""
NEW = OLD + (
    "        # [d2b] tiny-bin fine skip: when cnt[b*] <= 128 (all real decode\n"
    "        # cells measured, probe_d2b) replace the fine recursion + tail\n"
    "        # repair with a collect + warp0 exact select.\n"
    "        if p4_fine_skip is None:\n"
    "            p4_fine_skip = False\n"
    "        self.p4_fine_skip = (\n"
    "            bool(p4_fine_skip) and self.enable_p4_rank_scatter_exact\n"
    "        )\n")
assert src.count(OLD) == 1
src = src.replace(OLD, NEW)

# ------------------------------------------------- 3 wrap the exact body
# The exact branch body spans from the fbins const to the end of the
# radix-rewrite barrier, just before the APPROX `else:`. We wrap it in a
# CTA-uniform dynamic `if do_skip==0:` (barriers inside a uniform branch
# are legal — same pattern as `if do_cluster_sync:`), with the skip path
# in the `if do_skip==1:` arm above it.
START = """            # ---- EXACT: one fine-histogram recursion on the straddling bin b* ----
            if cutlass.const_expr(self.enable_p4_rank_scatter_exact):
"""
i = src.index(START)
END = """            else:
                # ---- APPROX rank-and-scatter (single pass), arbitrary straddling order ----
"""
j = src.index(END, i)
body = src[i + len(START):j]

SKIP = """                # [d2b] tiny-bin fine skip (CTA-uniform runtime gate).
                cntb_sk = smem_hist[b_star]
                need_sk = cutlass.Int32(kK) - rank_above
                do_skip = cutlass.Int32(0)
                if cutlass.const_expr(self.p4_fine_skip):
                    if cntb_sk <= cutlass.Int32(128) and need_sk > cutlass.Int32(0):
                        do_skip = cutlass.Int32(1)
                if do_skip == cutlass.Int32(1):
                    # counters: [4] = above-bin writer, [0] = pair collector
                    if tidx == cutlass.Int32(0):
                        s_iscalars[4] = cutlass.Int32(0)
                        s_iscalars[0] = cutlass.Int32(0)
                    cute.arch.barrier()
                    # one pass: scatter above-bin candidates, collect the
                    # b*-bin class as (order-preserving bits, idx) pairs
                    isk = tidx
                    while isk < cand_count:
                        vsk = smem_keys[isk]
                        bsk = cutlass.Int32((vsk - bmin_r) * inv1)
                        if bsk < cutlass.Int32(0):
                            bsk = cutlass.Int32(0)
                        if bsk > cutlass.Int32(kBins - 1):
                            bsk = cutlass.Int32(kBins - 1)
                        if bsk > b_star:
                            pos = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(4),
                                cutlass.Int32(1),
                            )
                            if pos < cutlass.Int32(kK):
                                if cutlass.const_expr(self.return_output_values):
                                    output_values_row[pos] = self.dtype(vsk)
                                output_indices_row[pos] = smem_vals[isk]
                        elif bsk == b_star:
                            osk = atomicAdd(
                                s_iscalars.iterator + cutlass.Int32(0),
                                cutlass.Int32(1),
                            )
                            if osk < cutlass.Int32(128):
                                smem_hist[osk + osk] = float_as_int32(vsk)
                                smem_hist[osk + osk + cutlass.Int32(1)] = smem_vals[isk]
                        isk = isk + cutlass.Int32(num_threads)
                    cute.arch.barrier()
                    # warp0 exact top-need select: repeated warp-max over the
                    # <=128 collected pairs (consumed flag = idx slot -1, so a
                    # genuine -FLT_MAX value stays selectable).
                    if warp_id == cutlass.Int32(0):
                        navl = s_iscalars[0]
                        if navl > cutlass.Int32(128):
                            navl = cutlass.Int32(128)
                        rsel = cutlass.Int32(0)
                        while rsel < need_sk:
                            bv = cutlass.Float32(self.NEG_FLT_MAX)
                            bs = cutlass.Int32(-1)
                            for jsl in cutlass.range_constexpr(4):
                                slot = lane + cutlass.Int32(jsl * 32)
                                if slot < navl:
                                    idx_s = smem_hist[slot + slot + cutlass.Int32(1)]
                                    if idx_s >= cutlass.Int32(0):
                                        bits_s = smem_hist[slot + slot]
                                        val_s = cutlass.Float32(
                                            llvm.bitcast(
                                                cutlass.Float32.mlir_type,
                                                bits_s.ir_value(),
                                            )
                                        )
                                        if bs < cutlass.Int32(0) or val_s > bv:
                                            bv = val_s
                                            bs = slot
                            wmax = self.warp_reduce_max_f32(bv)
                            pred_o = cutlass.Int32(0)
                            if bs >= cutlass.Int32(0):
                                if bv == wmax:
                                    pred_o = cutlass.Int32(1)
                            own = cute.arch.vote_ballot_sync(
                                pred_o != cutlass.Int32(0)
                            )
                            if own != cutlass.Uint32(0):
                                low_o = own & (cutlass.Uint32(0) - own)
                                src_o = cutlass.Int32(
                                    cute.arch.popc(low_o - cutlass.Uint32(1))
                                )
                                if lane == src_o:
                                    pos2 = rank_above + rsel
                                    if pos2 < cutlass.Int32(kK):
                                        if cutlass.const_expr(self.return_output_values):
                                            output_values_row[pos2] = self.dtype(bv)
                                        output_indices_row[pos2] = smem_hist[
                                            bs + bs + cutlass.Int32(1)
                                        ]
                                    smem_hist[bs + bs + cutlass.Int32(1)] = cutlass.Int32(-1)
                            rsel = rsel + cutlass.Int32(1)
                    cute.arch.barrier()
                else:
"""

new_body = START + SKIP + indent(body)
src = src[:i] + new_body + src[j:]

TARGET.write_text(src)
print("[d2b] spliced: ctor flag + tiny-bin skip path wrapping the exact body")
