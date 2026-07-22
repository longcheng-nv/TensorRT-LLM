# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Splice [p4sub] sub-P4 clock64 stamps into gvrpkgp4t_head (a copy of the
[ptime] twin gvrpkgtimed_head). Extends phase_ts from 8 to 16 int64 slots:

  t0..t7   unchanged ([ptime] whole-kernel phase boundaries)
  s8  after cluster handoff #2 (leader's arrive+wait for slowest peer collect)
  s9  after the leader's DSMEM peer gather (cs=1 / degrade: zero-width)
  s10 after candidate min/max block reduce         (P4 scan #1)
  s11 after coarse kNumBins histogram zero+build   (P4 scan #2, ATOMS)
  s12 after coarse 3-step high->low bin search
  s13 after fine 256-bin re-zero+build+search      (P4 scan #3)
  s14 after the classify+scatter writeback pass    (P4 scan #4)
  (tail = t6 - s14: output pad + p4_exact_tail / p4tt tie repair)

Sub-phases only cover phase4_rank_scatter (the path active at the PR head:
enable_r0 -> enable_p4_rank_scatter -> _exact all True, fp32). The
phase4_histogram_snap and APPROX paths are NOT stamped (const_expr-eliminated
at head config). Degenerate branches (cand_count == kK copy-out, < kK
underfill) collapse s10..s14 to one stamp so every launch writes all slots.

Idempotent-ish: refuses to run if '[p4sub]' already present. Every edit is an
exact-match replace; the script dies loudly if any anchor is missing.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "gvrpkgp4t_head" / "top_k" / "gvr_topk_decode.py"
src = TARGET.read_text()

if "[p4sub]" in src:
    raise SystemExit("[p4sub] already spliced; refusing to run twice")
assert "[ptime]" in src, "expected the [ptime] twin as the base"

EDITS = []


def edit(old, new, count=1, first_only=False):
    EDITS.append((old, new, count, first_only))


# ---------------------------------------------------------------- 1
# phase4_rank_scatter signature: add phase_ts_row param.
edit(
    """    def phase4_rank_scatter(
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
""",
    """    def phase4_rank_scatter(
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
        phase_ts_row,  # [p4sub] int64[16] leader-stamped sub-P4 timestamps
    ):
""",
)

# ---------------------------------------------------------------- 2
# cs=1 call site: pass phase_ts_row + stamp s9 (no gather at cs=1).
edit(
    """            if cutlass.const_expr(cluster_size == 1):
                # cs=1: the single CTA per row IS the leader.
                cand_count_p4 = min(s_iscalars[0], cutlass.Int32(self.kC))
                if cutlass.const_expr(self.enable_p4_rank_scatter):
                    self.phase4_rank_scatter(
                        smem_keys,
                        smem_vals,
                        smem_hist,
                        smem_wcnt,
                        s_thr,
                        s_iscalars,
                        output_values_row,
                        output_indices_row,
                        cand_count_p4,
                        tidx,
                        warp_id,
                        lane,
                    )
""",
    """            if cutlass.const_expr(cluster_size == 1):
                # cs=1: the single CTA per row IS the leader.
                # [p4sub] s9: no DSMEM gather at cs=1 (zero-width vs s8)
                if tidx == cutlass.Int32(0):
                    phase_ts_row[9] = cute.arch.clock64()
                cand_count_p4 = min(s_iscalars[0], cutlass.Int32(self.kC))
                if cutlass.const_expr(self.enable_p4_rank_scatter):
                    self.phase4_rank_scatter(
                        smem_keys,
                        smem_vals,
                        smem_hist,
                        smem_wcnt,
                        s_thr,
                        s_iscalars,
                        output_values_row,
                        output_indices_row,
                        cand_count_p4,
                        tidx,
                        warp_id,
                        lane,
                        phase_ts_row,  # [p4sub]
                    )
""",
)

# ---------------------------------------------------------------- 3
# s8: after cluster handoff #2 (both cs=1 pass-through and cs>1 wait).
edit(
    """            # Cluster handoff #2: leader's DSMEM gather of peer
            # smem_keys/smem_vals. Skipped at do_cluster_sync=False.
            if cutlass.const_expr(cluster_size > 1):
                if do_cluster_sync:
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
""",
    """            # Cluster handoff #2: leader's DSMEM gather of peer
            # smem_keys/smem_vals. Skipped at do_cluster_sync=False.
            if cutlass.const_expr(cluster_size > 1):
                if do_cluster_sync:
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

            # [p4sub] s8: after cluster handoff #2 (leader waited for the
            # slowest peer's collect); cs=1: zero-width vs t5.
            if is_leader & (tidx == cutlass.Int32(0)):
                phase_ts_row[8] = cute.arch.clock64()
""",
)

# ---------------------------------------------------------------- 4
# cs>1 leader call site: stamp s9 after gather (or degrade no-op) +
# pass phase_ts_row.
edit(
    """                    # ---- Phase 4: histogram snap + writeback ----
                    cand_count_p4 = min(s_iscalars[0], cutlass.Int32(self.kC))
                    if cutlass.const_expr(self.enable_p4_rank_scatter):
                        self.phase4_rank_scatter(
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            output_values_row,
                            output_indices_row,
                            cand_count_p4,
                            tidx,
                            warp_id,
                            lane,
                        )
""",
    """                    # ---- Phase 4: histogram snap + writeback ----
                    # [p4sub] s9: after DSMEM peer gather (degrade: no-op)
                    if tidx == cutlass.Int32(0):
                        phase_ts_row[9] = cute.arch.clock64()
                    cand_count_p4 = min(s_iscalars[0], cutlass.Int32(self.kC))
                    if cutlass.const_expr(self.enable_p4_rank_scatter):
                        self.phase4_rank_scatter(
                            smem_keys,
                            smem_vals,
                            smem_hist,
                            smem_wcnt,
                            s_thr,
                            s_iscalars,
                            output_values_row,
                            output_indices_row,
                            cand_count_p4,
                            tidx,
                            warp_id,
                            lane,
                            phase_ts_row,  # [p4sub]
                        )
""",
)

# ---------------------------------------------------------------- 5
# Degenerate cand_count == kK copy-out: collapse s10..s14.
edit(
    """        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i4] = self.dtype(smem_keys[i4])
                output_indices_row[i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
""",
    """        if cand_count == cutlass.Int32(kK):
            i4 = tidx
            while i4 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i4] = self.dtype(smem_keys[i4])
                output_indices_row[i4] = smem_vals[i4]
                i4 = i4 + cutlass.Int32(num_threads)
            # [p4sub] degenerate exact-fill: collapse s10..s14 (copy time
            # lands in the s9->s10 bucket = minmax column; tagged by
            # cand_count == K at analysis time via zero widths downstream).
            if tidx == cutlass.Int32(0):
                tdg = cute.arch.clock64()
                phase_ts_row[10] = tdg
                phase_ts_row[11] = tdg
                phase_ts_row[12] = tdg
                phase_ts_row[13] = tdg
                phase_ts_row[14] = tdg
""",
)

# ---------------------------------------------------------------- 6
# s10: after candidate min/max block reduce.
edit(
    """            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            cute.arch.barrier()
""",
    """            if bmax_r <= bmin_r:
                bmax_r = bmin_r + cutlass.Float32(1e-6)
            cute.arch.barrier()
            # [p4sub] s10: after candidate min/max block reduce
            if tidx == cutlass.Int32(0):
                phase_ts_row[10] = cute.arch.clock64()
""",
)

# ---------------------------------------------------------------- 7
# s11: after coarse histogram zero+build.
edit(
    """                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()
""",
    """                atomicAdd(smem_hist.iterator + bin_i, cutlass.Int32(1))
                i7 = i7 + cutlass.Int32(num_threads)
            cute.arch.barrier()
            # [p4sub] s11: after coarse kNumBins histogram zero + build
            if tidx == cutlass.Int32(0):
                phase_ts_row[11] = cute.arch.clock64()
""",
)

# ---------------------------------------------------------------- 8
# s12: after coarse 3-step search (b*/rank_above published).
edit(
    """            cute.arch.barrier()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]
""",
    """            cute.arch.barrier()
            # [p4sub] s12: after coarse 3-step high->low bin search
            if tidx == cutlass.Int32(0):
                phase_ts_row[12] = cute.arch.clock64()
            b_star = s_iscalars[3]
            rank_above = s_iscalars[2]
""",
)

# ---------------------------------------------------------------- 9
# s13: after fine 256-bin recursion (sb*/rank_above_fine published).
edit(
    """                cute.arch.barrier()
                sb_star = smem_hist[2]
                rank_above_fine = smem_hist[3]
""",
    """                cute.arch.barrier()
                # [p4sub] s13: after fine 256-bin re-zero + build + search
                if tidx == cutlass.Int32(0):
                    phase_ts_row[13] = cute.arch.clock64()
                sb_star = smem_hist[2]
                rank_above_fine = smem_hist[3]
""",
)

# ---------------------------------------------------------------- 10
# s14: after the classify+scatter writeback pass.
edit(
    """                cute.arch.barrier()
                cnt_strad = s_iscalars[1]
                filled = rank_above_fine + cnt_strad
""",
    """                cute.arch.barrier()
                # [p4sub] s14: after classify + scatter writeback pass
                if tidx == cutlass.Int32(0):
                    phase_ts_row[14] = cute.arch.clock64()
                cnt_strad = s_iscalars[1]
                filled = rank_above_fine + cnt_strad
""",
)

# ---------------------------------------------------------------- 11
# Degenerate cand_count < kK underfill: collapse s10..s14.
edit(
    """            i11 = cand_count + tidx
            while i11 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i11] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[i11] = cutlass.Int32(-1)
                i11 = i11 + cutlass.Int32(num_threads)
""",
    """            i11 = cand_count + tidx
            while i11 < cutlass.Int32(kK):
                if cutlass.const_expr(self.return_output_values):
                    output_values_row[i11] = self.dtype(self.NEG_FLT_MAX)
                output_indices_row[i11] = cutlass.Int32(-1)
                i11 = i11 + cutlass.Int32(num_threads)
            # [p4sub] degenerate underfill: collapse s10..s14
            if tidx == cutlass.Int32(0):
                tdg2 = cute.arch.clock64()
                phase_ts_row[10] = tdg2
                phase_ts_row[11] = tdg2
                phase_ts_row[12] = tdg2
                phase_ts_row[13] = tdg2
                phase_ts_row[14] = tdg2
""",
    count=2,  # 2nd hit = the identical underfill in phase4_histogram_snap
    first_only=True,  # rank_scatter's copy comes first in the file
)

for i, (old, new, count, first_only) in enumerate(EDITS, 1):
    n = src.count(old)
    assert n == count, f"edit {i}: anchor found {n} times, expected {count}"
    src = src.replace(old, new, 1 if first_only else n)

TARGET.write_text(src)
print(f"[p4sub] spliced {len(EDITS)} edits into {TARGET}")
