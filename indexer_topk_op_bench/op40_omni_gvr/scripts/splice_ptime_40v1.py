#!/usr/bin/env python3
"""Splice [ptime] clock64 phase timestamps into gvrpkg40t/top_k/gvr_topk_decode.py.

Idempotent-ish: refuses to run if '[ptime]' already present. Every edit is
an exact-string replacement asserted to match exactly once. The vendored
gvrpkg40b package is NOT touched.

phase_ts layout: int64[num_rows, 8] = [t0..t7], stamped by the leader CTA
(cta_in_cluster==0) thread 0 only, inside _run_phases:
  t0 entry | t1 post-P1 | t2 post-smem-stage | t3 post-P1b rungs |
  t4 threshold-final (post R0 admission + fb_fix refine / secant) |
  t5 post-P3 collect | t6 post-P4 (incl exact tail) | t7 end (post final barrier)
"""
import sys
from pathlib import Path

F = Path(__file__).resolve().parent.parent / "src" / "gvrpkg40v1t" / "top_k" / "gvr_topk_decode.py"
src = F.read_text()
if "[ptime]" in src:
    print("already spliced; aborting")
    sys.exit(1)

EDITS = []


def rep(old, new):
    EDITS.append((old, new))


# --- 1. gvr_topk_kernel signature: add phase_ts ---
rep(
    """        order_row: cute.Tensor,  # [batch_size] int32 (or None when seqlen_sorted=False)
    ):
        \"\"\"Thin entry: bidx → row_idx → run_one_row.""",
    """        order_row: cute.Tensor,  # [batch_size] int32 (or None when seqlen_sorted=False)
        phase_ts: cute.Tensor,  # [ptime] int64[num_rows, 8] phase timestamps
    ):
        \"\"\"Thin entry: bidx → row_idx → run_one_row.""",
)

# --- 2. gvr_topk_kernel -> run_one_row call: thread phase_ts ---
rep(
    """        self.run_one_row(
            row_idx,
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
        )""",
    """        self.run_one_row(
            row_idx,
            input_data,
            pre_idx,
            seq_lens,
            output_values,
            output_indices,
            phase_ts,  # [ptime]
        )""",
)

# --- 3. run_one_row signature ---
rep(
    """        output_indices: cute.Tensor,  # [numRows, top_k] int32
    ):
        \"\"\"Dispatch: compute per-row slice + cluster sync mode, call _run_phases.""",
    """        output_indices: cute.Tensor,  # [numRows, top_k] int32
        phase_ts: cute.Tensor,  # [ptime] int64[num_rows, 8]
    ):
        \"\"\"Dispatch: compute per-row slice + cluster sync mode, call _run_phases.""",
)

# --- 4. run_one_row: slice the per-row phase_ts view ---
rep(
    """        pre_idx_count = pre_idx.shape[1]

        griddepcontrol_wait()""",
    """        pre_idx_count = pre_idx.shape[1]
        phase_ts_row = phase_ts[row_idx, None]  # [ptime]

        griddepcontrol_wait()""",
)

# --- 5/6/7. three _run_phases call sites: append phase_ts_row ---
rep(
    """                        smem_gath,
                        tidx,
                        warp_id,
                        lane,
                    )
                else:
                    # Short row: only CTA 0 scans the full row; the other""",
    """                        smem_gath,
                        tidx,
                        warp_id,
                        lane,
                        phase_ts_row,  # [ptime]
                    )
                else:
                    # Short row: only CTA 0 scans the full row; the other""",
)
rep(
    """                            smem_gath,
                            tidx,
                            warp_id,
                            lane,
                        )""",
    """                            smem_gath,
                            tidx,
                            warp_id,
                            lane,
                            phase_ts_row,  # [ptime]
                        )""",
)
rep(
    """                    smem_gath,
                    tidx,
                    warp_id,
                    lane,
                )

        griddepcontrol_launch_dependents()""",
    """                    smem_gath,
                    tidx,
                    warp_id,
                    lane,
                    phase_ts_row,  # [ptime]
                )

        griddepcontrol_launch_dependents()""",
)

# --- 8. _run_phases signature ---
rep(
    """        tidx,
        warp_id,
        lane,
    ):
        \"\"\"Run Phase 1-4 + final cluster barrier on a given row slice.""",
    """        tidx,
        warp_id,
        lane,
        phase_ts_row,  # [ptime] int64[8] leader-stamped clock64 timestamps
    ):
        \"\"\"Run Phase 1-4 + final cluster barrier on a given row slice.""",
)

# --- 9. t0: entry ---
rep(
    """        is_leader = cta_in_cluster == cutlass.Int32(0)

        # ---- Phase 1: preIdx Min/Max/Mean ----""",
    """        is_leader = cta_in_cluster == cutlass.Int32(0)

        # [ptime] t0: entry of _run_phases (leader CTA, thread 0 only)
        if is_leader & (tidx == cutlass.Int32(0)):
            phase_ts_row[0] = cute.arch.clock64()

        # ---- Phase 1: preIdx Min/Max/Mean ----""",
)

# --- 10. t1: after phase1_preidx_stats ---
rep(
    """            s_mt_thr=s_mt_thr,  # r0_vseed: park pmean in the last rung column
        )

        # Degenerate threshold init:""",
    """            s_mt_thr=s_mt_thr,  # r0_vseed: park pmean in the last rung column
        )

        # [ptime] t1: after phase1_preidx_stats (phase1 ends with a CTA barrier)
        if is_leader & (tidx == cutlass.Int32(0)):
            phase_ts_row[1] = cute.arch.clock64()

        # Degenerate threshold init:""",
)

# --- 11a. degenerate-threshold branch (cs=1): collapse t2..t5 ---
rep(
    """                        je = je + cutlass.Int32(1)
            else:
                # cs>1: all cluster CTAs enter _run_phases; only leader writes.""",
    """                        je = je + cutlass.Int32(1)
                    # [ptime] degenerate bracket: collapse t2..t5
                    td = cute.arch.clock64()
                    phase_ts_row[2] = td
                    phase_ts_row[3] = td
                    phase_ts_row[4] = td
                    phase_ts_row[5] = td
            else:
                # cs>1: all cluster CTAs enter _run_phases; only leader writes.""",
)

# --- 11b. degenerate-threshold branch (cs>1, leader-guarded block) ---
rep(
    """                        je = je + cutlass.Int32(1)
        else:
            # Stage this CTA's slice into SMEM once before Phase 2's""",
    """                        je = je + cutlass.Int32(1)
                    # [ptime] degenerate bracket: collapse t2..t5
                    td = cute.arch.clock64()
                    phase_ts_row[2] = td
                    phase_ts_row[3] = td
                    phase_ts_row[4] = td
                    phase_ts_row[5] = td
        else:
            # Stage this CTA's slice into SMEM once before Phase 2's""",
)

# --- 12. t2: after smem slice stage (outside the const_expr if) ---
rep(
    """                self.load_slice_to_smem(
                    input_row,
                    slice_start,
                    slice_end,
                    smem_input,
                    tidx,
                )

            # ---- Phase 2: R0 histogram-ladder admission (single-CTA fast""",
    """                self.load_slice_to_smem(
                    input_row,
                    slice_start,
                    slice_end,
                    smem_input,
                    tidx,
                )

            # [ptime] t2: after smem slice stage (== t1 when cache disabled)
            if is_leader & (tidx == cutlass.Int32(0)):
                phase_ts_row[2] = cute.arch.clock64()

            # ---- Phase 2: R0 histogram-ladder admission (single-CTA fast""",
)

# --- 13a. t3: after P1b rungs (R0 path) ---
rep(
    """                self.block_count_ge_multi(
                    input_row,
                    slice_start,
                    slice_end,""",
    """                # [ptime] t3: after P1b rungs (P1b ends with a CTA barrier)
                if is_leader & (tidx == cutlass.Int32(0)):
                    phase_ts_row[3] = cute.arch.clock64()
                self.block_count_ge_multi(
                    input_row,
                    slice_start,
                    slice_end,""",
)

# --- 13b. t3 on the pure-secant path (enable_r0=False): no P1b -> t3==t2-ish ---
rep(
    """            else:
                self.phase2_secant_search(
                    input_row,
                    N,
                    slice_start,""",
    """            else:
                # [ptime] t3: no P1b on the pure-secant path
                if is_leader & (tidx == cutlass.Int32(0)):
                    phase_ts_row[3] = cute.arch.clock64()
                self.phase2_secant_search(
                    input_row,
                    N,
                    slice_start,""",
)

# --- 14. t4: threshold final, before cluster handoff #1 ---
rep(
    """            # Cluster handoff #1 (end of Phase 2). Skipped when""",
    """            # [ptime] t4: threshold final (post R0 admission + fb_fix refine
            # or secant search); just before Phase 3.
            if is_leader & (tidx == cutlass.Int32(0)):
                phase_ts_row[4] = cute.arch.clock64()

            # Cluster handoff #1 (end of Phase 2). Skipped when""",
)

# --- 15. t5: after phase3_collect_candidates ---
rep(
    """            # Cluster handoff #2: leader's DSMEM gather of peer""",
    """            # [ptime] t5: after phase3_collect_candidates
            if is_leader & (tidx == cutlass.Int32(0)):
                phase_ts_row[5] = cute.arch.clock64()

            # Cluster handoff #2: leader's DSMEM gather of peer""",
)

# --- 16. t6: after Phase 4 (incl. DSMEM gather + exact tail) ---
rep(
    """        # Final cluster barrier: keep peer CTAs (and their SMEM) alive""",
    """        # [ptime] t6: after Phase 4 (incl. cluster gather + p4 exact tail)
        if is_leader & (tidx == cutlass.Int32(0)):
            phase_ts_row[6] = cute.arch.clock64()

        # Final cluster barrier: keep peer CTAs (and their SMEM) alive""",
)

# --- 17. t7: end of _run_phases (after final cluster barrier) ---
rep(
    """                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

    # ------------------------------------------------------------------
    # Host-side launcher""",
    """                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

        # [ptime] t7: end of _run_phases
        if is_leader & (tidx == cutlass.Int32(0)):
            phase_ts_row[7] = cute.arch.clock64()

    # ------------------------------------------------------------------
    # Host-side launcher""",
)

# --- 18. __call__ signature ---
rep(
    """        order_row: cute.Tensor,  # or None when seqlen_sorted=False
        stream,
    ):
        num_rows = input_data.shape[0]""",
    """        order_row: cute.Tensor,  # or None when seqlen_sorted=False
        phase_ts: cute.Tensor,  # [ptime] int64[num_rows, 8]
        stream,
    ):
        num_rows = input_data.shape[0]""",
)

# --- 19. __call__ kernel invocation ---
rep(
    """            output_indices,
            order_row,
        ).launch(""",
    """            output_indices,
            order_row,
            phase_ts,  # [ptime]
        ).launch(""",
)

for i, (old, new) in enumerate(EDITS):
    n = src.count(old)
    assert n == 1, f"edit {i}: expected 1 match, got {n}\n---\n{old[:200]}"
    src = src.replace(old, new)

F.write_text(src)
print(f"spliced {len(EDITS)} edits into {F}")
