# P4 pipeline decomposition campaign (§9e follow-up)

Goal (user, 2026-07-22): starting from REPORT §9e (865-cell phase breakdown,
P4 select dominant 827/865, med 44%), decompose the P4 phase's internal
pipeline per case and account for the instruction-level cost inside P4.

Host: umbriel-b200-093 (8x B200 idle). Env: nv26.05 container + machine-local
cutlass 4.5.0 overlay (/tmp/gvrlayers), PYTHONNOUSERSITE=1, env -u tokens on
all nsys/ncu.

## Experiment A — sub-P4 clock64 pipeline breakdown (full 865 cells)

Timed twin `gvrpkgp4t_head` = §9e's `gvrpkgtimed_head` + `splice_p4sub.py`
(11 exact-anchor edits, markers `[p4sub]`): phase_ts int64[1,16], stamps
t0..t7 (unchanged) + s8..s14 splitting P4 select at its internal barrier
boundaries (phase4_rank_scatter path — the active path at the PR head):

| stage | window | code |
|---|---|---|
| p4_peer_wait | t5->s8 | cluster handoff #2 arrive+wait (leader waits for slowest peer collect); cs=1 zero-width |
| p4_dsmem_gather | s8->s9 | leader's DSMEM (mapa+ld.shared::cluster) gather of peer candidates; cs=1 zero-width |
| p4_minmax | s9->s10 | candidate min/max block reduce (P4 scan #1); degenerate cand==K/<K copy-out collapses here |
| p4_coarse_hist | s10->s11 | coarse kNumBins hist zero+build (scan #2, ATOMS) |
| p4_coarse_search | s11->s12 | 3-step high->low bin search (warp sums + t0 + target-warp walk) |
| p4_fine | s12->s13 | fine 256-bin re-zero+build+search (scan #3) |
| p4_scatter | s13->s14 | classify+scatter writeback pass (scan #4) |
| p4_tail | s14->t6 | output pad + p4_exact_tail / p4tt tie repair |

Methodology otherwise identical to §9e: 2 arms (pristine prod nsys anchor +
timed twin), 10 warmup + 20 cold-L2 launches, per-phase median cycles,
abs us = frac x prod nsys; per-cell gates = exact both arms, monotone chain
t0..t5,s8..s14,t6,t7, nsys overhead gate, PLUS p4_select-frac drift vs §9e
csv (stamp-tax check).

- Smoke 6/6 (cs1-small/cs1-mid/cs4/cs8/v32-K2048/degenerate): exact+mono;
  cs=1 top sub-stage = p4_fine (~15% of kernel), cs>=4 = p4_dsmem_gather
  (11-19%); degenerate pro_4k P4=4% collapse works.
- Full 865-cell 8-GPU sharded run launched 2026-07-22 ~02:20 (tag `full`,
  drive_p4pipe_shards.sh, logs shard_full_g*.log).

## Experiment B — NCU instruction accounting per case class (after A)

NEVER run while the grid sweep is on the GPUs (double-driver contamination).
`drive_ncu_p4.sh`: ncu --set full on ~11 representative cells (one per rung
x case class incl. degenerate + v32 tie-dense + PR-loss class), profile
launch #13 (skip 2 correctness + 10 warmup). Attribution trick: the twin's
15 executed clock64 stamps are CS2R landmarks in the SASS stream;
`parse_ncu_p4.py` segments the `--page source --csv` export between
consecutive executed CS2R rows and buckets inst_executed + opcode classes
(smem_load/store, smem_atomic, gmem, cluster_mapa, barrier, fp, int,
move_misc, control) + every PC-sampling column per phase / P4 sub-stage.
Validation: (a) executed-stamp count == expected (15), (b) segment SASS
order == program order (assert label sequence sane), (c) cross-check
segment inst_executed shares vs Experiment-A cycle shares per cell.

## Files

- splice_p4sub.py, gvrpkgp4t_head/ (spliced twin)
- measure_p4pipe_full.py, drive_p4pipe_shards.sh, aggregate_p4pipe.py
- ncu_p4_one.py, drive_ncu_p4.sh, parse_ncu_p4.py
- outputs: p4pipe_full_g*.json(l), p4pipe_full.csv, nsys_reps/ (NOT in git),
  ncu_reps/ (NOT in git)
