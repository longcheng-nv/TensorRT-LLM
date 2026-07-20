# op37 — dist_P4 v2: sync-reduction design sketch (pre-silicon)

Baseline = op36 A2 `phase4_rank_scatter_dist` (TRACKA2_DESIGN.md): exact,
wins N≈262K (BS1 1.19, BS256 1.41), loses N=65-131K. Attribution: 6 cluster
arrive/wait rounds vs the distributed O(cand) work; boundary ≈160K
candidates-scale. Goal: pull the win boundary down to N≈65K so dist_P4
covers the whole BS≤8 loss region at cs≥4.

## A2 sync ledger (per P4 invocation)

S1 entry: per-CTA (cmin,cmax) publish + capped-prefix        [arrive+wait]
S2 coarse hist red_add into leader                            [arrive+wait]
S3 leader coarse search -> publish b_star/rank_above          [arrive+wait]
S4 fine hist (256 sub-bins of b_star) red_add into leader     [arrive+wait]
S5 leader fine search -> publish sb_star/ra_fine              [arrive+wait]
S6 scatter counters + gmem writes                             [arrive+wait]
(+ final kernel-exit cluster barrier, pre-existing)

## v2 levers, in order of expected value/risk

1. **S1 elimination — fold into the P3-end handoff.** P3 already ends with a
   cluster handoff (peers publish per-CTA candidate counts; s_iscalars[5]
   stable). Move the (cmin,cmax) block-reduction + f32-bit publish to the
   tail of P3 (before its existing arrive), and the capped-prefix mapa scan
   to after its existing wait. Saves 1 full round. Risk: LOW (pure motion,
   same slots; verify slot liveness — s_dp4 written pre-arrive, read
   post-wait).

2. **S5+S6 merge — publish fine result inside the scatter round.** After S4,
   the leader's fine search is ~100 cycles of scalar work; peers' next need
   is sb_star/ra_fine. Replace S5's full round with: leader does fine search
   immediately after its own S4 wait, publishes, then ONE round (arrive+wait)
   releases peers straight into scatter. This is just re-phrasing S4/S5 as
   one round if peers do nothing between them — A2 already has peers idle
   there. Net: 6 rounds -> 4. Risk: LOW-MED (leader does search between its
   wait and its arrive; peers' wait covers both).

3. **Single-level wide hist — kill S4/S5 entirely.** Candidates span
   [cmin,cmax] (narrow, near-threshold band). Use one histogram of
   kFineBins=2048 (or 4096) bins over that band (SMEM budget: 2048×4B = 8KB,
   check headroom vs existing smem_hist kNumBins≥512 — may REUSE smem_keys
   space, dead post-stage?). Rounds: hist red_add (1) + leader search/publish
   (1, merged per lever 2) + scatter (1) + exit = 3. Boundary ties within one
   bin -> existing p4_exact_tail (ambiguity-gated radix, with p4tt tiny-tie
   fast path at current head) — fires more often than 2-level (resolution
   2048 vs 512×256); on real data boundary gaps are ~1e-5 of range, so
   single-level 2048 bins ≈ range/2048 per bin -> tail fires frequently at
   K2048? NEEDS a host-side fire-rate estimate on the 25 real cells BEFORE
   implementing (cheap: replay P4 binning in torch). Risk: MED.

4. **Spin-flag publish instead of full cluster round for S3/S5** (leader
   st.release to a DSMEM flag; peers poll mapa ld.acquire). Saves the
   all-CTA arrive cost for leader-only produce steps. Risk: HIGH (hand-rolled
   sync; the file convention is arrive/wait only; falsification history says
   cluster sync bugs are silent). Only if 1-3 insufficient.

## Decision protocol

- Gate 0: agent splice of A2 verbatim onto current head (gvrpkg37) — battery
  exact. Then nsys A/B on the BS≤8 loss cells (arm gvr_dp4 in ops_op37):
  reproduce the A2 shape at current head (expect: win ≥262K, lose 65-131K).
- Iterate levers 1→2 (→3 if fire-rate OK) as gvrpkg37 flags, nsys A/B same
  cells each time, ≤2-way, paired same-GPU. Ship rule: dist_p4_v2 ≥ pr on
  every routed cell (zero-regression), win region must extend to N=65538
  rung BS≤8 or the lever chain stops.
- Exactness battery re-run per lever (battery_dp4.py) + real-cell spot.
