# op21 learnings

## Inherited physics & red lines (condensed; full list in PLAN.md)
- op14: ALL DSv4 decode rows are L2-resident on B200 ⇒ pass-count reduction
  saves zero DRAM; levers = SMs/row, #rounds, barrier chain, per-row fixed cost.
- count_ge_multi_bench: M=2 fused count ≈ free (×1.00-1.06); M=4 ×1.01-1.46
  (grows with N); M counters must be static registers (no difference-array);
  one barrier M-column reduce; don't cache M× per-thread counts for P3.
- op20 iter4: slot-collect fused into the count scan is validated exact + wins
  at large N (P3 → O(candidates)); overflow fallback mandatory.
- Exact fixed-256-bin rank-scatter is a validated exact band-refine primitive.
- Real preIdx hit-rate: Pro 0.69-0.77, Flash 0.36-0.46, synth 0.6 — gathered
  prev-K value at sorted rank i has current global rank ≈ i/h ⇒ order-stat
  placement must be robust to h, or use g_min (rank K) as the guaranteed
  c ≥ K lower anchor (every gathered value ≥ g_min, K distinct positions).

## Iter log-worthy findings
- (iter0.5) P1 order-stat seeding law: gathered prev-K value at sorted rank i
  has current global rank ≈ i/h. Placement fractions must span below the
  worst-case hit-rate (0.35 real). g_min (rank K) is a GUARANTEED c≥K lower
  anchor — kills the `all_lt` failure mode entirely (0/55 rows).
- (iter0.5) miss mode is exclusively `all_ge` (5.5%) — round-2 must search
  ABOVE thr(f_min), i.e. inside the gathered top quarter; interp on the two
  highest counts.
- (iter0.5) speculative collect column is a per-K constant, not per-N: pro
  f=0.75 fits kC 30/30; flash needs lower f (h and K both smaller).
- (iter1) NEVER leave a per-bin serial tid0 loop in a phase: 256-bin cum-scan
  on tid0 = ~1800 dependent smem ops = +30µs fixed (2x the whole op at small
  N). In-place Hillis-Steele suffix-scan (8 double-barrier steps) + stateless
  parallel crossing check costs <1µs. Same trap as the red-lined P1 self-loop.
- (iter1) same-file p3-vs-p5 A/B isolates a regression to the new phase in
  one probe — cheaper than nsys/ncu when the suspect list is short.
- (iter1) phase1 stash: smem_keys (P3 slot buffer) is dead until the ladder
  pass — stashing the K gathered values there kills P1b's L2 re-gather for
  free (no extra smem). Sentinel = NEG_FLT_MAX (< v_lo == min of valid).
- (iter1) fused collect needs kC >= ~4K headroom: at K2048 (kC/K=2.5) slot
  overflow makes fuse a 13% LOSS at largeN; bigger kC is smem-capped (232KB).
  Production gate: `bs <= NUM_SMS and 4*K <= kC`.
- (iter1) event-vs-nsys axis gap is large at small cells (radix_cutedsl
  4096/64: 12.1µs event-graph vs 6.98µs nsys) — ratios vs rivals MUST come
  from the nsys axis (op20 red-card lesson re-confirmed).
- (iter2) vendored phase3_collect_candidates does NOT recount: it prefixes
  over smem_ptcnt (per-thread ge-counts at s_thr[0], same [0,N) striding).
  Any caller that changed threshold or scan domain MUST re-run
  block_count_ge first, or outputs get silent holes.
- (iter2) s_iscalars[1..3] belong to the vendored P3/P4 (done/cnt_lo/
  cnt_hi) — never borrow them as scratch across phase calls.
- (iter2) replicated seeding is the cheap multi-CTA trick: P1 gather + P1b
  are per-CTA deterministic on identical inputs, so C CTAs compute
  bit-identical thresholds with ZERO cross-CTA traffic; only the M counts
  need one DSMEM merge.
- (iter2) C=8 collapses at BS16 (128 CTAs oversubscribe: 43.8 vs C4 28.5µs)
  while gaining <=5% at BS1 — C=4 single-rule dispatch beats a two-tier
  rule.
- (iter2) real-capture gate catches what synth misses: pair=(0,1) fallback
  (h<0.5 at midsize N) never fired on synth h=0.6 65K+ cells but fires on
  pro L4 N=14.5K. Keep real captures in EVERY iteration's gate.
- (iter3) REPLICATED work across cluster CTAs on the SAME addresses is
  nearly free (first CTA misses, rest hit L2) — do NOT "optimize" it into
  a distributed version paying cluster barriers. Dist-P1 measured +0.6-1.7us
  worse everywhere. Cluster barriers are the expensive resource, not
  redundant L2 reads. (Same family as op14's "pass-count reduction saves
  zero DRAM" — L2 makes redundancy cheap.)
- (iter3) C8-vs-C4 is K-dependent, not just N/BS-dependent: only K2048's
  bigger K-proportional tail amortizes 8-way chunking. K is a compile-time
  dispatch key, so per-K C rules are production-legal.
- (iter4) no-op subclass overrides (@cute.jit methods overridden to pass)
  are a 20-minute phase-ablation harness — use BEFORE designing any
  fixed-cost optimization. They pinned P4 = 3.9-7us in one probe.
- (iter4) GENERALIZED red line from iter3+iter4: on this kernel family,
  "distribute the serial fixed part across the cluster" LOSES — the
  distributed version pays cluster barriers + replicated 256-bin scans
  that cost as much as the serial work saved, and the serial phase's own
  fixed machinery (snap setup) survives on the boundary anyway. The
  productive direction is making the serial phase CHEAPER (op8's
  rank-scatter-exact P4), not spreading it.
- (iter4) phase-cost breakdown at P0 BS1 (ablation): P4 3.9-7us >> P3 2-3us
  >> everything else is the scan+P1 floor (~20us incl. ~11us N-term at
  262K/C4). Rival floor is ~19-20us — the winnable margin is ALL in P4/P3.
