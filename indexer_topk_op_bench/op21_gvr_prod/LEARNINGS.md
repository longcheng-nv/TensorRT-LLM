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
- (iter5) umb-b200-035 GPU0 has degraded cooling (79C idle, GPU1 31C): a
  17-cell nsys sweep drifted +2.3-2.7us (~+13%) on the LATER cells vs the
  iter3 baseline while early cells improved — progressive load throttling,
  sw_thermal_slowdown "Not Active" at idle. Unpaired nsys on GPU0 is now
  untrustworthy on this node; canonical sweeps go GPU=1 (verified GPU1
  reproduces iter3-era numbers within 1.5%). Paired same-process event A/B
  ratios survive throttle; cross-run absolute comparisons do not. ALWAYS
  eyeball per-cell deltas vs the prior iter's table before accepting a
  verdict — the uniform-shift-with-run-order signature is thermal, not code.
- (iter5) same-process A/B via env-var read per _compile call + key
  inclusion (OP21_P4_RS/OP21_QBINS): both variants coexist in the compile
  cache, so every cell pairs A and B back-to-back on identical GPU state —
  drift-immune where separate-process A/B is not.
- (iter5) P1b QBINS=64 at highBS FALSIFIED (event gm 1.004, 14 P1 cells,
  all within ±2.5%): the 256-bin suffix scan (8 double-barrier steps) is
  NOT the per-row highBS bottleneck. The P1-grid SGLang gap lives in the
  scan/ladder structure itself (rows-per-SM serialization), not in P1b.
- (iter6) measure the DATA before designing the fast path: the host probe
  (cnt(b*) p50=2 max=4 on 68 rows) turned "small-bin fast path" from a
  speculative branch zoo into ONE dominant path (warp0 register ranking,
  <=32 members) + a never-firing fallback. 20 minutes of host replay
  killed weeks of wrong-shape kernel work.
- (iter6) event-axis A/B can produce a REPRODUCIBLE per-cell lie when the
  binary changes: K512 131K BS1 showed fast-path 0.957 across 5 paired
  reps, but nsys showed dead flat (16.74 vs 16.70us). Same-process
  pairing removes thermal/run drift, NOT codegen jitter (different SASS
  register allocation shifts whole-kernel behavior per cell). Single-cell
  event verdicts don't gate a lever that wins 13/14 elsewhere — nsys
  arbitrates.
- (iter6) warp-exact selection beats a fine histogram when the group is
  tiny: 32 constant-src shuffle_sync compares (tie by stash order) give
  exact ranks with zero atomics and zero extra smem passes; positions are
  computed, not contended. Constant-src shuffles avoid any doubt about
  dynamic-lane support in the DSL.
- (iter6) smem scratch aliasing rule for hist reuse: after the coarse
  search, smem_hist is dead EXCEPT via cnt(b*) — read it into a register
  BEFORE stashing into smem_hist[8..39], then the alias with b_star in
  [8,40) is harmless because no pass re-reads the coarse hist.
- (iter7) INLINE phases are ablation blind spots: iter6's "P3" cost was
  the slot walk only — the leader DSMEM band gather sat inline in the
  kernel body, so the no-op-subclass harness never saw its 1.7-2.4us and
  it masqueraded as "scan floor". Keep every phase in an overridable
  method; if an ablation table says the floor is at the rival bar, first
  ask what code is NOT behind a method.
- (iter7) no-op'ing a phase that PUBLISHES counts must still publish
  deterministic zeros: a bare `pass` walk fed garbage p_cnt (bounded
  0xFFFF) to the downstream gather and distorts the probe. Decompose with
  increments where the consumer is already no-op'd (noGat -> noWG).
- (iter7) remote-STORE push beats remote-LOAD gather for cluster band
  collection: st.shared::cluster is fire-and-forget (no round-trip stall)
  and the destination offset (global band prefix b_off) is already known
  BEFORE the walk from the ladder-count publish — so pushing during the
  walk is placement-free and deletes the whole gather pass + one cluster
  barrier pair + the count publish. Event gm 1.077 (14/14), nsys P0 gm
  1.125 -> 1.249. Distinct from the red-lined "distribute the serial
  part": no new barriers — one FEWER.
- (iter7) the campaign's biggest single-iter gain (+11% gm) came from a
  phase that had never appeared in any ablation table (see blind-spot
  note above) — re-run the ablation split after every structural change;
  the ranking moves.
- (iter5) op8's exact rank-scatter P4 ports cleanly onto the band refine:
  band range [thr1, thr0) is already known (op8's min/max pass drops out),
  rank target k_rem is runtime (vs op8's const K), all positions shift by
  m0. s_iscalars[0..4] are free scratch inside P4 (terminal at every call
  site) — but smem_hist[2]/[3] as post-fine-search scratch must respect
  the op8 comment: NOT [0]/[1], the last fine warp's reverse scan reads
  bins down to 0.
