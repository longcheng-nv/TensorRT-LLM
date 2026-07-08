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
- (iter8) falsifications can be DTYPE-conditional: C8-at-holes was
  falsified at fp32 (iter6, +0.6-1.3% noise) but is a 1.08-1.14x WIN at
  bf16/fp16 — halving the scan bytes re-weights the serial tail, so the
  C-scaling calculus flips. Re-test C/cluster geometry per dtype tier
  before inheriting an fp32 verdict. The win region compressed into ONE
  rule: `N >= 32768*BS` (collapse at 262K BS16 and the 131K BS8 marginal
  both fall out naturally).
- (iter8) 16-bit speeds up the RIVAL more than us at C4: radix drops ~35%
  (L2-BW-bound, flat N-scaling) while our C4 largeN smallBS time is flat
  (18.85 -> 18.66us) — the fused scan is instruction-bound there, not
  L2-BW-bound. C8 reclaims 2.5us; the residual gap is per-element cost
  (16->32 cvt + fp32 ladder) => 16-bit native compares is the lever, not
  more SMs.
- (iter8) real_data_v2's per-dtype refs make the 16-bit real gate free to
  write: dtype-truncated captures + tie-robust value_metrics — 360 checks
  caught zero issues, confirming the fp32-validated band/P4 semantics are
  dtype-independent (kNumBins differences included).
- (iter9) threshold-quantization is the key that unlocks native 16-bit
  compares EXACTLY: quantize every P1b column to the dtype grid ONCE at
  emit (thr_q = f32(dtype(thr))) and the 16-bit-domain ladder compares
  become bit-equivalent to the fp32 compares all other phases perform on
  the exactly-embedded values — no per-phase changes, no tie-semantics
  change (band inclusion of the tie group at thr is identical either
  side of quantization). Microbench counts matched on every config;
  real gate 360/360.
- (iter9) packed-16x2 count accumulate (set.ge + add.rn) needs a flush
  cadence bound, not luck: per-half growth <= pairs/iter (8 at 256-bit),
  bf16 integers exact to 256 => flush every 16 iters costs ~nothing and
  is provably exact at any N. The collect column CANNOT use the packed
  accumulator (slot cursor must be exact per element) — use the
  set.ge.u32 mask + rare half-extract; keeping PC out of the packed loop
  also avoids double-booking its count.
- (iter9) microbench-first paid again: 3 candidate compare schemes
  ranked in 20 minutes of .cu work (p2 1.73x, p1 1.31x, both ~1.0 at
  full occupancy) BEFORE touching the DSL — and the occupancy result
  predicted exactly where the production win would and wouldn't show
  (largeN smallBS yes, highBS canaries no).
- (iter9) fp16 event-axis showed reproducible ~5% single-cell
  regressions that nsys refuted at ~1% — third instance of the
  codegen-jitter lie (iter6, iter8 boundary probes, now iter9);
  single-cell event verdicts never gate a multi-cell winner.
- (iter5) op8's exact rank-scatter P4 ports cleanly onto the band refine:
  band range [thr1, thr0) is already known (op8's min/max pass drops out),
  rank target k_rem is runtime (vs op8's const K), all positions shift by
  m0. s_iscalars[0..4] are free scratch inside P4 (terminal at every call
  site) — but smem_hist[2]/[3] as post-fine-search scratch must respect
  the op8 comment: NOT [0]/[1], the last fine warp's reverse scan reads
  bins down to 0.

## Iter 10 (ship review + upstream assessment) findings

- **P4 path-C fixed-depth rank-scatter is NOT unconditionally exact** —
  found by cross-reading upstream `ec04147502` (2026-07-01, reverts
  rank-scatter-P4-default on the PR #15709 branch after 30 adversarial
  test failures): a fixed-depth histogram cannot separate two distinct
  values in the same (sub-)bin, so the straddling bin can emit a value
  below the true K-th rank. op21's `_p4_band_fine_scatter` (path C,
  1024×256 fixed depth, stash-order emit of the deepest straddling bin)
  carries the SAME latent mode. Paths A/B are exact; C never fires on
  real/synth probes (cnt(b*) max=4) and op21's gates never tripped it
  because they adversarialize preIdx, not logits collision structure
  (>32 distinct fp32 values inside one fine bin straddling the cut).
  16-bit duplicates are ties → tie-order emit stays exact.
  **Port rule: any upstream default-ON needs path C replaced by an exact
  fallback (snap on the residual bin, or recurse-until-≤32 + register
  ranking), then the upstream 30-case adversarial suite re-run.** Also:
  adopt upstream's adversarial multi-bucket logits cases into op21's own
  smoke suite — current gates have a blind spot there.
- Upstream production surface for the port: kernel+runner+dsa.py wiring
  all in origin/main (#14602/#15198/#15304); `enable_heuristic_topk` is
  the e2e GVR toggle; rank-scatter P4 sits opt-in on
  `fork/feat/gvr-rank-scatter-p4`. op21's top-3 levers (push, P4 fast
  paths, fused ladder) form a dependency chain rooted in the P1b/ladder
  redesign ⇒ incremental patching captures only dispatch+16-bit; the
  1.14-1.29× vs production needs the kernel-variant route
  (UPSTREAM_ASSESSMENT.md Strategy B).

## Iter 11 findings

- **Falsification confirmed at HEAD scale**: the iter10-predicted path-C
  inexactness is real and total — 72/72 adversarial cases FAIL pre-fix
  (vdiff always ULP-scale: the wrong subset differs only inside one fine
  bin). Fixed-depth sub-histograms are unfixably inexact as a P4
  terminal: a bin is a value interval, not a tie group. Only DATA-value
  terminals are exact: warp register ranking (path B) or the value-edge
  snap (block_band_snap_iter steps onto actual values). RED LINE: never
  reintroduce a fixed-depth scatter emit as a P4 terminal.
- **Adversarial harnesses must speak the production preIdx dialect**:
  cr=1 raw pointers get the kernel's +1 diagonal offset => every gathered
  value lands in the bulk => no straddle => the fail-soft baseline path
  (exact) silently absorbs the attack and the gate false-passes. First
  test version did exactly this (54/54 "ok" on cr=1). Convention: K512/
  K1024 = cr4 offset-0 (V4), K2048 = cr1 caller passes prev-1 (V3.2).
  Bonus result: whole-array off-pointer preIdx is fail-soft-exact.
- **`nsys profile -c cudaProfilerApi` exits 143 on SUCCESS** — any
  `set -e` driver dies silently after the first cell. The op21 drivers
  (`drive_nsys_*.sh`) survive only because they pipe the profile through
  `| tail -1` (pipeline status = tail's 0) — new A/B scripts must not
  `set -e` around a bare nsys invocation.

## Iter 12 findings (upstream-port PR-1 step 2: kernel assembly)

- **GVR output row ORDER is run-to-run nondeterministic — equivalence
  gates must compare sorted index SETS, never positions.** The P3
  direct-write cursor (`atomicAdd(s_iscalars[4], ...)`) and the P4
  emission cursors (`bp_gt`/`bp_eq` smem atomics) allocate output slots
  by warp-scheduling order: the BENCH kernel itself returns a different
  within-row permutation on back-to-back identical calls (verified 4x,
  K512 N262144; sorted sets bit-stable every time). This is inherited
  vendored-GVR behavior, not an op21 artifact. A naive `torch.equal`
  old-vs-new A/B "fails" 7/7 with 82-478 mismatched slots while the
  selection is exactly identical.
- **Line-anchored extraction scripts must be authored against the real
  sources, not memory**: assemble_ms.py's first draft had 2 wrong cut
  bounds (one clipping a trailing comment line, one overlapping the
  next method) and 3 stale edit anchors (multi-line def signature,
  banner wording) — every one caught by the content asserts, none by
  AST parsing. The content-assert-everything pattern paid for itself;
  keep it for any future re-assembly.
- **The next_n/varlen contract was already complete in the bench kernel
  body** (inherited verbatim from the vendored production kernel:
  row//next_n mapping, cr=1 diagonal preIdxOffset, per-row
  actual_kv_len) — the UPSTREAM_ASSESSMENT "op21 lacks next_n" gap was
  a VALIDATION gap, not a code gap. Closed by port/run_gate6_nextn.py
  (12/12: next_n {2,4} x varlen x {K512/cr4, K1024/cr4, K2048/cr1} x
  {ms, C4}).
- **vendored-vs-upstream import topology**: upstream main has the #15198
  cluster primitives unified into gvr_topk_decode.py; the bench vendored
  tree keeps them in gvr_topk_decode_cluster.py. The PR artifact imports
  the upstream layout; local validation goes through port/portshim/
  (re-export shim), never by editing the artifact.

## Iter 13 (HLS log-falsi) — mechanisms
- **Ladder counts make the fallback bracket FREE**: done=2 arrives with
  count(s_thr[1]) tracked in s_msti[0] and count(s_thr[2]) still resident
  in the last round's s_mt_cnt (ms), or cluster-merged global counts (msc).
  The legacy entry + hi-end full-row passes were re-measuring known
  numbers — 2 of the 4-8 fallback passes were pure waste.
- **Falsi aim exponent is silicon-invariant (alpha 0.1 vs 0.2 identical)**:
  the accepted-count size does NOT move the P4 cost enough to matter at
  these shapes; do not spend more probes on alpha.
- **The K2048 light-fallback pocket (-3..-6%) is codegen, not algorithm**:
  fast-path cells never execute the new code, yet per-binary deltas of
  BOTH signs (+5.5% K1024, -5.4% K2048) appear and reproduce across
  datasets — same-binary register-allocation lottery (iter6 precedent),
  with a mild systematic negative at K2048's kC=6144 smem/register
  ceiling. Do not chase per-binary +-5% without a phase-level mechanism.
- **worst-scenario (all_ge-mode) misses gain nothing from falsi**: with no
  count<K column, the bracket hi end is unknown -> hi-end pass + expansion
  still run; only the entry-skip helps. A future lever would need a
  cheap hi-end estimate (NOT another full pass).
- **HLS proto accounting held on silicon** for the ladder-seeded band
  misses (its headline case): kernel refine converges in ~1 falsi pass.
  The proto's known blind spot (leader-bound msc recount) is untouched —
  Step 2 (cluster-parallel recount via the iter7 st.shared::cluster
  primitives) is where the remaining 116us-at-1M tail lives.

## Iter 14 (distributed fallback) — mechanisms
- **Slice passes + count merge work because counts are ADDITIVE across
  slices** — the only cross-CTA state is one Int32 per pass. The collect
  bypasses the vendored full-row prefix contract entirely (slice compact
  + push at global prefix): never half-use a contract you can't satisfy
  (iter2 lesson applied in reverse).
- **Cluster-barrier cost did NOT kill it** (unlike dist-P1/P4, iters 3/4):
  those distributed FIXED us-scale serial work; here each barrier pair
  buys (1-1/C) of a FULL-ROW pass. The red line is about the ratio of
  saved-work to barrier, not about distribution per se.
- **Code mass is a real currency**: ~300 lines of fallback code taxed the
  msc fast path a systematic ~4% (P0 spot 3/5 cells one-signed) even
  though the fast path never executes it. iter13's pocket was lottery
  (mixed signs); iter14's was systematic — the distinction is sign
  consistency across independent binaries. Mitigation = dispatch-gate the
  COMPILATION (fb_dist keys on n), not runtime branches.
- **worst-scenario all_ge misses are distributed-fallback's best case**:
  hi-end + expansion passes (which log-falsi could not remove) all become
  P/C. iter13+iter14 together cover both miss geometries.
