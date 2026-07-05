# op21 — production-grade GVR redesign (omni-kernel campaign)

**Goal**: a GVR operator that beats ALL report.html ops (radix_cutedsl auto,
radix single/multi CUDA, SGLang StreamingTopK) on the composite metric, under
PRODUCTION constraints: low-dim dispatch (radix-op style 2-3 rules, no 240-key
tables), CUDA-graph compatible (host dispatch keys on buffer/max-N + BS only),
exact on REAL captures (not just synth), fail-soft under distribution shift.

**Priority weighting (user, 2026-07-05)**:
dtype/K: fp32·K1024 > fp32·K512/K2048 > bf16/fp16·all.
regime: smallBS·largeN > largeBS·smallN > {largeBS·largeN, smallBS·smallN}.
⇒ P0 grid = fp32 K1024 × N{65K,131K,262K} × BS{1,4,8,16}; P1 grid = fp32
K1024/512/2048 × N{4K..16K} × BS{64,256,1024}; verdict = nsys pure-kernel
cold-L2 vs per-cell best rival from report CSVs (B200).

## Where the bar is (nsys axis, from 2026-07-05 anchored analysis)
- P0 cells: gvr_x 0.87-0.99 vs radix_cutedsl(-multi); e.g. K1024 262K BS1/4:
  22.3-23.5µs vs 20.1-20.4µs (need −10~15%); K2048 262K BS1/4 0.79-0.81.
- P1 cells: SGLang bar 0.86-0.99 at midN highBS; radix bar 0.63-0.73 at N4-8K
  (smallBS·smallN deprioritized — structural wall, attack last).
- GVR already wins largeN midBS/highBS (gm 1.16-1.48): DO NOT REGRESS.

## Physics (corrected by op14 — read before proposing)
- ALL DSv4 decode rows fit L2 (N≤262144 fp32 = 1MB ≪ 126.5MB B200 L2). Pass-
  count reduction saves ZERO DRAM (ncu dram__bytes_read = 1× input for base).
  Real levers: (a) SMs-per-row (aggregate L2 BW), (b) #serial rounds (each = a
  full-row L2-limited scan), (c) barrier/merge chain length, (d) per-row fixed
  cost at high BS.
- count_ge_multi (count_ge_multi_bench, B200 nsys): M=2 costs ×1.00-1.06 of
  M=1 scan; M=4 ×1.01(4K)-1.46(256K) fp32. M-ary round splits bracket (M+1)-
  ways. M=1 secant with good seed ≈1.0-2.1 rounds today (K1024 1.3-2.1@N≥32K).
- Pro preIdx hit-rate 0.69-0.77 (real 32k/64k) ⇒ prev-K order statistics are
  high-quality bracket seeds. Flash 0.36-0.46 ⇒ design must fail-soft (extra
  round), never fail-wrong.

## The op21 algorithm (deltas vs op19/op20 sandwich)
1. **P1 order-statistics seeding (replaces offline straddle-fracs)**: gather
   prev-K values via preIdx (K loads, already done), then in-smem select order
   stats at ranks {K−δ, K+δ~} (δ≈K/16..K/4) → M=2 (or M=4) DATA-DRIVEN bracket
   thresholds. Distribution-free ⇒ production-robust; kills the offline-table
   dependency. (Non-convergent/garbage preIdx ⇒ falls back to pmin/pmean seeds.)
2. **P2 fused M-threshold bracket round**: ONE count_ge_multi<M=2/4> scan
   (sorted thr registers, M predicated adds, single barrier M-column reduce).
   Straddle hit (cnt_hi < K ≤ cnt_lo with cnt_lo−cnt_hi ≤ band_cap) in round 1
   at high probability; else 1 more M-ary round with secant/interp placement
   inside the surviving sub-bracket. Exactness authority unchanged.
3. **P3 fused slot-collect at thr_lo** (op20 iter4, validated): during the SAME
   scan, append v ≥ thr_loosest to per-thread smem slots ⇒ no separate collect
   pass in the happy path. Overflow ⇒ classic P3 fallback.
4. **P4 tiny-band exact refine**: defer-direct write ≥thr_hi winners; select
   (K−cnt_hi)-th among band (target band ≪ K via tight δ). Use fixed-256-bin
   exact rank-scatter (falsification history: validated exact, P4 1.11-2.12×)
   or in-smem bitonic when band ≤ 1024.
5. **Multi-CTA scale-out for P0 (smallBS·largeN)**: C CTAs chunk the row;
   count merge via GMEM-atomic 1-int per threshold (radix-style; falsification
   history flags this as the OPEN lever — clusters are GPC-capped, mcC16
   regressed via DSM merge). Slots per-CTA; direct-write via one prefix round.
6. **Fine-grain**: double-buffer/TMA on the streaming scan (flagged unmeasured
   in count_ge_multi report; cp.async was null on OLD CUDA kernel — must A/B
   fresh, drop fast if null); 16-bit native compares (skip fp32 cvt); guard
   smem_ptcnt STS when P3-reuse off (existing TODO); CTA width per N.

## Red lines (inherited — do NOT retry)
op14: pass-count/compaction moot (L2-resident); NO warp-collectives in the
per-element streaming loop; NO per-survivor scratch in gmem hot path.
op15/op20-iter6: smem-resident row staging (3× falsified, warm-L2 no-op).
op16: band-shrink inside secant, P2 sampling.
op12/op13: P4-internal reseeding (fine-hist, interp-seed); rank-scatter naive
(non-exact) version.
op9: complex per-row dispatchers lose to simple rules on mixed batches.
op20: level-2 sub-histogram; BS16 cluster fusion (bs×P over-extension);
fusP8T4 (>16-CTA cluster cap); mc at small N.
gvr_phase_timing: P1 self-loop model seeds; Opt-L per-element online slot
reserve (ballot chain); cluster DSM at high BS; L2-persistence/thread-coarsen.

## Gates (every iteration)
- exact_all synth ×3 seeds (all P0/P1 cells) AND **real-capture exactness**
  (pro 30 layers K1024 + flash 21 layers K512 + v32 K2048, via real_data_v2;
  tie-tolerant check; this is the op18/19 red-card lesson — synth exact ≠ real
  exact).
- nsys pure-kernel cold-L2 canonical from day 1 (NEVER event-only — op20's
  event bookkeeping hid 0.63-0.73 holes as 0.78-0.88). Warm-L2 A/B as cheap
  falsifier for any memory-tier claim.
- No-regression guard on GVR's winning regimes (largeN midBS/highBS).
- Production checklist per ship candidate: dispatch expressible in ≤3 rules on
  (dtype, K, cr, BS, max-N); no per-exact-N keys; graph-capture identity fixed.

## Iteration roadmap
- iter0: harness (nsys 3-way: op21 vs radix_cutedsl vs gvr_x ref) + P0 baseline.
- iter0.5 (HOST PROTOTYPE, before any kernel): on REAL pro/flash/v32 captures +
  synth: measure (a) P1 order-stat bracket straddle rate at M=2/4 vs δ,
  (b) band size distribution vs δ, (c) expected rounds. Go/no-go for design
  points 1-2. (op13 host-replay methodology.)
- iter1: single-CTA kernel v1 (P1 order-stats + M=2 fused round + slot collect
  + band refine), P1-priority cells first (chain-length win).
- iter2: multi-CTA C-chunk GMEM-atomic version for P0 cells.
- iter3: double-buffer/TMA A/B; CTA-width tune.
- iter4: K512/K2048 port; 16-bit native-compare port.
- iter5: dispatch distillation (≤3 rules), real-data exactness suite, no-regress
  full grid, B300 cross-check.

## Branch / bucket
Branch `omni/op21-gvr-prod`; bucket `indexer_topk_op_bench/op21_gvr_prod/`;
commit per iter `[op21 iter N]`. GPUs: umbriel-b200-035, 2× B200 idle.
