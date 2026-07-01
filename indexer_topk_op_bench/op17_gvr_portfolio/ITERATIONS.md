# op17 GVR threshold-portfolio — iteration log

**Goal (user):** starting from GVR (cuteDSL), on the report's exact synth data, lift
per-(seqlen,BS) perf by +40% avg with NO regression, keeping the GVR structure
(preIdx → secant → refine). Sanctioned new lever (user msg 2026-07-01): **multi-CTA
threshold portfolio** — launch G = NUM_SMS/num_rows CTAs, seed each with a different
initial threshold across [min,mean]∪[mean,max], each runs P2 secant ~2 iters, pick the
CTA whose cand-count is closest to K as leader, leader does P3→P4, others early-exit.

HW: B200 sm_100, NUM_SMS=148. Protocol: cold-L2 (512MB evict) + CUDA-graph + cudaEvent
median, matching harness/sweep.py (nsys-validate positive claims per repo rule).

Honest envelope (REVISED at iter 0 after user correction 2026-07-01):
- The portfolio has TWO distinct levers, hitting opposite ends of the N range:
  1. **P2-collapse** (large N, K1024/2048): parallel sweep pins the threshold in ~1 pass
     instead of 2-2.65 secant iters. Attacks the P2 42-54% share at N≥131K.
  2. **P4-shrink via free tight threshold** (small N, P4-dominated ~55%): because the
     148-threshold parallel sweep gives an EXACT count at each threshold for free, the
     leader can pick a threshold whose cand-count ≈ K+ε instead of the baseline's
     kFTarget-driven ~3.2-3.96×K over-collect → P4 histogram+snap working set (∝ cand)
     shrinks ~3×. **This attacks the small-N regime I wrongly wrote off.** Distinct from
     op13: op13 tightened the threshold via the secant and PAID a P2-eval tax that capped
     it at ~10-15%; the portfolio gets the tight threshold with NO extra P2 pass.
- Still NO headroom at high BS (SMs/row→1, no spare CTAs → degenerate to single-CTA).
- Net: gain may be more UNIFORM across N than first thought (small-N via P4-shrink,
  large-N via P2-collapse). Whether it reaches +40% is empirical and depends on:
  (a) how much of P4 is cand-linear vs fixed (kNumBins histogram + K writeback), and
  (b) the cross-CTA leader-selection sync cost (~1-3µs at BS=1) not eating the P4 saving.
  Both measured in iter 1 before any kernel rewrite. Ship as exact, no-regression dispatch.

---

## Iter 0 — 2026-07-01 — CRUX: is the redundant-read portfolio free at BS=1?

**What:** `scripts/crux_bandwidth.py` — Triton, grid=(G,), each program = 1 CTA (1024
threads, num_warps=32) grid-strides the full row doing count_ge. Cold-L2 median µs for
G ∈ {1,2,4,8,16,32,64,128,148} × N ∈ {8K..262K}.

**Result (fp32, cold-L2 µs):**
| N | G=1 | G=8 | G=16 | G=64 | G=148 | G* (≤1.5×base) |
|---|---|---|---|---|---|---|
| 8192 | 8.5 | 8.6 | 8.5 | 8.6 | 8.7 | 148 |
| 65536 | 14.0 | 14.1 | 12.8 | 12.6 | 13.3 | 148 |
| 131072 | 22.4 | 22.2 | 20.4 | 20.4 | 20.4 | 148 |
| 262144 | 37.7 | 37.3 | 34.7 | 34.6 | 34.8 | 148 |

**Verdict: PREMISE CONFIRMED, G*=148 at every N.** 148 independent full-N count_ge scans
cost the same wall-time as 1 (slightly LESS at large N — more CTAs hide latency). A single
1024-thread CTA at 262K sustains only ~27 GB/s ≈ 0.35% of B200 peak ⇒ ~148× redundancy
headroom. So a 148-threshold parallel sweep is ~free in wall-clock.

**Critical caveat (mechanism = L2 reuse, not HBM headroom):** every tested row ≤262K fp32
= ≤1MB ≪ ~50MB L2. The redundant reads are served from L2 (first-touch → L2 → reused), not
from spare HBM bandwidth. This is the op14/op15 "L2 trap": the same L2 residency that makes
the portfolio cheap ALSO makes the baseline's extra P2 passes cheap (they're L2 hits too).
⇒ collapsing P2 iters saves L2-cheap passes, not HBM passes → gain is capped, consistent
with op13's measured ~10-15% (P3 stays full-N-read-bound). NOT evidence of a 40% path.

**Next (iter 1) — REVISED per user P4 insight:** the decisive unknown is now the
**P4(cand_count) curve**. Measure, on the real cuteDSL GVR kernel, how P4 phase-µs varies
as cand_count is driven from baseline (~3.2×K) down toward ~1.1×K (sweep kFTarget/kC as
op13 did, reuse measure_cute_phases.py). This bounds the small-N lever directly:
  - If P4 is mostly cand-linear → tight threshold gives a large small-N win (user is right,
    small-N headroom is real, possibly the biggest single lever).
  - If P4 is mostly fixed (kNumBins-1024 histogram + K writeback + snap setup) → shrinking
    cand barely helps and the small-N regime stays capped.
Then estimate the cross-CTA leader-selection sync cost separately (cluster DSMEM reduction
over 148 scalars). Only if P4-shrink net-positive after sync → proceed to full kernel.

---

## Iter 1 / 1b — 2026-07-01 — P4(cand_count) decoupled from the P2 tax

**iter1** (`scripts/iter1_cand_sweep.py`): shrink acceptance ceiling kC via the SERIAL
secant (GvrP2C), K512 fp32, end-to-end cold-L2. Net best/base = 0.99/0.98/0.95/1.00× at
N=4K/8K/16K/65K — i.e. ~0% and a BLOWUP at 65K (24→33µs). But this pays the P2 tax the
portfolio avoids → it measures (P4-shrink − P2-tax), not the lever itself.

**iter1b** (`scripts/iter1b_p4_vs_cand.py`): RAW clock64 cycles, kC override, K512 fp32,
decoupled. Median cycles:
| N | P4 loose (kC5120) | P4 tight (kC768) | P4 shrink | P2 tax |
|---|---|---|---|---|
| 4096 | 10086 | 8394 | 0.83× | 2.06× |
| 8192 | 10292 | 7026 | 0.68× | 1.83× |
| 16384 | 18428 | 7798 | 0.42× | 2.52× |

**VERDICT: user is right — P4 is cand-linear, shrink grows with N (−58% @16K).** Fixed P4
floor ≈ 7500 cyc (kNumBins-1024 histogram + K writeback); the cand-proportional part above
it is what tight threshold removes. The serial P2 tax (~2×, +1 full-N pass) is exactly what
cancels this net (why op13/iter1 saw ~0%). **The portfolio's entire value = capture the P4
shrink WITHOUT the P2 tax.**

Estimated portfolio net ≈ (P4_loose − P4_tight) − leader_sync. At N=16384, P4 saving ≈ 7µs
of a 20.7µs kernel (~34%) IF leader_sync is cheap. So the whole idea now hinges on ONE
unmeasured term: the cross-CTA leader-selection sync at BS=1 (148 CTAs → pick leader).

**Next (iter 2):** measure leader-selection sync cost. 148 CTAs > cluster max (~16), so
cross-CTA over all SMs needs grid.sync() (cooperative launch) or a 2-kernel split
(sweep→global[G]; tail-kernel picks leader + P3/P4). Microbench both; net win exists only
if sync ≲ 5µs at small N. Also re-check large N (P2-collapse lever) separately.

---

## Iter 2 / 2b / 3 — 2026-07-01 — leader-sync cost + projected NET envelope

**iter2** (`scripts/iter2_sync_cost.py`): leader-selection sync (2-kernel proxy, conservative
upper bound). Overhead t_sweep+B − t_sweep = ~0µs (N≤8K), 1.95µs (16K), 3.42µs (65K), ~2µs
(131K/262K). Cheap. Also re-confirms t_sweep(G=148) ≈ t_base(G=1) (crux).

**iter2b** (`scripts/iter2b_projection.py`): K512 fp32 P4-shrink net (all terms measured):
| N | base µs | P4 µs | P4 shrink | sync | proj speedup |
|---|---|---|---|---|---|
| 4096 | 16.4 | 9.55 | 0.83 | 0.0 | 1.11× |
| 8192 | 16.4 | 9.40 | 0.69 | 0.0 | 1.22× |
| 16384 | 18.4 | 10.34 | 0.44 | 1.95 | 1.26× |
| 65536 | 26.6 | 10.47 | 0.67 | 3.42 | 1.00× (dispatch to baseline) |

**iter3** (`scripts/iter3_p2collapse.py`): large-N P2-collapse (sweep pins threshold in 1
pass vs baseline's P2_iters). fp32:
| K | N | proj speedup |
|---|---|---|
| 512 | 65K-262K | 0.89-0.96× (P2 already 1 iter → sync only cost → LOSES, dispatch to baseline) |
| 1024 | 131K / 262K | 1.27× / 1.17× |
| 2048 | 65K / 131K / 262K | 1.08× / 1.29× / 1.42× |

**Projected combined envelope (fp32, dispatched, no-regression via baseline fallback):**
- small/mid-N K512/1024: P4-shrink ~1.1-1.26×
- large-N K1024/2048: P2-collapse ~1.17-1.42×
- K512 large-N, K1024 65K, high BS: neutral (baseline)
- Peak ~1.42× (K2048/262K). Average across all cells ~1.10-1.20×. **NOT +40% universal.**

**CRITICAL STRATEGIC CAVEATS (must weigh before building the real kernel):**
1. **Large-N P2-collapse is DOMINATED by the existing PR#15198 cluster.** Report by-seqlen:
   GVR-multiCTA(cuteDSL) cluster is ~1.6×/2.2× vs GVR-CUDA at N=131K/262K BS=1 — far beyond
   the portfolio's projected 1.42× over single-CTA-cuteDSL. So the portfolio's large-N half
   does NOT beat what's already shipped; the cluster (cooperative scan) already exploits the
   same idle bandwidth better for the pure-scan-bound large-N regime.
2. **P2-collapse projection is OPTIMISTIC** (assumes 148 fixed thresholds pin as tightly as
   the secant's 2.65 iters in 1 pass). If band placement is coarse, real needs ~1.5 passes →
   gains ~halve. P4-shrink (small-N) numbers are solid (don't depend on sweep resolution).
3. ⇒ The portfolio's UNIQUE, defensible value narrows to **small/mid-N P4-shrink (~1.1-1.26×,
   K512/1024 fp32)** — a regime the cluster degenerates in, but which **op13 already ships**
   (~1.11-1.17×). Marginal incremental gain over op13, same regime.

**DECISION POINT (analysis phase complete):** the full cooperative multi-CTA portfolio kernel
is a large build (grid reduction + leader handoff + preIdx-band threshold placement + exact
tail P3/P4 + fallback). Its projected unique win (~1.1-1.26× small/mid-N K512/1024) modestly
beats op13 in op13's own regime; its large-N win is dominated by the shipped cluster; it does
not reach +40%. Recommend: confirm with user whether this envelope justifies the build, or
whether to fold the one clean delta (tax-free tight-cand P4) into op13 instead.

---

## Iter 4 — 2026-07-01 — single-CTA M-way P2: EXACT but net ≤1.0× (M-way ALU NOT free)

Built `src/gvr_portfolio_op.py`: GvrPortfolioKernel subclass, P2 replaced by one M-way
(M=16) multi-threshold count over band [pmean,pmax], pick tightest count∈[K,kC], secant
fallback. Compiles; **EXACT (vdiff=0, uniq=K) on all fp32 K512/1024/2048 cells.**

**A/B (`scripts/iter4_ab.py`, cold-L2, fp32) — SLOWER everywhere:**
| K | N=4K | 8K | 16K | 32K | 65K | 262K |
|---|---|---|---|---|---|---|
| 512 | 0.88× | 0.90× | 1.00× | 0.71× | 0.59× | 0.45× |
| 1024 | 0.88× | 0.77× | 0.82× | 0.72× | 0.41× | 0.49× |
| 2048 | 0.84× | 0.79× | 0.63× | — | 0.58× | 0.44× |

**ROOT CAUSE — the "M-way compare is free" assumption is FALSE for a single CTA.** The
crux's free-ness was bandwidth headroom across 148 CTAs each doing ONE compare. A single
CTA is memory-LATENCY-bound (few outstanding loads at 1/148 BW), and 16 compares × vec_w=4
= 64 ALU ops per 128-bit load exceed the latency-hiding slack → P2 becomes ~M×-ALU-bound.
The falsified "Opt-F 2-way wash" was free because 2 compares fit the slack; 16 do not.
Overhead grows with N (P2 fraction grows) → 0.45× at 262K. At best small/mid-N cell (16K)
M-way overhead ≈ P4 saving → net 1.00×. The iter2b projection assumed P2 stayed at baseline
cost; it does not.

**PIVOT:** the tight threshold must be obtained the way the crux proved free — **multi-CTA
redundant portfolio** (G CTAs, each ONE normal count_ge at its own threshold, memory-bound,
~1 pass wall-time per crux) + leader-select (~2µs, iter2) + tail. That is the user's original
proposal; my single-CTA M-way collapse was a wrong "optimization" that moved cost into
unhidden ALU. Next (iter5): build the multi-CTA redundant-threshold kernel.

**M-sweep {4,8} (small/mid N, fp32):** narrow single-CTA sweet spot only —
K512 N16K M=4 → 1.20× (M=8 1.17×), exact; K512 4K/8K ~1.00-1.01×; K1024 all 0.80-0.95×.
So single-CTA M-way's only real win is K512 mid-N with small M; not broad. Multi-CTA needed
to get the tight threshold FREE (removes the M-way ALU tax) and extend the win to K1024.

---

## Iter 5 — 2026-07-01 — multi-CTA redundant-threshold portfolio (BUILD, user chose A1)

Design (2 kernels, minimizes inter-kernel gap vs 3-kernel):
- **Kernel A (Triton, grid=G=148):** each program computes band=[pmean,pmax] from preIdx
  logits (redundant, cheap/L2), then ONE normal count_ge over full N at thr[pid]=band_lo+
  pid*(band_hi-band_lo)/(G-1) → counts[G]; pid0 writes band[2]. Free per crux (memory-bound,
  1 compare/elem, 148 CTAs = ~1 pass wall-time). This is the tight-threshold search with NO
  single-CTA ALU tax (the iter4 failure) and NO serial-secant tax (the op13 failure).
- **Kernel B (cuteDSL, copied gvr_topk_kernel body + counts_g[G]/band_g[2] params):** P1 kept
  (sets val_lo/val_hi for P3); phase2_seeded: tid0 walks counts high→low, picks tightest
  count≥K → thr*=band_lo+m*(band_hi-band_lo)/(G-1), s_thr[0]=thr*, done=1 (ZERO count passes);
  P3 collects tight cand at thr*; P4 snaps. Exact (count≥K guaranteed; P3 retry-shrink covers
  count>kC edge).
Expected: iter2b projection (~1.11-1.26× K512 small/mid N) MINUS the ~2µs 2-kernel gap →
net-positive mainly at mid N (16K-32K where P4 saving > gap); very small N may be gap-limited.
Cluster fusion (M-way in cooperative cluster scan, ONE kernel, no gap) = iter6 if iter5 shows
the gap is the limiter.

## Iter 5 result — 2-kernel multi-CTA: EXACT but net-negative (0.63-0.84×)

Fixed (band=[pmin,pmax] since count(pmin)≥K⟺v_K≥pmin; P3 only recounts/populates smem_ptcnt
when done≠1, so seeded P2 sets done=2 to force the recount). EXACT all K512/1024/2048 fp32.
A/B (`scripts/iter5_ab.py`, cold-L2): **0.63-0.84× everywhere** (K512 4K 0.77×, 16K 0.84×;
K2048 262K 0.77×).

**ROOT CAUSE (design-level, as predicted):** the 2-kernel design ADDS ≥1 full-N pass — the
Triton sweep is a SEPARATE kernel (its own launch + full-N read), and kernel B still needs a
count pass at thr* to populate smem_ptcnt for the collect. Baseline does P2(1 count, which
doubles as smem_ptcnt fill) + P3(collect). Portfolio does sweep(1) + B[P1 + recount(1) +
collect]. So +1 full-N pass + 1 extra launch, NOT offset (K512 P2 is already 1 iter → no
search saving). The crux's "free redundant reads" is free only as a REPLACEMENT within one
kernel; as an ADDED kernel it costs a full pass + launch.

**KEY INSIGHT for the only viable design:** in a SINGLE cooperative kernel, the WINNING CTA
(r==best_m) counted at thr* = ITS OWN threshold → its smem_ptcnt is ALREADY populated at thr*
→ it can do P3 collect + P4 with ZERO extra pass. The sweep REPLACES P2. This requires:
grid=(G,) cooperative, cross-CTA count share (DSMEM if G≤16 = one cluster; global+grid.sync
if G=148), a barrier, then winner does the tail. G=16 (one cluster, reuse the existing
gvr_topk_decode_cluster.py DSMEM machinery) is the tractable target — coarser resolution
(cand looser than G=148) but zero extra pass. THIS is the real A1 + cluster-fusion build.

**Status:** two tractable realizations falsified (iter4 single-CTA M-way = ALU not free;
iter5 2-kernel = +1 pass). Only the single-cooperative-cluster kernel (G=16, winner-does-tail)
can net-win, and only in small/mid-N (large-N dominated by the existing PR#15198 cluster).
Measured ceiling remains ~1.2×. That kernel is a large build on the 2292-line cluster file.

---

## Iter 6 — 2026-07-01 — cooperative-cluster portfolio (G=8): EXACT + NET-POSITIVE ✅

`src/gvr_portfolio_cluster_op.py`: single kernel, cluster of G=8 CTAs per row. Each CTA:
P1 (redundant) → thr_r = pmin + r*(pmax-pmin)/(G-1) → block_count_ge FULL N at thr_r (fills
its own smem_ptcnt + count). DSMEM-share counts (cluster_arrive_relaxed/wait +
mapa_shared_cluster/ld_shared_cluster_i32), pick best_m = highest r with count≥K; WINNER
(rank==best_m) sets s_thr[0]=thr_r, done=1 → P3 collect reuses its smem_ptcnt (ZERO recount)
→ P4 (tight cand). Others exit. Reuses single-CTA phase1/block_count_ge/phase3/phase4 (no
vendored edit); DSMEM helpers imported from the cluster module.

**EXACT (vdiff=0, uniq=K) all K512/1024/2048 fp32. A/B (`scripts/iter6_ab.py`, cold-L2):**
| K\N | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 | 262144 |
|---|---|---|---|---|---|---|---|
| 512  | 1.112 | 1.123 | 1.135 | 1.216 | 1.200 | 1.198 | 1.191 |
| 1024 | 0.988 | 1.123 | 0.997 | 1.231 | 1.263 | 1.204 | 1.305 |
| 2048 | –     | 1.035 | 1.082 | 1.104 | 1.248 | 1.324 | 1.487 |

**Avg ~1.18× over 21 cells; wins at LARGE N too (K2048/262K 1.49×, K1024/262K 1.31×).** Only
2 near-neutral: K1024 N=4096 (0.988×) + N=16384 (0.997×) — ~1% dips (fix via dispatch or G).
This VALIDATES B1: the cooperative single-kernel is the design that nets the P4-shrink with no
extra pass. Confirms the iter4/iter5 negatives were realization artifacts, not the idea.

**Next (iter7):** (a) try G=16 (finer band → tighter cand → lift the weak small-N cells);
(b) dispatch-guard the 2 K1024 dips to baseline for strict no-regression; (c) ×3-median +
nsys-validate the wins (repo rule); (d) extend bf16/fp16. Not +40% (avg ~1.18×) but a real,
broad, exact, ~no-regression win over the single-CTA baseline — the first shippable op17 form.

## Iter 7 — G=16: NO REGRESSIONS, avg ~1.21×, exact

Same kernel, G=16 (finer band → tighter cand). EXACT all cells. A/B (cold-L2):
| K\N | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 | 262144 |
|---|---|---|---|---|---|---|---|
| 512  | 1.148 | 1.145 | 1.422 | 1.180 | 1.230 | 1.130 | 1.148 |
| 1024 | 1.016 | 1.135 | 1.117 | 1.194 | 1.237 | 1.201 | 1.280 |
| 2048 | –     | 1.132 | 1.106 | 1.157 | 1.354 | 1.361 | 1.482 |

**min 1.016× (NO regression), avg ~1.21×, max 1.482× (K2048/262K); exact everywhere.** G=16
fixed the two G=8 dips (K1024 4K 0.988→1.016, 16K 0.997→1.117) and lifted K512/16K to 1.422×.
This is the first shippable op17 result: exact, no-regression, broad ~21% avg over single-CTA
gvr_cutedsl, up to +48%. Uses idle BS=1 SMs (free per crux); a low-BS/decode-regime win.

**Next:** ×3-median + nsys-validate (repo rule for positive claims); A/B vs the EXISTING
PR#15198 multicta cluster (portfolio adds tight-cand P4 on top of multi-CTA scan); bf16/fp16;
high-BS degeneration guard (dispatch G=1→baseline when BS·G>NUM_SMS). Not +40% avg but a real,
exact, no-regression broad win — B1 delivered.

## Iter 8 — ×3-median, all dtypes, vs BOTH baselines (scripts/iter8_validate.py)

Cooperative-cluster G=16, ×3-median cold-L2, EXACT all 33 cells (fp32/bf16/fp16).
port/base = vs single-CTA gvr_cutedsl (the target baseline); port/mc = vs PR#15198 multicta.

**vs single-CTA baseline — NO regression anywhere, all dtypes:**
- fp32: 1.00-1.48× (K512 1.17-1.37, K1024 1.00-1.27, K2048 1.03-1.48)
- bf16: 1.13-1.39×   fp16: 1.08-1.38×.  Avg ~1.22×.

**vs existing PR#15198 cluster:** WINS N≤65K (1.02-1.29×) but LOSES N=262144 (0.71-0.83×) —
multicta partitions the scan (cs=4, each CTA scans N/4) so its large-N collect beats the
portfolio's single-winner full-N collect. Crossover ~131K.

**Verdict:** solid no-regression ~1.22× win over the TARGET baseline (single-CTA gvr_cutedsl),
all dtypes, exact. NOT +40%. Production dispatch = portfolio for N≤~131K, existing multicta
cluster for N≥262K (or fuse partition+portfolio = deeper follow-up). Next: nsys-validate the
positive cells (repo rule); high-BS guard.

## Iter 9 — nsys pure-kernel validation (repo rule) — win CONFIRMED, LARGER than event

`scripts/nsys_run.py` (cold-L2 flush in cudaProfilerApi window), G=16, fp32, 100 iters,
cuda_gpu_kern_sum (evict kernel excluded), ×1:
| cell | base µs | port µs | nsys spdup | event spdup |
|---|---|---|---|---|
| K512  N16384  | 16.69 | 9.97  | 1.673× | 1.365× |
| K512  N65536  | 19.24 | 14.71 | 1.309× | 1.174× |
| K1024 N65536  | 24.27 | 17.91 | 1.355× | 1.250× |
| K2048 N262144 | 52.44 | 33.31 | 1.574× | 1.475× |
| K512  N262144 | 39.20 | 32.50 | 1.206× | 1.174× |

**nsys > event on ALL cells** → event timing penalizes the cluster kernel with launch
overhead that pure-kernel nsys omits ⇒ the ×3-median event ~1.22× avg is a CONSERVATIVE
lower bound; true pure-kernel win ~1.3–1.67× here. Repo rule satisfied (positive claim
nsys-validated). Reps: results/nsys/v2_*.nsys-rep.
