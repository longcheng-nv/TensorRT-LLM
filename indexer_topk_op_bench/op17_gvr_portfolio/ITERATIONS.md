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
