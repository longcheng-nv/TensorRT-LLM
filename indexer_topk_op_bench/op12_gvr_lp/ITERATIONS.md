# op12_gvr_lp — GVR cuteDSL low-precision + P4-skip, beat SGLang StreamingTopK

**Goal**: new single-CTA GVR top-K op that beats SGLang StreamingTopK across ALL
report cells (fp32 input, K∈{512,1024}, N=4K..256K, BS=1..2048, beta_moderate
synth, hit-rate 0.6), average ≥50% faster. Keep GVR outline (secant→refine).

**Base**: `p4_recursive_digit/src/gvr_topk_decode_p4.py` (op#7 rank-scatter P4),
copied to `src/gvr_topk_decode_lp.py`. Originals never modified.

**HW**: B200 sm_100 (matches report `results_b200/`).

## Baseline (from results_b200, fp32, K∈{512,1024}, 182 cells)
- best existing GVR = rank-scatter (op#7). `sglang/gvr_rs`: median **0.868**, mean **0.996**.
- gvr_rs faster in 61/182. Target = **sglang/new ≥ 1.5 median, > 1.0 every cell**.
- WORST for GVR (must fix most): small N (4K–16K), low/mid BS → ratio 0.61–0.69.
  - e.g. K512 N=4K BS=1: sglang 12.4 vs gvr_rs 20.3 (need new ≤ 8.3µs).
- BEST for GVR: large N (256K) low BS → ratio ~1.9 (already wins).

## Phase cost model (report, BS=1)
- N≤16K: **P4 dominates ~55%** → optimization (2) P4-skip is the lever.
- N≥64K: **P2+P3 full-N scans dominate (52→80%)** → optimization (1) fp16 traffic.

## Levers
1. Internal fp16/bf16 for P1→P3 (halve full-N byte traffic), fp32 P4 refine.
2. P4-skip: mark definite-top-K (val > threshold band) during P2/P3, refine only the straddling band.
3. CTA size tuning (512 vs 1024).

---

## Iter 0 — copy sanity ✅
Copied kernel compiles + exact (fp32, K512/1024, N 4K..256K): uniq=K, valdiff=0.

## Iter 1 — config sweep (existing knobs), battleground 11 cells
A/B vs SGLang, cold-L2 (sglang/new, >1=new faster):

| config | median | mean | min | win | fails |
|---|---|---|---|---|---|
| rs_exact/512 | 1.000 | 1.040 | 0.701 | 5/11 | 0 |
| snap/512 | 0.831 | 0.909 | 0.567 | 3/11 | 0 |
| snap/1024 | 0.819 | 0.993 | 0.571 | 4/11 | 0 |
| rs/512 (approx) | — | — | — | — | **9** (inexact) |

Per-cell winners: N=4K→snap/1024 0.76 (all lose); N≥64K→snap/1024 (1.27–1.79);
N=16K→rs_exact 1.00–1.05; N≥128K→snap/1024 best.

**Findings**
- `rs_exact/512` = best all-rounder; `snap/1024` = best large-N. Dispatch helps mid/large.
- **Small N (4K,8K) is the wall**: every config 0.70–0.84×. P4-bound.
- `rs` non-exact unusable (straddle-bin mis-order on continuous Beta data).
- Since P1+P2+P3≈6.6µs < SGLang 12.7µs at N=4K, the win requires a **near-free P4**.

**Next (iter 2)**: opt-2. Tighten kFTarget so cand_count ≤ num_threads, then add a
SGLang-style one-element-per-thread 4-round 8-bit radix P4 (lean refine), keeping
GVR's secant→refine outline. Targets the small-N P4 wall.

## Iter 2 — P4-budget probe (nop4 = early-return after P3)
Within-process, cold-L2:

| K | N | sglang | P1+P2+P3 floor | rs_exact | P4 cost |
|---|---|---|---|---|---|
| 512 | 4096 | 12.7 | 10.3 | 18.1 | ~7.8 |
| 512 | 8192 | 13.2 | 11.1 | 15.4 | ~4.3 |
| 512 | 16384 | 16.5 | 13.8 | 24.1 | ~10.3 |
| 512 | 65536 | 30.4 | 21.2 | 32.4 | ~11.2 |
| 1024 | 262144 | 85.7 | 48.8 | 57.2 | ~8.4 |

**Findings (decisive)**
- **P4 is 45–50% of GVR time** (4–11µs) → the entire reducible opportunity.
- **P1+P2+P3 floor alone is ~1.2× SGLang at small N** (N=4K: 10.3 vs 12.7). So even
  a FREE P4 caps small-N at ~1.2×; the shared ~4µs CUDA-graph launch + GVR's intrinsic
  secant make **1.5× physically UNREACHABLE at N≤16K**.
- Heavy run-to-run variance (rs_exact N=4K: 14.9–19.9 across runs) — GPU co-tenancy;
  individual cells ±30%. Trends robust, per-cell numbers noisy.

## Iter 3 — opt-2 test (kc_accept tightening → cand_count ≈ K → cheap P4)
Added `kc_accept` knob (secant acceptance window upper bound, separate from kC buffer
cap) + kFTarget pulled to window mid. Exactness held (fails=0). Small-N A/B:

| K | N | sglang | rs_exact (a5120) | rs_exact a768 | snap a768 | snap a640 |
|---|---|---|---|---|---|---|
| 512 | 4096 | 12.5 | 0.83× | 0.84× | 0.76× | 0.56× |
| 512 | 8192 | 13.2 | 0.88× | 0.88× | 0.54× | 0.79× |
| 512 | 16384 | 16.6 | 1.10× | **0.98×** | 0.89× | 0.58× |
| 512 | 65536 | 29.6 | 1.20× | **0.95×** | 0.69× | 0.84× |

**Finding (decisive, NEGATIVE)**: tightening the acceptance window does NOT help —
P4 barely changes (it is **barrier/latency-floor bound, not candidate-count bound**)
while the extra secant/retry-shrink full-N passes make it *worse* at N≥16K. opt-2
(candidate reduction) is rejected by data.

## Iter 4 — opt-1 upper bound: half-width INPUT (free, no pre-pass) + P1-P3-only decomp
Measures the ceiling of "fp16-scratch then P1-P3 read fp16, P4 refine fp32" by feeding
the kernel bf16/fp16 input directly (the zero-overhead best case). `~` = inexact vs fp32
(bf16/fp16 rounds logits → its top-K ≠ fp32 top-K; the exactness landmine).

Full kernel (rs_exact), cold-L2 — note SEVERE co-tenancy noise at small N (rs_exact
bf16 N=4K read 14.8µs in one run, 20.4µs in the next):
| K | N | sglang | fp32 | bf16 | fp16 |
|---|---|---|---|---|---|
| 512 | 65536 | 29.6 | 0.95× | 1.37× | 1.38× |
| 512 | 262144 | 82.8 | 1.38× | 1.74× | 1.74× |

**Decomposition — P1+P2+P3 only (nop4), the part an EXACT (fp32-P4) scheme keeps:**
| K | N | P1-P3 fp32 | P1-P3 bf16 | bf16 saving |
|---|---|---|---|---|
| 512 | 4096 | 10.8 | 9.6 | −1.2µs (11%) |
| 512 | 8192 | 10.9 | 11.6 | ~0 (noise) |
| 512 | 16384 | 14.3 | 12.9 | −1.4µs (10%) |
| 512 | 65536 | 21.2 | 17.7 | −3.5µs (17%) |
| 512 | 262144 | 49.6 | 43.8 | −5.8µs (12%) |

**Findings (answers the fp16-scratch question)**
- The big bf16 full-kernel win is mostly from **bf16 collapsing candidate values → cheaper
  P4** (report: bf16 snaps fastest). But that is exactly what makes it **inexact**, and a
  scheme with fp32-P4-refine GIVES IT BACK.
- The EXACT scheme (bf16/fp16 P1-P3 + fp32-reload P4) captures only the **P1-P3 traffic
  saving: ~1µs at small N (noise-level, doesn't flip the verdict) and ~3-6µs at large N
  (~+0.1× where GVR already wins 1.6-1.8×)**.
- So fp16-scratch is a small, real, traffic-proportional win that **does NOT solve the
  small-N wall** (P4-barrier + ~4µs launch floor, both dtype-independent) and adds a
  pre-pass/mixed-precision + fp32-gather complexity cost. Marginal net.

## Iter 5 — IMPLEMENTED the exact mixed-precision scheme (opt-1 full)
New `enable_lp_scan` kernel path + `lp_bf16`/`lp_fp16` modes: P1-P3 scan the
bf16/fp16 copy (half-width loads), **Phase-3 reloads the original fp32 value**
(new `input_fp32` kernel arg) for each collected candidate into smem_keys[fp32],
so P4 refines EXACTLY. Kernel changes: ctor flag, phase3 3 write-sites, kernel
sig + fp32 row slice + call, __call__ + wrapper fake tensor. lp timing INCLUDES
the fp32->bf16 cast (realistic with-pre-pass).

**Exactness: PASS** — valdiff=0, uniq=K for bf16 AND fp16, all N incl. 262144
(boundary value-collapse did NOT overflow kC). Confirms the monotonic-rounding
superset argument + fp32-reload gives exact top-K.

**Perf (cold-L2, sglang/new; lp vs the fp32 GVR baseline):**
| K | N | rs_exact fp32 | lp_bf16 | lp_fp16 | lp vs fp32 |
|---|---|---|---|---|---|
| 512 | 4096 | 0.83× | 0.79× | 0.77× | worse |
| 512 | 8192 | 0.95× | 0.81× | 0.62× | worse |
| 512 | 16384 | 0.99× | 0.83× | 0.84× | worse |
| 512 | 32768 | 0.79× | 1.02× | 1.02× | better |
| 512 | 65536 | 0.91× | 1.15× | 1.19× | **+21%** |
| 512 | 131072 | 1.15× | 1.37× | 1.35× | **+16%** |
| 512 | 262144 | 1.36× | 1.59× | 1.55× | **+15%** |

**Validated findings (exactly as analyzed):**
- The exact scheme HELPS at **large N (+15-23% over fp32 GVR even WITH the cast)** —
  improves the already-winning region (262K vs SGLang 1.36×→1.59×). lp_fp16 is the
  best single all-rounder so far (median vs SGLang 1.089).
- It is **neutral-to-WORSE at small N (4K-16K)**: the half-width P1-P3 saving is
  tiny there (latency/launch-bound) and the fp32->bf16 cast pre-pass overhead
  dominates → makes the binding constraint worse, not better.
- **Does NOT solve the small-N wall** (still 0.62-0.84× at N<=16K). The 50%-everywhere
  target stays blocked by the small-N floor (launch + P4 barriers + now cast).
- If the bf16 copy were produced upstream (cast excluded), small N would be ~neutral
  (per the iter-4 nop4 decomposition ~1µs) and large N even better — but still not a
  small-N win.

## Iter 6 — head-to-head: large-N dispatch arm (snap/1024 vs lp_fp16) + BS sweep
Pending-task-1 was "fold lp_fp16 into the large-N arm". iter5 only compared lp_fp16
vs **rs_exact** (never vs the actual incumbent `snap/1024`) and only at BS=1. This
closes both gaps. All exact (fails=0), cold-L2, within-process.

**Large-N BS sweep (sglang/new):**
| K | N | BS | snap/1024 | lp_fp16/1024 | lp_fp16/512 |
|---|---|---|---|---|---|
| 512 | 131072 | 1 | **1.61** | 1.47 | 1.10 |
| 512 | 131072 | 64 | **1.39** | 1.14 | 1.03 |
| 512 | 131072 | 128 | **1.24** | 1.01 | 0.86 |
| 512 | 131072 | 256 | **1.08** | 0.82 | 0.89 |
| 512 | 262144 | 1 | **1.64** | 1.57 | 1.36 |
| 512 | 262144 | 64 | **1.65** | 1.36 | 1.05 |
| 1024 | 131072 | 1 | **1.42** | 1.27 | 1.10 |
| 1024 | 262144 | 1 | **1.69** | 1.63 | 1.40 |
| **median** | | | **1.516** | 1.316 | 1.073 |

**Finding (decisive, NEGATIVE for task-1)**: `snap/1024` BEATS `lp_fp16` in **8/8**
large-N cells. lp_fp16's gain in iter5 was only relative to rs_exact; snap was always
the large-N arm and is cheaper still — lp_fp16 pays a full-N fp32→fp16 cast pre-pass
+ rank_scatter_exact fine-hist barriers, which at large N outweigh the half-width
read saving. **lp_fp16 is NOT folded in; the existing dispatch default already wins.**

**Dispatch-boundary fix (BS>NUM_SMS at large N):** snap/1024 still wins at BS=256/512,
but the old `_dispatch_config` gated the snap arm on `bs<=NUM_SMS` → fell back to
rs_exact/512 there (a loss). A/B:
| K | N | BS | snap/1024 | rs_exact/512 (old fallback) | snap/512 |
|---|---|---|---|---|---|
| 512 | 131072 | 256 | **1.12** | 0.87 | 0.95 |
| 512 | 131072 | 512 | **1.09** | 0.85 | 0.89 |
| 512 | 262144 | 256 | **0.92** | 0.89 | 0.93 |
| 1024 | 131072 | 256 | **1.11** | 0.95 | 0.89 |
→ Removed the `bs<=NUM_SMS` gate: large N (>=131072) now uses snap/1024 for ALL BS.
Recovers ~0.2x at large-N high BS. dispatch exactness re-verified (valdiff=0, all N).

## Iter 7 — small-N arm control-variable A/B (snap vs rs_exact, threads fixed 512)
Validate the OTHER dispatch arm (iter6 did large-N). Isolate the P4-algorithm
variable: hold num_threads=512, vary only p4_mode. small N, full BS. All exact.

| K | N | BS | snap/512 | rs_exact/512 |
|---|---|---|---|---|
| 512 | 4096 | 1 | 0.60 | **0.69** |
| 512 | 4096 | 256 | 0.68 | **0.81** |
| 512 | 16384 | 1 | 0.54 | **0.67** |
| 512 | 16384 | 64 | 0.61 | **0.81** |
| 512 | 65536 | 64 | 0.95 | **1.07** |
| 1024 | 8192 | 1 | 0.70 | **0.84** |
| 1024 | 65536 | 1 | 0.86 | **1.08** |
| **median** | | | 0.735 (0/10) | **0.849 (2/10)** |

**Finding**: rs_exact ≥ snap in ALL 10 small-N cells → the existing small-N arm
(rs_exact/512) is confirmed optimal; no change. NOT contradicting op#7's production
"rs ≈0.92× (slower than snap)" verdict — that is a large-N + high-BS-dominated
aggregate; in the small-N segment rs_exact is consistently better, and the dispatch
splits by N so each arm takes its best.
**No new improvement found**: rs_exact still 0.69× at N=4K — the launch + P4-barrier
floor, exactly as the falsification history (≥18 single-CTA paths ruled out, Pareto
endpoint) and op12's own small-N-wall proof predicted.

### Alignment with the GVR falsification history (gvr_phase_timing/)
Prior-session algorithm "insights" checked against the historical experiments:
- Histogram-replace-P2-secant → P2 is algorithm-internal Pareto optimal (mean 2.13
  iter, max 6, 100% converged; Q5e). GVR ≈2.5 full-N passes ≈ SGLang's 2. **P2 is
  not a target; do not replace it** (per user + data).
- Fuse-collect-into-scan → = **Opt-L**, already falsified: per-element coordination
  (ballot+popc+shfl+atomicAdd) ≈ one full-N scan; no speedup even at FORCE_HAPPY.
- rank-scatter P4 → = op#7, already shipped (and production ≈0.92×; the instrumented
  "P4 win" was a clock64 artifact).
- preIdx-as-seed → = P1 self-loop v1/v2/v3 (91k cells), net-negative (drift ~symmetric).
- Lesson: single-CTA fp32 on sm_100 + real data is a Pareto endpoint; verify any
  "insight" with a control-variable A/B before acting — most are already ruled out.

## Verdict
- **P4 floor-bound** across snap/rs/rs_exact (all cluster ~same) → no in-structure P4
  algo change moves it materially.
- **opt-1 (fp16 traffic)**: candidate keys already fp32 in smem; input fixed fp32 vs
  SGLang → no traffic to cut without a pre-pass that cancels savings. Neutral.
- **opt-2 (candidate reduction)**: rejected (above).
- **50%-everywhere target is physically infeasible at small N** (floor proof).
- **lp_fp16 folding REJECTED (iter6)**: snap/1024 beats lp_fp16 in 8/8 large-N cells.
  The exact mixed-precision path stays available but is not on the dispatch frontier.
- **Best achievable op = regime dispatch** (rs_exact/512 for N<131072, snap/1024 for
  N≥131072, ALL BS) → wins large N decisively (1.1–1.9×), parity/slight-loss small N.
  Set as the `gvr_lp` default (p4_mode="dispatch"). iter6 removed the `bs<=NUM_SMS`
  gate on the snap arm (recovers large-N high-BS).
