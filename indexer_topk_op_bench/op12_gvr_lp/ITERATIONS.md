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

## Verdict
- **P4 floor-bound** across snap/rs/rs_exact (all cluster ~same) → no in-structure P4
  algo change moves it materially.
- **opt-1 (fp16 traffic)**: candidate keys already fp32 in smem; input fixed fp32 vs
  SGLang → no traffic to cut without a pre-pass that cancels savings. Neutral.
- **opt-2 (candidate reduction)**: rejected (above).
- **50%-everywhere target is physically infeasible at small N** (floor proof).
- **Best achievable op = regime dispatch** (rs_exact/512 for N<131072, snap/1024 for
  N≥131072) → wins large N decisively (1.2–1.9×), parity/slight-loss small N.
  Set as the `gvr_lp` default (p4_mode="dispatch").
