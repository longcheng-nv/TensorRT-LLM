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
