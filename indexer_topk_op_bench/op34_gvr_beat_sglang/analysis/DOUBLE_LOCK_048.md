# op34 — DOUBLE-LOCK: 30% beat of sglang is INFEASIBLE within the GVR skeleton (v4cap, BS=1)

Node 048, nsys pure-kernel cold-L2, real pro decode data. Pre-authorized negative conclusion
(AUTONOMY.md). This is the FRESH v4cap double-lock the paper feasibility analysis lacked.

## The decisive decomposition (nsys pure-kernel cold, µs) — results/decomp2
| pro cell (N) | hr | sglang FULL | col_hint | **col_orac (UB)** | mcta_orac (col+tail) | orac/sgl |
|---|---|---|---|---|---|---|
| 1024k L20 (262127) | 0.76 | 17.60 | 29.34 | **17.01** | 51.12 | 2.90 |
| 1024k L32 | 0.39 | 18.77 | 28.74 | **16.16** | 49.95 | 2.66 |
| 1024k L44 | 0.18 | 15.26 | 29.67 | **16.93** | 49.77 | 3.26 |
| 256k L20 (65539) | 0.74 | 12.66 | 24.89 | **12.21** | 47.27 | 3.73 |
| 256k L32 | 0.49 | 11.68 | 24.29 | **11.63** | 46.37 | 3.97 |
| 256k L44 | 0.46 | 12.22 | 24.73 | **12.31** | 48.55 | 3.97 |

col_orac = fused scan+collect at the ORACLE threshold (t = true K-th value → minimal candidate
set M≈K), at max MLP C=64, WITHOUT the rank tail. This is the theoretical UPPER BOUND on any
single-pass GVR-skeleton scan+collect.

## LOCK 1 — UB / math floor (relaxed to the extreme, still only parity)
col_orac (16–17µs @1024k, 12µs @256k) ≈ sglang's ENTIRE top-K kernel (15–19 / 12µs). So the best
the GVR skeleton's scan+collect can EVER do — given the impossible oracle threshold and maximum
CTA parallelism and skipping the tail — merely MATCHES sglang. Since an exact top-K MUST then rank
the M candidates (a nonzero phase), GVR_total = collect + rank > col_orac ≈ sglang. Even a perfect
zero-cost rank yields parity, never sglang/1.30. There is NO 30% headroom at the information/UB
floor. (Leanest conceivable collect = bare scan ~7–9µs nsys + leanest rank ~6µs ≈ 14µs ≈ sglang;
still parity, and still needs the unavailable oracle threshold.)

## LOCK 2 — relaxed-constraint control on THIS data (three independent, all fail)
1. **Oracle-threshold multi-CTA (this campaign, the strongest possible relaxation):** given the
   ANSWER as the threshold + max MLP, full kernel = 47–51µs = 2.7–4.0× SLOWER than sglang. The
   torch.topk tail alone is ~33µs; a lean CuTe rank would cut that, but LOCK 1 shows even a
   zero-cost tail only reaches parity.
2. **Real hint threshold (op34_mcta):** 76–125µs = 4–8× sglang (results/harvest_pro). The hint
   cannot place an exact-safe tight threshold: t=hint.min is exact but admits M=16K–100K
   candidates on real data (weak hint at hit_rate<1) → collect degenerates to a full scan+heavy
   tail. A tighter hint-quantile rung gives small M but MISSES exactness (count<K on most cells,
   /tmp/qsweep) — the classic GVR threshold-miss that forces op26's 2-pass measure-then-collect.
3. **op26_r0auto (best in-tree 2-pass GVR):** 2.2–4.6× slower than sglang across these cells
   (results/harvest_pro). op31 HBE-C (hint on sglang's own cluster) already reached only parity
   in-envelope. All three land at parity-or-worse; none at +30%.

## Root cause (precise, mechanism-level)
Under cold-L2 the first HBM read dominates and BOTH kernels do exactly ONE cold read (NCU_CRUX).
The kernels are LATENCY-bound (<1% DRAM/SM peak); sglang's edge is 8-CTA MLP. The GVR **hint** can
only save sglang's histogram pass — but that pass is L2-HOT (cheap), so eliminating it buys almost
nothing under cold-L2. Meanwhile the GVR skeleton (a) needs an exact-safe threshold the hint can't
give without a measurement pass, and (b) SEPARATES collect and rank into distinct barrier-bound
phases, whereas sglang fuses collect+rank in one lean kernel at the cost GVR needs for collect
alone. The saved (L2-hot) histogram pass < the added (threshold-safety + phase-separation) cost.

## VERDICT: STOP. sglang_v2 remains best. Beating it by 30% on BS=1 within the GVR threshold
skeleton is double-locked INFEASIBLE. No conditional +30% win region exists (even oracle col-only
does not beat sglang by 30% at any cell). Harvest = none for +30%; the honest ceiling is PARITY at
large N with an impossible oracle threshold. Recommendation: keep sglang_v2 as the production top-K
at BS=1; the GVR family's value is elsewhere (BS>1 batched throughput, where multi-row saturation
changes the MLP calculus — out of this campaign's BS=1 scope).
