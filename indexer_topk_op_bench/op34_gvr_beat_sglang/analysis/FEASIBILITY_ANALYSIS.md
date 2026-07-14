# op34 — Feasibility analysis: can a GVR-skeleton kernel beat sglang_v2 by 30% (BS=1 real)?

Date 2026-07-14. Evidence: Phase-1 gap map + crux + kC-probe (this campaign) + master GVR
falsification ledger + sglang v2 source read (ops/sglang_v2/sgl_kernel/deepseek_v4/topk_impl.cuh).

## What sglang_v2 actually does (source-confirmed)
Header: "fp16 coarse histogram -> threshold bin -> fp32-boundary collect -> exact radix tie-break."
Three dispatch paths (by seq_len = compressed valid N):
- **N ≤ 8192 — TopKRegister**: loads the ENTIRE row into registers (`local_vecs`), builds the
  histogram from registers, finds the threshold bin, COLLECTS FROM THE SAME REGISTERS (no 2nd
  global read), tie-breaks. => **ONE global read, register-resident.** ~4–6us.
- **8192 < N — TopKStreaming**: "two vectorized passes over global memory" (histogram pass +
  collect pass). Single block.
- **large N — TopKCluster<8>**: same 2-pass histogram-select but across **8 cooperating CTAs**
  (8× the HBM/L2 parallelism).

Key: sglang is NOT magic — it is a histogram-select (radix family), the SAME family as GVR's
count+collect. Its advantages are (a) register-residency at small N, (b) minimal phase overhead
(one lean kernel, few barriers), (c) 8-CTA parallelism at large N.

## The one real GVR lever vs sglang: hint-driven pass ELIMINATION
GVR's hint (prev-topK) lets P1b place a threshold from K values WITHOUT a full-N histogram pass.
So GVR could do 1 full-N pass (collect at the hint threshold + inline multi-count verify) vs
sglang streaming/cluster's 2 passes. THIS is the user's single-scan idea, and it is NOT Opt-L
(Opt-L kept the count pass and fused collect into it; this DROPS the count pass).

## Why it still (very likely) can't reach +30% — the double-lock
1. **Exactness read-floor (information-theoretic).** An exact top-K must read every element ≥ once
   (any unread element could be in the top-K; hint hit-rate < 1 — flash 0.30–0.76 — so winners
   exist outside the hint and MUST be found by scanning N). So GVR's floor = 1 full-N read = the
   SAME as sglang. The hint cannot get below 1 read while exact. GVR's ceiling vs sglang = parity.
2. **Relaxed-constraint controls (three independent, all land at ~parity, none at +30%):**
   - **op31 HBE-C** already built EXACTLY this pass-elimination (hint threshold → 1 collect pass,
     "one find_threshold dropped + 1-cmp pass") on the BETTER (cluster) skeleton: CLOSED at
     **parity in-envelope (geomean 0.991), wins only N≥524288** (outside envelope). Ledger.
   - **op29** same fused pass on base: **1.03–1.13× vs BASE** at N≥65536 — but base is 2.36× slower
     than sglang, so 1.13× vs base is still ~2× slower than sglang. Pass-elimination helps a little;
     the GVR phase-chain + single-CTA overhead is the wall, not the pass count.
   - **op34 kC-probe** (this campaign): the P4/fast-write mechanism = ≤14% (small N) to NEGATIVE
     (large N). P4 is not the BS=1 cost.
3. **Arithmetic ceiling** (op34 kC-probe absolute us): flash/1024k GVR 2-pass = 34.5us, sglang =
   16.5us ≈ exactly half. A PERFECT 1-pass GVR (free append, zero phase overhead) => ~17us ≈
   sglang parity. Real GVR has non-zero phase overhead => parity-or-worse. No +30% headroom.

## Verdict (pre-authorized negative conclusion, AUTONOMY.md)
Beating sglang_v2 by **30% on the BS=1 grand average, exact, within the GVR skeleton, is very
likely structurally infeasible.** sglang already sits at the exact-top-K information floor
(1-pass register-resident at small N; lean 2-pass multi-CTA at large N). The hint's only real
lever (pass-elimination) reaches PARITY at best (op31 measured it), because the GVR skeleton's
fixed phase-chain + single-CTA overhead exceeds sglang's lean design — and that overhead, not the
pass count, is the wall.

## Residual uncertainty worth ONE empirical test
op31's parity verdict was on synth/realcap on the CLUSTER skeleton. The NEW v4cap **pro** data has
unusually HIGH hit-rate (0.6–1.0). A hint-driven single-scan on high-hr pro large-N cells is the
ONE untested combination that could, in principle, do better than op31's aggregate parity. If any
regime beats sglang it is here. => The honest next step is to BUILD the single-scan on op26_r0 and
MEASURE this specific regime, OR accept the double-locked wall + harvest the parity improvement.

## Harvest (positive outcome even under the wall)
The single-scan pass-elimination CAN move op26_r0auto from its current 1.6–2.4× deficit toward
~parity — a real GVR-family improvement (just not a sglang beat). That is shippable value.
