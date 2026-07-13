# op29 — GVR-HBE (hint-boundary-exact): beat SGLang v2 across the op22 grid

Date: 2026-07-13. Node: umbriel-b200-027 (GPU0 co-tenant, GPU1 foreign 16GB
resident — blacklist both; use GPU2-7). Branch: omni/op21-gvr-prod.
Protocol: omni-kernel v2. Insights source: ../op28_ext_topk/INSIGHTS_GVR_NEXT.md.

## Objective triple

```yaml
objective:
  incumbent: gvr_ms_auto @ HLS-op27 (production default, op21_gvr_prod/src)
  rival_to_beat: sglang_v2 (op28 arm — new fastest op22-REPORT arm; fp32-only)
  also_tracked: [flashinfer_topk, radix_cutedsl, gvr_cutedsl]
  envelope: {K: [512,1024,2048], dtype: fp32 (rival is fp32-only; 16-bit =
             secondary axis vs incumbent only), N: 4096..262144 primary
             (512K/1M stress), BS: 1..2048, scenarios: [best, worst, real]}
  verdict_axes: [worst, real, best]
  ship_rule: "geomean >= sglang_v2 on EVERY (scenario x K) slice, per-cell
              losses < 10% and attributed; zero regression vs op27 incumbent
              (its cells stay reachable via dispatch); exactness 3-track green;
              dispatch rules <= 3 new (shape-keyed only, no data-keyed)"
  hard_constraints: [exact top-K (tie-aware value-multiset), CUDA-graph safe,
                     fail-soft to 2-pass on hint miss, baseline immutable
                     (new op = new files; flags default-off in shared files)]
```

## Core thesis (what we have that sglang v2 cannot)

sglang v2 = 2 full-N passes (histogram -> threshold bin; collect by bin) +
<=2048-candidate boundary-exact tie select. It sits AT the hint-blind
information floor (measured ~6.8 TB/s DRAM on 2 passes at high BS).

GVR's exclusive asset = preIdx hint. HBE: predict the threshold bin from the
hint's K-th value (K-element gather, ~free), then run ONE fused pass =
collect-by-predicted-bin + inline full histogram build. Invariant check
(count_gt < K <= count_gt+count_eq) passes -> done in 1 DRAM pass (2x the
bandwidth-bound ceiling vs rival; 1 fewer serial phase at latency-bound BS=1).
Fails -> the inline histogram already gives the TRUE bin -> one redo collect
pass = exactly rival's cost. Exactness is unconditional (same boundary-exact
tie machinery); hint quality only moves speed. Worst-scenario floor = parity
by construction (fallback IS the rival's structure).

This is primitive recomposition, not invention: GVR's speculative fused
collect (P1/P3, verified in HLS) x sglang's bin-exactness (verified in op28
gate 459/459) x tiered tie-select. Ledger check: NOT Opt-L (no online
slot-reserve ballot chain — plain atomicAdd cursors); NOT P1-model-seed (uses
the hint directly, not a model); NOT op15 smem-residency; NOT P2 refine (P2
is deleted, not refined).

## Iteration queue

- iter0: fork vendored sglang_v2 -> `gvr29` op (same 4-path dispatch,
  hooks for hint). Gate + L1 parity check vs sglang_v2 arm (sanity: fork
  reproduces rival perf +-3%).
- iter1 (CRUX, host, no GPU): hint->bin prediction accuracy on op22rr bundles.
  For each (scenario, K, N): quantile of hint VALUES at rank K -> coarse bin
  b_hat; compare vs true threshold bin b*. Metrics: P(b_hat == b*),
  P(invariant holds @ b_hat), expected passes = 1*hit + 2*miss.
  GO if real-scenario expected passes <= ~1.3 (=> beats rival's 2.0);
  worst-scenario expected ~2.0 = parity floor (pre-authorized).
- iter2: implement HBE fast path in gvr29 (flag OP29_HINT=1):
  fused collect@b_hat + inline histogram + miss redo. Gate 3-track.
- iter3: L1 pilot grid (subset cells x 3 scenarios) -> L2 nsys verdict on
  ship-candidate cells vs sglang_v2 SAME-BATCH.
- iter4+: long-row cluster variant of the same trick (skip Phase1+DSMEM
  all-reduce on hit); dispatch-floor tuning on OUR envelope (sglang's
  65536 BS16-30 streaming pocket is beatable by clustering earlier);
  P5 engineering if needed.

## Red lines (from FALSIFIED.md seed)

See FALSIFIED.md — op15 smem-residency, Opt-L online slot-reserve fusion,
Opt-B high-BS cluster, Opt-F/P2-multithreshold, P4-internal reseed, P1
model-seed. HBE touches none (checked 2026-07-13).

## Envelope ruling provenance

Deployment envelope N<=262K (user/memory 2026-07); rival fp32-only => primary
axis fp32; 16-bit tracked vs incumbent only. Set by loncheng 2026-07-13.
