# op34 Phase-1 — Real V4 decode-capture data characterization + gap map

Date 2026-07-14 · node umbriel-b200-074 · B200 SM100 (148 SM) · branch omni/op21-gvr-prod
Data: `../E2E_exp/indexer_decode_capture/data/{flash,pro}`, 18 (model,ISL) cells,
BS=1 fp32 cold-L2. Scripts: `analyze_realdata.py`, `gap_map.py` (committed).

## A. Temporal hit-rate (the GVR preIdx-hint core assumption)

`hr(s) = |topk[s] ∩ topk[s-1]| / K`. Benchmark uses the LAST decode step, so the
relevant hr is the **mature steady-state** value (`last-hr` below), NOT the
step-0 transient (which is ~0.002 — first decode step vs prefill, irrelevant).

| regime | flash last-hr med | pro last-hr med |
|---|---|---|
| small N (4K–16K)  | 0.51–0.76 | 0.76–0.998 |
| mid N (32K–128K)  | 0.34–0.70 | 0.60–0.79 |
| large N (256K–1M) | 0.30–0.51 | 0.63–0.64 |

- **pro hit-rate is high everywhere** (med ≥0.60, up to 0.998 at 4K where K/N≈1).
- **flash hit-rate is lower & noisier** (min 0.023 at 512K L?, med dips to 0.30).
- Per-step: hr RAMPS from ~0.002 (step0) to the mature value as context stabilizes.
  Benchmark's last-step hr is representative of steady-state decode.

**Implication:** the hint is genuinely informative (esp. pro) → a threshold
seeded from the hint's order statistics should be near-correct in ONE pass.

## B. Boundary difficulty (why threshold methods pay a refine tax)

`gap_rel = (v[K-1] − v[K]) / (vmax − vmin)` = separation of the K-th from the
(K+1)-th largest, relative to the value span.

- **gap_rel median 2e-5 … 2e-4, min down to 2e-7.** The selection boundary is
  razor-thin: the threshold must be placed to ~1e-5 relative precision or the
  count is wrong → secant refinement / undershoot risk. This is the mechanism
  behind op26_r0auto's 2.7e-6 boundary-precision requirement.
- sglang_v2 (hint-blind register/streaming/cluster top-K) is INSENSITIVE to
  gap_rel — it never places a scalar threshold; it does cooperative selection.

## C. Current gap map (BS=1 fp32 cold-L2, geomean over layers) — THE TARGET

```
GRAND geomean:  sgl=7.88us  r0(op26_r0auto)=12.31us  base=18.60us
                r0/sgl=1.561   base/sgl=2.359
op34 target:    new GVR <= sgl/1.30 = 6.06us  => need 2.03x over r0 / 3.07x over base
WIN LAYERS: 0/459 — NO in-tree GVR arm beats sglang on ANY (model,ISL,layer).
```

Per-regime deficit (need× over r0 to beat sgl by 30%):
- small N (4K–16K): **2.2–2.6×** (hardest) — sgl 4–5us vs r0 7.5–10us.
- mid N (32K–128K): 1.75–2.1×.
- large N (256K–1M): **1.6–1.8×** (least hard) — sgl 10–16us vs r0 15–21us.

## D. Bottleneck hypothesis (drives Phase-3 probes)

Two DISTINCT regimes with different walls:

1. **Small N (≤16K) = launch/fixed-overhead bound.** sglang's lean raw-CUDA
   kernel (~4–5us) beats the cuteDSL GVR kernel's fixed launch+setup (~7.5us
   floor). At N=1027 there is almost no per-element work; the gap is pure
   overhead. **Likely a structural wall for cuteDSL GVR** unless launch/setup
   is slashed. Candidate: a lean single-pass CUDA GVR for the small-N tail, or
   accept the wall here.

2. **Large N (≥128K) = pass-count bound.** Both sgl and GVR must read all N once
   (N·4B ≤ 1MB ≤ L2, so not HBM-bound except 512K/1M). sgl does ~1 selection
   pass; GVR does threshold-count + secant-refine (>1 effective pass over N).
   **The lever: exploit the high hint hit-rate to place a near-perfect threshold
   from the hint's order statistics so GVR reads N ONCE (1 counting pass +
   collect), matching sgl's pass count.** This is where pro (hr 0.6–1.0) is most
   promising and the deficit is smallest (1.6–1.8×).

## E. Feasibility verdict (Phase-1, pre-probe)

The headline 30%-average target is **very aggressive** (2.03× over the best
existing GVR; 0/459 current win-layers). It is NOT yet double-locked infeasible —
the large-N high-hr regime has a concrete unexploited lever (single-pass
hint-seeded threshold). Plan: attack large-N/high-hr first (best odds), treat
small-N as a probable overhead wall to be bounded (Phase-5 UB/LB), and report
the regime map honestly. The pre-authorized negative conclusion applies if the
double-lock (math floor + relaxed-constraint control) confirms a wall.
```
