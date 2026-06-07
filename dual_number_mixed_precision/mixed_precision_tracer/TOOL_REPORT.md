# Tool Report — `mptracer`: a modular agent/harness for mixed-precision error tracing

**Date:** 2026-06-07 · **Status:** built & verified (regression PASS on local B200 host) ·
**Provenance:** productized from the DSV4 MoE dual-number harness (8 iterations, `../dsv4_moe_harness/`).
**Chinese version:** `TOOL_REPORT_zh.md`.

---

## 1. What was built and why

The DSV4 MoE effort proved a methodology (forward dual-number error attribution + ρ-gated escalation
+ roofline/fusion-gated precision search) on one real operator. This tool **modularizes that
methodology** — per the Anthropic agent-engineering analysis — into a reusable, semi-general
package: an **operator-agnostic harness core** (the deterministic environment), **pluggable
twins/backends**, **five Skills** (the固化 procedural knowledge), and a **specialist agent**.

The architectural keystone is the **trust boundary**, now enforced structurally rather than by
convention: every number lives in a `MeasureResult`; a proposer's numeric justification can only
*reference* a result by id (`PolicyProposal.result_ref`). This is the same propose-verify pattern
Claude Code itself is built on — the model proposes, a deterministic environment adjudicates.

## 2. Architecture (the seam that makes it general)

```
            ┌──────────────── trust boundary (type-enforced) ────────────────┐
 agent /    │ proposes: twin code · dual rules · precision policy · next probe │
 5 skills   │ reads:    MeasureResult (rho, budget, fidelity, latency, accept) │
            └──────────────────────────────┬───────────────────────────────────┘
                                            ▼
   ┌─────────────────────── mptracer (operator-AGNOSTIC core) ───────────────────────┐
   │ core.measure() · types(JSON) · metrics · roofline · escalation · policy_search   │
   └───────────────┬───────────────────────────────────────────┬─────────────────────┘
                   ▼ (plugin)                                    ▼ (plugin)
            twins/  (per-operator dual twin)            backends/ (host fake-quant │ silicon)
            moe_gemm: FC1(SwiGLU)/FC2 ……                torch _scaled_mm fp8/fp4 │ fused cute_dsl op
```

| Layer | Generality | Where |
|---|---|---|
| Harness core (measure / metrics / escalation / roofline / policy search) | **operator-agnostic** | `mptracer/*.py` |
| Twin plugin (primitive sequence + source-tagged dual channels) | per-operator (same template for the GEMM/attention/MLP family) | `mptracer/twins/` |
| Silicon backend (quant recipe + kernel calls) | DSL/version-specific (isolated, pinned) | `mptracer/backends/`, `silicon-precision-oracle` skill |
| Methodology (escalation ladder, fidelity rules, fusion gate) | **fully general** | the 5 Skills |

## 3. The five Skills (modularized methodology)

1. **`mixed-precision-error-tracing`** — top playbook: attribute → ρ-gate → roofline-gate → search →
   silicon-validate. Encodes the escalation ladder and the single-sided-ρ diagnostic.
2. **`dual-twin-authoring`** — how to add a new operator: the primitive propagation contract,
   escalation channels, and the 3 mandatory self-validations. This is the generality lever.
3. **`precision-policy-search`** — attribution-guided greedy / LLM-in-loop policy search; proposer
   orders knobs, measured error accepts.
4. **`silicon-precision-oracle`** — the version-fragile fp8/fp4 `_scaled_mm` + fused-kernel recipes,
   correct twin-fidelity, GEMM-only-vs-quant timing, and the decode-fusion gate.
5. **`autoresearch-loop`** — the gap-board / acceptance-gate / `/loop` protocol for autonomous,
   self-documenting multi-step runs.

Plus the **`mixed-precision-specialist`** agent that routes among them and enforces the trust
boundary — designed to slot beside PerfBot's triton/cudeepy/tileir specialists.

## 4. Verification (harness-owned)

`tests/test_regression.py` reproduces the validated iteration numbers after the refactor:

| Check | Result | Source iteration |
|---|---|---|
| FC2 single-sided nvf4 dual == fp64 first-order | `rho = 2.54e-8` ✅ | iter 1 |
| one-pass budget vs leave-one-out | `cosine = 1.000000` ✅ | iter 1 |
| FC2 double-sided + cross_term closes the bilinear gap | `rho = 2.55e-8` ✅ | iters 2/6 |
| FC1 (K=512) + cross + SwiGLU-2nd-order | `rho = 0.046` ✅ | iter 5 |

The refactor is correctness-preserving. The regression also **surfaced a new honest caveat**: the
SwiGLU 2nd-order residual is **not dimension-invariant** — at real DSV4 K=4096 it is `rho = 0.147`,
not 0.046, because the gate magnitude sits deeper in the silu-curvature region. So on the nonlinear
path, validate at the real contraction dim and expect a ρ-gated ablation fallback at production K.

## 5. Generality envelope (honest)

- **Generalizes cleanly:** the methodology (trust boundary, ρ-gate, escalation ladder, roofline /
  fusion gating, loop protocol) is operator-independent; the twin template covers the
  linear/bilinear-matmul + smooth-nonlinear-epilogue family (GEMM, attention, MLP, MoE) — most
  TRT-LLM hotspots.
- **Breaks down:** arbitrary control flow / exotic fused epilogues / heavy non-smooth routing
  (top-k, sparsity) exceed first-order + 2nd-order Taylor; the SwiGLU residual's K-dependence shows
  the nonlinear path needs care at real dims.
- **Operational risks (isolated by design):** twin maintenance (auto-twin synthesis is the parked
  Phase-2 lever) lives in `twins/`; DSL/recipe fragility (the fp4 `_scaled_mm` recipe is torch/
  torchao version-bound) lives in `backends/` + the `silicon-precision-oracle` stack pin.

## 6. How to use it

- **Library:** `import mptracer; mptracer.measure(...)` / `greedy_policy_search(...)`.
- **As an agent:** delegate to `mixed-precision-specialist`; it routes the five skills and keeps the
  trust boundary.
- **Autonomously:** `/loop` with an `autoresearch-loop` `PROGRAM.md` gap board.
- **New operator:** follow `dual-twin-authoring`, drop a `twins/<op>.py`, add a regression row.

## 7. Maturity & next steps

Done: operator-agnostic core, MoE twin plugin, escalation ladder, policy search, roofline/fusion
gating, 5 skills, specialist agent, passing regression, trust boundary as a type invariant.

Remaining for full production generality:
1. **Auto-twin synthesis** (AST / operator-overloading) → twins regenerate from the kernel; removes
   the hand-authoring bottleneck (the decisive generality step).
2. **Second built-in twin** (attention / FlashAttention) to prove the template transfers beyond MoE.
3. **Silicon backend module** packaging the iter3/6/8 drivers behind a uniform `backends/` API with
   the CI/headless `silicon-pending` fallback.
4. **PerfBot integration:** register the specialist in the performance-optimization routing table.

## 8. Assessment

The methodology was already correct and validated on real silicon; this tool makes it **reusable,
semi-general, and trust-safe by construction**. It is a defensible *primitive-level error-attribution
+ precision-search specialist* for the GEMM/attention/MoE family — not an any-kernel silver bullet,
and it says so. The two real risks (twin drift, DSL fragility) are quarantined to the two plugin
layers, which is exactly where an agent/harness engineer wants them.
