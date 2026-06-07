---
name: mixed-precision-error-tracing
description: >
  Top-level playbook for primitive-level forward dual-number error tracing and mixed-precision
  optimization of GPU kernels (GEMM / attention / MLP families). Use when the user wants to know
  WHICH tensor/primitive/precision decision drives a low-precision kernel's error, WHETHER a
  precision demotion is worth it, or to run an attribution-guided precision search. Routes to the
  other mptracer skills and drives the deterministic mptracer harness.
metadata:
  tool: mptracer
  trust_boundary: the harness owns every number; you never assert a measured value
---

# Mixed-Precision Error Tracing — playbook

You orchestrate the `mptracer` harness (`mixed_precision_tracer/mptracer/`). **The harness owns
every number.** You propose artifacts (twins, policies, next shapes) and read `MeasureResult`s; you
never state a numerical result that did not come from `measure()`.

## The loop (per operator / workload)

1. **Identify the operator** and register/select its twin (`available_twins()`; if absent →
   `dual-twin-authoring` skill).
2. **Attribute** — one `measure()` with the candidate policy gives the per-source budget and the
   trust gate `rho`. Read which source dominates.
3. **Gate by rho** (the escalation ladder — see below).
4. **Gate by roofline** — only propose a precision demotion for latency where
   `roofline_regime == "compute"` (`precision-policy-search` SOL-gates this).
5. **Search** the precision policy via `precision-policy-search` (attribution-guided, harness-verified).
6. **Validate on silicon** via `silicon-precision-oracle` (twin-fidelity + measured latency/SOL).
7. For autonomous multi-step work, wrap in the `autoresearch-loop` protocol.

## The escalation ladder (rho-gated — this is the core technique)

`rho = ||measured - predicted|| / ||measured||` is the trust gate. Escalate ONLY where it fires:

| rho / signal | meaning | escalation |
|---|---|---|
| small, flip_risk low | first-order explains the error | trust the point dual attribution |
| large on a **linear/bilinear** op | dropped `δA·δB` cross term | `escalation="cross_term"` (matmul-exact) |
| large on a **smooth-nonlinear epilogue** (SwiGLU, GELU) | epilogue curvature | `escalation="taylor2"` (adds the 2nd-order epilogue channel) |
| `flip_risk` high (clamp/argmax) | non-smooth decision flip | interval / stochastic dual |
| large & nothing above explains it | model breakdown | exact leave-one-out ablation (ground truth) |

**Diagnostic rule:** to tell a matmul-cross-term miss from an epilogue-curvature miss, check
single-sided rho (quantize one operand only → `δA·δB ≡ 0`). If single-sided rho is still large,
the miss is the epilogue, not the matmul.

## Hard rules (verified lessons)

- **TF32 off** for every reference; use fp64 for the twin cross-check.
- For linear ops, `rho`/budget are **dimension-invariant** ratios — you may validate at reduced
  dims. For **nonlinear epilogues this is NOT true** (the operating point on the curve shifts with
  K); validate the nonlinear residual at the real K.
- A precision speedup only materializes **compute-bound**; never propose a demotion for decode
  latency on roofline alone — see `silicon-precision-oracle` for the activation-quant-fusion gate.
