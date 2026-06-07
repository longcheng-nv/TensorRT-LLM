---
name: dual-twin-authoring
description: >
  Procedure to add a NEW operator to the mptracer harness by writing a source-tagged forward
  dual-number twin and registering it. Use when mptracer has no twin for the target operator
  (available_twins() lacks it) and you need primitive-level error attribution for it. Covers the
  primitive propagation contract, the escalation channels, and finite-difference self-validation.
metadata:
  tool: mptracer
  target: mixed_precision_tracer/mptracer/twins/
---

# Authoring a dual twin for a new operator

A twin mirrors the production kernel's **primitive sequence** carrying source-tagged dual channels.
It is registered in `mptracer/twins/`; the harness core stays operator-agnostic. This is the seam
that makes twin maintenance "regenerate-and-revalidate".

## The Twin protocol (implement these three)

```python
class MyTwin:
    name = "my_op"
    def generate_inputs(self, rng, shape, distribution): -> (A, B)
    def reference(self, A, B, shape, ref_dtype): -> Y_ref (fp64)   # TF32 off
    def real_and_dual(self, A, B, policy, shape, with_cross): -> (Y_real, {tag: dual}, extra)
register_twin("my_op", MyTwin())
```

## Primitive propagation contract

Inject the concrete residual `δz = z - quantize(z)` (a finite difference, NOT a derivative — this
is running-error analysis, so it is exact at the injection node, unlike AD). Propagate per primitive:

- **convert/quantize:** real = rounded value; add residual to the source's channel.
- **matmul C=A·B:** `C_dA = δA·B_real`, `C_dB = A_real·δB`, `C_mma` = fp32-accum residual.
- **add/scale (linear):** pass channels through linearly — **exact**, independent of ‖δ‖.
- **smooth nonlinear f (softmax/SiLU/GELU/1/√):** propagate via the Jacobian `f'(real)·channel`
  — **first-order**, bounded, ρ-measurable.
- **store:** inject the output-dtype residual into `D_store_round`.

## Escalation channels (add when `with_cross`)

- **`cross_AB`** = `δA·δB` (the bilinear term first-order drops). For a linear op this makes it
  **exact**; for a nonlinear epilogue propagate it through the Jacobian too.
- **`<op>_2nd`** = the epilogue's 2nd-order Taylor term. For SwiGLU `out=silu(g)·u`:
  `0.5·silu''(g)·δg²·u + silu'(g)·δg·δu`, `silu''(x)=s(1-s)[2+x(1-2s)]`.
- Mark non-smooth nodes (clamp/argmax/ReLU) and expose `flip_risk` = fraction of elements whose
  distance to the breakpoint is below the propagated perturbation.

## Self-validation (mandatory before registering)

1. **Linear-path exactness:** single-sided quantize on a linear sub-op → `rho ≤ 1e-6` vs fp64.
2. **Budget correctness:** one-pass per-source budget vs leave-one-out → cosine ≥ 0.999.
3. **Real-dim nonlinear residual:** validate the nonlinear-path `rho` at the REAL contraction dim
   (not reduced) — the curvature residual is not dimension-invariant.

Add a row to the harness regression test (`tests/test_regression.py`) pinning these.
