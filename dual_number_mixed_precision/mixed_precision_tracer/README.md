# mixed_precision_tracer (`mptracer`)

A professional, modular tool for **primitive-level forward dual-number error tracing and
mixed-precision optimization** of GPU kernels — productized from the DSV4 MoE harness effort
(`../dsv4_moe_harness/`). Operator-agnostic harness core + pluggable twins/backends + five Skills +
a specialist agent. **The harness owns every number; the LLM only proposes and reads.**

## Layout

```
mptracer/                 operator-agnostic harness (the environment)
  core.py                 measure() + attribution_vs_leave_one_out()  — the one call
  types.py                MeasureRequest / MeasureResult / PrecisionPolicy / PolicyProposal (JSON contract)
  quant.py                block-scaled fake-quant (fp4/fp8/bf16)
  metrics.py              rho / Higham μ_F / budget / cosine / twin_fidelity
  escalation (in twin)    point → cross_AB → taylor2 channels
  roofline.py             regime classification + demotion gate
  policy_search.py        attribution-guided greedy search (Goal-4 loop)
  twins/                  PLUGIN: registry + per-operator dual twins (moe_gemm built in)
skills/                   5 SKILL.md (the固化 methodology)
agents/                   mixed-precision-specialist.md
tests/test_regression.py  reproduces the validated iteration numbers
TOOL_REPORT.md / _zh.md   the tool report (EN / 中文)
```

## Quick start

```python
import mptracer as mp
fc2 = mp.Shape("decode_bs32_fc2", "moe_gemm:FC2", M=192, K=2048, N=4096, n_groups=8)
r = mp.measure(mp.MeasureRequest(fc2, mp.PrecisionPolicy("nvf4"), ref_dtype="fp64", seed=42))
print(r.rho, [(b.source, round(b.budget,2)) for b in r.budget_per_source])

# attribution-guided precision search (compute-bound shape)
props = mp.greedy_policy_search(
    mp.MeasureRequest(mp.Shape("p","moe_gemm:FC2",512,2048,4096), mp.PrecisionPolicy("bf16","fp32")),
    demote_format="nvf4", error_budget=0.2)
```

```bash
python3 tests/test_regression.py     # PASS reproduces iter1 (2.5e-8, cos 1.0) + iter5 (0.046)
```

## Adding a new operator

Implement the `Twin` protocol (3 methods) in `mptracer/twins/<op>.py`, `register_twin(...)`, and add a
regression row. See the `dual-twin-authoring` skill. The harness core never changes.

## What it is / is NOT

- **IS:** a deterministic error-attribution + precision-search environment for linear/bilinear ops
  with smooth nonlinear epilogues (GEMM/attention/MLP/MoE) under block-scaled fp8/fp4, with the
  trust boundary enforced by the type contract.
- **IS NOT:** an any-kernel auto-optimizer. Arbitrary control flow / exotic epilogues / heavy
  non-smooth routing exceed the first-order model. Twin maintenance and DSL-recipe fragility are the
  known risks (isolated to `twins/` and `backends/`).

See `TOOL_REPORT.md` (English) / `TOOL_REPORT_zh.md` (中文) for the full report.
