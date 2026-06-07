# Phase-0 Design: Unified Harness API for DSV4 MoE GEMM Dual-Number Error Tracing

> **Status:** design (Phase 0 of the LLM-native build plan). **Target operator:** DeepSeek-V4
> MoE contiguous grouped GEMM (`Sm100BlockScaledContiguousGroupedGemmKernel`). **Scope:**
> `SCOPE_DSV4_MOE_BS1-512.md`. **Loop:** `PROGRAM.md`.

## 摘要 (Chinese executive summary)

Phase-0 的目标是把当前散落的实验脚本压缩成 **一个确定性、可复现、被 agent 反复调用的 harness
API**——它就是整个 agentic loop 的"环境"。核心是 **信任边界**:LLM 只提议 artifact(twin、精度策略),
**harness 拥有每一个数字**(measured error / ρ / budget / latency / SOL / 接受判定)。本文档定义这个 API
的契约、数据结构(JSON schema)、确定性保证(bf16/fp32/fp64 参考、TF32 off)、twin 注册表、ρ-gated 升级、
以及如何用 `/loop` skill 持续迭代。首个目标算子是 DSV4 MoE grouped GEMM(hidden=4096, intermediate=2048,
256 experts, top_k=6),BS 范围 1–512。

---

## 1. Why a single harness API is Phase 0

The prior `research_loop/` produced 13 strong experiments, but each is a self-contained script
that re-reads files, re-implements references, and prints prose numbers. For an **LLM-native**
build that is the wrong granularity: an agent cannot iterate against 13 scripts, only against an
**environment** it can call with structured inputs and that returns structured ground truth.

Phase 0 therefore builds the environment. This mirrors how Claude Code itself works — the model
proposes tool calls, a deterministic environment (filesystem, compiler, tests) returns truth, and
the model iterates on truth. Here the "environment" is `harness.measure(...)`.

```
        ┌─────────────────────────── trust boundary ───────────────────────────┐
LLM /   │  proposes:  twin code · dual rules · precision policy · next shape      │
agent   │  reads:     measured error · rho · budget · latency · SOL · accept      │
        └──────────────────────────────┬─────────────────────────────────────────┘
                                        │  (never asserts a number)
                                        ▼
                         ┌──────────────────────────────┐
                         │   harness.measure(request)    │  ← deterministic, owns every number
                         └──────────────────────────────┘
```

## 2. The trust boundary (non-negotiable invariant)

| Side | May produce | May NEVER produce |
|---|---|---|
| **LLM / proposer** | twin source, dual-propagation rules, candidate precision policies, demotion order, next (shape, dist, policy) to probe, prose explanations of *why* | any numerical result, any pass/fail verdict, any "rho is small" claim |
| **Harness (this API)** | measured error, ρ, per-source budget, budget cosine, twin fidelity, latency, SOL, flip-risk, accept/reject | heuristic judgement, model opinions |

Every number that appears in a report or proposal edit must be traceable to one `MeasureResult`
emitted by a committed, re-runnable harness call. This is the same guardrail as
`research_loop/PROGRAM.md` ("the harness owns every number"), now enforced by a typed API rather
than convention.

## 3. The core API

A single entry point, plus thin convenience wrappers. Pure function of its request +
pinned RNG seed → fully reproducible.

```python
# dsv4_moe_harness/harness.py

def measure(request: MeasureRequest) -> MeasureResult:
    """The one call. Deterministic given (request, seed). Owns every number.

    Pipeline:
      1. materialize inputs for `shape` under `distribution` (seeded)
      2. compute high-precision reference  Y_ref   (ref_dtype: bf16 | fp32 | fp64)
      3. run the candidate low/mixed-precision REAL path (twin or real kernel)
      4. run the dual twin to get per-source predicted error channels
      5. compute measured / predicted / residual / rho / budget / fidelity
      6. optionally measure silicon latency + SOL for `policy`
      7. emit a single typed, JSON-serializable MeasureResult
    """
```

### 3.1 Request

```python
@dataclass(frozen=True)
class Shape:
    name: str                 # "decode_bs32_fc1"
    gemm: str                 # "FC1" | "FC2"
    M_total: int              # routed token-expert pairs on this rank
    K: int                    # 4096 (FC1) | 2048 (FC2)
    N: int                    # 4096 (FC1 gate_up fused) | 4096 (FC2)
    n_groups: int             # local experts (= 256 / EP)
    group_sizes: tuple[int]   # per-expert M (sums to M_total); None → balanced
    phase: str                # "decode" | "prefill"

@dataclass(frozen=True)
class PrecisionPolicy:
    ab_format: str            # "bf16" | "mxf8" | "nvf4" | "mxf4"
    sf_dtype: str | None      # "e8m0" | "e4m3" | None
    sf_vec_size: int | None   # 16 (nvf4) | 32 (mxf8/mxf4) | None
    acc_dtype: str            # "fp32" (kernel-fixed) — tracked for completeness
    out_dtype: str            # "bf16" | "fp16" | "fp8_e4m3"
    # per-source tags the twin must carry for this policy:
    source_tags: tuple[str]   # ("A_input_round","B_input_round","mma_accum",
                              #  "swiglu_epi","D_store_round", ...)

@dataclass(frozen=True)
class MeasureRequest:
    kernel: str               # "dsv4_moe_grouped_gemm"
    twin: str                 # registered twin id (see §6); "" → real-kernel-only
    shape: Shape
    policy: PrecisionPolicy
    distribution: str         # "normal" | "shifted" | "laplace" | "outlier_channel" | "rmsnorm_real"
    ref_dtype: str            # "bf16" | "fp32" | "fp64"
    seed: int                 # pinned; result is a pure function of (request, seed)
    measure_latency: bool     # if True and on real HW, also time the silicon kernel
    escalation: str           # "none" | "cross_term" | "interval" | "stochastic" | "ablation"
```

### 3.2 Result (the JSON contract)

```python
@dataclass
class SourceBudget:
    source: str               # tag, e.g. "B_input_round"
    l2: float                 # ‖dual channel‖_2
    relative_impact: float    # ‖Y_dual,i‖ / ‖Y_real‖
    budget: float             # ‖Y_dual,i‖ / Σ_j ‖Y_dual,j‖
    max_abs: float

@dataclass
class MeasureResult:
    # --- identity / reproducibility ---
    request: dict             # echoed MeasureRequest
    harness_version: str
    git_commit: str
    timestamp_utc: str        # stamped by caller, not inside the script

    # --- accuracy (the validation core) ---
    measured_rel: float       # ‖Y_ref − Y_real‖ / ‖Y_ref‖
    predicted_rel: float      # ‖Σ dual channels‖ / ‖Y_ref‖
    rho: float                # ‖measured − predicted‖ / ‖measured‖   (trust gate)
    higham_mu_F: float | None # GEMM backward error ‖D−D̂‖_F / ‖|A||B|‖_F (cancellation-robust)
    cos_pred_measured: float  # direction agreement

    # --- attribution ---
    budget_per_source: list[SourceBudget]
    cos_vs_reference: float | None   # budget cosine vs leave-one-out / Shapley (if computed)
    ranking_vs_reference: list[str] | None

    # --- twin fidelity (does the twin stand in for silicon?) ---
    twin_fidelity: float | None      # ‖Y_twin − Y_silicon‖ / ‖Y_silicon‖
    noise_floor: float | None        # kernel fp-compute floor; rho meaningful only above it

    # --- non-smooth guard ---
    flip_risk: float | None          # fraction of elements within decision margin (swiglu_limit clamp, etc.)
    escalation_used: str

    # --- performance (secondary; gates proposals, not accuracy) ---
    latency_us: float | None
    sol_pct: float | None
    arithmetic_intensity: float | None
    roofline_regime: str | None      # "memory" | "launch" | "compute"

    # --- verdict (computed by rule, not by model) ---
    accepted: bool | None            # measured_rel ≤ budget ∧ speedup target / SOL ceiling
    notes: list[str]                 # machine-generated flags (e.g. "rho below noise floor")
```

Example serialized result (the shape an agent reads back):

```json
{
  "request": {"kernel":"dsv4_moe_grouped_gemm","shape":{"name":"decode_bs32_fc1","M_total":192,"K":4096,"N":4096},
              "policy":{"ab_format":"nvf4","out_dtype":"bf16"},"distribution":"outlier_channel","ref_dtype":"fp32"},
  "measured_rel": 1.31e-1, "predicted_rel": 1.28e-1, "rho": 1.9e-2, "higham_mu_F": 3.4e-3,
  "budget_per_source": [
    {"source":"B_input_round","relative_impact":1.1e-1,"budget":0.86,"max_abs":2.3e-2},
    {"source":"A_input_round","relative_impact":1.4e-2,"budget":0.11,"max_abs":4.1e-3},
    {"source":"mma_accum","relative_impact":7e-6,"budget":0.00,"max_abs":9e-5}
  ],
  "twin_fidelity": 1.6e-7, "noise_floor": 5e-6, "flip_risk": 0.0,
  "latency_us": null, "sol_pct": null, "roofline_regime": "memory",
  "accepted": true, "notes": ["B dominates budget 0.86 → B precision is the lever"]
}
```

### 3.3 Convenience wrappers (sugar over `measure`)

```python
sweep(requests: list[MeasureRequest]) -> list[MeasureResult]   # parallel-friendly, deterministic
exhaustive_accuracy(shape, policies, dist, seed) -> Table       # ground-truth accuracy axis
attribution(shape, policy, dist, seed) -> list[SourceBudget]    # one-pass dual ranking
verify_twin(twin, shape, policy) -> twin_fidelity               # twin vs real cute_dsl kernel
```

## 4. Determinism & reference correctness (hard requirements)

Lessons already paid for in `research_loop` are baked in as preconditions:

1. **TF32 off for every reference.** `torch.backends.cuda.matmul.allow_tf32 = False`,
   `…cudnn.allow_tf32 = False`. A silent `fp32@fp32`-in-TF32 invalidates the reference (iter1
   finding #2). `ref_dtype="fp64"` is available for the twin cross-check.
2. **Native reference for the real kernel is bf16** (the kernel's working type); `fp32`/`fp64`
   references are for the *twin*, with `twin_fidelity` bridging twin↔silicon.
3. **Pinned seed.** `(request, seed)` fully determines outputs. No `Date.now()`/unseeded RNG
   inside the measured path.
4. **Real scale capture.** The injected residual `δz` is computed under the *real* (per-block /
   per-channel) quantization scale the kernel uses, so attribution is granularity-correct
   (iter6: dual is granularity-agnostic when δz uses the real scale).
5. **Noise-floor gating.** `rho` is reported but flagged meaningful only where
   `rho_twin · measured > twin_fidelity` (iter5/7 caveat) — encoded in `notes`, not hidden.

## 5. Twin ↔ silicon: how the real cute_dsl kernel enters the loop

Three execution backends behind the same API, chosen by `MeasureRequest.twin` / `measure_latency`:

| Backend | Role | When |
|---|---|---|
| **numpy/torch twin** | source-tagged dual channels, fast, fp64-checkable | attribution + ρ on any host |
| **real `Sm100BlockScaledContiguousGroupedGemmKernel`** | silicon ground truth for accuracy + latency | when CUDA + B200/B300 present |
| **`torch._scaled_mm` / fake-quant** | cheap silicon-ish accuracy for formats without a direct call | bridge / sanity |

The twin must reproduce the kernel's **primitive sequence**: TMA-loaded A/B with block scales →
tcgen05 MMA (fp32 acc) → SwiGLU epilogue (FC1) / store (FC2). `twin_fidelity` quantifies the gap;
the loop only trusts twin attribution where fidelity ≪ measured error.

## 6. Twin registry (decoupling twin drift from the API)

Twins are registered, not hard-coded, so regeneration (Phase 2 auto-twin synthesis) is a
drop-in:

```python
TWIN_REGISTRY = {
  "moe_grouped_gemm_v1": MoEGroupedGemmTwin,   # hand-written, primitive-aligned
  # "moe_grouped_gemm_auto": <LLM/AST-generated>,  # Phase 2 — same interface, regenerated
}
```

A twin implements one contract:

```python
class Twin(Protocol):
    def real_and_dual(self, A, B, policy, shape) -> tuple[Y_real, dict[tag, Y_dual]]: ...
    def primitives(self) -> list[str]   # for audit vs the production kernel
```

This is exactly where the proposal's "twin auto-synthesis" plugs in: swap the registry entry, the
harness re-validates `twin_fidelity` and `rho` automatically — twin maintenance becomes
*regenerate-and-revalidate*.

## 7. ρ-gated escalation (the trust mechanism, not just a self-check)

`measure` consults `escalation`; the loop escalates only where the cheap point dual is flagged:

```
point dual  ──►  rho small & flip_risk low  ──►  trust attribution
     │
     └──► rho large (bilinear)      ──► escalation="cross_term"  (add A×B channel; exact for GEMM)
     └──► flip_risk high (swiglu clamp / argmax) ──► "interval" (rigorous bound) | "stochastic" (magnitude)
     └──► rho large & non-smooth     ──► "ablation"   (exact leave-one-out, N passes; ground truth)
```

For this MoE GEMM the dominant escalation is **`cross_term`** (double-sided fp4/fp8 drops the
`δA·δB` term — the measured ~1.9 % residual). SwiGLU's `swiglu_limit=10` clamp is the non-smooth
node `flip_risk` watches.

## 8. How `/loop` drives it (continuous iteration)

`PROGRAM.md` holds the gap board + acceptance gate. One `/loop` tick = one iteration:

1. **PROPOSE** (LLM) — pick the highest-leverage gap; state a falsifiable hypothesis + the
   `MeasureRequest`(s) that test it; pre-register the threshold.
2. **EXECUTE** (harness) — `sweep([...])`; deterministic, minutes-scale; shrink shapes not rigor.
3. **EVALUATE** (harness) — the typed `MeasureResult` is the decision metric. LLM never asserts.
4. **DECIDE** — accept/reject by the gate → RESEARCH_LOG row (+ folded proposal edit only if a
   claim changed). Negative results are kept (cf. MQA "40 % unreachable").
5. **REPEAT** — `ScheduleWakeup` / `/loop` re-fires; rotate or deepen the gap.

The dynamic `/loop` (self-paced) fits: each tick is a self-contained harness sweep that notifies
on completion; the loop schedules the next PROPOSE.

## 9. Build order for Phase 0 (concrete, testable steps)

| Step | Deliverable | Done-when (harness-checked) |
|---|---|---|
| 0.1 | `harness.py` skeleton + dataclasses + JSON (de)serialize | round-trips a `MeasureResult` |
| 0.2 | `MoEGroupedGemmTwin` (FC1+SwiGLU, FC2), source tags A/B/mma_accum/swiglu_epi/store | dual vs fp64 first-order < 1e-6 on bf16 inputs |
| 0.3 | reference + real path (bf16 ref, TF32 off; fake-quant nvf4/mxf8/mxf4) | `exhaustive_accuracy` table for one shape reproduces prior NVF4 vs MXF8 finding |
| 0.4 | metrics: measured/predicted/rho/Higham μ_F/budget/cosine | `rho ≤ 2e-2` on benign nvf4; budget cosine ≥ 0.999 vs leave-one-out |
| 0.5 | silicon hook: drive real `Sm100…GroupedGemmKernel` + latency/SOL | `twin_fidelity ≤ 1e-3` twin vs kernel; latency reproducible per-count |
| 0.6 | ρ-gated escalation (`cross_term` first) | cross-term cuts double-fp4 rho ~10× |
| 0.7 | `PROGRAM.md` gap board + first `/loop` tick wired | one accepted/rejected RESEARCH_LOG row, end-to-end |

Steps 0.1–0.4 need no GPU (host twin + fake-quant); 0.5–0.6 need B200/B300. This lets the agent
make real progress headless and gate the silicon-only steps behind hardware availability.

## 10. What this explicitly is and is NOT

- **IS:** a deterministic, typed environment that turns dual-number error tracing for the DSV4 MoE
  GEMM into a single re-runnable call an agent can iterate against, with the trust boundary enforced
  by the type system.
- **IS NOT:** a modification of the production kernel, a full-engine integration, or a tool that
  ever lets the LLM assert a numerical result. Those are later phases (3–5 of the build plan).

The defensible Phase-0 claim: *a primitive-level error-attribution environment for the DSV4 MoE
GEMM that explains **which operand / primitive / precision decision** drives the error and **whether
a demotion can pay off** on the roofline — with every number owned by the harness.*
