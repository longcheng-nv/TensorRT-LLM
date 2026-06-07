# Scope — DSV4 MoE GEMM Dual-Number Mixed-Precision Harness (BS = 1…512)

> Companion to `HARNESS_API_DESIGN.md` and `PROGRAM.md`. This file fixes **what kernel,
> what shapes, what precisions, and what targets** the Phase-0 harness is built around.
> All architecture facts are read from the live editable install, cited inline.

## 1. Target operator (verified from source)

The production kernel under analysis is the CuTe-DSL block-scaled **contiguous grouped GEMM**:

- Class `Sm100BlockScaledContiguousGroupedGemmKernel`
  — `…/DSV4/TensorRT-LLM/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm.py:63`
- Driven through the cute_dsl custom op
  — `…/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`
- It is the per-expert MoE FC1/FC2 GEMM for DeepSeek-V4 (`DeepseekV4MoE`,
  `…/_torch/models/modeling_deepseekv4.py:1445`; weights `w1=gate_proj`, `w3=up_proj`,
  `w2=down_proj`, `modeling_deepseekv4.py:223-225`).

### Kernel facts (cited)

| Property | Value | Source |
|---|---|---|
| MMA | tcgen05 block-scaled UMMA (SM100/Blackwell) | `blockscaled_contiguous_grouped_gemm.py:53,212` |
| Accumulator | `Float32` | `:132` |
| MMA tiler (M,N) | configurable, e.g. `(256,128)`; 2-CTA when M=256 | `:107,133,138` |
| Pipeline | multi-stage TMA load A/B/SFA/SFB → MMA → epilogue → TMA store; `num_ab_stage / num_acc_stage / num_c_stage` | `:288-339,452` |
| Output dtype | taken from C tensor `element_type` (bf16 typical); `epilogue_op` lambda hook | `:372,409` |
| **MXF8** | A/B fp8 (E5M2 / E4M3FN) + SF **E8M0FNU**, `sv=32` | `:78` |
| **MXF4** | A/B fp4 (E2M1FN) + SF **E8M0FNU**, `sv=32` | `:79` |
| **NVF4** | A/B fp4 (E2M1FN) + SF **E8M0FNU / E4M3FN**, `sv=16` | `:80` |

"Contiguous grouped" = routed tokens for all local experts are concatenated along **M**, with
per-expert group offsets; one launch sweeps all local experts. So the GEMM's M dimension is
`M_total = (#routed token-expert pairs assigned to this rank)`, **not** per-expert M.

## 2. DSV4 MoE config (verified, `configs/deepseekv4.py`)

| Field | Value | line |
|---|---|---|
| `hidden_size` | **4096** | `:23` |
| `moe_intermediate_size` | **2048** | `:25` |
| `n_routed_experts` | **256** | `:31` |
| `num_experts_per_tok` (top_k) | **6** | `:42` |
| `n_shared_experts` | 1 | `:30` |
| `num_hidden_layers` | 43 | `:26` |
| `first_k_dense_replace` | 3 → **40 MoE layers** | `:44` |
| `hidden_act` | `silu` → SwiGLU | `:47` |
| `routed_scaling_factor` | 1.5 | `:33` |
| `swiglu_limit` | 10.0 | `:72` |
| `ep_size` (default) | 1 (production sweeps EP) | `:32` |

## 3. The two GEMMs per expert

SwiGLU MoE has two grouped GEMMs. With **gate+up fused** (`params_map: gate_up_proj =
[gate_proj, up_proj]`, `modeling_deepseekv4.py:790`):

| GEMM | Role | K | N | Notes |
|---|---|---|---|---|
| **FC1** | gate_up proj, then SwiGLU | `4096` (hidden) | `2×2048 = 4096` | output halved by SwiGLU → 2048 |
| **FC2** | down proj | `2048` (intermediate) | `4096` (hidden) | |

Per-expert weight bytes dominate decode bandwidth (this is the headline of the prior
`moe_gemm_dual_tracing` study: B weights ≈ 86 % of traffic, already at fp4).

## 4. BS = 1…512 → M_total mapping

Let `T` = number of tokens in the batch this step, `top_k = 6`, `E_local = 256 / EP` local
experts. Total routed pairs `P = T · 6`, spread (assume balanced) over `E_local` groups, so the
**mean per-group M ≈ P / E_local**, and the grouped-GEMM `M_total = P` per rank.

### Decode (1 token / request, `T = BS`)

| BS | P = BS·6 | M_total | per-group M @ EP=32 (E_local=8) | regime |
|---:|---:|---:|---:|:--|
| 1 | 6 | 6 | <1 | extreme memory/launch bound (tiny M, full weight load) |
| 8 | 48 | 48 | 6 | memory bound |
| 32 | 192 | 192 | 24 | memory bound |
| 64 | 384 | 384 | 48 | memory bound, M < MMA-tile |
| 128 | 768 | 768 | 96 | transitional |
| 256 | 1536 | 1536 | 192 | transitional → compute |
| 512 | 3072 | 3072 | 384 | approaching compute bound |

### Prefill (`T = BS · S`, here illustrate `S` folded into T)

Prefill drives `M_total` into the thousands–tens-of-thousands → **compute bound**, where fp4/fp8
SOL gains are realizable (cf. roofline iter11: fp8 win only when compute-bound).

| Effective T·6 | M_total | regime |
|---:|---:|:--|
| 768 | 768 | transitional |
| 3072 | 3072 | compute |
| 12288 | 12288 | compute bound |

**Scope decision:** sweep BS ∈ {1, 8, 32, 64, 128, 256, 512} for **decode**, plus
{128, 512, 2048} effective-token **prefill** points, on **both FC1 (K=4096,N=4096) and FC2
(K=2048,N=4096)**. EP ∈ {1, 32} to cover dense-group vs sharded-group M. This spans the full
memory-bound → compute-bound arc that decides whether a precision demotion can pay off.

## 5. Precision-policy search space

Anchored to what the kernel actually supports (§1) plus epilogue/output knobs:

- **A/B format** ∈ { bf16(ref), MXF8 (sv32), NVF4 (sv16), MXF4 (sv32) }
- **scale granularity** is intrinsic to format (sv16 vs sv32); scale dtype e8m0 vs e4m3 for NVF4
- **output dtype** ∈ { bf16, fp16, fp8_e4m3 } (epilogue store)
- **per-tensor knob**: FC1 vs FC2 may take different formats (the grouped GEMM is launched twice)

This is a small, enumerable space (≈ 4 formats × 3 outputs × 2 GEMMs), so the harness can hold
**exhaustive ground truth** for the accuracy axis and use dual attribution only to *rank* and to
*explain*, never to assert the accepted number.

## 6. Overall goal & targets (the loop's acceptance bar)

**North-star:** maintain accuracy parity with the **NVF4 baseline** while delivering the proposal's
**average ~40 % speedup** target over the current cute_dsl NVF4 GEMM — and, where that is *not*
reachable by precision alone (cf. the MQA-logits negative result), **prove the ceiling** with a
measured roofline argument instead of guessing.

Per-iteration decision metrics (identical family to the proposal's trust gauge):

1. **`rho = ‖measured − predicted‖ / ‖measured‖`** — dual-model trust gate (`measured = Y_bf16ref −
   Y_lowp`, `predicted = Σ dual channels`). Reference is bf16 here (kernel native), with an fp32/fp64
   cross-check on the twin.
2. **budget-vector cosine** vs exact leave-one-out / Shapley when a per-source ranking is claimed.
3. **twin fidelity** `‖Y_twin − Y_silicon‖/‖Y_silicon‖` — the twin must stand in for the real kernel.
4. **measured latency / SOL** on B200/B300 — gates whether a demotion is worth proposing.

**Accuracy budget:** end-to-end MoE-output relative error ≤ NVF4-baseline error (per shape).
**Acceptance:** a policy is KEPT only if (runs on real HW) ∧ (measured error ≤ budget) ∧
(measured speedup ≥ target or a proven SOL ceiling) ∧ (reproducible by re-running the script).

## 7. Out of scope (Phase 0)

- Rewriting the production kernel. The harness builds a **twin** + drives the existing kernel; it
  does not modify `blockscaled_contiguous_grouped_gemm.py`.
- Full DSV4 engine integration (single MoE layer / single grouped-GEMM is the unit here).
- CuTe/CTM native dual embedding — the prior loop proved embedding in Triton; here we drive the
  real cute_dsl kernel as the silicon oracle and trace via an aligned twin.
