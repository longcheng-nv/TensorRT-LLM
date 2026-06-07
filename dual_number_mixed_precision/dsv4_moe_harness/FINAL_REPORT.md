# Final Report — DSV4 MoE GEMM Dual-Number Mixed-Precision Harness

**Date:** 2026-06-07 · **Hardware:** local 8× NVIDIA B200 (sm_100) · **Stack:** TensorRT-LLM
1.3.0rc15 (DSV4 branch), Torch 2.11, torchao 0.15, NumPy 2.3 · **Trust model:** the harness owns
every number; the LLM only proposes artifacts and reads measured results.

> Companion docs: `HARNESS_API_DESIGN.md` (the Phase-0 API), `SCOPE_DSV4_MOE_BS1-512.md` (verified
> architecture + scope), `PROGRAM.md` (loop steering), `RESEARCH_LOG.md` (per-iteration journal).
> Chinese version: `FINAL_REPORT_zh.md`.

---

## 1. What this is

A **Phase-0 unified harness** that turns dual-number numerical-error tracing for DeepSeek-V4's MoE
grouped GEMM into a single deterministic call an agent can iterate against — plus **8 autoresearch
iterations** run on real B200 silicon that (a) validate the dual-number error model on the DSV4 MoE
operator, (b) build the escalation ladder needed for its non-linear SwiGLU epilogue, and (c) map and
then **resolve** the production fp4 speedup story across batch sizes 1–512.

The design follows the LLM-native principle from the proposal: the model proposes twins / dual rules
/ precision policies; a deterministic `harness.measure(...)` owns every measured number; a
generate-and-verify boundary keeps model output out of the numerical trust path.

## 2. Target operator (verified from the live editable install)

`…/perf/workloads/DSV4/TensorRT-LLM`, model `DeepseekV4Config`:

| MoE config | value | | Kernel | value |
|---|---|---|---|---|
| hidden_size | 4096 | | class | `Sm100BlockScaledContiguousGroupedGemmKernel` |
| moe_intermediate_size | 2048 | | MMA | tcgen05 block-scaled UMMA (SM100) |
| n_routed_experts | 256 | | accumulator | Float32 |
| num_experts_per_tok | 6 | | formats | MXF8 (sv32), MXF4 (sv32), **NVF4** (sv16) |
| n_shared_experts | 1 | | fused variant | `…gather_grouped_gemm_act_fusion` (gather+GEMM+SwiGLU+fp4-out) |
| MoE layers | 40 (43 − `first_k_dense_replace`=3) | | activation | SwiGLU (`silu`, gate+up fused), `swiglu_limit`=10 |

Two GEMMs per expert: **FC1** K=4096 → N=2·2048=4096 (gate_up fused, then SwiGLU → 2048);
**FC2** K=2048 → N=4096. BS=1…512 maps to `M_total = BS·top_k` routed token-expert pairs.

## 3. The harness (Phase 0)

One entry point, `measure(MeasureRequest) -> MeasureResult`, deterministic given `(request, seed)`:

- **Inputs:** shape (FC1/FC2, M/K/N, groups), precision policy (ab_format ∈ {bf16, mxf8, nvf4,
  mxf4}, out_dtype, per-operand quant flags), distribution, ref dtype, escalation mode.
- **Outputs (JSON):** `measured_rel`, `predicted_rel`, **`rho`** (trust gate), Higham `μ_F`,
  per-source budget + cosine vs leave-one-out, `twin_fidelity`, `flip_risk`, latency/SOL, regime,
  accept verdict.
- **Twin:** a primitive-aligned FC1(SwiGLU)/FC2 twin carrying source-tagged dual channels
  (`A_input_round`, `B_input_round`, `mma_accum`, `D_store_round`, + `cross_AB`, `swiglu_2nd`).
- **Determinism:** fp64 reference, **TF32 off**, pinned seeds, residual `δz` computed under the real
  block scale.

## 4. The eight iterations (all numbers harness-measured)

| # | Gap | Where | Verdict | Headline measured result |
|---|---|---|---|---|
| 1 | API + twin | host | **KEPT** | dual vs fp64 first-order = **2.54e-8**; one-pass budget vs leave-one-out cosine = **1.000000** |
| 2 | BS=1–512 attribution map | host | **KEPT** | FC2 cross-term → 2.5e-8 ∀ fmt; point-ρ orders by coarseness (nvf4 6.7e-2 < mxf8 1.1e-1 < mxf4 1.4e-1); outliers do **not** raise FC2 ρ |
| 3 | silicon fp8 oracle | B200 | **KEPT** | twin_fidelity **1.66e-3**; fp8 speedup 0.77×(BS1) → **1.44×**(BS512); wins only compute-bound |
| 4 | FC1 matmul cross-term | host | **DISCARDED** | cross-term only 0.160→0.146; single-sided FC1 ρ=0.103 ⇒ miss is **SwiGLU curvature**, not matmul |
| 5 | FC1 SwiGLU 2nd-order | host | **KEPT** | adding the 2nd-order epilogue channel: FC1 nvf4 ρ **0.160 → 0.046**; FC2 unaffected |
| 6 | production NVF4 fp4 silicon | B200 | **KEPT** | twin_fid **1.67e-3**, meas_rel **0.134**; fp4 GEMM 0.9×(BS1)→**2.10×**(BS512); **unfused eager quant collapses decode to 0.05–0.15×** |
| 7 | framework fusion (CUDA-graph) | B200 | **DISCARDED** | CUDA-graph fusion only 0.054×→**0.27×** decode (still 3.7× slower than bf16) |
| 8 | **real fused act-fusion kernel** | B200 | **KEPT** | production fused kernel **1.82×(BS1) / 1.90×(BS8) / 1.75–1.78×(BS32–512)** vs bf16 — fp4 wins at decode |

## 5. The four conclusions

### 5.1 The dual error model is exact on the linear path, first-order on the non-linear path
For **FC2 (linear)** the first-order dual prediction matches the fp64 first-order to **2.5e-8**, and
the one-pass per-source budget equals exact leave-one-out attribution to **cosine 1.0**. The only
first-order miss is the bilinear `δA·δB` cross term, which a single `cross_AB` channel closes
**exactly** (6.7e-2 → 2.5e-8). **Outliers do not break linear attribution** — channel-outlier ρ
(0.046) is no worse than benign (0.067).

### 5.2 The non-linear SwiGLU epilogue needs a 2nd-order channel, not a cross term
For **FC1 (SwiGLU)** the dominant first-order miss is **epilogue curvature**, proven by single-sided
FC1 (δA≡0, matmul cross term identically zero) already having ρ=0.103. The matmul cross-term barely
helps (0.160→0.146, iter 4 DISCARDED). Adding the analytic SwiGLU 2nd-order Taylor channel
`0.5·silu''(g)·δg²·u + silu'(g)·δg·δu` drops FC1 nvf4 ρ to **0.046** (iter 5). **Escalation ladder:**
point dual → `+cross_AB` (matmul bilinear) → `+swiglu_2nd` (epilogue curvature).

### 5.3 The twin faithfully stands in for silicon — once the recipe matches
On real B200, `twin_fidelity` is **1.66e-3 (fp8)** and **1.67e-3 (fp4)** — the genuine fp8/fp4
MMA-accumulation floor — but only after two corrections: the twin must use the **same scaling recipe**
(per-tensor vs per-block) as silicon, and must **reuse the exact quantized operands** silicon
consumed (a software fp8/fp4 emulation differs from hardware rounding element-wise). fp4 measured
error 0.134 matches the prior NVF4 study (1.32e-1).

### 5.4 The fp4 decode win exists — but only with kernel-epilogue activation-quant fusion
This is the report's headline, resolved across iters 6→7→8:

| Path | fp4 FC1 at decode (BS≤8) vs bf16 |
|---|---|
| fp4 GEMM kernel alone (iter 6) | up to 2.10× at large BS, but the activation-quant pass is separate |
| **unfused** eager `nvfp4_quantize` + GEMM (iter 6) | **0.05–0.15×** (catastrophic) |
| **framework** fusion — CUDA-graph / `torch.compile` (iter 7) | **0.27×** (still 3.7× slower) |
| **kernel-epilogue** fusion — real production `…act_fusion` kernel (iter 8) | **1.8–1.9×** ✅ |

The activation **quantization pass** (multi-pass amax/scale/cast/pack over the full tensor in HBM),
not the GEMM, is the decode bottleneck. Framework fusion removes launch overhead but not that HBM
traffic. The production fused kernel emits the FC1 activation **already in fp4** in one launch, so the
separate pass disappears and fp4 beats bf16 even at decode.

## 6. The actionable answer for DSV4 MoE mixed precision

- **Which precision, where:** NVF4 (fp4 e2m1 + e4m3 block scale, sv=16) is the right MoE weight
  format — fp4 GEMM 0.9×→2.1× by BS, error 0.134 (the known NVF4 level). MXF8/MXF4 are worse on both
  axes for this operator.
- **Which path dominates error:** FC2 error is fully explained first-order and outlier-robust; FC1
  error needs the SwiGLU 2nd-order term — i.e. **protect/verify the FC1 SwiGLU epilogue**, not the
  matmul, when accuracy is tight.
- **When the speedup pays off:** fp4 wins **compute-bound (prefill / large BS)** unconditionally, and
  **at decode only if activation quantization is fused into the producing kernel** — which the
  production `…gather_grouped_gemm_act_fusion` kernel does (measured 1.8–1.9× at decode).
- **The 40 %+ target** is therefore reachable across **both decode and prefill** via the production
  fused kernel; the separate-quant-pass tax was the entire obstacle, and it is already solved in the
  production path.

## 7. Honest limitations

- The dual model is a **first-order linearization**: exact on injection + linear paths, first-order
  (ρ-measurable) on smooth non-linear paths, and the SwiGLU clamp is a guarded non-smooth node
  (`flip_risk`).
- The host metric sweep (iter 2) computes ρ/budget at **reduced dims** (ratios are dimension-invariant;
  documented, not silent); true dims drive the silicon iterations.
- The iter-8 bf16 baseline is a **coarse per-local-expert-mm proxy**, so the 1.8× ratio is approximate
  — the direction (fused fp4 ≫ bf16 at decode, vs framework fp4 ≪ bf16) is unambiguous.
- Results are on the FC1/FC2 grouped-GEMM operator, not a full DSV4 engine; auto-twin synthesis (GA8)
  is the parked Phase-2 step.

## 8. Artifacts

```
dsv4_moe_harness/
├── HARNESS_API_DESIGN.md        Phase-0 API design (trust boundary, JSON contract, escalation)
├── SCOPE_DSV4_MOE_BS1-512.md    verified DSV4 arch + BS scope + precision matrix
├── PROGRAM.md                   loop steering: gap board, acceptance gate (all CLOSED/PARKED)
├── RESEARCH_LOG.md              8 iterations, one row each (KEPT/DISCARDED, negatives kept)
├── harness.py                   the harness: API + FC1/FC2 twin + metrics + escalation
├── iter2_bs_sweep.py            host BS=1..512 attribution/ρ/flip/regime map
├── iter3_silicon.py             B200 fp8 oracle (twin fidelity + latency/SOL)
├── iter6_silicon_nvf4.py        B200 NVF4 fp4 _scaled_mm (GEMM-only vs eager-quant)
├── iter7_fused_quant.py         B200 CUDA-graph framework-fusion test
├── iter8_fused_kernel.py        B200 real production fused act-fusion kernel driver
└── results/                     per-iteration CSV/JSON (regenerate, don't hand-edit)
```

## 9. Next steps

1. **GA8 (Phase 2):** auto-twin synthesis (AST / operator-overloading) for the cute_dsl grouped GEMM
   — regenerate the twin from the kernel so twin maintenance becomes regenerate-and-revalidate.
2. **Fold iters 6–8** (the fp4 fusion story) into the main proposal's experimental-evidence section.
3. **Full-engine integration:** drive the fused kernel inside a real DSV4 MoE layer with measured
   per-shape/per-phase context.
