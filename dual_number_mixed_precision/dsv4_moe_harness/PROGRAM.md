# PROGRAM.md — DSV4 MoE GEMM Mixed-Precision Autoresearch Loop

> Steers an autonomous `/loop`-driven research loop for the DSV4 MoE grouped GEMM, against the
> Phase-0 harness (`HARNESS_API_DESIGN.md`). This file is the **only** place strategy lives.
> The loop never edits the proposal except to fold in numbers a `MeasureResult` confirmed.
> Sibling of `research_loop/PROGRAM.md`; same protocol, new target operator.

## North-star objective

Build and exercise the **Phase-0 unified harness** for DeepSeek-V4's MoE contiguous grouped GEMM
(`Sm100BlockScaledContiguousGroupedGemmKernel`), then use it to find, **with every number owned by
the harness**, the mixed-precision policy that holds NVF4-baseline accuracy while approaching the
~40 % speedup target across **BS = 1…512** — or to *prove the ceiling* where precision alone cannot
reach it (a valid, valuable negative result, cf. the MQA-logits study).

Scope, shapes, formats, targets: `SCOPE_DSV4_MOE_BS1-512.md`.

## The loop (one `/loop` tick = one iteration)

1. **PROPOSE** — pick the top Gap-Board item. State a falsifiable hypothesis and the exact
   `MeasureRequest`(s) that test it. **Pre-register the threshold before running.**
2. **EXECUTE** — `harness.sweep([...])` on the real B200/B300 (or host for 0.1–0.4 steps).
   Budget: minutes, not hours. If it won't fit, shrink shapes, not rigor.
3. **EVALUATE** — the typed `MeasureResult` *is* the decision metric. The LLM never asserts a
   number; the harness prints it.
4. **DECIDE — keep / discard / park** by the acceptance gate. A kept result earns a RESEARCH_LOG
   row and (only if it changes a claim) a folded proposal edit. Negative results are kept.
5. **REPEAT** — update the Gap Board; `ScheduleWakeup`/`/loop` re-fires the next PROPOSE.

## Decision metric (fixed across iterations)

Primary (from the harness `MeasureResult`):
- **`rho`** = ‖measured − predicted‖ / ‖measured‖  (trust gate; lower better)
- **budget cosine** vs exact leave-one-out / Shapley (when a ranking is claimed)
- **twin fidelity** ‖Y_twin − Y_silicon‖ / ‖Y_silicon‖ (when the twin stands in for the kernel)

Secondary (gating only for perf claims): measured latency, SOL %, roofline regime.

## Acceptance gate (keep/discard)

KEPT iff: (1) runs on real HW (or explicitly design-only), AND (2) the pre-registered threshold is
met, AND (3) re-running the committed harness call reproduces it. Else DISCARDED or PARKED
(hypothesis logged, no proposal edit). A discarded result still earns a RESEARCH_LOG row.

## Gap Board (priority queue)

| ID | Gap | Phase-0 step | Status |
|----|-----|------|--------|
| GA1 | No single typed harness; experiments are scattered scripts | 0.1 | **CLOSED (iter 1)** — `harness.py` typed API + JSON |
| GA2 | No primitive-aligned MoE twin (FC1+SwiGLU, FC2) with source tags | 0.2 | **CLOSED (iter 1)** — dual vs fp64 = 2.5e-8; budget cos = 1.0 |
| GA3 | No reference/real path with TF32-off + fake-quant nvf4/mxf8/mxf4 | 0.3 | **CLOSED-host (iter 1)** — fp64 ref + block-scaled fake-quant; silicon ref still GA5 |
| GA4 | Metrics (measured/predicted/rho/Higham μ_F/budget/cosine) not unified | 0.4 | **CLOSED (iter 1)** — all in `MeasureResult` |
| GA5 | Drive a real silicon fp8 GEMM as oracle (twin fidelity + latency/SOL) | 0.5 | **CLOSED (iter 3, local B200)** — twin_fid 1.66e-3 (fp8-accum floor); fp8 speedup 0.77×→1.44× by BS |
| GA6 | No ρ-gated cross-term escalation for double-sided fp4/fp8 | 0.6 | **CLOSED (iter 1)** — FC2 rho 6.7e-2 → 2.5e-8; FC1 SwiGLU cross-term deferred |
| GA7 | BS=1…512 sweep + roofline regime per shape | 0.5/0.7 | **CLOSED** — host map (iter 2) + silicon latency/SOL arc (iter 3) |
| GA8 | Auto-twin synthesis (regenerate from kernel) — Phase 2, deferred | — | PARKED |
| GA9 | FC1 matmul cross-term δA·δB through SwiGLU Jacobian | 0.6+ | **DISCARDED (iter 4)** — barely helps (rho 0.160→0.146); FC1 miss is epilogue curvature |
| GA9′ | FC1 SwiGLU **2nd-order** (hyper-dual / Taylor-2) epilogue channel — single-sided FC1 rho=0.103 is pure SwiGLU curvature | 0.6+ | **CLOSED (iter 5)** — nvf4 rho 0.160→0.046; FC2 unaffected |
| GA10 | Production path is NVF4 **fp4** weights (pre-quantized), not per-tensor fp8 — drive fp4 `_scaled_mm` / real `Sm100…GroupedGemmKernel` for the true 40 % story | 0.5+ | **CLOSED (iter 6, local B200)** — twin_fid 1.67e-3; fp4 GEMM speedup 0.9×→2.10× by BS; unfused act-quant kills decode (0.05–0.15×) |
| GA11 | Framework-level fusion (CUDA-graph/`torch.compile`) of act-quant + fp4 GEMM recovers decode | 0.5+ | **DISCARDED (iter 7)** — only 0.054×→0.27× (still 3.7× slower than bf16); HBM traffic of the quant survives framework fusion |
| GA11′ | KERNEL-EPILOGUE act-quant fusion — upstream op emits fp4 activations directly (no separate quant pass); = the production `…gather_grouped_gemm_act_fusion` kernel | 0.5+ | **CLOSED (iter 8, local B200)** — real fused kernel 1.82×(BS1)/1.90×(BS8)/1.75–1.78×(BS32–512) vs bf16; resolves iter7 (0.27× framework) — fp4 wins at decode when act-quant is kernel-fused |

## Loop status: COMPLETE (fp4 story closed end-to-end as of iter 8)

8 iterations on **local 8× B200 (sm_100)** (silicon via `CUDA_VISIBLE_DEVICES=<free gpu>`,
no Slurm). Only GA8 (auto-twin synthesis, Phase 2) remains PARKED → stop condition met.

- **CLOSED:** GA1/GA2/GA3/GA4/GA6 (iter 1) · GA7 host (iter 2) · GA5+GA7 silicon (iter 3) ·
  GA9′ SwiGLU 2nd-order (iter 5) · GA10 NVF4 fp4 silicon (iter 6) · **GA11′ real fused
  act-fusion kernel (iter 8)**.
- **DISCARDED (kept as evidence):** GA9 (matmul cross-term ≠ FC1 fix — epilogue curvature) ·
  GA11 (framework CUDA-graph fusion only 0.054×→0.27×, still <bf16).
- **PARKED:** GA8 (auto-twin synthesis, Phase 2).

**The fp4 decode story, resolved (iters 6→7→8):**
- iter 6: fp4 GEMM kernel alone 0.9×→2.10× by BS, but **unfused eager act-quant collapses
  decode to 0.05–0.15×**.
- iter 7: even CUDA-graph/`torch.compile` framework fusion only reaches **0.27×** at decode —
  the quant's HBM traffic survives.
- iter 8: the **real production kernel-epilogue-fused** `…gather_grouped_gemm_act_fusion` runs
  fp4 FC1 at **1.8–1.9× bf16 even at decode**. ⇒ **fp4 wins at decode iff activation quant is
  fused into the producing kernel.** The 40 %+ target is reachable across decode AND prefill via
  the production fused kernel — the separate-quant-pass tax was the entire obstacle.

### To resume later (Phase 2, needs explicit scoping)
- **GA8** — auto-twin synthesis (AST / operator-overloading) for the cute_dsl grouped GEMM, per
  the prior research_loop's Triton auto-tracer. Add as an OPEN Gap-Board row and re-run `/loop`.

## Guardrails

- The harness owns every number. No prose number without a committed, re-runnable `MeasureRequest`.
- Keep the predicted-vs-measured-vs-residual contract intact (per repo CLAUDE.md).
- `results/` and plots are generated — regenerate, don't hand-edit. Matplotlib forced to Agg.
- Each iteration writes its own CSV/PNG/MD under a local `results/`.
- Do NOT modify the production kernel or `dkg_latest_snapshot/`. The harness builds a twin and
  *drives* the real kernel read-only.
- Silicon-only steps (GA5/GA7) stay BLOCKED until B200/B300 is confirmed; host steps proceed.
