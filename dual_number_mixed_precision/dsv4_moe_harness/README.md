# dsv4_moe_harness — Phase-0 unified harness for DSV4 MoE GEMM dual-number error tracing

The **environment** an LLM/agent iterates against: one deterministic call,
`harness.measure(MeasureRequest) -> MeasureResult`, that owns every number. Targets the
DeepSeek-V4 MoE contiguous grouped GEMM (`Sm100BlockScaledContiguousGroupedGemmKernel`).

## Files

| File | What it is |
|---|---|
| `FINAL_REPORT.md` / `FINAL_REPORT_zh.md` | **The complete report (EN / 中文).** Phase-0 design + all 8 iterations + the four conclusions + the resolved fp4-decode-fusion story. |
| `HARNESS_API_DESIGN.md` | **The design doc.** Trust boundary, typed API + JSON contract, determinism, twin registry, ρ-gated escalation, `/loop` integration, build order. |
| `SCOPE_DSV4_MOE_BS1-512.md` | Verified DSV4 arch facts (cited), the two GEMM shapes, BS=1..512 → M_total map, precision-policy search space, goals/targets. |
| `PROGRAM.md` | Autoresearch loop: north-star, decision metric, acceptance gate, Gap Board, guardrails. |
| `RESEARCH_LOG.md` | Append-only journal; one row per iteration (negative results kept). |
| `harness.py` | The harness: typed API, FC1(SwiGLU)/FC2 source-tagged twin, metrics, cross-term escalation. Runs iteration-1 validation as `__main__`. |
| `iter2_bs_sweep.py` | Iteration 2: BS=1..512 × format × distribution attribution/rho/flip/regime map. |
| `results/` | Generated CSV/JSON — regenerate, don't hand-edit. |

## Run (host-only; no GPU needed for iter 1–2)

```bash
cd dsv4_moe_harness
python3 harness.py          # iteration 1: twin validation -> results/iteration1.json
python3 iter2_bs_sweep.py   # iteration 2: BS sweep        -> results/iter2_*.{csv,json}
```

## Status (see RESEARCH_LOG.md / PROGRAM.md Gap Board)

- **CLOSED (iter 1):** typed harness API (GA1), primitive-aligned MoE twin (GA2, dual vs fp64
  = 2.5e-8, budget cos = 1.0), host fake-quant + fp64 ref (GA3-host), unified metrics (GA4),
  ρ-gated cross-term (GA6, FC2 rho 6.7e-2 → 2.5e-8).
- **Mapped (iter 2):** BS=1..512 regime arc (launch/memory/compute); FC2 attribution exact;
  FC1 SwiGLU is the residual-bearing path (rho 0.16–0.48) → opened **GA9** (FC1 cross-term).
- **BLOCKED on B200/B300:** drive the real cute_dsl kernel as silicon oracle + measured
  latency/SOL (GA5), and the true-dimension BS sweep (GA7-silicon).

## Continue the loop (`/loop`)

Each iteration = PROPOSE (pick top Gap-Board item, pre-register threshold) → EXECUTE
(`harness.measure`/`sweep`) → EVALUATE (the `MeasureResult`) → DECIDE (gate → RESEARCH_LOG row).

- **Next host iteration:** GA9 — add the SwiGLU-Jacobian cross-term channel so FC1 double-fp4 rho
  drops the way FC2's did.
- **Next silicon iteration:** GA5 — once a B200/B300 node is allocated (use the
  `tmux-slurm-srun` or `computelab-sc-01-launch` skills), wire `measure(..., measure_latency=True)`
  to the real `Sm100BlockScaledContiguousGroupedGemmKernel` and validate `twin_fidelity` + SOL.

To run continuously: `/loop` with this PROGRAM.md as the steering file (self-paced; each tick is
one self-contained harness iteration that writes its own RESEARCH_LOG row).
