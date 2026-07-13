# PLAN — rmsnorm_campaign (omni-kernel v2 Tier-B trial, dense class)

## Objective triple (human-supplied 2026-07-13 via ../KICKOFF.md; agent may not relax)
```yaml
objective:
  incumbent: flashinfer.norm.rmsnorm (flashinfer 0.6.11)   # TRT-LLM production default
  rivals: [eager torch RMSNorm (fp32 upcast), torch.compile RMSNorm]
  envelope: {hidden: 7168, tokens: [1, 16, 256, 4096, 16384], dtype: [bf16], BS: n/a}
  verdict_axes: [worst, geomean, best]   # over the token grid; no real-capture axis
  ship_rule: "geomean >= 1.00 vs incumbent AND no cell < 0.98 AND exactness green
              (dense bf16 atol/rtol 1e-2) AND dispatch rules <= 3"
  hard_constraints: [CUDA-graph compatible, out-of-place, no incumbent source edits]
budget: {iterations_max: 5, wallclock_max: 2h, gpu: CUDA_VISIBLE_DEVICES=2 umbriel-b200-027}
pre_authorized_negative_conclusion: >
  If flashinfer.norm.rmsnorm remains best on the envelope, say so plainly with
  numbers. A clean FALSIFIED/INFEASIBLE verdict is a fully successful outcome.
```

## Red lines (ledger grep at kickoff)
- FALSIFIED.md / WALLS.md start EMPTY for this campaign (grep run 2026-07-13, no hits —
  this is a fresh dense-class campaign; the GVR ledgers are selection-class and their
  entries do not transfer per SKILL Phase 6 note, but M/C/S/E learnings do apply).
- Applicable cross-campaign learnings: M1-M7, C1-C3 (dense criterion), S1, E1-E7.

## Feasibility priors (Phase 1.4 — numbers filled after characterization)
- Environment: B200 SM100, 148 SMs, L2 = 132.6 MB, HBM3e peak ~8.0 TB/s.
- Traffic model (single-pass op): bytes(T) = 2 * T * 7168 * 2 B (read x + write y; w = 14 KB, negligible).
  - T=1: 28.7 KB · T=16: 459 KB · T=256: 7.34 MB · T=4096: 117.4 MB · T=16384: 469.8 MB
- Math floor: theoretical @8 TB/s: T=16384 = 58.7 µs, T=4096 = 14.7 µs. MEASURED
  same-traffic torch-elementwise ceiling (nsys x3, iter0): T=1 1.64 · T=16 2.25 ·
  T=256 3.64 · T=4096 21.54 · T=16384 75.09 µs. Incumbent nsys: 2.83 / 2.96 /
  4.04 / 21.82 / 71.79 µs → at T=4096 incumbent = 99% of ceiling; at T=16384 it
  BEATS torch's elementwise kernel (6.54 TB/s); headroom concentrates at T<=256.
- RMSNorm is information-theoretically a 1-read-1-write op; NO pass-count lever exists.
  Any win must come from launch config / occupancy / vectorization / latency, not traffic.
- L2-trap note: not applicable in its op14 form (no re-read passes to save), but
  T<=4096 fits in L2 → cold-vs-warm gap expected large at mid cells.
- Occupancy structure: grid is 1 CTA/row for both incumbent and naive candidate →
  T=1 covers 1/148 SMs, T=16 covers 16/148 → occupancy STRUCTURAL at small T;
  only more-CTAs-per-row (split-row) forms can move those cells.

## Anchor cell
- cell: T=4096, hidden=7168, bf16 · impl: incumbent (flashinfer) · nsys pure-kernel
- expected: 21.82 µs ± 3% (set at iter0 on umbriel-b200-027 GPU2, nsys x3 median)

## Probe plan (Phase 3 ladder)
| # | Hypothesis | Crux question | Rung-0 tool | GO/NO-GO criterion |
|---|---|---|---|---|
| P1 | Large-T cells are un-winnable (incumbent at BW SOL) | What % of HBM peak does flashinfer hit at T=16384? | ncu_attrib.sh on incumbent, KERNEL_REGEX set | >=90% of achievable copy BW → large-T = margin defense only |
| P2 | Small-T cells (1/16) sit on a latency/occupancy floor a Triton kernel can also hit | Incumbent kernel µs at T=1/16 vs empty-kernel + 28 KB traffic floor | nsys on incumbent + a trivial copy | if incumbent ≈ floor → small-T also margin defense; else GO split-row |
| P3 | A Triton 1-CTA/row autotuned kernel matches incumbent everywhere | full-grid L1 A/B | bench_cold.py | geomean >= 0.98 to proceed to nsys |
| P4 | torch.compile rival is not competitive (data point only) | its kernel µs on grid | bench_cold.py | informational |
