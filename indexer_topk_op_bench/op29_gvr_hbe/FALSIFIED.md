# FALSIFIED — campaign falsification ledger

> Checkpoint rule (SKILL Phase 2.5): before implementing ANY hypothesis, grep
> this file and WALLS.md. On a hit, cite the revival condition or drop the idea.
> Entries are scoped: a falsification can be conditional (e.g. "noise at fp32,
> a 1.08-1.14x WIN at 16-bit") — record the domain, not just the verdict.

| # | Hypothesis | Conclusion | Condition domain (K/N/dtype/BS/arch) | Evidence strength | Root cause | Revival condition |
|---|---|---|---|---|---|---|
| 1 | <what was proposed> | FALSIFIED / conditional | <where it holds> | host / event / nsys / NCU | structural-wall / measurement-artifact / complexity-backfire | <what would have to change> |

## Root-cause class reference
- **structural-wall**: physics/architecture forbids it (occupancy structure,
  pass-count floor, phase-chain latency). Move the entry's wall to WALLS.md.
- **measurement-artifact**: the "win" or "loss" was the harness lying
  (instrumented baseline, event bias, thermal drift, anchor noise). Record the
  artifact in the Anti-Patterns catalog if new.
- **complexity-backfire**: mechanism real, but coordination/register/barrier
  cost exceeds the saving (e.g. reg spill, extra barrier pairs).

## Seed red-lines (imported from June campaign ledger, 2026-07-13)
- (smem row-residency saves nothing, {BS=1, N<=262K, fp32/16bit, B200}, nsys+warm-L2) structural-wall — op15; warm-L2 still slower.
- (P3-into-P2 online slot-reserve fusion, {all}, nsys) complexity-backfire — Opt-L; ballot+popc+shfl+atomicAdd chain ~= a full pass.
- (cluster DSM at high BS, {BS>~SMs}, nsys) structural-wall — Opt-B/Q5f 0.36-0.45x; GPC wave-cap contention.
- (P2 multi-threshold / M-ary refine, {all}, nsys) complexity-backfire — Opt-F + op8 + op27 ledger #3; secant/logfalsi already ~1 pass.
- (P4-internal reseed / fine-hist / interp, {all}, nsys) complexity-backfire — p4_recursive_digit; P4 is barrier-bound.
- (P1 model-driven seed, {all}, 91k-cell host+silicon) structural-wall — drift ~symmetric, unfixable at P1.
- (torch.randn for 16-bit selection inputs, {bf16/fp16}, gate) measurement-artifact — collapses to ~256 levels, tie storms.
- (event-axis ship claims, {all}, protocol) measurement-artifact — >=5 fabricated wins in record; nsys only.
