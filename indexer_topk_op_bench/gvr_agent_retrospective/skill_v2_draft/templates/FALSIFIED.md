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
