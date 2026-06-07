---
name: autoresearch-loop
description: >
  Run the mptracer error-tracing / precision-optimization work as an autonomous propose→measure→
  decide loop with a gap board and acceptance gate. Use when the task is multi-step exploration of a
  kernel's precision behavior (not a single query) and the user wants iterative, self-documenting
  progress where every kept claim is backed by a harness number. Pairs with the /loop command.
metadata:
  tool: mptracer
  artifacts: PROGRAM.md (gap board + gate), RESEARCH_LOG.md (append-only journal)
---

# Autoresearch loop protocol

Faithful to propose → execute on a fixed budget → evaluate one objective metric → keep/discard →
repeat. The harness owns every number; negative results are kept (they are evidence).

## One iteration

1. **PROPOSE** — pick the top OPEN gap from `PROGRAM.md`'s Gap Board. State a **falsifiable**
   hypothesis and the exact `MeasureRequest`(s) that test it. **Pre-register the threshold before
   running.**
2. **EXECUTE** — run `measure()`/`greedy_policy_search()` (host needs no GPU; silicon iters run on a
   FREE GPU via `CUDA_VISIBLE_DEVICES`, check `nvidia-smi`, never a busy one). Keep it minutes-scale:
   **shrink shapes, not rigor** — but validate nonlinear-path residuals at the real contraction dim
   (they are not dimension-invariant).
3. **EVALUATE** — the typed `MeasureResult` IS the decision metric (`rho`, budget cosine,
   `twin_fidelity`, latency/SOL). Never assert a number yourself.
4. **DECIDE — keep/discard/park** by the gate: KEPT iff (runs / explicitly design-only) ∧
   (pre-registered threshold met) ∧ (re-running reproduces it). Append one `RESEARCH_LOG.md` row.
   A DISCARDED result still earns a row — and often refines the gap (a negative result is a finding).
5. **REPEAT** — update the Gap Board; rotate or deepen. Stop when all gaps are CLOSED/PARKED or no
   minutes-scale falsifiable iteration remains.

## Gap Board / acceptance gate

Maintain `PROGRAM.md` with a Gap Board (id, gap, status ∈ OPEN/CLOSED/DISCARDED/PARKED) and the
acceptance gate. Each kept result that changes a claim folds one verified number into the report —
never prose numbers without a committed re-runnable `MeasureRequest`.

## Driving it with /loop

`/loop` (dynamic mode) with `PROGRAM.md` as the steering file: each tick is one self-contained
iteration that writes its own `RESEARCH_LOG.md` row, then schedules the next. Capture session
learnings (recipes, dimension-dependence caveats, negative results) so they accrue across runs.
