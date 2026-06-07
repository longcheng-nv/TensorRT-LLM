---
name: precision-policy-search
description: >
  Attribution-guided mixed-precision policy search (the "error-indicating compiler + LLM" loop).
  Use when the user wants the fastest precision policy that holds an accuracy budget for an operator
  the mptracer harness can measure. Demotes operands in dual-attribution order, accepts by measured
  error, and SOL-gates by roofline regime. The proposer never asserts a number.
metadata:
  tool: mptracer
  trust_boundary: proposer orders knobs; harness-measured error is the only acceptance criterion
---

# Precision-policy search

Turn `2^K` blind precision trials into a `~K`-trial guided walk up the accuracy–cost Pareto front.

## Mechanism

1. **Propose** — one-pass dual **attribution** (`measure(...).budget_per_source`) ranks each operand's
   impact → demotion order (demote the lowest-budget operand first). An LLM may sit in this slot; it
   proposes the order/policy only, never a number.
2. **SOL-gate** — `roofline.demotion_worth_it(regime)`: only pursue a demotion for latency when the
   probe shape is **compute-bound**. At launch/memory-bound decode, report "SOL-gated" and stop
   (precision won't pay off there — unless the activation quant is kernel-fused, see
   `silicon-precision-oracle`).
3. **Verify** — for each candidate, `measure()` returns end-to-end `measured_rel`; accept iff
   `measured_rel <= error_budget`. Acceptance is the measured number, never a proposer claim.
4. **Refine** — greedily continue demoting while accepted; stop at the first rejection per operand.

Use `greedy_policy_search(base_req, demote_format, error_budget, sol_gate=True)` →
list of `PolicyProposal`, each carrying the `result_ref` id of the `MeasureResult` that justifies it.

## Swapping in an LLM proposer

Keep the trust boundary: give the LLM only the attribution + the measuring harness + the budget.
It outputs `PolicyProposal(policy, rationale, result_ref=<MeasureResult.id>)`. The harness validates
each proposal's measured error. Demonstrated equivalent to exhaustive search in ~K trials.

## Output

A Pareto-front of accepted policies. Pair with `silicon-precision-oracle` to replace the cost proxy
with measured latency, and to confirm whether the chosen low-precision format needs a fused
activation-quant kernel to actually win at the target batch size.
