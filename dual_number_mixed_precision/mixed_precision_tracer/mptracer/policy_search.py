"""Attribution-guided mixed-precision policy search (Goal-4 loop, operator-agnostic).

One-pass dual attribution ranks which operand is safest to demote; greedily demote while
measured error stays under budget. The proposer (here a deterministic greedy; an LLM Skill
plugs into the same slot) never asserts a number — acceptance is the harness-measured error.
SOL-gating (roofline) decides whether a demotion is worth proposing for latency at all.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List

from .core import measure
from .roofline import demotion_worth_it
from .types import MeasureRequest, PolicyProposal


def greedy_policy_search(
    base_req: MeasureRequest, demote_format: str, error_budget: float, sol_gate: bool = True
) -> List[PolicyProposal]:
    """Attribution-guided greedy precision demotion under an accuracy budget.

    Demote operands (A, then B) to `demote_format` in attribution order while measured
    rel error <= error_budget. Returns accepted proposals, each carrying the result id
    that justifies it. Returns [] (with a rationale) if SOL-gated out.
    """
    accepted: List[PolicyProposal] = []

    if sol_gate:
        probe = measure(base_req)
        if not demotion_worth_it(probe.roofline_regime or ""):
            return [
                PolicyProposal(
                    base_req.policy,
                    f"SOL-gated: regime={probe.roofline_regime} not compute-bound "
                    f"(demotion won't pay off)",
                    probe.id,
                )
            ]

    # attribution order: demote the LOWEST-budget operand first
    attr = measure(base_req)
    order = [
        b.source
        for b in sorted(attr.budget_per_source, key=lambda b: b.budget)
        if b.source in ("A_input_round", "B_input_round")
    ]

    policy = base_req.policy
    demoted = {"A_input_round": False, "B_input_round": False}
    for src in order:
        trial = replace(policy, ab_format=demote_format)  # demote whole op step (simplified)
        if src == "A_input_round":
            trial = replace(trial, quantize_A=True)
        r = measure(replace(base_req, policy=trial))
        if r.measured_rel <= error_budget:
            policy = trial
            demoted[src] = True
            accepted.append(
                PolicyProposal(
                    trial,
                    f"demote {src} ({demote_format}); "
                    f"measured_rel={r.measured_rel:.3e} <= {error_budget}",
                    r.id,
                )
            )
        # else: stop demoting this operand (rejected by measured error)
    return accepted
