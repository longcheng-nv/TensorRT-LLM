"""Roofline regime classification + precision-demotion gate.

Encodes the iter 3/6/11 lesson: a precision demotion only pays off where the kernel is
compute-bound; at launch/memory-bound decode it does not (and may lose). The harness reports
the regime so the policy-search skill can SOL-gate proposals.
"""

from __future__ import annotations


def regime(M_effective: int) -> str:
    if M_effective < 16:
        return "launch"
    if M_effective < 128:
        return "memory"
    return "compute"


def demotion_worth_it(regime_str: str) -> bool:
    """A precision demotion is worth proposing for latency only when compute-bound."""
    return regime_str == "compute"
