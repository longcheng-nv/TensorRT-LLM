"""mptracer — primitive-level forward dual-number error tracing + mixed-precision search.

Operator-agnostic harness core (the environment an agent iterates against), pluggable
twins/ and backends/. The trust boundary is structural: the harness owns every number
(MeasureResult); proposers may only reference results by id (PolicyProposal.result_ref).

Public API:
    measure(MeasureRequest) -> MeasureResult          # the one deterministic call
    attribution_vs_leave_one_out(req) -> (cos, rank)  # exact per-source reference
    greedy_policy_search(...)                          # attribution-guided demotion (G4)
    register_twin / get_twin / available_twins         # operator plugins
"""

from .core import HARNESS_VERSION, attribution_vs_leave_one_out, measure
from .policy_search import greedy_policy_search
from .twins import available_twins, get_twin, register_twin
from .types import (
    MeasureRequest,
    MeasureResult,
    PolicyProposal,
    PrecisionPolicy,
    Shape,
    SourceBudget,
)

__version__ = HARNESS_VERSION
__all__ = [
    "measure",
    "attribution_vs_leave_one_out",
    "greedy_policy_search",
    "register_twin",
    "get_twin",
    "available_twins",
    "Shape",
    "PrecisionPolicy",
    "MeasureRequest",
    "MeasureResult",
    "SourceBudget",
    "PolicyProposal",
    "HARNESS_VERSION",
]
