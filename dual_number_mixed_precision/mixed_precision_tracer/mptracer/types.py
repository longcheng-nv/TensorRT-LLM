"""Typed contract for the harness — the JSON schema that crosses the trust boundary.

The LLM/proposer produces a MeasureRequest (and PrecisionPolicy proposals); the harness
returns a MeasureResult. Every NUMBER lives in MeasureResult. A PolicyProposal's numeric
fields may only *reference* a MeasureResult by id (result_ref) — the proposer never fills a
measured number itself. This makes "the harness owns every number" a type invariant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Shape:
    name: str
    op: str  # operator key, e.g. "moe_gemm:FC1" | "moe_gemm:FC2"
    M: int
    K: int
    N: int
    n_groups: int = 1
    phase: str = "decode"
    meta: dict = field(default_factory=dict)  # operator-specific extras (true dims, etc.)


@dataclass(frozen=True)
class PrecisionPolicy:
    ab_format: str  # "bf16" | "mxf8" | "nvf4" | "mxf4"
    out_dtype: str = "bf16"
    sf_dtype: Optional[str] = None
    sf_vec_size: Optional[int] = None
    acc_dtype: str = "fp32"
    quantize_A: bool = True
    quantize_B: bool = True


@dataclass(frozen=True)
class MeasureRequest:
    shape: Shape
    policy: PrecisionPolicy
    twin: str = ""  # "" -> inferred from shape.op
    distribution: str = "normal"
    ref_dtype: str = "fp64"
    seed: int = 0
    escalation: str = "none"  # none | cross_term | taylor2 | interval | stochastic | ablation
    measure_latency: bool = False


@dataclass
class SourceBudget:
    source: str
    l2: float
    relative_impact: float
    budget: float
    max_abs: float


@dataclass
class MeasureResult:
    id: str  # stable hash of (request, harness_version) — the trust handle
    request: dict
    harness_version: str
    # accuracy
    measured_rel: float = 0.0
    predicted_rel: float = 0.0
    rho: float = 0.0
    higham_mu_F: Optional[float] = None
    cos_pred_measured: float = 0.0
    # attribution
    budget_per_source: List[SourceBudget] = field(default_factory=list)
    cos_vs_reference: Optional[float] = None
    ranking_vs_reference: Optional[List[str]] = None
    # twin <-> silicon
    twin_fidelity: Optional[float] = None
    noise_floor: Optional[float] = None
    # non-smooth guard
    flip_risk: Optional[float] = None
    escalation_used: str = "none"
    # performance
    latency_us: Optional[float] = None
    sol_pct: Optional[float] = None
    roofline_regime: Optional[str] = None
    # verdict (rule-computed, never model-asserted)
    accepted: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyProposal:
    """An LLM-proposed precision policy.

    Numeric justification MUST reference a measured result id — the proposer cannot
    assert a number.
    """

    policy: PrecisionPolicy
    rationale: str
    result_ref: Optional[str] = None  # id of the MeasureResult that justifies this
