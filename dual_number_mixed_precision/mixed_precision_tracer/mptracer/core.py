"""The harness core — operator-agnostic. THE single deterministic call.

measure(MeasureRequest) -> MeasureResult. Owns every number. Orchestrates:
  generate inputs -> reference -> twin real+dual -> metrics -> budget -> roofline -> verdict.
Operator specifics live entirely in the registered twin; this module never names an operator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import List

import numpy as np

from . import metrics
from .roofline import regime as _regime
from .twins import get_twin
from .types import MeasureRequest, MeasureResult, PrecisionPolicy, SourceBudget

HARNESS_VERSION = "1.0.0"


def _req_dict(req: MeasureRequest) -> dict:
    return {
        "shape": asdict(req.shape),
        "policy": asdict(req.policy),
        "twin": req.twin,
        "distribution": req.distribution,
        "ref_dtype": req.ref_dtype,
        "seed": req.seed,
        "escalation": req.escalation,
    }


def _result_id(req: MeasureRequest) -> str:
    blob = json.dumps(_req_dict(req), sort_keys=True) + "|" + HARNESS_VERSION
    return "mr_" + hashlib.sha1(blob.encode()).hexdigest()[:16]


def _twin_name(req: MeasureRequest) -> str:
    return req.twin or req.shape.op.split(":", 1)[0]


def measure(req: MeasureRequest) -> MeasureResult:
    twin = get_twin(_twin_name(req))
    rng = np.random.default_rng(req.seed)
    A, B = twin.generate_inputs(rng, req.shape, req.distribution)

    Y_ref = twin.reference(A, B, req.shape, req.ref_dtype)
    want_cross = (
        req.escalation in ("cross_term", "taylor2")
        and req.policy.quantize_A
        and req.policy.quantize_B
    )
    Y_real, chans, extra = twin.real_and_dual(A, B, req.policy, req.shape, with_cross=want_cross)
    escalation_used = "cross_term" if (want_cross and "cross_AB" in chans) else "none"

    measured = Y_ref - Y_real.astype(np.float64)
    predicted = np.sum([c.astype(np.float64) for c in chans.values()], axis=0)
    m_norm, ref_norm = metrics.l2(measured), metrics.l2(Y_ref)

    chan_l2 = {k: metrics.l2(v) for k, v in chans.items()}
    total = sum(chan_l2.values()) or 1.0
    y_real_l2 = metrics.l2(Y_real) or 1.0
    budgets: List[SourceBudget] = sorted(
        [
            SourceBudget(
                k,
                chan_l2[k],
                chan_l2[k] / y_real_l2,
                chan_l2[k] / total,
                float(np.max(np.abs(chans[k]))) if chans[k].size else 0.0,
            )
            for k in chans
        ],
        key=lambda b: -b.l2,
    )

    higham = metrics.higham_mu_F(measured, A, B) if "moe_gemm" in twin.name else None
    M_eff = req.shape.M // max(req.shape.n_groups, 1)

    res = MeasureResult(
        id=_result_id(req),
        request=_req_dict(req),
        harness_version=HARNESS_VERSION,
        measured_rel=m_norm / ref_norm if ref_norm > 0 else 0.0,
        predicted_rel=metrics.l2(predicted) / ref_norm if ref_norm > 0 else 0.0,
        rho=metrics.rho(measured, predicted),
        higham_mu_F=higham,
        cos_pred_measured=metrics.cos(predicted, measured),
        budget_per_source=budgets,
        flip_risk=extra.get("flip_risk"),
        escalation_used=escalation_used,
        roofline_regime=_regime(M_eff),
    )
    if budgets:
        res.notes.append(f"{budgets[0].source} dominates budget {budgets[0].budget:.2f}")
    return res


def attribution_vs_leave_one_out(req: MeasureRequest):
    """Exact per-source reference: quantize one operand at a time. Returns (budget cosine, ranking)."""
    twin = get_twin(_twin_name(req))
    rng = np.random.default_rng(req.seed)
    A, B = twin.generate_inputs(rng, req.shape, req.distribution)
    Y_ref = twin.reference(A, B, req.shape, req.ref_dtype)

    def loo(qa, qb):
        p = PrecisionPolicy(
            req.policy.ab_format,
            "fp32",
            req.policy.sf_dtype,
            req.policy.sf_vec_size,
            quantize_A=qa,
            quantize_B=qb,
        )
        Yr, _, _ = twin.real_and_dual(A, B, p, req.shape, with_cross=False)
        return metrics.l2(Y_ref - Yr.astype(np.float64))

    loo_vec = {"A_input_round": loo(True, False), "B_input_round": loo(False, True)}
    _, chans, _ = twin.real_and_dual(
        A,
        B,
        PrecisionPolicy(req.policy.ab_format, "fp32", req.policy.sf_dtype, req.policy.sf_vec_size),
        req.shape,
        with_cross=False,
    )
    dual_vec = {
        "A_input_round": metrics.l2(chans["A_input_round"]),
        "B_input_round": metrics.l2(chans["B_input_round"]),
    }
    keys = ["A_input_round", "B_input_round"]
    return (
        metrics.cos([dual_vec[k] for k in keys], [loo_vec[k] for k in keys]),
        sorted(keys, key=lambda k: -dual_vec[k]),
    )
