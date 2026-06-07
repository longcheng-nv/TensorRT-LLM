#!/usr/bin/env python3
"""Phase-0 unified harness for DSV4 MoE GEMM dual-number error tracing.

THE single deterministic environment an agent iterates against. The trust boundary
(HARNESS_API_DESIGN.md) is enforced here: this module *owns every number*; the
LLM/proposer only supplies a MeasureRequest and reads a MeasureResult.

Target operator: DeepSeek-V4 MoE contiguous grouped GEMM
(Sm100BlockScaledContiguousGroupedGemmKernel). Host steps 0.1-0.4 need no GPU.

Block-scaled fake-quant helpers (fp4 e2m1 / fp8 e4m3 / per-block scale) are vendored
from the validated `moe_gemm_dual_tracing/moe_gemm_dual_tracing.py` so this harness is
self-contained. See SCOPE_DSV4_MOE_BS1-512.md for shapes/formats/targets.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

HARNESS_VERSION = "0.1.0"

# ══════════════════════════════════════════════════════════════════════════════
# Block-scaled fake quant (vendored & validated; fp32-exact dual injection)
# ══════════════════════════════════════════════════════════════════════════════

_FP4_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_FP4_MAX = 6.0
_FP4_THRESHOLDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def bf16_round(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    # round-to-nearest-even to 8-bit mantissa (bf16)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u = (u + rounding_bias) & 0xFFFF0000
    return u.view(np.float32)


def fp4_round_vectorized(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    ax = np.abs(x)
    idx = np.clip(np.searchsorted(_FP4_THRESHOLDS, ax), 0, 7)
    return sign * _FP4_GRID[idx]


def fp8_e4m3_round_vectorized(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    sign = np.sign(x32)
    ax = np.abs(x32).clip(0, 448.0)
    exp = np.where(ax > 0, np.floor(np.log2(np.maximum(ax, 2.0**-6))), -127.0).astype(np.int32)
    exp = np.clip(exp, -6, 8)
    base = np.exp2(exp.astype(np.float32))
    mant = np.round((ax / np.maximum(base, 1e-30) - 1.0) * 8.0) / 8.0
    mant = np.clip(mant, 0.0, 7.0 / 8.0)
    normal = ax >= 2.0**-6
    sub_step = 2.0 ** (-6) / 8.0
    sub_val = np.round(ax / max(sub_step, 1e-30)) * sub_step
    result = np.where(normal, base * (1.0 + mant), sub_val)
    return (sign * np.minimum(result, 448.0)).astype(np.float32)


# map PrecisionPolicy.ab_format -> (ab_dtype, default sf_vec, default sf_dtype)
_FORMAT_MAP = {
    "bf16": ("bf16", 1, None),
    "mxf8": ("fp8_e4m3", 32, "e8m0"),
    "mxf4": ("fp4_e2m1", 32, "e8m0"),
    "nvf4": ("fp4_e2m1", 16, "e4m3"),
}


def _block_scale_quantize(x, sf_vec_size, ab_dtype, sf_dtype):
    flat = x.ravel().astype(np.float32)
    n = len(flat)
    n_groups = (n + sf_vec_size - 1) // sf_vec_size
    pad_n = n_groups * sf_vec_size
    padded = np.zeros(pad_n, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(n_groups, sf_vec_size)
    max_abs = np.max(np.abs(blocks), axis=1)
    fp_max = _FP4_MAX if ab_dtype == "fp4_e2m1" else 448.0
    safe_max = np.where(max_abs > 0, max_abs, 1.0)
    raw_sf = safe_max / fp_max
    if sf_dtype == "e8m0":
        exp = np.floor(np.log2(np.maximum(raw_sf, 1e-30))).astype(np.int32)
        sf_q = np.exp2(exp.astype(np.float32))
    elif sf_dtype == "e4m3":
        sf_q = fp8_e4m3_round_vectorized(raw_sf)
        sf_q = np.where(sf_q == 0, raw_sf, sf_q)
        sf_q = np.where(raw_sf == 0, 1.0, sf_q)
    else:
        sf_q = raw_sf
    sf_q = np.where(max_abs == 0, 1.0, sf_q)
    sf_expanded = sf_q[:, np.newaxis]
    scaled = blocks / np.maximum(sf_expanded, 1e-30)
    if ab_dtype == "fp4_e2m1":
        quant_scaled = fp4_round_vectorized(scaled)
    elif ab_dtype == "fp8_e4m3":
        quant_scaled = fp8_e4m3_round_vectorized(scaled)
    else:
        quant_scaled = bf16_round(scaled)
    quant_vals = (quant_scaled * sf_expanded).reshape(pad_n)[:n]
    return quant_vals.reshape(x.shape)


def quantize_ab(x: np.ndarray, ab_format: str, sf_dtype, sf_vec_size) -> np.ndarray:
    """Return the dequantized low-precision value (fp32) of x under ab_format."""
    ab_dtype, def_sv, def_sf = _FORMAT_MAP[ab_format]
    if ab_dtype == "bf16":
        return bf16_round(x)
    sv = sf_vec_size or def_sv
    sf = sf_dtype or def_sf
    return _block_scale_quantize(x, sv, ab_dtype, sf)


def quantize_output(d: np.ndarray, out_dtype: str) -> np.ndarray:
    if out_dtype == "bf16":
        return bf16_round(d)
    if out_dtype == "fp16":
        return d.astype(np.float16).astype(np.float32)
    if out_dtype == "fp8_e4m3":
        mx = float(np.max(np.abs(d)))
        if mx == 0.0:
            return d.astype(np.float32)
        scale = mx / 448.0
        return fp8_e4m3_round_vectorized(d / scale) * scale
    return d.astype(np.float32)  # fp32 = no store rounding


# ══════════════════════════════════════════════════════════════════════════════
# Typed API (the contract — HARNESS_API_DESIGN.md §3)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Shape:
    name: str
    gemm: str  # "FC1" | "FC2"
    M_total: int
    K: int
    N: int  # FC1: 2*intermediate (gate_up fused); FC2: hidden
    n_groups: int = 1  # local experts
    phase: str = "decode"


@dataclass(frozen=True)
class PrecisionPolicy:
    ab_format: str  # "bf16" | "mxf8" | "nvf4" | "mxf4"
    out_dtype: str = "bf16"  # "bf16" | "fp16" | "fp8_e4m3" | "fp32"
    sf_dtype: Optional[str] = None
    sf_vec_size: Optional[int] = None
    acc_dtype: str = "fp32"  # kernel-fixed
    # which operands carry low precision (lets us model single- vs double-sided):
    quantize_A: bool = True
    quantize_B: bool = True


@dataclass(frozen=True)
class MeasureRequest:
    shape: Shape
    policy: PrecisionPolicy
    kernel: str = "dsv4_moe_grouped_gemm"
    twin: str = "moe_grouped_gemm_v1"
    distribution: str = "normal"
    ref_dtype: str = "fp64"
    seed: int = 0
    measure_latency: bool = False
    escalation: str = "none"  # "none" | "cross_term"


@dataclass
class SourceBudget:
    source: str
    l2: float
    relative_impact: float
    budget: float
    max_abs: float


@dataclass
class MeasureResult:
    request: dict
    harness_version: str = HARNESS_VERSION
    measured_rel: float = 0.0
    predicted_rel: float = 0.0
    rho: float = 0.0
    higham_mu_F: Optional[float] = None
    cos_pred_measured: float = 0.0
    budget_per_source: List[SourceBudget] = field(default_factory=list)
    cos_vs_reference: Optional[float] = None
    ranking_vs_reference: Optional[List[str]] = None
    twin_fidelity: Optional[float] = None
    noise_floor: Optional[float] = None
    flip_risk: Optional[float] = None
    escalation_used: str = "none"
    latency_us: Optional[float] = None
    sol_pct: Optional[float] = None
    roofline_regime: Optional[str] = None
    accepted: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=float)


# ══════════════════════════════════════════════════════════════════════════════
# Input generation (seeded, deterministic)
# ══════════════════════════════════════════════════════════════════════════════


def _gen(rng, shape, distribution, scale=0.5):
    if distribution == "normal":
        x = rng.normal(0.0, scale, size=shape)
    elif distribution == "shifted":
        x = rng.normal(0.0, scale, size=shape) + 1.0
    elif distribution == "laplace":
        x = rng.laplace(0.0, scale, size=shape)
    elif distribution == "outlier_channel":
        x = rng.normal(0.0, scale, size=shape)
        if x.ndim == 2:  # spike a few K-columns (channel outliers)
            cols = rng.choice(x.shape[1], max(1, x.shape[1] // 64), replace=False)
            x[:, cols] *= 50.0
    else:
        x = rng.normal(0.0, scale, size=shape)
    return x.astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _silu(x):
    return x * _sigmoid(x)


def _silu_prime(x):
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _silu_pp(x):  # silu''(x) = s(1-s)[2 + x(1-2s)]
    s = _sigmoid(x)
    return s * (1.0 - s) * (2.0 + x * (1.0 - 2.0 * s))


# ══════════════════════════════════════════════════════════════════════════════
# The twin: source-tagged dual channels through FC1(SwiGLU) / FC2(linear)
# ══════════════════════════════════════════════════════════════════════════════


def _twin(
    A, B, policy: PrecisionPolicy, gemm: str, swiglu_limit: float = 10.0, with_cross: bool = False
):
    """Return (Y_real, dual_channels: dict[tag->ndarray], flip_risk).

    Primitive sequence mirrors the production kernel: (block-scaled A,B) -> fp32 MMA
    -> [FC1: SwiGLU epilogue] -> store. Dual channels carry the first-order error of
    each source through that exact sequence (eq.14 for the MMA, SwiGLU Jacobian for
    the FC1 epilogue).

    with_cross=True adds the bilinear cross channel δA·δB (the term first-order drops).
    For FC2 it is matmul-exact; for FC1 it is propagated through the SwiGLU Jacobian
    (so only the SwiGLU 2nd-order Taylor term remains).
    """
    _A64, _B64 = A.astype(np.float64), B.astype(np.float64)
    # --- low-precision (real) operands + injected residuals ---
    Aq = (
        quantize_ab(A, policy.ab_format, policy.sf_dtype, policy.sf_vec_size)
        if policy.quantize_A
        else A
    )
    Bq = (
        quantize_ab(B, policy.ab_format, policy.sf_dtype, policy.sf_vec_size)
        if policy.quantize_B
        else B
    )
    dA = (A - Aq).astype(np.float64)  # δA = original − rounded
    dB = (B - Bq).astype(np.float64)
    Aq64, Bq64 = Aq.astype(np.float64), Bq.astype(np.float64)

    # --- MMA (fp32 accumulate over a fp64 product) ---
    Z_real_f64 = Aq64 @ Bq64
    Z_real = Z_real_f64.astype(np.float32)
    # dual of the matmul output, per source
    Z_dA = dA @ Bq64  # A_input_round
    Z_dB = Aq64 @ dB  # B_input_round
    Z_mma = Z_real_f64 - Z_real.astype(np.float64)  # mma_accum (fp32 rounding)
    Z_cross = (dA @ dB) if with_cross else None  # bilinear cross term δA·δB

    flip_risk = 0.0
    if gemm == "FC1":
        # gate_up fused: split N in half
        half = Z_real.shape[1] // 2
        gate, up = Z_real[:, :half], Z_real[:, half:]
        gate_dA, up_dA = Z_dA[:, :half], Z_dA[:, half:]
        gate_dB, up_dB = Z_dB[:, :half], Z_dB[:, half:]
        gate_mma, up_mma = Z_mma[:, :half], Z_mma[:, half:]
        # SwiGLU: out = silu(gate) * up   (swiglu_limit is the non-smooth clamp guard)
        sg = _silu(gate)
        sgp = _silu_prime(gate)
        Y_real = (sg * up).astype(np.float32)

        def epi(dg, du):  # SwiGLU Jacobian: d(silu(g)*u) = silu'(g)dg*u + silu(g)du
            return (sgp * dg * up + sg * du).astype(np.float32)

        chans = {
            "A_input_round": epi(gate_dA, up_dA),
            "B_input_round": epi(gate_dB, up_dB),
            "mma_accum": epi(gate_mma, up_mma),
        }
        if Z_cross is not None:
            chans["cross_AB"] = epi(Z_cross[:, :half], Z_cross[:, half:])
            # GA9′: SwiGLU 2nd-order epilogue Taylor term — the dominant FC1 miss is
            # epilogue curvature (single-sided FC1 rho=0.103 with δA≡0), not the matmul
            # bilinearity. δg/δu are the TOTAL gate/up perturbations entering the epilogue.
            dg = gate_dA + gate_dB + Z_cross[:, :half]
            du = up_dA + up_dB + Z_cross[:, half:]
            sgpp = _silu_pp(gate)
            chans["swiglu_2nd"] = (0.5 * sgpp * dg * dg * up + sgp * dg * du).astype(np.float32)
        # flip-risk (decision-margin guard): fraction of pre-activations whose
        # DISTANCE to the nearest ±swiglu_limit clamp boundary is smaller than the
        # propagated perturbation — i.e. the perturbation could cross the boundary.
        # (distance-to-threshold, NOT "already saturated"; gate is the clamped operand.)
        dist_to_clamp = np.abs(np.abs(gate) - swiglu_limit)
        perturb = np.abs(gate_dA + gate_dB)
        flip_risk = float(np.mean(dist_to_clamp < perturb))
    else:  # FC2 linear, no epilogue nonlinearity
        Y_real = Z_real
        chans = {
            "A_input_round": Z_dA.astype(np.float32),
            "B_input_round": Z_dB.astype(np.float32),
            "mma_accum": Z_mma.astype(np.float32),
        }
        if Z_cross is not None:
            chans["cross_AB"] = Z_cross.astype(np.float32)

    # --- store (output dtype rounding) ---
    Y_stored = quantize_output(Y_real, policy.out_dtype)
    chans["D_store_round"] = (Y_real - Y_stored).astype(np.float32)

    # zero out channels for non-quantized operands
    if not policy.quantize_A:
        chans["A_input_round"] = np.zeros_like(Y_real)
    if not policy.quantize_B:
        chans["B_input_round"] = np.zeros_like(Y_real)
    return Y_stored, chans, flip_risk


def _reference(A, B, gemm: str, ref_dtype: str):
    dt = np.float64 if ref_dtype == "fp64" else (np.float32 if ref_dtype == "fp32" else None)
    if dt is None:  # bf16 reference
        Ar, Br = bf16_round(A).astype(np.float64), bf16_round(B).astype(np.float64)
    else:
        Ar, Br = A.astype(dt), B.astype(dt)
    Z = Ar @ Br
    if gemm == "FC1":
        half = Z.shape[1] // 2
        gate, up = Z[:, :half], Z[:, half:]
        Y = _silu(gate.astype(np.float64)) * up.astype(np.float64)
    else:
        Y = Z
    return Y.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# metrics helpers
# ══════════════════════════════════════════════════════════════════════════════


def _l2(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).ravel()))


def _cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ══════════════════════════════════════════════════════════════════════════════
# THE call
# ══════════════════════════════════════════════════════════════════════════════


def measure(req: MeasureRequest) -> MeasureResult:
    rng = np.random.default_rng(req.seed)
    s = req.shape
    A = _gen(rng, (s.M_total, s.K), req.distribution)
    B = _gen(rng, (s.K, s.N), req.distribution)

    Y_ref = _reference(A, B, s.gemm, req.ref_dtype)
    # ρ-gated cross-term escalation: the twin adds the bilinear δA·δB channel (matmul-
    # exact for FC2; SwiGLU-Jacobian-propagated for FC1). Only the SwiGLU 2nd-order term
    # then remains for FC1.
    want_cross = req.escalation == "cross_term" and req.policy.quantize_A and req.policy.quantize_B
    Y_real, chans, flip_risk = _twin(A, B, req.policy, s.gemm, with_cross=want_cross)
    escalation_used = "cross_term" if (want_cross and "cross_AB" in chans) else "none"

    measured = Y_ref - Y_real.astype(np.float64)
    predicted = np.sum([c.astype(np.float64) for c in chans.values()], axis=0)
    residual = measured - predicted
    m_norm, ref_norm = _l2(measured), _l2(Y_ref)
    rho = _l2(residual) / m_norm if m_norm > 0 else 0.0

    # per-source budget
    chan_l2 = {k: _l2(v) for k, v in chans.items()}
    total = sum(chan_l2.values()) or 1.0
    budgets = [
        SourceBudget(
            source=k,
            l2=chan_l2[k],
            relative_impact=chan_l2[k] / (_l2(Y_real) or 1.0),
            budget=chan_l2[k] / total,
            max_abs=float(np.max(np.abs(chans[k]))) if chans[k].size else 0.0,
        )
        for k in chans
    ]
    budgets.sort(key=lambda b: -b.l2)

    # Higham backward error (cancellation-robust)
    denom = _l2(np.abs(A.astype(np.float64)) @ np.abs(B.astype(np.float64)))
    higham = (_l2(measured) / denom) if denom > 0 else None

    res = MeasureResult(
        request=_req_to_dict(req),
        measured_rel=m_norm / ref_norm if ref_norm > 0 else 0.0,
        predicted_rel=_l2(predicted) / ref_norm if ref_norm > 0 else 0.0,
        rho=rho,
        higham_mu_F=higham,
        cos_pred_measured=_cos(predicted, measured),
        budget_per_source=budgets,
        flip_risk=flip_risk,
        escalation_used=escalation_used,
        roofline_regime=_regime(s),
    )
    if budgets:
        res.notes.append(f"{budgets[0].source} dominates budget {budgets[0].budget:.2f}")
    return res


def _regime(s: Shape) -> str:
    # crude per-shape roofline classification (refined with measured SOL on silicon)
    per_group_m = s.M_total / max(s.n_groups, 1)
    if per_group_m < 16:
        return "launch"
    if per_group_m < 128:
        return "memory"
    return "compute"


def _req_to_dict(req: MeasureRequest) -> dict:
    return {
        "shape": asdict(req.shape),
        "policy": asdict(req.policy),
        "kernel": req.kernel,
        "twin": req.twin,
        "distribution": req.distribution,
        "ref_dtype": req.ref_dtype,
        "seed": req.seed,
        "escalation": req.escalation,
    }


def attribution_vs_leave_one_out(req: MeasureRequest) -> Tuple[float, List[str]]:
    """Exact per-source reference for the dual budget.

    Quantize one operand at a time, measure its marginal output error.
    Returns (budget-cosine, dual-ranking).
    """
    rng = np.random.default_rng(req.seed)
    s = req.shape
    A = _gen(rng, (s.M_total, s.K), req.distribution)
    B = _gen(rng, (s.K, s.N), req.distribution)
    Y_ref = _reference(A, B, s.gemm, req.ref_dtype)

    def loo(qa, qb):
        p = PrecisionPolicy(
            req.policy.ab_format,
            "fp32",
            req.policy.sf_dtype,
            req.policy.sf_vec_size,
            quantize_A=qa,
            quantize_B=qb,
        )
        Yr, _, _ = _twin(A, B, p, s.gemm)
        return _l2(Y_ref - Yr.astype(np.float64))

    loo_vec = {"A_input_round": loo(True, False), "B_input_round": loo(False, True)}
    _, chans, _ = _twin(
        A,
        B,
        PrecisionPolicy(req.policy.ab_format, "fp32", req.policy.sf_dtype, req.policy.sf_vec_size),
        s.gemm,
    )
    dual_vec = {
        "A_input_round": _l2(chans["A_input_round"]),
        "B_input_round": _l2(chans["B_input_round"]),
    }
    keys = ["A_input_round", "B_input_round"]
    cos = _cos([dual_vec[k] for k in keys], [loo_vec[k] for k in keys])
    ranking = sorted(keys, key=lambda k: -dual_vec[k])
    return cos, ranking


# ══════════════════════════════════════════════════════════════════════════════
# Iteration 1 (pre-registered in PROGRAM.md): host-only validation of the twin
# ══════════════════════════════════════════════════════════════════════════════


def _iteration1():
    print("=" * 78)
    print("DSV4 MoE harness — iteration 1 (GA1+GA2): twin validation, host-only")
    print("=" * 78)
    results = {}

    # --- Check 1: dual vs fp64 first-order EXACT on the linear FC2, single-sided ---
    # only B quantized → measured error must equal the B_input_round channel to fp64.
    fc2 = Shape("decode_bs32_fc2", "FC2", M_total=192, K=2048, N=4096, n_groups=8)
    req = MeasureRequest(
        shape=fc2,
        policy=PrecisionPolicy("nvf4", out_dtype="fp32", quantize_A=False),
        distribution="normal",
        ref_dtype="fp64",
        seed=42,
    )
    r = measure(req)
    print("\n[Check 1] FC2 single-sided nvf4 (B only), out=fp32")
    print(f"  measured_rel = {r.measured_rel:.3e}  rho = {r.rho:.3e}  (want rho < 1e-6)")
    results["check1_rho"] = r.rho

    # --- Check 2: budget cosine vs leave-one-out >= 0.999, double-sided FC2 ---
    req2 = MeasureRequest(
        shape=fc2,
        policy=PrecisionPolicy("nvf4", out_dtype="fp32"),
        distribution="normal",
        ref_dtype="fp64",
        seed=42,
    )
    cos, ranking = attribution_vs_leave_one_out(req2)
    print("\n[Check 2] FC2 double-sided nvf4 budget vs leave-one-out")
    print(f"  budget cosine = {cos:.6f}  (want >= 0.999)   ranking={ranking}")
    results["check2_cos"] = cos

    # --- Context: double-sided rho (the bilinear cross-term floor) + escalation ---
    r3 = measure(req2)
    r3b = measure(
        MeasureRequest(
            shape=fc2,
            policy=PrecisionPolicy("nvf4", out_dtype="fp32"),
            distribution="normal",
            ref_dtype="fp64",
            seed=42,
            escalation="cross_term",
        )
    )
    print(
        f"\n[Context] FC2 double-sided nvf4 rho = {r3.rho:.3e} -> with cross_term = {r3b.rho:.3e}"
    )
    print(
        "  per-source budget: "
        + ", ".join(f"{b.source}={b.budget:.2f}" for b in r3.budget_per_source)
    )
    results["check3_rho_double"] = r3.rho
    results["check3_rho_crossterm"] = r3b.rho

    # --- FC1 SwiGLU: first-order (nonlinear) rho + flip-risk guard ---
    fc1 = Shape("decode_bs32_fc1", "FC1", M_total=192, K=4096, N=4096, n_groups=8)
    r4 = measure(
        MeasureRequest(
            shape=fc1,
            policy=PrecisionPolicy("nvf4", out_dtype="bf16"),
            distribution="normal",
            ref_dtype="fp64",
            seed=42,
        )
    )
    print(
        f"\n[Context] FC1 SwiGLU nvf4: measured_rel={r4.measured_rel:.3e} rho={r4.rho:.3e} "
        f"flip_risk={r4.flip_risk:.3f}"
    )
    print(
        "  per-source budget: "
        + ", ".join(f"{b.source}={b.budget:.2f}" for b in r4.budget_per_source)
    )

    # --- verdict against pre-registered thresholds ---
    passed = results["check1_rho"] <= 1e-6 and results["check2_cos"] >= 0.999
    print("\n" + "=" * 78)
    print("PRE-REGISTERED THRESHOLD: check1 rho<=1e-6 AND check2 cos>=0.999")
    print(
        f"RESULT: {'KEPT ✅' if passed else 'DISCARDED ❌'}  "
        f"(rho={results['check1_rho']:.2e}, cos={results['check2_cos']:.5f})"
    )
    print("=" * 78)

    # persist
    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(exist_ok=True)
    (outdir / "iteration1.json").write_text(
        json.dumps(
            {
                "results": results,
                "passed": bool(passed),
                "check1_result": asdict(r),
                "check4_fc1": asdict(r4),
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {outdir / 'iteration1.json'}")
    return passed


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    _iteration1()
