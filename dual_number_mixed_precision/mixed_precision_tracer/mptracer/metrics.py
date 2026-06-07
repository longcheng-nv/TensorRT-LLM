"""Operator-agnostic metrics — the only place numbers are computed."""

from __future__ import annotations

import numpy as np


def l2(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).ravel()))


def cos(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if (na == 0 or nb == 0) else float(np.dot(a, b) / (na * nb))


def rho(measured, predicted) -> float:
    m = l2(measured)
    return (
        l2(np.asarray(measured, np.float64) - np.asarray(predicted, np.float64)) / m
        if m > 0
        else 0.0
    )


def higham_mu_F(measured, A, B):
    """Cancellation-robust GEMM backward error ||C-Ĉ||_F / || |A||B| ||_F."""
    denom = l2(np.abs(np.asarray(A, np.float64)) @ np.abs(np.asarray(B, np.float64)))
    return (l2(measured) / denom) if denom > 0 else None
