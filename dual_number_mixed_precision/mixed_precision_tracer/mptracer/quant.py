"""Block-scaled fake quantization (operator-agnostic).

Validated fp4-e2m1 / fp8-e4m3 / bf16 + per-block scale emulation, vendored from the
dsv4_moe_harness iterations. The injected residual `delta = original - rounded` is computed
under the REAL block scale, so dual attribution is granularity-correct.
"""

from __future__ import annotations

import numpy as np

_FP4_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_FP4_MAX = 6.0
_FP4_THRESHOLDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)

# ab_format -> (ab_dtype, default sf_vec_size, default sf_dtype)
FORMAT_MAP = {
    "bf16": ("bf16", 1, None),
    "mxf8": ("fp8_e4m3", 32, "e8m0"),
    "mxf4": ("fp4_e2m1", 32, "e8m0"),
    "nvf4": ("fp4_e2m1", 16, "e4m3"),
}


def bf16_round(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = (u + (((u >> 16) & 1) + 0x7FFF)) & 0xFFFF0000
    return u.view(np.float32)


def fp4_round(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    idx = np.clip(np.searchsorted(_FP4_THRESHOLDS, np.abs(x)), 0, 7)
    return sign * _FP4_GRID[idx]


def fp8_e4m3_round(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    sign = np.sign(x32)
    ax = np.abs(x32).clip(0, 448.0)
    exp = np.where(ax > 0, np.floor(np.log2(np.maximum(ax, 2.0**-6))), -127.0).astype(np.int32)
    exp = np.clip(exp, -6, 8)
    base = np.exp2(exp.astype(np.float32))
    mant = np.clip(np.round((ax / np.maximum(base, 1e-30) - 1.0) * 8.0) / 8.0, 0.0, 7.0 / 8.0)
    sub_step = 2.0 ** (-6) / 8.0
    sub = np.round(ax / max(sub_step, 1e-30)) * sub_step
    res = np.where(ax >= 2.0**-6, base * (1.0 + mant), sub)
    return (sign * np.minimum(res, 448.0)).astype(np.float32)


def _block_scale_quant(x, sf_vec, ab_dtype, sf_dtype):
    flat = x.ravel().astype(np.float32)
    n = len(flat)
    ng = (n + sf_vec - 1) // sf_vec
    padded = np.zeros(ng * sf_vec, dtype=np.float32)
    padded[:n] = flat
    blocks = padded.reshape(ng, sf_vec)
    max_abs = np.max(np.abs(blocks), axis=1)
    fp_max = _FP4_MAX if ab_dtype == "fp4_e2m1" else 448.0
    raw_sf = np.where(max_abs > 0, max_abs, 1.0) / fp_max
    if sf_dtype == "e8m0":
        sf_q = np.exp2(
            np.floor(np.log2(np.maximum(raw_sf, 1e-30))).astype(np.int32).astype(np.float32)
        )
    elif sf_dtype == "e4m3":
        sf_q = fp8_e4m3_round(raw_sf)
        sf_q = np.where(sf_q == 0, raw_sf, sf_q)
        sf_q = np.where(raw_sf == 0, 1.0, sf_q)
    else:
        sf_q = raw_sf
    sf_q = np.where(max_abs == 0, 1.0, sf_q)
    scaled = blocks / np.maximum(sf_q[:, None], 1e-30)
    if ab_dtype == "fp4_e2m1":
        q = fp4_round(scaled)
    elif ab_dtype == "fp8_e4m3":
        q = fp8_e4m3_round(scaled)
    else:
        q = bf16_round(scaled)
    return (q * sf_q[:, None]).reshape(ng * sf_vec)[:n].reshape(x.shape)


def quantize_ab(x: np.ndarray, ab_format: str, sf_dtype=None, sf_vec_size=None) -> np.ndarray:
    """Dequantized low-precision value of x under ab_format (fp32 carrier)."""
    ab_dtype, def_sv, def_sf = FORMAT_MAP[ab_format]
    if ab_dtype == "bf16":
        return bf16_round(x)
    return _block_scale_quant(x, sf_vec_size or def_sv, ab_dtype, sf_dtype or def_sf)


def quantize_output(d: np.ndarray, out_dtype: str) -> np.ndarray:
    if out_dtype == "bf16":
        return bf16_round(d)
    if out_dtype == "fp16":
        return d.astype(np.float16).astype(np.float32)
    if out_dtype == "fp8_e4m3":
        mx = float(np.max(np.abs(d)))
        if mx == 0.0:
            return d.astype(np.float32)
        s = mx / 448.0
        return fp8_e4m3_round(d / s) * s
    return d.astype(np.float32)
