"""MoE grouped-GEMM dual twin: FC1 (gate_up fused + SwiGLU) and FC2 (linear).

Numeric logic is identical to the validated dsv4_moe_harness iterations (iter 1/2/4/5):
- source-tagged dual channels A_input_round / B_input_round / mma_accum / D_store_round
- escalation channels: cross_AB (matmul bilinear) and swiglu_2nd (SwiGLU 2nd-order Taylor)
- SwiGLU clamp is the guarded non-smooth node (flip_risk).
"""

from __future__ import annotations

import numpy as np

from ..quant import bf16_round, quantize_ab, quantize_output


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _silu(x):
    return x * _sigmoid(x)


def _silu_p(x):
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _silu_pp(x):
    s = _sigmoid(x)
    return s * (1.0 - s) * (2.0 + x * (1.0 - 2.0 * s))


def _gen(rng, shape2d, distribution, scale=0.5):
    if distribution == "normal":
        x = rng.normal(0.0, scale, size=shape2d)
    elif distribution == "shifted":
        x = rng.normal(0.0, scale, size=shape2d) + 1.0
    elif distribution == "laplace":
        x = rng.laplace(0.0, scale, size=shape2d)
    elif distribution == "outlier_channel":
        x = rng.normal(0.0, scale, size=shape2d)
        if x.ndim == 2:
            cols = rng.choice(x.shape[1], max(1, x.shape[1] // 64), replace=False)
            x[:, cols] *= 50.0
    else:
        x = rng.normal(0.0, scale, size=shape2d)
    return x.astype(np.float32)


class MoEGemmTwin:
    name = "moe_gemm"
    SWIGLU_LIMIT = 10.0

    @staticmethod
    def _gemm_kind(shape):
        return shape.op.split(":", 1)[1] if ":" in shape.op else "FC2"

    def generate_inputs(self, rng, shape, distribution):
        A = _gen(rng, (shape.M, shape.K), distribution)
        B = _gen(rng, (shape.K, shape.N), distribution)
        return A, B

    def reference(self, A, B, shape, ref_dtype):
        dt = np.float64 if ref_dtype == "fp64" else (np.float32 if ref_dtype == "fp32" else None)
        if dt is None:
            Ar, Br = bf16_round(A).astype(np.float64), bf16_round(B).astype(np.float64)
        else:
            Ar, Br = A.astype(dt), B.astype(dt)
        Z = Ar @ Br
        if self._gemm_kind(shape) == "FC1":
            half = Z.shape[1] // 2
            return _silu(Z[:, :half].astype(np.float64)) * Z[:, half:].astype(np.float64)
        return Z.astype(np.float64)

    def real_and_dual(self, A, B, policy, shape, with_cross: bool):
        kind = self._gemm_kind(shape)
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
        dA = (A - Aq).astype(np.float64)
        dB = (B - Bq).astype(np.float64)
        Aq64, Bq64 = Aq.astype(np.float64), Bq.astype(np.float64)

        Z_real_f64 = Aq64 @ Bq64
        Z_real = Z_real_f64.astype(np.float32)
        Z_dA = dA @ Bq64
        Z_dB = Aq64 @ dB
        Z_mma = Z_real_f64 - Z_real.astype(np.float64)
        Z_cross = (dA @ dB) if with_cross else None

        flip_risk = 0.0
        if kind == "FC1":
            half = Z_real.shape[1] // 2
            gate, up = Z_real[:, :half], Z_real[:, half:]
            gate_dA, up_dA = Z_dA[:, :half], Z_dA[:, half:]
            gate_dB, up_dB = Z_dB[:, :half], Z_dB[:, half:]
            gate_mma, up_mma = Z_mma[:, :half], Z_mma[:, half:]
            sg, sgp = _silu(gate), _silu_p(gate)
            Y_real = (sg * up).astype(np.float32)

            def epi(dg, du):
                return (sgp * dg * up + sg * du).astype(np.float32)

            chans = {
                "A_input_round": epi(gate_dA, up_dA),
                "B_input_round": epi(gate_dB, up_dB),
                "mma_accum": epi(gate_mma, up_mma),
            }
            if Z_cross is not None:
                chans["cross_AB"] = epi(Z_cross[:, :half], Z_cross[:, half:])
                dg = gate_dA + gate_dB + Z_cross[:, :half]
                du = up_dA + up_dB + Z_cross[:, half:]
                chans["swiglu_2nd"] = (0.5 * _silu_pp(gate) * dg * dg * up + sgp * dg * du).astype(
                    np.float32
                )
            dist_to_clamp = np.abs(np.abs(gate) - self.SWIGLU_LIMIT)
            flip_risk = float(np.mean(dist_to_clamp < np.abs(gate_dA + gate_dB)))
        else:
            Y_real = Z_real
            chans = {
                "A_input_round": Z_dA.astype(np.float32),
                "B_input_round": Z_dB.astype(np.float32),
                "mma_accum": Z_mma.astype(np.float32),
            }
            if Z_cross is not None:
                chans["cross_AB"] = Z_cross.astype(np.float32)

        Y_stored = quantize_output(Y_real, policy.out_dtype)
        chans["D_store_round"] = (Y_real - Y_stored).astype(np.float32)
        if not policy.quantize_A:
            chans["A_input_round"] = np.zeros_like(Y_real)
        if not policy.quantize_B:
            chans["B_input_round"] = np.zeros_like(Y_real)
        return Y_stored, chans, {"flip_risk": flip_risk}


from . import register_twin  # noqa: E402

register_twin("moe_gemm", MoEGemmTwin())
