#!/usr/bin/env python3
"""Regression test: the modular mptracer reproduces the validated dsv4_moe_harness numbers.

iter1: FC2 single-sided nvf4 dual==fp64 first-order (rho ~2.5e-8); budget vs leave-one-out cos=1.0
iter5: FC1 nvf4 + cross+2nd-order escalation drops rho to ~0.046
iter6-analog: FC2 cross_term closes double-sided rho to ~2.5e-8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mptracer import (
    MeasureRequest,
    PrecisionPolicy,
    Shape,
    attribution_vs_leave_one_out,
    available_twins,
    measure,
)


def main():
    print("available twins:", available_twins())
    fc2 = Shape("decode_bs32_fc2", "moe_gemm:FC2", M=192, K=2048, N=4096, n_groups=8)
    fc1 = Shape("decode_bs32_fc1", "moe_gemm:FC1", M=192, K=4096, N=4096, n_groups=8)
    ok = True

    # iter1 check 1: FC2 single-sided nvf4 -> dual == fp64 first-order
    r = measure(
        MeasureRequest(
            fc2,
            PrecisionPolicy("nvf4", "fp32", quantize_A=False),
            distribution="normal",
            ref_dtype="fp64",
            seed=42,
        )
    )
    c1 = r.rho <= 1e-6
    print(f"[iter1.1] FC2 single-sided rho = {r.rho:.3e}  (<=1e-6: {c1})  id={r.id}")
    ok &= c1

    # iter1 check 2: budget vs leave-one-out cosine
    cos, rank = attribution_vs_leave_one_out(
        MeasureRequest(
            fc2, PrecisionPolicy("nvf4", "fp32"), distribution="normal", ref_dtype="fp64", seed=42
        )
    )
    c2 = cos >= 0.999
    print(f"[iter1.2] budget cosine = {cos:.6f}  (>=0.999: {c2})  ranking={rank}")
    ok &= c2

    # iter6-analog: FC2 double-sided cross_term closes rho
    r3 = measure(
        MeasureRequest(
            fc2,
            PrecisionPolicy("nvf4", "fp32"),
            distribution="normal",
            ref_dtype="fp64",
            seed=42,
            escalation="cross_term",
        )
    )
    c3 = r3.rho <= 1e-6
    print(f"[xterm ] FC2 double-sided +cross_term rho = {r3.rho:.3e}  (<=1e-6: {c3})")
    ok &= c3

    # iter5: FC1 nvf4 + cross+2nd-order escalation -> rho ~0.046 AT THE VALIDATED CONFIG (K=512).
    # NOTE (dimension dependence): unlike the LINEAR FC2 path (rho dim-invariant), the SwiGLU
    # 2nd-order residual depends on K — at real DSV4 K=4096 the gate magnitude sits deeper in the
    # silu curvature region, so the 3rd+-order residual is larger. We pin the regression to iter5's
    # validated reduced config and report the real-dim value separately (honest caveat).
    fc1_val = Shape("iter5_fc1", "moe_gemm:FC1", M=128, K=512, N=512, n_groups=8)
    r4 = measure(
        MeasureRequest(
            fc1_val,
            PrecisionPolicy("nvf4", "fp32"),
            distribution="normal",
            ref_dtype="fp64",
            seed=42,
            escalation="cross_term",
        )
    )
    c4 = r4.rho <= 0.05
    print(
        f"[iter5 ] FC1(K=512) +cross+2nd rho = {r4.rho:.3e}  (<=0.05: {c4})  flip_risk={r4.flip_risk:.3f}"
    )
    ok &= c4

    r4_real = measure(
        MeasureRequest(
            fc1,
            PrecisionPolicy("nvf4", "fp32"),
            distribution="normal",
            ref_dtype="fp64",
            seed=42,
            escalation="cross_term",
        )
    )
    print(
        f"[caveat] FC1(K=4096 real) +cross+2nd rho = {r4_real.rho:.3e}  "
        f"(2nd-order residual grows with K — NOT dim-invariant on the nonlinear path)"
    )

    print("\nREGRESSION:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
