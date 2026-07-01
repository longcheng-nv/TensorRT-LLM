# SPDX-License-Identifier: NVIDIA
# op17 iter3: project the LARGE-N P2-collapse half of the portfolio.
#
# The 148-threshold parallel sweep evaluates a 148-point CCDF sample of the row in
# ONE full-N pass and picks the threshold with count closest-above-K — equivalent
# to what baseline's serial secant reaches in P2_iters full-N passes. So:
#   portfolio_P2_us ≈ baseline_P2_us / P2_iters   (one pass instead of P2_iters)
#   proj_total ≈ base − P2_us*(1 − 1/P2_iters) + sync
# P3/P4 unchanged (at large N cand is already lean-ish; P4 is a small fraction).
#
# base_us, P2_us : measured (measure_cute_phases). P2_iters : report fp32 counts
# (report/REPORT.md "P2 secant iters by seq-len × K"). sync : iter2 proxy.
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
from measure_cute_phases import measure  # noqa: E402

# report/REPORT.md fp32 P2 secant iters by (K, N)
P2_ITERS = {
    (1024, 65536): 1.29, (1024, 131072): 1.94, (1024, 262144): 2.04,
    (2048, 65536): 2.08, (2048, 131072): 2.44, (2048, 262144): 2.65,
    (512, 65536): 1.00, (512, 131072): 1.00, (512, 262144): 1.00,
}
SYNC_US = {65536: 3.42, 131072: 2.21, 262144: 2.02}

if __name__ == "__main__":
    print("PROJECTED portfolio P2-collapse (large N, fp32). proj = base - P2*(1-1/iters) + sync")
    print(f"{'K':>5} {'N':>8} | base_us  P2_us  iters  sync | proj_us  speedup")
    for K, cr_val in ((512, 4), (1024, 4), (2048, 1)):
        for N in (65536, 131072, 262144):
            r = measure(K, torch.float32, N, cr_val, reps=25)
            base_us, p2_us = r["total_us"], r["P2_us"]
            it = P2_ITERS[(K, N)]
            sync = SYNC_US[N]
            proj = base_us - p2_us * (1 - 1.0 / it) + sync
            print(f"{K:>5} {N:>8} | {base_us:6.1f} {p2_us:6.2f}  {it:4.2f}  {sync:4.2f} | "
                  f"{proj:6.1f}  {base_us / proj:5.3f}x")
