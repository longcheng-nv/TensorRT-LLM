# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 mechanism crossover — does the best-scenario GVR slowdown follow the
preIdx (threshold-init mechanism) or the logits row (value-density mechanism)?

2x2 swap at the headline cell (K2048 fp32 N=1048576, + K512 fp32 N=1048576):
time gvr_ms_auto and gvr_cutedsl on {best,real} logits x {best,real} preIdx.
The GVR ladder/secant init is a function of the stash logits[preIdx], so if
cost follows the preIdx column, the init-undershoot mechanism is confirmed
end-to-end on the real kernels; if it follows the logits row, it is a value
distribution (histogram band) effect instead.

CUDA-event timing, screening-only (an ~8x effect; canonical numbers stay
nsys). Run on the idle GPU while the nsys grid owns the other one.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "op21_gvr_prod" / "src"))
sys.path.insert(0, str(BENCH / "ops"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE))

from gvr_msc_op import gvr_ms_auto  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
import bundle_data  # noqa: E402

REPS = 50
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32,
                     device="cuda")


def time_op(fn, cold=True):
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True))
          for _ in range(REPS)]
    fn()
    torch.cuda.synchronize()
    for s, e in ev:
        if cold:
            _EVICT.random_()
            torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) * 1e3 for s, e in ev)
    return ts[len(ts) // 2]


def main():
    for K, N in ((2048, 1048576), (512, 1048576)):
        dt = torch.float32
        bb = bundle_data.get_bundle("best", K, dt, N)
        br = bundle_data.get_bundle("real", K, dt, N)
        cr = bb["cr"]
        print(f"\n=== K={K} fp32 N={N} cr={cr} "
              f"(best hr={bb['kernel_hit_rate']:.3f} L{bb['row_meta']['layer']}"
              f" / real hr={br['kernel_hit_rate']:.3f}"
              f" L{br['row_meta']['layer']}) median cold-us x{REPS} ===")
        seq = torch.full((1,), N * cr, dtype=torch.int32, device="cuda")
        for op_name, op in (("gvr_ms_auto",
                             lambda lg, pi: gvr_ms_auto(lg, pi, seq, K,
                                                        compress_ratio=cr)),
                            ("gvr_cutedsl",
                             lambda lg, pi: gvr_cutedsl(lg, pi, seq, K, cr))):
            for lg_src, lg in (("best", bb), ("real", br)):
                for pi_src, pib in (("best", bb), ("real", br)):
                    logits = lg["logits"].contiguous()
                    pre = pib["preIdx"].contiguous()
                    us = time_op(lambda: op(logits, pre))
                    print(f"  {op_name:12s} logits={lg_src:4s} "
                          f"preIdx={pi_src:4s} -> {us:8.1f} us", flush=True)


if __name__ == "__main__":
    main()
