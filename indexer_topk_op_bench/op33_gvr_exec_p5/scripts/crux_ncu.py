# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op33 iter0 CRUX — NCU attribution on the two incumbents (op26_r0auto,
gvr_ms_auto) at op22-identical bundles. Answers, per regime, whether the P5
execution levers have headroom BEFORE any kernel is written:

  dram %      -> memory headroom (L2-trap veto if idle & input<<L2)
  issue %     -> issue-bound? (op29 iter11 saw 81-84%)
  warps %     -> achieved occupancy (structural if grid<<SM)
  reg/thread  -> occupancy pressure (falsi-hist: 44-61 -> 50% occ)
  bytes/sector-> global-load vectorization headroom (32B = fully packed)

Run ONE cell per process under ncu with profiler start/stop so only the single
measured call is profiled (warm-compile happens before start()).

  env -u GITHUB_TOKEN -u HF_TOKEN ncu --profile-from-start off \
    --metrics <list> --csv --page raw \
    python crux_ncu.py --op op26_r0auto --scenario real --sweep seqlen \
      --K 512 --dtype fp32 --N 65536 --BS 1
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
_OP22 = _BENCH / "op22_temporal_fixed_hr_bench"
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_OP22))

import bundle_data                                  # noqa: E402
from sweep_nsys import build_call                   # noqa: E402

_DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--scenario", default="real", choices=["best", "worst", "real"])
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--dtype", default="fp32", choices=list(_DT))
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--BS", type=int, default=1)
    args = ap.parse_args()

    dtype = _DT[args.dtype]
    b = bundle_data.get_bundle(args.scenario, args.K, dtype, args.N)
    logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
    call, keep, extra = build_call(args.op, args.K, dtype, args.N, args.BS, cr,
                                   logits_row, preidx_row)
    # warm-compile + warmup OUTSIDE the profiled window
    for _ in range(12):
        call()
    torch.cuda.synchronize()

    prof.start()
    call()                       # the single profiled invocation
    torch.cuda.synchronize()
    prof.stop()
    # keep refs alive
    del keep
    print(f"# done op={args.op} scen={args.scenario} K={args.K} "
          f"dt={args.dtype} N={args.N} BS={args.BS} extra={extra}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
