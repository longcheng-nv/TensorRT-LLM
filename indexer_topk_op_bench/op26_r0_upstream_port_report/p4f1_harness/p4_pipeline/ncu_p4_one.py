# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-cell kernel driver for NCU instruction-level P4 accounting.

Launches the [ptime]+[p4sub] timed twin (gvrpkgp4t_head) on ONE real decode
cell so ncu can profile it. The 15 executed clock64 stamps (t0..t7 +
s8..s14) act as SASS address landmarks: parse_ncu_p4.py segments the
per-instruction source-page export between consecutive executed CS2R
(clock) instructions and buckets inst_executed + PC-sampling stalls per
kernel phase / P4 sub-stage.

Usage (profile launch #13 = first steady-state launch after 2 correctness
+ 10 warmup):
  ncu --set full --replay-mode kernel -s 12 -c 1 -f -o <rep> \
      python3 ncu_p4_one.py --cell flash_128k_L02
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
P4F1 = HERE.parent
REPORT = P4F1.parent
BENCH = REPORT.parent

sys.path.insert(0, str(REPORT / "kf_campaign" / "gvrpkg_head"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE))

import cutlass  # noqa: E402

assert cutlass.__version__.startswith("4.5.0"), cutlass.__version__

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from measure_p4pipe_full import timed_compile, launch_cfg, exact_set  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="model_isl_LNN")
    ap.add_argument("--launches", type=int, default=13)
    args = ap.parse_args()

    model, isl, lpart = args.cell.split("_")
    layer = int(lpart[1:])
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, layer, "fp32")
    logits = b["logits"].contiguous()
    pre = b["preIdx"].contiguous()
    N, K, cr = b["N"], b["K"], b["cr"]
    sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
    out_t = torch.empty(1, K, dtype=torch.int32, device=DEV)
    ts = torch.zeros(1, 16, dtype=torch.int64, device=DEV)

    cfg = launch_cfg(logits, N)
    timed = timed_compile(K, cr, cfg)
    print(f"[ncu_one] {args.cell} N={N} K={K} cs={cfg['cluster_size']} "
          f"T={cfg['num_threads']}", flush=True)

    for i in range(args.launches):
        timed(logits, pre, sl, None, out_t, None, ts)
        if i == 0:
            torch.cuda.synchronize()
            ok = exact_set(out_t, b)
            print(f"[ncu_one] exact={ok}", flush=True)
            assert ok, "timed arm inexact on this cell"
    torch.cuda.synchronize()
    print("[ncu_one] done", flush=True)


if __name__ == "__main__":
    main()
