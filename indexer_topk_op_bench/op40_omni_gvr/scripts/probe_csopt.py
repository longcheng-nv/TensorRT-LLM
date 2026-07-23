# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cs-choice probe (rung 2): on large-N real cells, time arm v5best at
cluster_size in {1, 2, 4, 8} back-to-back on ONE GPU under nsys, cold-L2.
pick_config currently forces cs=8 at N>=131072 / cs=4 at 64-128K (BS=1);
v1-class P4 shifted the cluster-tax-vs-scan-length balance — re-locate the
optimum.

  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=<g> nsys profile \
    -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -f true -o results/csopt/csopt python3 scripts/probe_csopt.py
"""
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
BENCH = OP40.parent
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(HERE))

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from ab40 import compile_arm, exact_set, launch_cfg  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
WARMUP, REPS = 10, 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

CELLS = [("flash", "256k", 2), ("flash", "512k", 2), ("flash", "1024k", 2),
         ("pro", "256k", 30), ("pro", "512k", 30), ("pro", "1024k", 30),
         ("v32", "64k", 34), ("v32", "128k", 34), ("v32", "256k", 34),
         ("pro", "128k", 30), ("v32", "32k", 34)]
CSS = [1, 2, 4, 8]


def main():
    prof.start()
    for model, isl, L in CELLS:
        RD = RV32 if model == "v32" else RV4
        b = RD.get_bundle(model, isl, L, "fp32")
        logits, pre = b["logits"].contiguous(), b["preIdx"].contiguous()
        N, K, cr = b["N"], b["K"], b["cr"]
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        uuid = f"{model}_{isl}_L{L:02d}"
        for cs in CSS:
            cfg = launch_cfg(logits, N)
            cfg["cluster_size"] = cs
            fn = compile_arm("v5best", K, cr, cfg)
            oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
            fn(logits, pre, sl, None, oi, None)
            torch.cuda.synchronize()
            ok = exact_set(oi, logits[0], K, N)
            for _ in range(WARMUP):
                fn(logits, pre, sl, None, oi, None)
            torch.cuda.synchronize()
            for _ in range(REPS):
                _EVICT.random_()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"c|cs{cs}|{uuid}|fp32")
                fn(logits, pre, sl, None, oi, None)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            print(f"{uuid} cs={cs} exact={ok}", flush=True)
        del logits, pre
        RV4._bundle_cache.clear()
        RV32._bundle_cache.clear()
    prof.stop()


if __name__ == "__main__":
    main()
