#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter11 L3: single-cell driver for ncu attribution (NOT a timing baseline).

Runs ONE (op, K, N, BS, scenario) cell: build + a few kernel launches for ncu
to attach to. Profile with a kernel-name filter, e.g.:

  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=<g> \
    ncu --kernel-name 'regex:topk_main_kernel|gvr29_hbe_kernel' -s 2 -c 3 \
    --metrics <...> --csv python3 scripts/ncu_cell_op29.py \
    --op gvr29_hbe --K 2048 --N 262144 --BS 1024 --scenario real
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "harness"))
sys.path.insert(0, str(HERE.parents[1] / "op22_temporal_fixed_hr_bench"))

from sglang_v2_op import topk_v2, plan as v2_plan  # noqa: E402
from gvr29_op import gvr29_topk, plan as g29_plan  # noqa: E402
import bundle_data_rr  # noqa: E402

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True,
                    choices=["sglang_v2", "gvr29_hbe", "gvr29_off"])
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--BS", type=int, required=True)
    ap.add_argument("--scenario", default="real")
    ap.add_argument("--launches", type=int, default=5)
    args = ap.parse_args()

    b = bundle_data_rr.get_bundle(args.scenario, args.K, torch.float32,
                                  args.N, device=DEV)
    logits = b["logits"].to(torch.float32).expand(args.BS, -1).contiguous()
    pre = b["preIdx"].to(torch.int32).expand(args.BS, -1).contiguous()
    sl = torch.full((args.BS,), args.N, dtype=torch.int32, device=DEV)
    out = torch.empty((args.BS, args.K), dtype=torch.int32, device=DEV)

    if args.op == "sglang_v2":
        md = v2_plan(sl)
        call = lambda: topk_v2(logits, sl, args.K, out=out, metadata=md,  # noqa: E731
                               max_seq_len=args.N)
    else:
        md = g29_plan(sl)
        hbe = args.op == "gvr29_hbe"
        call = lambda: gvr29_topk(logits, sl, args.K, pre, out=out,  # noqa: E731
                                  metadata=md, max_seq_len=args.N,
                                  use_hbe=hbe)
    for _ in range(args.launches):
        call()
    torch.cuda.synchronize()
    print(f"CELL DONE {args.op} K={args.K} N={args.N} BS={args.BS}",
          flush=True)


if __name__ == "__main__":
    main()
