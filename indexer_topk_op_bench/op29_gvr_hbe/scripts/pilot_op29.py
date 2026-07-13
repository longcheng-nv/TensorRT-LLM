#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op29 pilot: nsys same-batch paired A/B on HBE-engaged cells.

Arms: sglang_v2 (RIVAL, untouched vendored build) | gvr29_hbe | gvr29_off
(fork parity control). Protocol identical to op28 measure_cell (10 warmup,
50 warm-L2, 20 cold-L2 512MB-evict, NVTX c|/w| ranges, cudaProfilerApi).

Run under nsys (see pilot launch cmds in RESUME_PROMPT.md):
  CUDA_VISIBLE_DEVICES=<g> nsys profile -t cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o <rep> -f true \
    python3 pilot_op29.py --scenario real --out <jsonl>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "harness"))
sys.path.insert(0, str(HERE.parents[1] / "op22_temporal_fixed_hr_bench"))

from sweep_nsys import measure_cell  # noqa: E402
from sglang_v2_op import topk_v2, plan as v2_plan  # noqa: E402
from gvr29_op import gvr29_topk, plan as g29_plan  # noqa: E402
import bundle_data_rr  # noqa: E402

DEV = "cuda"
# (N, BS) HBE-engaged pilot cells
CELLS = [(32768, 4), (32768, 64), (65536, 64),
         (32768, 1024), (65536, 1024), (131072, 1024), (262144, 1024),
         (65536, 2048), (262144, 2048)]
KS = [512, 2048]


def build(op, K, N, BS, b):
    logits = b["logits"].to(torch.float32).expand(BS, -1).contiguous()
    pre = b["preIdx"].to(torch.int32).expand(BS, -1).contiguous()
    sl = torch.full((BS,), N, dtype=torch.int32, device=DEV)
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, pre, sl, out]
    if op == "sglang_v2":
        md = v2_plan(sl)
        keep.append(md)
        topk_v2(logits, sl, K, out=out, metadata=md, max_seq_len=N)
        return (lambda: topk_v2(logits, sl, K, out=out, metadata=md,
                                max_seq_len=N)), keep
    md = g29_plan(sl)
    keep.append(md)
    hbe = (op == "gvr29_hbe")
    gvr29_topk(logits, sl, K, pre, out=out, metadata=md, max_seq_len=N,
               use_hbe=hbe)
    return (lambda: gvr29_topk(logits, sl, K, pre, out=out, metadata=md,
                               max_seq_len=N, use_hbe=hbe)), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    args = ap.parse_args()
    f = open(args.out, "a")
    prof.start()
    try:
        for K in KS:
            for N, BS in CELLS:
                if N <= 2 * K:
                    continue
                b = bundle_data_rr.get_bundle(args.scenario, K,
                                              torch.float32, N, device=DEV)
                for op in ("sglang_v2", "gvr29_hbe", "gvr29_off"):
                    base = f"{op}|{K}|fp32|{N}|{BS}"
                    rec = {"op": op, "K": K, "N": N, "BS": BS,
                           "scenario": args.scenario,
                           "range_cold": f"c|{base}",
                           "range_warm": f"w|{base}"}
                    try:
                        call, keep = build(op, K, N, BS, b)
                        measure_cell(call, base, args.reps, args.reps_warm)
                        del call, keep
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {str(e)[:100]}"
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    torch.cuda.empty_cache()
                print(f"[{args.scenario}] K={K} N={N} BS={BS} done",
                      flush=True)
    finally:
        prof.stop()
    f.close()
    print("PILOT DONE", flush=True)


if __name__ == "__main__":
    main()
