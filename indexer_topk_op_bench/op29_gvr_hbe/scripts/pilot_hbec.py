#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op31 HBE-C pilot: nsys same-batch paired A/B on the cluster domain.

Arms: gvr_cutedsl (ANCHOR) | sglang_v2 (RIVAL, untouched vendored build) |
gvr29_hbec (tier-5, GVR29_HBEC=1 env set here) | gvr29_hbec_off (fork-parity
control: same build, use_hbe=False -> stock cluster dispatch).
Cells (DESIGN §6 rung 3): N {131072..1048576} x BS {1,16,64,256,512}.
Protocol identical to pilot_op29 (measure_cell: 10 warmup, warm-L2, 512MB-
evict cold-L2, NVTX c|/w| ranges, cudaProfilerApi window).

Run under nsys:
  CUDA_VISIBLE_DEVICES=<g> env -u GITHUB_TOKEN -u HF_TOKEN nsys profile \
    -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o <rep> -f true python3 pilot_hbec.py --scenario real --out <jsonl>
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["GVR29_HBEC"] = "1"          # before first transform (static read)

import torch  # noqa: E402
import torch.cuda.profiler as prof  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "harness"))
sys.path.insert(0, str(HERE.parents[1] / "op22_temporal_fixed_hr_bench"))

from sweep_nsys import measure_cell  # noqa: E402
from ops_ext29 import build_call_ext29  # noqa: E402
from gvr29_op import gvr29_topk, plan as g29_plan, _spill_buf  # noqa: E402
import bundle_data_rr  # noqa: E402

DEV = "cuda"
CELLS = [(N, BS) for N in (131072, 262144, 524288, 1048576)
         for BS in (1, 16, 64, 256, 512)]
KS = [512, 1024, 2048]
ARMS = ("gvr_cutedsl", "sglang_v2", "gvr29_hbec", "gvr29_hbec_off")


def build(op, K, N, BS, b):
    cr = b["cr"]
    logits_row = b["logits"][:, :N].to(torch.float32)
    pre_row = b["preIdx"].to(torch.int32)
    if op in ("gvr29_hbec", "gvr29_hbec_off"):
        if cr == 1:
            pre_row = (pre_row + 1) % N       # production kernel-read conv
        hbe = (op == "gvr29_hbec")
        logits = logits_row.expand(BS, -1).contiguous()
        pre = pre_row.expand(BS, -1).contiguous()
        sl = torch.full((BS,), N, dtype=torch.int32, device=DEV)
        out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
        md = g29_plan(sl)
        spill = _spill_buf(BS, K, DEV, False)
        keep = [logits, pre, sl, out, md, spill]
        gvr29_topk(logits, sl, K, pre, out=out, metadata=md, max_seq_len=N,
                   use_hbe=hbe, spill=spill)
        return (lambda: gvr29_topk(logits, sl, K, pre, out=out, metadata=md,
                                   max_seq_len=N, use_hbe=hbe,
                                   spill=spill)), keep
    call, keep, _ = build_call_ext29(op, K, torch.float32, N, BS, cr,
                                     logits_row, pre_row)
    return call, keep


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
                try:
                    b = bundle_data_rr.get_bundle(args.scenario, K,
                                                  torch.float32, N,
                                                  device=DEV)
                except (FileNotFoundError, AssertionError) as e:
                    print(f"skip K={K} N={N}: {type(e).__name__}",
                          flush=True)
                    continue
                for op in ARMS:
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
