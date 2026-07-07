# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op21 iter13 — nsys A/B of the HLS log-falsi fallback on the op22 bundles.

Same-process PAIRED A/B (throttle-immune): for each cell both arms are
compiled in one process (OP21_FB_LOGFALSI is part of the _compile key; the
env var is flipped between build_call invocations) and measured back-to-back
inside one cudaProfilerApi window with distinct NVTX ranges:
    c|old|<cell>  /  c|new|<cell>   (cold-L2, 512MB evict outside the range)

Cells = where the HLS model says the fallback tail lives: the op22 stress
scenarios (best hr=.90 / worst hr=.05, static-placement fast rate 0%) plus
real (fast rate 69%), at large N. Exactness cross-check per cell: sorted
index sets of old vs new vs torch.topk values (GVR row order is runtime-
nondeterministic — op21 iter12 LEARNINGS — so only sorted sets compare).

Usage (one nsys run per scenario):
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o <rep> python3 scripts/ab_hls_logfalsi.py --scenario best \
    --out <jsonl> [--reps 30]
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))

from sweep import DTYPES  # noqa: E402
from sweep_nsys import build_call, _EVICT  # noqa: E402
import bundle_data  # noqa: E402

# (K, N, BS) grid: seqlen tail (BS=1) + one msc BS spot per K. N >= 65536 is
# the msc cluster regime; 262144/1048576 are where the model tail peaks;
# 16384/32768 cover the single-CTA ms fallback (fused gate 4K <= kC).
CELLS = []
for K in (512, 1024, 2048):
    for N in (16384, 65536, 262144, 1048576):
        if N > 2 * K:
            CELLS.append((K, N, 1))
    CELLS.append((K, 262144, 16))

REF_CHECK_BS_ROWS = (0, -1)


def sorted_vals(logits_row_f32, idx):
    return logits_row_f32[idx.long()].sort().values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    dtype = DTYPES[args.dtype]

    f = open(args.out, "a")
    prof.start()
    try:
        for (K, N, BS) in CELLS:
            b = bundle_data.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            base = f"{args.scenario}|{K}|{args.dtype}|{N}|{BS}"
            rec = {"scenario": args.scenario, "K": K, "dtype": args.dtype,
                   "N": N, "BS": BS, "cr": cr,
                   "hit_rate": b["kernel_hit_rate"], "seed": b["seed"],
                   "reps": args.reps}
            try:
                arms = {}
                for arm, env in (("old", "0"), ("new", "1")):
                    call, keep, extra = build_call(
                        "gvr_ms_auto", K, dtype, N, BS, cr,
                        logits_row, preidx_row)

                    # gvr_ms_auto re-reads OP21_FB_LOGFALSI at EVERY
                    # invocation (_compile is lazy + env-keyed), so the env
                    # must be pinned per-call, not per-build.
                    def wrapped(_c=call, _e=env):
                        os.environ["OP21_FB_LOGFALSI"] = _e
                        _c()
                    arms[arm] = (wrapped, keep)
                    rec.setdefault("ms_path", extra.get("ms_path"))
                # exactness cross-check (pre-timing; sorted-set criterion)
                ref = torch.topk(logits_row[0, :N].float(), K).values.sort().values
                for arm, (call, keep) in arms.items():
                    call(); torch.cuda.synchronize()
                    out_idx = keep[3]
                    for r in REF_CHECK_BS_ROWS:
                        got = sorted_vals(logits_row[0, :N].float(),
                                          out_idx[r])
                        if not torch.equal(got, ref):
                            rec[f"exact_{arm}"] = "FAIL"
                            break
                    else:
                        rec[f"exact_{arm}"] = "ok"
                # paired cold-L2 measurement, arms interleaved per rep
                for arm, (call, keep) in arms.items():
                    for _ in range(args.warmup):
                        call()
                torch.cuda.synchronize()
                for _ in range(args.reps):
                    for arm, (call, keep) in arms.items():
                        _EVICT.uniform_(0, 1)
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push(f"c|{arm}|{base}")
                        call()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                del arms
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"done {base}", flush=True)
    finally:
        prof.stop()
    f.close()
    print("AB BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
