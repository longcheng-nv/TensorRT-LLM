# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 Step 2 — nsys A/B of the S1a screened ladder (wide4b + slot_scale=2)
on the op22rr bundles.

Same-process PAIRED three-arm A/B (throttle-immune), iter13 protocol:
    base  = gvr_ms_auto, OP25_QFRACS=base OP25_SLOTCAP=1  (bit-path of HEAD)
    ship  = gvr_ms_auto, defaults (per-K wide4b table + slot_scale 2)
    radix = radix_cutedsl (the rival the worst scenario loses to)
NVTX ranges c|<arm>|<cell>, cold-L2 512MB evict outside the range.
Exactness: sorted VALUE sets vs torch.topk per arm (GVR row order is
runtime-nondeterministic).

Cells: deployment envelope (N <= 262144), BS=1 tail + BS spots.

Usage (one nsys run per scenario):
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o <rep> python3 ab_qfracs.py --scenario worst \
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
BENCH = HERE.parents[0]
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))

from sweep import DTYPES  # noqa: E402
from sweep_nsys import build_call, _EVICT  # noqa: E402
import bundle_data_rr  # noqa: E402

CELLS = []
for K in (512, 1024, 2048):
    for N in (8192, 16384, 65536, 131072, 262144):
        CELLS.append((K, N, 1))
    CELLS.extend([(K, 8192, 512), (K, 65536, 64), (K, 131072, 16)])

# arm -> (op, {env}) ; None value = pop the var (ship defaults)
ARMS = (
    ("base", "gvr_ms_auto", {"OP25_QFRACS": "base", "OP25_SLOTCAP": "1"}),
    ("ship", "gvr_ms_auto", {"OP25_QFRACS": None, "OP25_SLOTCAP": None}),
    ("radix", "radix_cutedsl", {}),
)


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
            b = bundle_data_rr.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            base_tag = f"{args.scenario}|{K}|{args.dtype}|{N}|{BS}"
            rec = {"scenario": args.scenario, "K": K, "dtype": args.dtype,
                   "N": N, "BS": BS, "cr": cr,
                   "hit_rate": b["kernel_hit_rate"], "reps": args.reps}
            try:
                arms = {}
                for arm, op, env in ARMS:
                    call, keep, extra = build_call(
                        op, K, dtype, N, BS, cr, logits_row, preidx_row)

                    # gvr_ms_auto re-reads the knobs at EVERY invocation
                    # (_compile is lazy + env-keyed) -> pin per-call.
                    def wrapped(_c=call, _e=dict(env)):
                        for k, v in _e.items():
                            if v is None:
                                os.environ.pop(k, None)
                            else:
                                os.environ[k] = v
                        _c()
                    arms[arm] = (wrapped, keep)
                    if extra.get("ms_path"):
                        rec.setdefault("ms_path", extra.get("ms_path"))
                # exactness (pre-timing; sorted-value-set criterion)
                ref = torch.topk(logits_row[0, :N].float(),
                                 K).values.sort().values
                for arm, (call, keep) in arms.items():
                    call(); torch.cuda.synchronize()
                    out_idx = keep[3]
                    ok = True
                    for r in (0, -1):
                        got = logits_row[0, :N].float()[
                            out_idx[r].clamp(min=0).long()].sort().values
                        if not torch.equal(got, ref):
                            ok = False
                            break
                    rec[f"exact_{arm}"] = "ok" if ok else "FAIL"
                # paired cold-L2, arms interleaved per rep
                for arm, (call, keep) in arms.items():
                    for _ in range(args.warmup):
                        call()
                torch.cuda.synchronize()
                for _ in range(args.reps):
                    for arm, (call, keep) in arms.items():
                        _EVICT.uniform_(0, 1)
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push(f"c|{arm}|{base_tag}")
                        call()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                del arms
            except Exception as e:  # noqa: BLE001 — record and continue
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"done {base_tag}", flush=True)
    finally:
        prof.stop()
    f.close()
    print("AB BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
