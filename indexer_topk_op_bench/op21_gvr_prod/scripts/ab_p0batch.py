# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op21 P0-batch — same-process paired multi-arm nsys A/B on the op22 bundles.

Extends ab_hls_logfalsi.py (iter13) to a THREE-arm direct verdict:
    orig    = gvr_cutedsl (original single-CTA production GVR)   [--with-orig]
    legacy  = gvr_ms_auto, OP21_FB_LOGFALSI=0 + OP21_FB_DIST=0 (pre-HLS op21)
    shipped = gvr_ms_auto, OP21_FB_LOGFALSI=1 + OP21_FB_DIST unset (N-rule
              default: dist fires iff n >= 524288) — the HLS-GVR ship config.

All arms are compiled in ONE process (the OP21_* knobs are part of the lazy
_compile key and are re-read per invocation, so each arm's wrapper pins its
env before every call) and measured back-to-back inside one cudaProfilerApi
window, interleaved per rep (throttle-immune), cold-L2 (512MB evict outside
the NVTX range):
    c|orig|<cell> / c|legacy|<cell> / c|shipped|<cell>

Cell sets:
    tail   = iter13 grid (BS=1 seqlen tail + 262144 BS16 spot per K)
    highbs = BS 64/256/1024 coverage incl. 1M BS64 (single-CTA ms path:
             gvr_ms_auto dispatches C=4 only when 4*BS <= NUM_SMS)

Exactness per cell per arm: sorted VALUE sets vs torch.topk on the same
dtype-truncated row (GVR output order is runtime-nondeterministic — iter12
LEARNINGS — so only sorted comparisons are valid).

Usage (one nsys run per scenario x dtype x cellset):
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o <rep> python3 scripts/ab_p0batch.py --scenario best \
    --dtype fp32 --cells tail --with-orig --out <jsonl> [--reps 30]
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

KNOB_F = "OP21_FB_LOGFALSI"
KNOB_D = "OP21_FB_DIST"


def cells_tail():
    cells = []
    for K in (512, 1024, 2048):
        for N in (16384, 65536, 262144, 1048576):
            if N > 2 * K:
                cells.append((K, N, 1))
        cells.append((K, 262144, 16))
    return cells


def cells_highbs():
    cells = []
    for K in (512, 1024, 2048):
        for (N, BS) in ((4096, 256), (4096, 1024),
                        (16384, 64), (16384, 256), (16384, 1024),
                        (65536, 64), (65536, 256),
                        (262144, 64), (1048576, 64)):
            if N > 2 * K:
                cells.append((K, N, BS))
    return cells


CELLSETS = {"tail": cells_tail, "highbs": cells_highbs}


def _pin_env(falsi, dist):
    """falsi/dist: '0'/'1' to pin, None to unset (kernel default rule)."""
    for var, val in ((KNOB_F, falsi), (KNOB_D, dist)):
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--cells", default="tail", choices=list(CELLSETS))
    ap.add_argument("--with-orig", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    dtype = DTYPES[args.dtype]

    # (arm, op, falsi, dist) — env pinned before EVERY call (lazy env-keyed
    # compile) and also during build_call (warm compile happens there for
    # non-ms ops).
    ARMS = []
    if args.with_orig:
        ARMS.append(("orig", "gvr_cutedsl", None, None))
    ARMS.append(("legacy", "gvr_ms_auto", "0", "0"))
    ARMS.append(("shipped", "gvr_ms_auto", "1", None))

    # resumability: skip cells already recorded in the jsonl
    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["K"], r["N"], r["BS"]))
            except Exception:
                pass

    f = open(args.out, "a")
    prof.start()
    try:
        for (K, N, BS) in CELLSETS[args.cells]():
            if (K, N, BS) in done:
                print(f"skip {K}/{N}/{BS} (done)", flush=True)
                continue
            b = bundle_data.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            base = f"{args.scenario}|{K}|{args.dtype}|{N}|{BS}"
            rec = {"scenario": args.scenario, "K": K, "dtype": args.dtype,
                   "N": N, "BS": BS, "cr": cr,
                   "hit_rate": b["kernel_hit_rate"], "seed": b["seed"],
                   "reps": args.reps, "arms": [a[0] for a in ARMS]}
            try:
                arms = {}
                for arm, op, falsi, dist in ARMS:
                    _pin_env(falsi, dist)
                    call, keep, extra = build_call(
                        op, K, dtype, N, BS, cr, logits_row, preidx_row)

                    def wrapped(_c=call, _f=falsi, _d=dist):
                        _pin_env(_f, _d)
                        _c()
                    arms[arm] = (wrapped, keep)
                    if extra.get("ms_path"):
                        rec.setdefault("ms_path", extra["ms_path"])
                # exactness cross-check (pre-timing; sorted-set criterion)
                ref = torch.topk(logits_row[0, :N].float(),
                                 K).values.sort().values
                row_f32 = logits_row[0, :N].float()
                for arm, (call, keep) in arms.items():
                    call(); torch.cuda.synchronize()
                    out_idx = keep[3]
                    for r in (0, -1):
                        got = row_f32[out_idx[r].long()].sort().values
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
