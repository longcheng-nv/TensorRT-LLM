# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr — nsys pure-kernel sweep, 5 arms, radix-relative scenarios.

Arms (baseline first; all arms read the SAME bundle row per cell, so
inputs are byte-identical across ops at every test point):
    gvr_cutedsl   original single-CTA production GVR   [BASELINE]
    op21_legacy   gvr_ms_auto, OP21_FB_LOGFALSI=0 + OP21_FB_DIST=0 (pre-HLS)
    op21_hls      gvr_ms_auto, OP21_FB_LOGFALSI=1 + OP21_FB_DIST unset
                  (HLS ship defaults at HEAD; iter16: dist gate n>=65536)
    radix_cutedsl
    sglang_streaming   (fp32-only AND K<=1024)

The two op21 arms are compiled in ONE process: the OP21_* knobs are part
of the lazy compile key and re-read per invocation, so each arm's wrapper
pins its env before build AND before every call (ab_p0batch pattern).

Scenarios = op24 radix-relative grid-average definitions via
bundle_data_rr (best/worst regenerated, real = original op22 bundles).

Timing protocol IDENTICAL to sweep_op22.py / harness sweep_nsys.py
(measure_cell: 10 warmup, warm-L2 "w|" reps, 512MB-evict cold-L2 "c|"
reps, eager+sync in range, cudaProfilerApi window). Exactness: at BS=1,
each arm's output index set is validated (sorted VALUES vs torch.topk)
once per (K, dtype, N); recorded per-arm in the jsonl.

Run UNDER nsys via drive_nsys_op22rr.sh.
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
sys.path.insert(0, str(HERE.parents[0] / "harness"))

from sweep import DTYPES, KS                      # noqa: E402
from sweep_nsys import build_call, measure_cell   # noqa: E402

import bundle_data_rr                              # noqa: E402

KNOB_F, KNOB_D = "OP21_FB_LOGFALSI", "OP21_FB_DIST"
# arm name -> (harness op, falsi, dist); None = unset (kernel default)
ARMS = [
    ("gvr_cutedsl", "gvr_cutedsl", None, None),
    ("op21_legacy", "gvr_ms_auto", "0", "0"),
    ("op21_hls", "gvr_ms_auto", "1", None),
    ("radix_cutedsl", "radix_cutedsl", None, None),
    ("sglang_streaming", "sglang_streaming", None, None),
]
# Extra arms NOT in the original 5-arm campaign; selectable only via the
# OP22RR_ARMS env filter so default driver invocations are unchanged.
# gvr_multicta_cutedsl = "GVR multi-CTA (cuteDSL, PR#15198)" — the cluster
# kernel with the PR's host-side cluster_size auto-dispatch (build_call
# records the picked cluster_size per cell).
ARMS_EXTRA = [
    ("gvr_multicta_cutedsl", "gvr_multicta_cutedsl", None, None),
]
# OP22RR_ARMS="gvr_cutedsl,gvr_multicta_cutedsl" -> run only those arms
# (order preserved from ARMS + ARMS_EXTRA). Unset -> original 5 arms.
_ARM_FILTER = os.environ.get("OP22RR_ARMS")
if _ARM_FILTER:
    _sel = [a.strip() for a in _ARM_FILTER.split(",") if a.strip()]
    _by_name = {a[0]: a for a in ARMS + ARMS_EXTRA}
    unknown = [a for a in _sel if a not in _by_name]
    assert not unknown, f"OP22RR_ARMS unknown arms: {unknown}"
    ARMS = [_by_name[a] for a in _sel]

N_SEQ_MAIN = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
N_HUGE = [524288, 1048576]
N_SEQ_EXT = N_SEQ_MAIN + N_HUGE
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BS_HUGE = [2, 4, 8, 16, 32, 64]

SUBDIR = {"seqlen": "seqlen_sweep", "bs": "bs_scaling", "bs_hugeN": "bs_hugeN"}


def arms_for(dtype_name, K):
    out = [a for a in ARMS if a[0] != "sglang_streaming"]
    sgl = next((a for a in ARMS if a[0] == "sglang_streaming"), None)
    if sgl is not None and dtype_name == "fp32" and K <= 1024:
        out.append(sgl)
    return out


def _pin_env(falsi, dist):
    for var, val in ((KNOB_F, falsi), (KNOB_D, dist)):
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["sweep"], r["op"], r["K"], r["dtype"],
                          r["N"], r["BS"]))
            except Exception:
                pass
    return done


def run_batch(sweep, scenario, cells, out_path, reps_cold, reps_warm):
    done = _load_done(out_path)
    f = open(out_path, "a")
    total = len(cells)
    exact_done = set()
    prof.start()
    try:
        for i, (K, dt_name, N, BS) in enumerate(cells):
            dtype = DTYPES[dt_name]
            b = bundle_data_rr.get_bundle(scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            for arm, op, falsi, dist in arms_for(dt_name, K):
                key = (sweep, arm, K, dt_name, N, BS)
                if key in done:
                    continue
                base = f"{arm}|{K}|{dt_name}|{N}|{BS}"
                rec = {"sweep": sweep, "op": arm, "harness_op": op,
                       "K": K, "dtype": dt_name, "N": N, "BS": BS,
                       "cr": cr, "scenario": scenario,
                       "data_src": b["cfg"],
                       "hit_rate": b["kernel_hit_rate"],
                       "layer": b["row_meta"].get("layer"),
                       "seed": b["seed"],
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "reps_cold": reps_cold, "reps_warm": reps_warm}
                try:
                    _pin_env(falsi, dist)
                    call, keep, extra = build_call(op, K, dtype, N, BS, cr,
                                                   logits_row, preidx_row)
                    rec.update(extra)

                    def wrapped(_c=call, _f=falsi, _d=dist):
                        _pin_env(_f, _d)
                        _c()
                    # one-time exactness per (arm, K, dt, N) at BS=1
                    if BS == 1 and (arm, K, dt_name, N) not in exact_done:
                        exact_done.add((arm, K, dt_name, N))
                        wrapped()
                        torch.cuda.synchronize()
                        ref = torch.topk(logits_row[0, :N].float(),
                                         K).values.sort().values
                        row_f32 = logits_row[0, :N].float()
                        got = row_f32[keep[3][0].long()].sort().values
                        rec["exact"] = ("ok" if torch.equal(got, ref)
                                        else "FAIL")
                    measure_cell(wrapped, base, reps_cold, reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            if (i + 1) % 4 == 0 or i + 1 == total:
                print(f"[{scenario}/{sweep} K={cells[0][0]} {cells[0][1]}] "
                      f"{i+1}/{total} (N={N} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()


def cells_for(sweep, K, dt):
    if sweep == "seqlen":
        return [(K, dt, N, 1) for N in N_SEQ_EXT if N > 2 * K]
    if sweep == "bs":
        return [(K, dt, N, BS) for N in N_SEQ_MAIN if N > 2 * K
                for BS in BS_GRID]
    if sweep == "bs_hugeN":
        return [(K, dt, N, BS) for N in N_HUGE for BS in BS_HUGE]
    raise ValueError(sweep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=list(SUBDIR), required=True)
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--K", type=int, required=True, choices=KS)
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20, help="cold-L2 reps")
    ap.add_argument("--reps-warm", type=int, default=50, help="warm-L2 reps")
    args = ap.parse_args()

    K, dt = args.K, args.dtype
    sub = SUBDIR[args.sweep]
    results = Path(args.out_root)
    (results / sub).mkdir(parents=True, exist_ok=True)
    out_path = results / sub / f"results_K{K}_{dt}.jsonl"
    cells = cells_for(args.sweep, K, dt)
    print(f"# op22rr nsys batch: scenario={args.scenario} sweep={args.sweep} "
          f"K={K} dt={dt} cells={len(cells)} "
          f"arms={[a[0] for a in arms_for(dt, K)]} "
          f"reps_cold={args.reps} reps_warm={args.reps_warm}", flush=True)
    run_batch(args.sweep, args.scenario, cells, out_path,
              args.reps, args.reps_warm)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
