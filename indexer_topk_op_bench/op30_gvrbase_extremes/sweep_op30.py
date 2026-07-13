# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op30 — nsys pure-kernel sweep, 10 arms, GVR-base-relative scenarios.

Arms (baseline first; all arms read the SAME bundle row per cell, so inputs
are byte-identical across ops at every test point):
    gvr_cutedsl            original single-CTA production GVR   [BASELINE]
    gvr_multicta_cutedsl   PR#15198 cluster kernel + host auto cluster_size
    radix_cutedsl
    radix_single_cuda
    radix_multi_cuda
    op25_hls               gvr_ms_auto @HEAD, OP27_K2048_TAIL=0 (op25 ship:
                           w3a ladder + slot_scale2 + fp32 C8; stock K2048)
    op27_hls               gvr_ms_auto @HEAD, OP27_K2048_TAIL=1 (op27 ship:
                           + K2048 tail ladder 0.75/0.45/0.048)
    op26_r0auto            op26 R0 dispatch (1cta/mc auto + small-N gate)
    sglang_v2              sglang@main 2026-07 topk v2   (fp32-only)
    flashinfer_topk        flashinfer.top_k 0.6.11       (fp32-only)

op25_hls / op27_hls coexist in ONE process: OP27_K2048_TAIL is re-read per
call inside _qfracs_for and qfracs is part of both compile-cache keys
(gvr_ms_op.py:2056, gvr_msc_op.py:1596); each arm's wrapper pins its env
before build AND before every call (ab_p0batch pattern, as sweep_op22rr).

Scenarios = op30 GVR-base absolute-time extremes via bundle_data_op30
(scen_op30.json from the phase-1 calibration).

Timing protocol IDENTICAL to sweep_op22rr.py (measure_cell: 10 warmup,
warm-L2 "w|" reps, 512MB-evict cold-L2 "c|" reps, eager+sync in range,
cudaProfilerApi window). Exactness: at BS=1 each arm's output index set is
validated (sorted VALUES vs torch.topk) once per (K, dtype, N); recorded
per-arm in the jsonl.

Run UNDER nsys via drive_nsys_op30.sh.
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
sys.path.insert(0, str(HERE.parents[0] / "op28_ext_topk"))

from sweep import DTYPES, KS                      # noqa: E402
from sweep_nsys import measure_cell, _load_done   # noqa: E402
from ops_ext import build_call_ext                # noqa: E402

import bundle_data_op30                            # noqa: E402

# env knobs owned by the gvr_ms_auto arms; pinned (set or unset) per arm
KNOBS = ("OP21_FB_LOGFALSI", "OP21_FB_DIST", "OP27_K2048_TAIL")
FP32_ONLY = {"sglang_v2", "flashinfer_topk"}

# arm name -> (harness op, env pin dict; KNOBS keys absent -> unset)
ARMS = [
    ("gvr_cutedsl", "gvr_cutedsl", {}),
    ("gvr_multicta_cutedsl", "gvr_multicta_cutedsl", {}),
    ("radix_cutedsl", "radix_cutedsl", {}),
    ("radix_single_cuda", "radix_single_cuda", {}),
    ("radix_multi_cuda", "radix_multi_cuda", {}),
    ("op25_hls", "gvr_ms_auto",
     {"OP21_FB_LOGFALSI": "1", "OP27_K2048_TAIL": "0"}),
    ("op27_hls", "gvr_ms_auto",
     {"OP21_FB_LOGFALSI": "1", "OP27_K2048_TAIL": "1"}),
    ("op26_r0auto", "op26_r0auto", {}),
    ("sglang_v2", "sglang_v2", {}),
    ("flashinfer_topk", "flashinfer_topk", {}),
]

_ARM_FILTER = os.environ.get("OP30_ARMS")
if _ARM_FILTER:
    _sel = [a.strip() for a in _ARM_FILTER.split(",") if a.strip()]
    _by_name = {a[0]: a for a in ARMS}
    unknown = [a for a in _sel if a not in _by_name]
    assert not unknown, f"OP30_ARMS unknown arms: {unknown}"
    ARMS = [_by_name[a] for a in _sel]

N_SEQ_MAIN = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
N_HUGE = [524288, 1048576]
N_SEQ_EXT = N_SEQ_MAIN + N_HUGE
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BS_HUGE = [2, 4, 8, 16, 32, 64]

_N_FILTER = os.environ.get("OP30_NS")
if _N_FILTER:
    _ns = {int(x) for x in _N_FILTER.split(",") if x.strip()}
    N_SEQ_MAIN = [n for n in N_SEQ_MAIN if n in _ns]
    N_HUGE = [n for n in N_HUGE if n in _ns]
    N_SEQ_EXT = N_SEQ_MAIN + N_HUGE

_BS_FILTER = os.environ.get("OP30_BS")
if _BS_FILTER:
    _bss = {int(x) for x in _BS_FILTER.split(",") if x.strip()}
    BS_GRID = [b for b in BS_GRID if b in _bss]
    BS_HUGE = [b for b in BS_HUGE if b in _bss]

SUBDIR = {"seqlen": "seqlen_sweep", "bs": "bs_scaling", "bs_hugeN": "bs_hugeN"}


def arms_for(dtype_name, K):
    return [a for a in ARMS
            if dtype_name == "fp32" or a[0] not in FP32_ONLY]


def _pin_env(env):
    for var in KNOBS:
        val = env.get(var)
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def _exact_idx(arm, keep):
    """Output index row 0 for the exactness check, per arm keep layout."""
    if arm == "sglang_v2":
        return keep[2][0]
    if arm == "flashinfer_topk":
        import flashinfer
        return flashinfer.top_k(keep[0][:1], keep[2].shape[1])[1][0]
    return keep[3][0]


def run_batch(sweep, scenario, cells, out_path, reps_cold, reps_warm):
    done = _load_done(out_path)
    f = open(out_path, "a")
    total = len(cells)
    exact_done = set()
    prof.start()
    try:
        for i, (K, dt_name, N, BS) in enumerate(cells):
            dtype = DTYPES[dt_name]
            b = bundle_data_op30.get_bundle(scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            for arm, op, env in arms_for(dt_name, K):
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
                    _pin_env(env)
                    call, keep, extra = build_call_ext(op, K, dtype, N, BS,
                                                       cr, logits_row,
                                                       preidx_row)
                    rec.update(extra)

                    def wrapped(_c=call, _e=env):
                        _pin_env(_e)
                        _c()
                    if BS == 1 and (arm, K, dt_name, N) not in exact_done:
                        exact_done.add((arm, K, dt_name, N))
                        wrapped()
                        torch.cuda.synchronize()
                        ref = torch.topk(logits_row[0, :N].float(),
                                         K).values.sort().values
                        row_f32 = logits_row[0, :N].float()
                        got = row_f32[_exact_idx(arm, keep).long()
                                      ].sort().values
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
    ap.add_argument("--scenario", choices=["best", "worst"], required=True)
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
    print(f"# op30 nsys batch: scenario={args.scenario} sweep={args.sweep} "
          f"K={K} dt={dt} cells={len(cells)} "
          f"arms={[a[0] for a in arms_for(dt, K)]} "
          f"reps_cold={args.reps} reps_warm={args.reps_warm}", flush=True)
    run_batch(args.sweep, args.scenario, cells, out_path,
              args.reps, args.reps_warm)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
