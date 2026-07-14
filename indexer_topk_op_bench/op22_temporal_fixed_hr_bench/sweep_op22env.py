# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 ENV — unified 9-arm nsys pure-kernel sweep on the LATEST-skill GVR
performance-envelope bundles (bundle_data_env; best/worst fixed-hr).

All 9 arms read the SAME bundle row per cell (inputs byte-identical), and ALL
run in ONE process on ONE node -> per-cell ratios need NO cross-node anchor
transfer. Arms (baseline first):

  gvr_cutedsl            GVR (cuteDSL) — base                     [BASELINE]
  radix_cutedsl          Radix (cuteDSL)
  gvr_multicta_cutedsl   GVR multi-CTA (cuteDSL, PR#15198)
  radix_single_cuda      Radix single-CTA (CUDA)
  radix_multi_cuda       Radix multi-CTA (CUDA)
  op27_hls               GVR op#21 ms_auto (HLS-op27)  = gvr_ms_auto,
                         OP21_FB_LOGFALSI=1 + OP21_FB_DIST unset, OP27 tail ON
  op26_r0auto            GVR op#26 R0 (auto 1CTA/MC dispatch)
  sglang_v2              SGLang v2 top-K (main 2026-07)   [external, fp32]
  flashinfer_topk        FlashInfer top_k (0.6.11)        [external, fp32]

fp32 ONLY (sglang_v2 / flashinfer are fp32-only). K 512/1024/2048.
sglang_v2 + flashinfer are run at all K (matching op28).

Timing protocol IDENTICAL to sweep_op22rr.py / harness/sweep_nsys.py:
10 warmup, 50 warm-L2 reps ("w|"), 20 cold-L2 reps with 512MB evict ("c|"),
eager + sync inside the NVTX range, whole loop inside the cudaProfilerApi
window. Run UNDER nsys via drive_nsys_op22env.sh.
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

import bundle_data_env                            # noqa: E402

KNOB_F, KNOB_D = "OP21_FB_LOGFALSI", "OP21_FB_DIST"
# (arm label, harness op, OP21_FB_LOGFALSI, OP21_FB_DIST); None -> unset.
ARMS9 = [
    ("gvr_cutedsl", "gvr_cutedsl", None, None),
    ("radix_cutedsl", "radix_cutedsl", None, None),
    ("gvr_multicta_cutedsl", "gvr_multicta_cutedsl", None, None),
    ("radix_single_cuda", "radix_single_cuda", None, None),
    ("radix_multi_cuda", "radix_multi_cuda", None, None),
    ("op27_hls", "gvr_ms_auto", "1", None),
    ("op26_r0auto", "op26_r0auto", None, None),
    ("sglang_v2", "sglang_v2", None, None),
    ("flashinfer_topk", "flashinfer_topk", None, None),
]

N_SEQ_MAIN = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
N_HUGE = [524288, 1048576]
N_SEQ_EXT = N_SEQ_MAIN + N_HUGE
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BS_HUGE = [2, 4, 8, 16, 32, 64]
SUBDIR = {"seqlen": "seqlen_sweep", "bs": "bs_scaling", "bs_hugeN": "bs_hugeN"}

# opt-in arm filter (comma list) — default all 9
_AF = os.environ.get("OP22ENV_ARMS")
if _AF:
    _sel = [a.strip() for a in _AF.split(",") if a.strip()]
    _by = {a[0]: a for a in ARMS9}
    ARMS9 = [_by[a] for a in _sel]


def _pin_env(falsi, dist):
    for var, val in ((KNOB_F, falsi), (KNOB_D, dist)):
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def run_batch(sweep, scenario, cells, out_path, reps_cold, reps_warm):
    done = _load_done(out_path)
    f = open(out_path, "a")
    total = len(cells)
    prof.start()
    try:
        for i, (K, dt_name, N, BS) in enumerate(cells):
            dtype = DTYPES[dt_name]
            b = bundle_data_env.get_bundle(scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            for arm, op, falsi, dist in ARMS9:
                key = (sweep, arm, K, dt_name, N, BS)
                if key in done:
                    continue
                base = f"{arm}|{K}|{dt_name}|{N}|{BS}"
                rec = {"sweep": sweep, "op": arm, "harness_op": op,
                       "K": K, "dtype": dt_name, "N": N, "BS": BS, "cr": cr,
                       "scenario": scenario, "data_src": b["cfg"],
                       "hit_rate": b["kernel_hit_rate"],
                       "layer": b["row_meta"].get("layer"),
                       "seed": b["seed"],
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "reps_cold": reps_cold, "reps_warm": reps_warm}
                try:
                    _pin_env(falsi, dist)
                    call, keep, extra = build_call_ext(op, K, dtype, N, BS, cr,
                                                       logits_row, preidx_row)
                    rec.update(extra)

                    def wrapped(_c=call, _f=falsi, _d=dist):
                        _pin_env(_f, _d)
                        _c()
                    measure_cell(wrapped, base, reps_cold, reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            if (i + 1) % 2 == 0 or i + 1 == total:
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
    ap.add_argument("--scenario", choices=["best", "worst", "slowbase"],
                    required=True)
    ap.add_argument("--K", type=int, required=True, choices=KS)
    ap.add_argument("--dtype", default="fp32", choices=["fp32"])
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
    print(f"# op22env nsys batch: scenario={args.scenario} sweep={args.sweep} "
          f"K={K} dt={dt} cells={len(cells)} arms={[a[0] for a in ARMS9]} "
          f"reps_cold={args.reps} reps_warm={args.reps_warm}", flush=True)
    run_batch(args.sweep, args.scenario, cells, out_path,
              args.reps, args.reps_warm)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
