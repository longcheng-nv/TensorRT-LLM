# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op30 phase 1 — calibration measurement: gvr_cutedsl (object) +
radix_cutedsl (data-insensitivity control) over the calib cfg×hr×N grid,
fp32 BS=1, nsys pure-kernel protocol (harness measure_cell verbatim).

Run UNDER nsys via drive_calib_op30.sh, one rep per model shard:
  CUDA_VISIBLE_DEVICES=g nsys profile -t cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o <rep> -f true \
    python3 calib_op30.py --model v4pro --out-root ../results_b200_op30_calib
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))

from sweep_nsys import build_call, measure_cell, _load_done  # noqa: E402

from gen_calib_bundles_op30 import (MODELS, CAL_CFGS, CAL_HRS, CAL_NS,  # noqa: E402
                                    calib_dir, hr_tag)

ARMS = ["gvr_cutedsl", "radix_cutedsl"]


def load_calib_bundle(model, cfg, hr, N, device="cuda"):
    d = calib_dir(model, cfg, hr, N)
    meta = json.loads((d / "meta.json").read_text())
    logits = torch.load(d / "logits.pt", map_location=device)
    preIdx = torch.load(d / "preIdx.pt", map_location=device)
    cr = meta["compress_ratio"]
    assert meta["seq_lens_val"] == N * cr
    return (logits.contiguous(), preIdx.to(torch.int32).contiguous(), cr,
            meta["realised_hr_mean"], meta["rows"][0].get("layer"),
            meta["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=20)
    args = ap.parse_args()

    model, K = args.model, MODELS[args.model]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"calib_{model}.jsonl"
    done = _load_done(out_path)   # key: (sweep, op, K, dtype, N, BS)

    cells = [(N, cfg, hr) for N in CAL_NS if N > 2 * K
             for cfg in CAL_CFGS for hr in CAL_HRS
             if (calib_dir(model, cfg, hr, N) / "meta.json").exists()]
    print(f"# calib_op30: model={model} K={K} cells={len(cells)} "
          f"arms={ARMS} reps={args.reps}/{args.reps_warm}", flush=True)

    f = open(out_path, "a")
    exact_done = set()
    prof.start()
    try:
        for i, (N, cfg, hr) in enumerate(cells):
            logits_row, preidx_row, cr, real_hr, layer, seed = \
                load_calib_bundle(model, cfg, hr, N)
            for op in ARMS:
                # encode (cfg, hr) in the fake "dtype" slot of the done-key
                dt_slot = f"{cfg}|{hr_tag(hr)}"
                if ("calib", op, K, dt_slot, N, 1) in done:
                    continue
                base = f"{op}|{model}|{cfg}|{hr_tag(hr)}|{N}"
                rec = {"sweep": "calib", "op": op, "K": K, "dtype": dt_slot,
                       "model": model, "cfg": cfg, "target_hr": hr,
                       "realised_hr": real_hr, "layer": layer, "seed": seed,
                       "N": N, "BS": 1, "cr": cr,
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "reps_cold": args.reps, "reps_warm": args.reps_warm}
                try:
                    call, keep, extra = build_call(op, K, torch.float32, N, 1,
                                                   cr, logits_row, preidx_row)
                    rec.update(extra)
                    if (op, cfg, hr, N) not in exact_done:
                        exact_done.add((op, cfg, hr, N))
                        call()
                        torch.cuda.synchronize()
                        ref = torch.topk(logits_row[0, :N].float(),
                                         K).values.sort().values
                        got = logits_row[0, :N].float()[
                            keep[3][0].long()].sort().values
                        rec["exact"] = ("ok" if torch.equal(got, ref)
                                        else "FAIL")
                    measure_cell(call, base, args.reps, args.reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            if (i + 1) % 6 == 0 or i + 1 == len(cells):
                print(f"[calib {model}] {i+1}/{len(cells)} "
                      f"(N={N} {cfg} hr={hr})", flush=True)
    finally:
        prof.stop()
    f.close()
    print("CALIB BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
