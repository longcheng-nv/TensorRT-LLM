# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 backlog-1: qfracs UH4 vs M2D on-silicon paired nsys A/B.

Host screening (screen_r0_qfracs.py, ITERATIONS.md L196-204) showed
uh4 (0.90,0.65,0.40,0.15) with higher static admission than the shipped
M2D (0.85,0.35). Pass count != latency (iter5 lesson) -> silicon check.

No new harness arm: both arms call the SAME wrapper with the qfracs=
parameter (compile-key includes qfracs), so paired deltas are purely the
ladder. Representative grid per RESUME_POST_ITER7.md section 2:

  mc port   (gvr_r0_mc_op26):  K1024 fp32 N=131072 BS 1-16
                               K2048 fp16 N=65536..262144 BS 1-8
            + win-guard band   N in {8192,16384,32768} BS {1,8}
  1cta port (gvr_r0_op26):     K2048 fp16 N in {16384,32768} BS {1,8}
            (the only small-N band where prod r0auto still runs the
             1cta R0 ladder after the 07-13 small-N gate)

Timing protocol identical to sweep_op22rr.py (measure_cell: 10 warmup,
warm-L2 reps, 512MB-evict cold-L2 reps, cudaProfilerApi window).
Run UNDER nsys via drive_nsys_qfracs.sh.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[0]
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(HERE / "src"))

from sweep import DTYPES                          # noqa: E402
from sweep_nsys import measure_cell               # noqa: E402
import bundle_data_rr                             # noqa: E402
from gvr_op26_r0_op import gvr_r0_op26            # noqa: E402
from gvr_op26_r0mc_op import (gvr_r0_mc_op26,     # noqa: E402
                              picked_cluster_size_r0mc)

DEV = "cuda"
UH4 = (0.90, 0.65, 0.40, 0.15)
ARMS = [("m2d", None), ("uh4", UH4)]   # qfracs=None -> shipped default M2D

# (K, dtype) -> list of (port, N, BS)
GRIDS = {
    (1024, "fp32"):
        [("mc", 131072, bs) for bs in (1, 2, 4, 8, 16)] +
        [("mc", n, bs) for n in (8192, 16384, 32768) for bs in (1, 8)],
    (2048, "fp16"):
        [("mc", n, bs) for n in (65536, 131072, 262144)
         for bs in (1, 2, 4, 8)] +
        [("mc", n, bs) for n in (8192, 16384, 32768) for bs in (1, 8)] +
        [("r1cta", n, bs) for n in (16384, 32768) for bs in (1, 8)],
}


def _build(port, K, dtype, N, BS, cr, logits_row, preidx_row, qf):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_div, pre, out]
    if port == "mc":
        cs = picked_cluster_size_r0mc(logits, K, cr)
        gvr_r0_mc_op26(logits, pre, seq_div, K, compress_ratio=cr,
                       out=out, qfracs=qf)
        return (lambda: gvr_r0_mc_op26(logits, pre, seq_div, K,
                                       compress_ratio=cr, out=out,
                                       qfracs=qf)), keep, {"cluster_size": cs}
    gvr_r0_op26(logits, pre, seq_div, K, compress_ratio=cr,
                out=out, qfracs=qf)
    return (lambda: gvr_r0_op26(logits, pre, seq_div, K, compress_ratio=cr,
                                out=out, qfracs=qf)), keep, {}


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["port"], r["op"], r["K"], r["dtype"],
                          r["N"], r["BS"]))
            except json.JSONDecodeError:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    args = ap.parse_args()

    K, dt_name = args.K, args.dtype
    dtype = DTYPES[dt_name]
    cells = GRIDS[(K, dt_name)]
    out_dir = Path(args.out_root) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_K{K}_{dt_name}.jsonl"
    done = _load_done(out_path)
    print(f"# qfracs A/B batch: scenario={args.scenario} K={K} dt={dt_name} "
          f"cells={len(cells)} x {len(ARMS)} arms "
          f"reps_cold={args.reps} reps_warm={args.reps_warm}", flush=True)

    f = open(out_path, "a")
    exact_done = set()
    prof.start()
    try:
        for i, (port, N, BS) in enumerate(cells):
            b = bundle_data_rr.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            for arm, qf in ARMS:
                key = (port, arm, K, dt_name, N, BS)
                if key in done:
                    continue
                base = f"{port}_{arm}|{K}|{dt_name}|{N}|{BS}"
                rec = {"port": port, "op": arm, "K": K, "dtype": dt_name,
                       "N": N, "BS": BS, "cr": cr,
                       "scenario": args.scenario, "data_src": b["cfg"],
                       "hit_rate": b["kernel_hit_rate"], "seed": b["seed"],
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "reps_cold": args.reps, "reps_warm": args.reps_warm}
                try:
                    call, keep, extra = _build(port, K, dtype, N, BS, cr,
                                               logits_row, preidx_row, qf)
                    rec.update(extra)
                    if BS == 1 and (port, arm, K, dt_name, N) not in exact_done:
                        exact_done.add((port, arm, K, dt_name, N))
                        call()
                        torch.cuda.synchronize()
                        row_f32 = logits_row[0, :N].float()
                        ref = torch.topk(row_f32, K).values.sort().values
                        got = row_f32[keep[3][0].long()].sort().values
                        rec["exact"] = ("ok" if torch.equal(got, ref)
                                        else "FAIL")
                    measure_cell(call, base, args.reps, args.reps_warm)
                    del call, keep
                except Exception as e:  # record + continue: smem blowup etc.
                    rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            print(f"[{args.scenario} K={K} {dt_name}] {i+1}/{len(cells)} "
                  f"(port={port} N={N} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
