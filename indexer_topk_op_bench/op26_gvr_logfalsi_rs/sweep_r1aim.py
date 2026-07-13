# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 backlog-2: K2048 fp32 edge-aim R1 paired nsys A/B.

The mc-port R1 inline falsi shot aims at the geometric center
sqrt(kK*kCC) (self.log2_r1aim). The iter5d 1cta aim table showed edge vs
center flips by (K, dtype, N) but K2048 fp32 was never checked on the mc
port. Arms = gvr_r0_mc_op26(r1aim="center") vs r1aim="edge" (=log2(kK));
r1aim is in the compile key so both arms pair in one process.

R1 only runs on R0 miss — real-axis static admission ~0.96 so the win
ceiling is small; worst axis is where meat could be. Expected fast close.

Grid per RESUME_POST_ITER7.md section 3: K2048 fp32 N in
{131072, 262144} BS 1-16, scenarios real+worst.
Protocol identical to sweep_qfracs.py; run via drive_nsys_r1aim.sh.
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
sys.path.insert(0, str(HERE / "src"))

from sweep import DTYPES                          # noqa: E402
from sweep_nsys import measure_cell               # noqa: E402
import bundle_data_rr                             # noqa: E402
from gvr_op26_r0mc_op import (gvr_r0_mc_op26,     # noqa: E402
                              picked_cluster_size_r0mc)

DEV = "cuda"
ARMS = [("center", "center"), ("edge", "edge")]
CELLS = [(n, bs) for n in (131072, 262144) for bs in (1, 2, 4, 8, 16)]
# R1AIM_BS="1,2" -> restrict the BS grid (confirm-pass pattern, same as
# the harness OP22RR_BS filter).
_BS_FILTER = os.environ.get("R1AIM_BS")
if _BS_FILTER:
    _bss = {int(x) for x in _BS_FILTER.split(",") if x.strip()}
    CELLS = [(n, bs) for (n, bs) in CELLS if bs in _bss]


def _build(K, dtype, N, BS, cr, logits_row, preidx_row, ra):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_div, pre, out]
    cs = picked_cluster_size_r0mc(logits, K, cr)
    gvr_r0_mc_op26(logits, pre, seq_div, K, compress_ratio=cr,
                   out=out, r1aim=ra)
    return (lambda: gvr_r0_mc_op26(logits, pre, seq_div, K,
                                   compress_ratio=cr, out=out,
                                   r1aim=ra)), keep, {"cluster_size": cs}


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["op"], r["N"], r["BS"]))
            except json.JSONDecodeError:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    args = ap.parse_args()

    K, dt_name = args.K, args.dtype
    dtype = DTYPES[dt_name]
    out_dir = Path(args.out_root) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_K{K}_{dt_name}.jsonl"
    done = _load_done(out_path)
    print(f"# r1aim A/B batch: scenario={args.scenario} K={K} dt={dt_name} "
          f"cells={len(CELLS)} x {len(ARMS)} arms", flush=True)

    f = open(out_path, "a")
    exact_done = set()
    prof.start()
    try:
        for i, (N, BS) in enumerate(CELLS):
            b = bundle_data_rr.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            for arm, ra in ARMS:
                if (arm, N, BS) in done:
                    continue
                base = f"{arm}|{K}|{dt_name}|{N}|{BS}"
                rec = {"op": arm, "K": K, "dtype": dt_name, "N": N, "BS": BS,
                       "cr": cr, "scenario": args.scenario,
                       "data_src": b["cfg"], "hit_rate": b["kernel_hit_rate"],
                       "seed": b["seed"], "range_cold": f"c|{base}",
                       "range_warm": f"w|{base}", "reps_cold": args.reps,
                       "reps_warm": args.reps_warm}
                try:
                    call, keep, extra = _build(K, dtype, N, BS, cr,
                                               logits_row, preidx_row, ra)
                    rec.update(extra)
                    if BS == 1 and (arm, N) not in exact_done:
                        exact_done.add((arm, N))
                        call()
                        torch.cuda.synchronize()
                        row_f32 = logits_row[0, :N].float()
                        ref = torch.topk(row_f32, K).values.sort().values
                        got = row_f32[keep[3][0].long()].sort().values
                        rec["exact"] = ("ok" if torch.equal(got, ref)
                                        else "FAIL")
                    measure_cell(call, base, args.reps, args.reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            print(f"[{args.scenario} K={K} {dt_name}] {i+1}/{len(CELLS)} "
                  f"(N={N} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
