# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op26 backlog-3: kC-diet K512@1536 paired nsys A/B.

Host screen (screen_kc_diet.py, 2026-07-13) at kC=1536: real/best static
admission 87.5% (misses all bracket-type = one extra fb_fix falsi pass),
accepted cand med ~880 << 1536 -> silicon-worthy. SMEM saving =
(5120-1536)*(4+4)B ~ 28KB per CTA -> occupancy; helps the 1cta port most
(one CTA per row: occupancy binds when many rows are resident), mc port
is latency-bound and expected mostly insensitive.

Arms = stock kC (5120) vs kc_override=1536 via the wrapper param
(compile-key includes it, paired in-process).

Grid (prod-visible 1cta R0 bands after the 07-13 small-N gate):
  1cta 16-bit band : fp16 N in {16384,32768} BS {1,8,64,256}
  1cta BS>=128 route: fp32 N in {65536,131072} BS {128,256}
  mc spot guard    : fp32 N in {65536,131072} BS {1,8}
Scenarios real+worst. Protocol identical to sweep_qfracs.py.
Run via drive_nsys_kcdiet.sh.
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
from gvr_op26_r0_op import gvr_r0_op26            # noqa: E402
from gvr_op26_r0mc_op import (gvr_r0_mc_op26,     # noqa: E402
                              picked_cluster_size_r0mc)

DEV = "cuda"
K = 512
# KCDIET_KC picks the diet window (default 1536 = the original backlog-3
# candidate; 3072 = the tie-safe iteration after gate Suite C killed
# 1536). Explicit values on BOTH arms so the stock arm stays a true
# baseline regardless of the wrapper's dispatch default.
KC_DIET = int(os.environ.get("KCDIET_KC", "1536"))
ARMS = [("kc5120", 5120), (f"kc{KC_DIET}", KC_DIET)]

# (port, dtype, N, BS). KCDIET_DT16 swaps the 16-bit band dtype
# (bf16 spot-confirm pass); KCDIET_ONLY16=1 drops the fp32/mc cells.
_DT16 = os.environ.get("KCDIET_DT16", "fp16")
CELLS = (
    [("r1cta", _DT16, n, bs) for n in (16384, 32768)
     for bs in (1, 8, 64, 256)] +
    ([] if os.environ.get("KCDIET_ONLY16") else
     [("r1cta", "fp32", n, bs) for n in (65536, 131072)
      for bs in (128, 256)] +
     [("mc", "fp32", n, bs) for n in (65536, 131072) for bs in (1, 8)])
)


def _build(port, dtype, N, BS, cr, logits_row, preidx_row, kc):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_div, pre, out]
    if port == "mc":
        cs = picked_cluster_size_r0mc(logits, K, cr)
        gvr_r0_mc_op26(logits, pre, seq_div, K, compress_ratio=cr,
                       out=out, kc_override=kc)
        return (lambda: gvr_r0_mc_op26(logits, pre, seq_div, K,
                                       compress_ratio=cr, out=out,
                                       kc_override=kc)), keep, \
            {"cluster_size": cs}
    gvr_r0_op26(logits, pre, seq_div, K, compress_ratio=cr,
                out=out, kc_override=kc)
    return (lambda: gvr_r0_op26(logits, pre, seq_div, K, compress_ratio=cr,
                                out=out, kc_override=kc)), keep, {}


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["port"], r["op"], r["dtype"], r["N"], r["BS"]))
            except json.JSONDecodeError:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_K{K}.jsonl"
    done = _load_done(out_path)
    print(f"# kcdiet A/B batch: scenario={args.scenario} K={K} "
          f"cells={len(CELLS)} x {len(ARMS)} arms", flush=True)

    f = open(out_path, "a")
    exact_done = set()
    prof.start()
    try:
        for i, (port, dt_name, N, BS) in enumerate(CELLS):
            dtype = DTYPES[dt_name]
            b = bundle_data_rr.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            for arm, kc in ARMS:
                if (port, arm, dt_name, N, BS) in done:
                    continue
                base = f"{port}_{arm}|{K}|{dt_name}|{N}|{BS}"
                rec = {"port": port, "op": arm, "K": K, "dtype": dt_name,
                       "N": N, "BS": BS, "cr": cr,
                       "scenario": args.scenario, "data_src": b["cfg"],
                       "hit_rate": b["kernel_hit_rate"], "seed": b["seed"],
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "reps_cold": args.reps, "reps_warm": args.reps_warm}
                try:
                    call, keep, extra = _build(port, dtype, N, BS, cr,
                                               logits_row, preidx_row, kc)
                    rec.update(extra)
                    ek = (port, arm, dt_name, N)
                    if BS == 1 and ek not in exact_done:
                        exact_done.add(ek)
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
            print(f"[{args.scenario} kcdiet] {i+1}/{len(CELLS)} "
                  f"(port={port} {dt_name} N={N} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
