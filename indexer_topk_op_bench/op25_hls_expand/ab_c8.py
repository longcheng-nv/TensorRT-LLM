# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 Step 4 probe A — msc cluster width C=8 vs C=4 for fp32 K512/K1024
at low BS (the R2 bandwidth-undersubscription region), vs radix.

The old fp32-C8 falsification predates iter16 (dist fallback) and op25 S1a
(admission 0.30->0.96): both changed what the extra CTAs amortize. If C=8
wins here, the ms_auto dispatch rule extends before any >8-CTA work.

Arms: c4 = gvr_msc(C=4) / c8 = gvr_msc(C=8) / radix = radix_cutedsl.
Protocol identical to ab_qfracs.py (paired, cold-L2, sorted-value sets).

Usage:
  env -u GITHUB_TOKEN -u HF_TOKEN CUDA_VISIBLE_DEVICES=$GPU \
    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end=stop \
    -f true -o <rep> python3 ab_c8.py --scenario real --out <jsonl>
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
sys.path.insert(0, str(BENCH / "op21_gvr_prod" / "src"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))

from sweep import DTYPES  # noqa: E402
from sweep_nsys import build_call, _EVICT  # noqa: E402
from gvr_msc_op import gvr_msc  # noqa: E402
import bundle_data_rr  # noqa: E402

CELLS = [(K, N, BS)
         for K in (512, 1024)
         for N in (65536, 131072, 262144)
         for BS in (1, 2, 4, 8)]


def build_msc_call(C, K, dtype, N, BS, cr, logits_row, preidx_row):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device="cuda")
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device="cuda")
    keep = [logits, seq_div, pre, out]
    gvr_msc(logits, pre, seq_div, K, compress_ratio=cr, out=out, C=C)
    return (lambda: gvr_msc(logits, pre, seq_div, K, compress_ratio=cr,
                            out=out, C=C)), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    dtype = DTYPES["fp32"]

    f = open(args.out, "a")
    prof.start()
    try:
        for (K, N, BS) in CELLS:
            b = bundle_data_rr.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            tag = f"{args.scenario}|{K}|fp32|{N}|{BS}"
            rec = {"scenario": args.scenario, "K": K, "N": N, "BS": BS,
                   "cr": cr, "hit_rate": b["kernel_hit_rate"],
                   "reps": args.reps}
            try:
                arms = {}
                for C in (4, 8):
                    call, keep = build_msc_call(C, K, dtype, N, BS, cr,
                                                logits_row, preidx_row)
                    arms[f"c{C}"] = (call, keep)
                call, keep, _ = build_call("radix_cutedsl", K, dtype, N, BS,
                                           cr, logits_row, preidx_row)
                arms["radix"] = (call, keep)
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
                for arm, (call, keep) in arms.items():
                    for _ in range(args.warmup):
                        call()
                torch.cuda.synchronize()
                for _ in range(args.reps):
                    for arm, (call, keep) in arms.items():
                        _EVICT.uniform_(0, 1)
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push(f"c|{arm}|{tag}")
                        call()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                del arms
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"done {tag}", flush=True)
    finally:
        prof.stop()
    f.close()
    print("AB BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
