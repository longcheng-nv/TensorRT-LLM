# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 Step 5 (S3a) — small-N kC diet silicon A/B.

s3a_screen.py: kC 50% is admission-free at N<=32K on the S1a ladder.
This measures whether the halved band/collect budget (smaller P3
over-collect feeding P4, -20KB smem_keys/vals) moves the small-N floor,
against the rivals that own R1 (radix, sglang).

Arms: ship = gvr_ms (spec kC) / kc50 = gvr_sw(kC=spec//2, same ladder)
      / radix = radix_cutedsl / sglang = sglang_streaming.
Cells: K512/K1024 x N {4096..32768} x BS {1, 256}; fp32 real+best.
Protocol = ab_qfracs.py (paired cold-L2, sorted-value sets).
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
import gvr_ms_op as MS  # noqa: E402
import bundle_data_rr  # noqa: E402

SPEC_KC = {512: 5120, 1024: 5120}
CELLS = [(K, N, BS)
         for K in (512, 1024)
         for N in (4096, 8192, 16384, 32768)
         for BS in (1, 256)]


def build_ms_call(K, dtype, N, BS, cr, logits_row, preidx_row, kc=None):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device="cuda")
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device="cuda")
    keep = [logits, seq_div, pre, out]
    if kc is None:
        fn = lambda: MS.gvr_ms(logits, pre, seq_div, K, compress_ratio=cr,  # noqa: E731
                               out=out)
    else:
        qf = MS._qfracs_for(K)
        fuse = bool(BS <= MS.NUM_SMS and 4 * K <= 5120)
        fn = lambda: MS.gvr_sw(logits, pre, seq_div, K, compress_ratio=cr,  # noqa: E731
                               out=out, M=len(qf) + 1, R=1, band_acc=64,
                               place_mode=5, kC=kc, fuse=fuse, qfracs=qf,
                               slot_scale=MS._slot_scale())
    fn()
    return fn, keep


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
                c, k = build_ms_call(K, dtype, N, BS, cr, logits_row,
                                     preidx_row)
                arms["ship"] = (c, k)
                c, k = build_ms_call(K, dtype, N, BS, cr, logits_row,
                                     preidx_row, kc=SPEC_KC[K] // 2)
                arms["kc50"] = (c, k)
                for name, op in (("radix", "radix_cutedsl"),
                                 ("sglang", "sglang_streaming")):
                    try:
                        call, keep, _ = build_call(op, K, dtype, N, BS, cr,
                                                   logits_row, preidx_row)
                        arms[name] = (call, keep)
                    except Exception as e:  # noqa: BLE001 — rival optional
                        rec[f"skip_{name}"] = str(e)[:80]
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
