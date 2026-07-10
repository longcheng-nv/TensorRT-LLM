# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op27 iter1 — small-N worst-loss DECOMPOSITION A/B (paired, same-process).

Host-replay falsified the all_ge-probe hypothesis for K512/K1024 (w3a's
0.048 column already brackets the worst pole; fallback = 1 pass). The
op22rr silicon losses at small N must therefore come from machinery:
op25 loses to pre-op25 HLS even on BEST (pure fast path) at K512 4-16K,
i.e. a data-independent w3a/slot2 tax; K2048 worst is all_ge at every N
(stock ladder has no tail column).

Arms (env pinned per call; compile key includes the knobs):
  plain     gvr_cutedsl                                  [floor reference]
  stock_s1  gvr_ms_auto  OP25_QFRACS=base OP25_SLOTCAP=1 [pre-op25 HLS]
  w3a_s1    gvr_ms_auto  OP25_QFRACS=<ship> OP25_SLOTCAP=1   [K512/K1024]
  ship      gvr_ms_auto  defaults (w3a + slot2)
  tail_s1   gvr_ms_auto  OP25_QFRACS=0.75,0.45,0.048 OP25_SLOTCAP=1 [K2048]

Splits: slot2 tax = ship vs w3a_s1 (K512/1024) / ship vs stock_s1 (K2048);
w3a delta = w3a_s1 vs stock_s1; K2048 tail-column value = tail_s1 vs
stock_s1 on worst, regression check on best/real.

Protocol identical to op25 ab_qfracs.py: NVTX c|<arm>|<cell>, 512MB evict
outside the range, exactness sorted-value-set, reps interleaved.
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

from sweep import DTYPES  # noqa: E402
from sweep_nsys import build_call, _EVICT  # noqa: E402
import bundle_data_rr  # noqa: E402

CELLS = []
for K in (512, 1024):
    for N in (4096, 8192, 16384, 32768, 65536):
        CELLS.append((K, N, 1))
for N in (8192, 16384, 32768, 65536, 131072):
    CELLS.append((2048, N, 1))

TAIL_K2048 = "0.75,0.45,0.048"


def arms_for(K):
    if K < 2048:
        return (
            ("plain", "gvr_cutedsl", {}),
            ("stock_s1", "gvr_ms_auto",
             {"OP25_QFRACS": "base", "OP25_SLOTCAP": "1"}),
            ("w3a_s1", "gvr_ms_auto",
             {"OP25_QFRACS": None, "OP25_SLOTCAP": "1"}),
            ("ship", "gvr_ms_auto",
             {"OP25_QFRACS": None, "OP25_SLOTCAP": None}),
        )
    return (
        ("plain", "gvr_cutedsl", {}),
        ("stock_s1", "gvr_ms_auto",
         {"OP25_QFRACS": "base", "OP25_SLOTCAP": "1"}),
        ("ship", "gvr_ms_auto",
         {"OP25_QFRACS": None, "OP25_SLOTCAP": None}),
        ("tail_s1", "gvr_ms_auto",
         {"OP25_QFRACS": TAIL_K2048, "OP25_SLOTCAP": "1"}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["best", "worst", "real"],
                    required=True)
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    dtype = DTYPES[args.dtype]

    f = open(args.out, "a")
    prof.start()
    try:
        for (K, N, BS) in CELLS:
            b = bundle_data_rr.get_bundle(args.scenario, K, dtype, N)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            base_tag = f"{args.scenario}|{K}|{args.dtype}|{N}|{BS}"
            rec = {"scenario": args.scenario, "K": K, "dtype": args.dtype,
                   "N": N, "BS": BS, "cr": cr,
                   "hit_rate": b["kernel_hit_rate"], "reps": args.reps}
            try:
                arms = {}
                for arm, op, env in arms_for(K):
                    # pin knobs BEFORE build too (compile key reads env)
                    for k, v in env.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                    call, keep, extra = build_call(
                        op, K, dtype, N, BS, cr, logits_row, preidx_row)

                    def wrapped(_c=call, _e=dict(env)):
                        for k, v in _e.items():
                            if v is None:
                                os.environ.pop(k, None)
                            else:
                                os.environ[k] = v
                        _c()
                    arms[arm] = (wrapped, keep)
                    if extra.get("ms_path"):
                        rec.setdefault("ms_path", extra.get("ms_path"))
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
                        torch.cuda.nvtx.range_push(f"c|{arm}|{base_tag}")
                        call()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                del arms
            except Exception as e:  # noqa: BLE001 — record and continue
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"done {base_tag}", flush=True)
    finally:
        prof.stop()
    f.close()
    print("AB BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
