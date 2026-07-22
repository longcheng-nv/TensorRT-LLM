# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pass-2 of the compB BS-scaling study: re-measure the EXTERNAL baselines
LOCALLY on this node (umbriel-b200-019), same envelope as bs_kf.py.

Motivation: cross-node PR-arm normalization proved unsafe here — local gvr_pr
runs ~1.07x (BS<=32) to ~1.7x (BS>=256) slower than the REPORT b200-027 rival
run, and the node effect is asymmetric between one-big-batched-kernel arms and
compB's sequential small launches. So every arm in the final comparison is
measured on THIS node, paired per (model,isl,L) batch on one GPU.

Arms: gvr_pr2 (identical build to pass-1 gvr_pr — pass-to-pass stability gate)
+ sglang_v2 + radix_cutedsl + flashinfer_topk (REPORT-verbatim ops_rival
builds). fp32 only (rival CSV envelope). Results in results_rv/,
reps in nsys_reps_rv/.   python3 bs_rv.py --batch "flash 4k 22"  |  --list
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BENCH = REPORT.parent
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(REPORT / "harness"))
sys.path.insert(0, str(REPORT / "gvrpkg_snapshot"))
sys.path.insert(0, str(REPORT / "rival_harness"))

import cutlass  # noqa: E402

assert cutlass.__version__.startswith("4.5.0"), cutlass.__version__

from sweep_nsys import measure_cell                     # noqa: E402
from ops_rival import build_call_rival                  # noqa: E402
import real_data_v4cap as RV4                           # noqa: E402
import real_data_v32 as RV32                            # noqa: E402
from bs_kf import gvr_call, exact_rows                  # noqa: E402

DEV = "cuda"
REPS_COLD, REPS_WARM = 20, 50
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
LAYERS = {"flash": [10, 22, 34], "pro": [14, 30, 46], "v32": [14, 34, 54]}
ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
        "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
        "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
OPS = ["gvr_pr2", "sglang_v2", "radix_cutedsl", "flashinfer_topk"]


def run_batch_cells(model, isl, L):
    tag = f"{model}_{isl}_L{L}"
    out_path = HERE / "results_rv" / f"rv_{tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["op"], r["BS"]))
            except Exception:
                pass
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, L, "fp32")
    K, N, cr = b["K"], b["N"], b["cr"]
    hit = b.get("hit_rate")
    f = open(out_path, "a")
    prof.start()
    for BS in BS_GRID:
        for op in OPS:
            if (op, BS) in done:
                continue
            base = f"{op}|{model}|{isl}|L{L}|{BS}"
            rec = {"model": model, "isl": isl, "L": L, "N": N, "K": K,
                   "cr": cr, "BS": BS, "op": op,
                   "hit": round(float(hit), 4) if hit is not None else None,
                   "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                   "reps_cold": REPS_COLD, "reps_warm": REPS_WARM}
            try:
                if op == "gvr_pr2":
                    call, keep, extra, out = gvr_call(b, BS)
                    getter = lambda: out                     # noqa: E731
                else:
                    call, keep, extra, getter = build_call_rival(
                        op, K, torch.float32, N, BS, cr,
                        b["logits"], b["preIdx"])
                rec.update(extra or {})
                rec["exact"] = exact_rows(getter(), b, BS)
                measure_cell(call, base, REPS_COLD, REPS_WARM)
                del call, keep, getter
            except Exception as e:  # record, never fake
                rec["error"] = f"{type(e).__name__}: {e}"
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{tag}] {op} BS={BS} "
                  f"{'ERR ' + rec['error'] if 'error' in rec else 'exact=' + str(rec['exact'])}",
                  flush=True)
    prof.stop()
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    batches = [f"{m} {isl} {L}" for m in ("flash", "pro", "v32")
               for isl in ISLS[m] for L in LAYERS[m]]
    if args.list:
        print("\n".join(batches))
        return
    model, isl, L = args.batch.split()
    run_batch_cells(model, isl, int(L))


if __name__ == "__main__":
    main()
