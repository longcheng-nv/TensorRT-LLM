# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FINAL verdict sweep for the BS>1 optimization campaign: run_batch_auto
(unified dispatcher: v4 single-wave / tp3 / tp2) paired vs gvr_pr.

Target envelope (gates: pooled gm >= 2.0x, min case >= 1.2x, no case < 1.0):
  flash_512k_L22 + pro_512k_L30 (N=131075), BS {2..1024 pow2}.
Generalization cells (reported, not gated): flash_256k_L22 (N=65536),
pro_1024k_L30 (N=262144).
Protocol == bs_ext.py (nsys cold-L2, 20c+50w, real row replicated);
exactness: ALL rows for auto.
  python3 bs_final.py --batch "flash 512k 22"   |   --list
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bs_ext as X  # noqa: E402
from bs_pq import exact_all_rows  # noqa: E402

DEV = "cuda"
REPS_COLD, REPS_WARM = 20, 50
BS_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CELLS = [("flash", "512k", 22, "target"), ("pro", "512k", 30, "target"),
         ("flash", "256k", 22, "gen"), ("pro", "1024k", 30, "gen")]
ARMS = ["gvr_pr", "auto"]


def auto_call(b, BS, mod):
    K, N = b["K"], b["N"]
    lg = X.padded_batch(b, BS)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    pick = int(mod.auto_pick(N, BS))
    call = lambda: mod.run_batch_auto(lg, N, out)   # noqa: E731
    call()
    torch.cuda.synchronize()
    return call, [lg, out], {"pick": pick}, out


def run_batch_cells(model, isl, L):
    cell = next(c for c in CELLS if c[:3] == (model, isl, L))
    kind = cell[3]
    tag = f"{model}_{isl}_L{L}"
    out_path = HERE / "results" / f"final_{tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add((rec["op"], rec["BS"]))
            except Exception:
                pass
    ext = X.load_compb_ext()
    b = X.real_bundle(model, isl, L)
    K, N, cr = b["K"], b["N"], b["cr"]
    f = open(out_path, "a")
    prof.start()
    for BS in BS_GRID:
        for op in ARMS:
            if (op, BS) in done:
                continue
            base = f"{op}|{model}|{isl}|L{L}|{BS}"
            rec = {"model": model, "isl": isl, "L": L, "N": N, "K": K,
                   "cr": cr, "BS": BS, "op": op, "kind": kind,
                   "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                   "reps_cold": REPS_COLD, "reps_warm": REPS_WARM}
            try:
                if op == "gvr_pr":
                    call, keep, extra, out = X.gvr_call(b, BS)
                else:
                    call, keep, extra, out = auto_call(b, BS, ext)
                rec.update(extra)
                rec["exact"] = (exact_all_rows(out, b, BS) if op == "auto"
                                else X.exact_rows(out, b, BS))
                X.measure_cell(call, base, REPS_COLD, REPS_WARM)
                del call, keep, out
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
    if args.list:
        print("\n".join(f"{m} {isl} {L}" for m, isl, L, _ in CELLS))
        return
    model, isl, L = args.batch.split()
    run_batch_cells(model, isl, int(L))


if __name__ == "__main__":
    main()
