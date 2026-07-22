# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register-diet sweep for the compB BS>1 extension large-n row-team path.

Diagnosis (fast_stats, b200-039): topk_fast<1> = 56 regs / 41312B smem /
active=2 (register-bound; carveout no-op). topk_fast<3> = 40 regs + 16B
local spill / active=3. topk_fast<4> = 32 regs, ZERO spill / active=4
-> cap 592, rows_per_wave 9 at team=65.

Arms paired per cell on ONE GPU: gvr_pr (local anchor), ext_v1 (validated
baseline, active=2), ext_v3, ext_v4. Cells flash_512k_L22 / pro_512k_L30
(N=131075), BS {1,2,4,8,16,32}. Protocol == bs_ext.py (nsys cold-L2,
20c+50w, real row replicated, exactness rows {0, BS//2, BS-1}).
  python3 bs_diet.py --batch "flash 512k 22"   |   --list
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bs_ext as X  # noqa: E402  (reuses loaders, gvr arm, exactness)

DEV = "cuda"
REPS_COLD, REPS_WARM = 20, 50
BS_GRID = [1, 2, 4, 8, 16, 32]
CELLS = [("flash", "512k", 22), ("pro", "512k", 30)]
ARMS = ["gvr_pr", "ext_v1", "ext_v3", "ext_v4"]


def ext_v_call(b, BS, mod, minb):
    K, N = b["K"], b["N"]
    lg = X.padded_batch(b, BS)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    stats = [int(x) for x in mod.fast_stats(minb)]
    call = lambda: mod.run_batch_ext_v(lg, N, out, minb)   # noqa: E731
    call()
    torch.cuda.synchronize()
    cap = stats[4] * 148
    team = (N + 2047) // 2048
    rpw = max(1, cap // team)
    return call, [lg, out], {"minb": minb, "regs": stats[0],
                             "active": stats[4], "cap": cap, "rpw": rpw,
                             "waves": (BS + rpw - 1) // rpw}, out


def run_batch_cells(model, isl, L):
    tag = f"{model}_{isl}_L{L}"
    out_path = HERE / "results" / f"diet_{tag}.jsonl"
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
    hit = b.get("hit_rate")
    f = open(out_path, "a")
    prof.start()
    for BS in BS_GRID:
        for op in ARMS:
            if (op, BS) in done:
                continue
            base = f"{op}|{model}|{isl}|L{L}|{BS}"
            rec = {"model": model, "isl": isl, "L": L, "N": N, "K": K,
                   "cr": cr, "BS": BS, "op": op, "kind": "B",
                   "hit": round(float(hit), 4) if hit is not None else None,
                   "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                   "reps_cold": REPS_COLD, "reps_warm": REPS_WARM}
            try:
                if op == "gvr_pr":
                    call, keep, extra, out = X.gvr_call(b, BS)
                else:
                    minb = int(op[-1])
                    call, keep, extra, out = ext_v_call(b, BS, ext, minb)
                rec.update(extra)
                rec["exact"] = X.exact_rows(out, b, BS)
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
        print("\n".join(f"{m} {isl} {L}" for m, isl, L in CELLS))
        return
    model, isl, L = args.batch.split()
    run_batch_cells(model, isl, int(L))


if __name__ == "__main__":
    main()
