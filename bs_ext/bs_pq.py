# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B' persistent-queue sweep: pq_v4 (one launch, teams loop rows, per-team
slice reuse guarded by __ldcg + widened sense tokens + inter-row barrier)
vs ext_v4 (chunked single-wave champion) vs gvr_pr. Protocol == bs_diet.py.
  python3 bs_pq.py --batch "flash 512k 22"   |   --list
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
from bs_diet import ext_v_call  # noqa: E402

DEV = "cuda"
REPS_COLD, REPS_WARM = 20, 50
BS_GRID = [1, 2, 4, 8, 16, 32, 64]
CELLS = [("flash", "512k", 22), ("pro", "512k", 30)]
ARMS = ["gvr_pr", "ext_v4", "pq_v4"]


def pq_call(b, BS, mod, minb=4):
    K, N = b["K"], b["N"]
    lg = X.padded_batch(b, BS)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    stats = [int(x) for x in mod.pq_stats(minb)]
    call = lambda: mod.run_batch_pq(lg, N, out, minb)   # noqa: E731
    call()
    torch.cuda.synchronize()
    cap = stats[4] * 148
    team = (N + 2047) // 2048
    nteams = min(max(1, cap // team), BS)
    return call, [lg, out], {"minb": minb, "regs": stats[0],
                             "local": stats[2], "active": stats[4],
                             "nteams": nteams,
                             "iters": (BS + nteams - 1) // nteams}, out


def exact_all_rows(out, b, BS):
    lg = b["logits"][0, :b["N"]].float()
    ref = torch.topk(lg, b["K"]).values.sort().values
    for r in range(BS):
        idx = out[r].long()
        if (idx.numel() != b["K"] or torch.unique(idx).numel() != b["K"]
                or not torch.equal(lg[idx].sort().values, ref)):
            return False
    return True


def run_batch_cells(model, isl, L):
    tag = f"{model}_{isl}_L{L}"
    out_path = HERE / "results" / f"pq_{tag}.jsonl"
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
                   "cr": cr, "BS": BS, "op": op, "kind": "B",
                   "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                   "reps_cold": REPS_COLD, "reps_warm": REPS_WARM}
            try:
                if op == "gvr_pr":
                    call, keep, extra, out = X.gvr_call(b, BS)
                elif op == "ext_v4":
                    call, keep, extra, out = ext_v_call(b, BS, ext, 4)
                else:
                    call, keep, extra, out = pq_call(b, BS, ext, 4)
                rec.update(extra)
                rec["exact"] = (exact_all_rows(out, b, BS) if op == "pq_v4"
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
        print("\n".join(f"{m} {isl} {L}" for m, isl, L in CELLS))
        return
    model, isl, L = args.batch.split()
    run_batch_cells(model, isl, int(L))


if __name__ == "__main__":
    main()
