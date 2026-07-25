# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41 sweep worker: v3mt (per-K rung fractions) on the §7b fp32 envelope.
Cloned from op38/bs42_nsys.py; ONLY the kernel build differs.

Protocol mirrors perf_3arm_real_bs.py (the report's §7b): same-row batch
(expand -> contiguous), 10 warmup, 50 reps with a 256MB L2.zero_() evict
before each timed call. Each timed call sits in its own NVTX range
c|cand|<cell>|BS<n> (sync inside), so evict kernels fall outside ranges.
Exactness: every row, tie-robust, per (cell, BS). Run under nsys
(cudaProfilerApi window). argv: --cells m:isl:L[,m:isl:L...] | --shard i/m
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent          # op41/scripts
OP41 = HERE.parent
BENCH = OP41.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE))

from probe import bundle, make_batch, exact_rows  # noqa: E402
from champ_gate import build_champ  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

BS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def all_cells():
    import csv
    seen = []
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" /
                                 "bs_real_layers.csv")):
        key = (r["model"], r["isl"], int(r["L"]))
        if key not in seen:
            seen.append(key)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    cells = all_cells()
    if args.cells:
        want = [tuple(c.split(":")) for c in args.cells.split(",")]
        cells = [c for c in cells if (c[0], c[1], str(c[2])) in
                 [(m, i, L) for m, i, L in want]]
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        cells = cells[i::m]
    print(f"[bs42] {len(cells)} cells", flush=True)

    mod = build_champ()
    evict = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    exact_log = {}
    prof.start()
    for model, isl, L in cells:
        b = bundle(model, isl, L)
        K, N = b["K"], b["N"]
        cname = f"{model}_{isl}_L{L:02d}"
        for bs in BS_LIST:
            lg, pre = make_batch(b, bs)
            out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
            mod.run(lg, pre, N, out)
            torch.cuda.synchronize()
            bad = exact_rows(b, out, bs)
            exact_log[f"{cname}|BS{bs}"] = bad or "OK"
            if bad:
                print(f"[bs42] INEXACT {cname} BS{bs}: {bad}", flush=True)
            for _ in range(args.warmup):
                mod.run(lg, pre, N, out)
            torch.cuda.synchronize()
            rname = f"c|cand|{cname}|BS{bs}"
            for _ in range(args.reps):
                evict.zero_()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(rname)
                mod.run(lg, pre, N, out)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            del lg, pre, out
        print(f"[bs42] {cname} done", flush=True)
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()
    prof.stop()
    (OP41 / 'results' / f"exact_{args.tag}.json").write_text(json.dumps(exact_log, indent=1))
    n_bad = sum(1 for v in exact_log.values() if v != "OK")
    print(f"[bs42] ALL DONE, inexact {n_bad}/{len(exact_log)}", flush=True)


if __name__ == "__main__":
    main()
