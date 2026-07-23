# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 envelope sweep worker: production arm v2 over the §7b fp32 envelope
(75 cells x BS 1-1024), op38 protocol (NVTX c|arm|<cell>|BS<n>, 256MB evict,
10 warmup / 50 reps, exactness every row). Sharded like bs38_nsys.

  nsys profile ... python3 bs39_nsys.py --shard i/8 --tag e1_s<i>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch, exact_rows  # noqa: E402
from bs38_nsys import all_cells, BS_LIST  # noqa: E402
from arm2_gate import build_arm2, bufs  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    cells = all_cells()
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        cells = cells[i::m]
    print(f"[bs39] {len(cells)} cells", flush=True)
    arm = build_arm2()
    evict = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    exact_log = {}
    prof.start()
    for model, isl, L in cells:
        b = bundle(model, isl, L)
        K, N = b["K"], b["N"]
        cname = f"{model}_{isl}_L{L:02d}"
        for bs in BS_LIST:
            lg, pre = make_batch(b, bs)
            bb = bufs(bs, K)
            from probe import timeit as _t
            best = None
            for ch in sorted({1, 2, 4, max(1, 296 // bs), max(1, 592 // bs)}):
                arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                        bb["done"], bb["ovf"], bb["resc"], bb["out"], ch)
                torch.cuda.synchronize()
                us = _t(lambda: arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"],
                                        bb["cnt"], bb["done"], bb["ovf"],
                                        bb["resc"], bb["out"], ch), reps=5)
                if best is None or us < best[0]:
                    best = (us, ch)
            chunks = best[1]
            arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                    bb["done"], bb["ovf"], bb["resc"], bb["out"], chunks)
            torch.cuda.synchronize()
            bad = exact_rows(b, bb["out"], bs)
            exact_log[f"{cname}|BS{bs}"] = bad or "OK"
            if bad:
                print(f"[bs39] INEXACT {cname} BS{bs}: {bad}", flush=True)
            for _ in range(args.warmup):
                arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                        bb["done"], bb["ovf"], bb["resc"], bb["out"], chunks)
            torch.cuda.synchronize()
            rname = f"c|arm|{cname}|BS{bs}"
            for _ in range(args.reps):
                evict.zero_()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(rname)
                arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                        bb["done"], bb["ovf"], bb["resc"], bb["out"], chunks)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            del lg, pre, bb
        print(f"[bs39] {cname} done", flush=True)
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()
    prof.stop()
    (HERE.parent / "results" / f"exact_{args.tag}.json").write_text(
        json.dumps(exact_log, indent=1))
    n_bad = sum(1 for v in exact_log.values() if v != "OK")
    print(f"[bs39] ALL DONE, inexact {n_bad}/{len(exact_log)}", flush=True)


if __name__ == "__main__":
    main()
