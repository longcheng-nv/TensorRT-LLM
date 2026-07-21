# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nsys A/B worker: KernelFactory candidate vs PR-head GVR, NVTX cold/warm-L2.

Run UNDER nsys (see run_nsys_ab.sh). For each selected cell it runs
measure_cell (house protocol: 10 warmup, w| warm reps, c| cold reps with
512MB evict outside the range) for arms {gvr_pr, kf_cand}; exactness is
checked pre-timing and written to exact_<tag>.json.

  python3 nsys_ab.py --cand <dir> [--entry E] [--cells all|uuid,uuid,...]
                     [--arms gvr_pr,kf_cand] [--tag t0]
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
sys.path.insert(0, str(HERE / "gvrpkg_head"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE))

from sweep_nsys import measure_cell  # noqa: E402
import quick_ab as Q  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand")
    ap.add_argument("--entry", default=None)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--arms", default="gvr_pr,kf_cand")
    ap.add_argument("--tag", default="t0")
    ap.add_argument("--reps-cold", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=20)
    ap.add_argument("--grid", default="subset", choices=["subset", "full"],
                    help="subset = 28 campaign cells; full = 865-cell §4 grid")
    ap.add_argument("--shard", default=None, help="i/m stripe over the cell list")
    args = ap.parse_args()

    arms = args.arms.split(",")
    cmod, entry = None, None
    if "kf_cand" in arms:
        cmod, entry = Q.build_candidate(args.cand)
        if args.entry:
            entry = args.entry
        print(f"[nsys_ab] candidate entry: {entry}", flush=True)

    if args.grid == "full":
        import csv as _csv
        rows = []
        for r in _csv.DictReader(open(REPORT / "real_3arm_layers_full.csv")):
            rows.append(dict(uuid=f"{r['model']}_{r['isl']}_L{int(r['layer']):02d}",
                             model=r["model"], isl=r["isl"], layer=r["layer"]))
    else:
        rows = Q.load_cells()
    if args.cells != "all":
        want = set(args.cells.split(","))
        rows = [r for r in rows if r["uuid"] in want]
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        rows = rows[i::m]
    print(f"[nsys_ab] {len(rows)} cells", flush=True)

    exact_log = {}
    prof.start()
    for r in rows:
        b = Q.bundle_for(r)
        for arm in arms:
            if arm == "gvr_pr":
                call, out = Q.pr_call(b)
            else:
                call, out = Q.cand_call(b, cmod, entry)
            call()
            torch.cuda.synchronize()
            ok, why = Q.exact(b, out)
            exact_log[f"{r['uuid']}|{arm}"] = (ok, why)
            if not ok:
                print(f"[nsys_ab] INEXACT {r['uuid']} {arm}: {why}", flush=True)
            measure_cell(call, f"{arm}|{r['uuid']}",
                         args.reps_cold, args.reps_warm)
        del b
        # keep GPU bundle caches bounded on the full grid
        import real_data_v4cap as _v4
        import real_data_v32 as _v32
        _v4._bundle_cache.clear()
        _v32._bundle_cache.clear()
    prof.stop()
    (HERE / f"exact_{args.tag}.json").write_text(json.dumps(exact_log, indent=1))
    n_bad = sum(1 for v in exact_log.values() if not v[0])
    print(f"[nsys_ab] done, inexact {n_bad}/{len(exact_log)}", flush=True)


if __name__ == "__main__":
    main()
