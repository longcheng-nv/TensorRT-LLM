# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op38 v3 confirmation probe: for each (model, npad, BS) key where the ladder
found a non-prod winner, time {prod} U {candidate cfgs seen at that key} on
EVERY layer of the §7b envelope sharing the key. A key switches to cfg c only
if c >= prod on all layers (min-gain maximizer, threshold 1.02 on gm).

Output: v3_table.csv (model, npad, BS, cfg, gm_gain, min_gain, n_layers).
  python3 confirm_v3.py --shard i/m
"""
import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCH / "harness"))

from probe import build, bundle, make_batch, exact_rows, timeit  # noqa: E402
from bs38_nsys import all_cells  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402


def candidate_keys():
    """{(model, npad, BS): set(cfg strings)} from the ladder probe."""
    keys = defaultdict(set)
    for f in sorted(glob.glob(str(HERE / "v3_probe_s*.csv"))):
        for r in csv.DictReader(open(f)):
            if r["best_cfg"] != "prod":
                keys[(r["cell"].split("_")[0], int(r["Npad"]),
                      int(r["BS"]))].add(r["best_cfg"])
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    i, m = (int(x) for x in args.shard.split("/"))

    keys = candidate_keys()
    # group envelope cells by (model, npad)
    by_mn = defaultdict(list)
    npad_of = {}
    for model, isl, L in all_cells():
        b = bundle(model, isl, L)
        by_mn[(model, b["Npad"])].append((model, isl, L))
        npad_of[(model, isl)] = b["Npad"]
    v4._bundle_cache.clear()
    v32._bundle_cache.clear()
    torch.cuda.empty_cache()

    work = sorted(keys.items())[i::m]
    print(f"[confirm_v3] shard {args.shard}: {len(work)} keys", flush=True)
    mod = build("kernel_bs")

    rows = []
    for (model, npad, bs), cfgs in work:
        cells = by_mn[(model, npad)]
        per_cfg = {c: [] for c in cfgs}
        for cm, isl, L in cells:
            b = bundle(cm, isl, L)
            K, N = b["K"], b["N"]
            lg, pre = make_batch(b, bs)
            out = torch.empty(bs, K, dtype=torch.int32, device="cuda")
            mod.run(lg, pre, N, out)
            torch.cuda.synchronize()
            assert not exact_rows(b, out, bs)
            for _ in range(3):
                mod.run(lg, pre, N, out)
            torch.cuda.synchronize()
            t_prod = timeit(lambda: mod.run(lg, pre, N, out), reps=11)
            for c in cfgs:
                tb, cs, mv, ar, hs = (int(x) for x in c.split(","))
                try:
                    mod.run_cfg(lg, pre, N, out, tb, cs, mv, ar, hs)
                    torch.cuda.synchronize()
                except RuntimeError:
                    torch.cuda.synchronize()
                    per_cfg[c].append(0.0)
                    continue
                if exact_rows(b, out, bs):
                    per_cfg[c].append(0.0)
                    continue
                for _ in range(3):
                    mod.run_cfg(lg, pre, N, out, tb, cs, mv, ar, hs)
                torch.cuda.synchronize()
                us = timeit(lambda: mod.run_cfg(lg, pre, N, out,
                                                tb, cs, mv, ar, hs), reps=11)
                per_cfg[c].append(t_prod / us)
            del lg, pre, out
            v4._bundle_cache.clear()
            v32._bundle_cache.clear()
            torch.cuda.empty_cache()
        # pick min-gain maximizer among cfgs with min >= 1.0
        best = None
        for c, gains in per_cfg.items():
            if not gains or min(gains) < 1.0:
                continue
            gm = statistics.geometric_mean(gains)
            if best is None or min(gains) > best[1]:
                best = (c, min(gains), gm, len(gains))
        if best and best[2] >= 1.02:
            rows.append(dict(model=model, npad=npad, BS=bs, cfg=best[0],
                             min_gain=round(best[1], 3),
                             gm_gain=round(best[2], 3), n_layers=best[3]))
            print(f"SWITCH {model} npad={npad} BS{bs}: {best[0]} "
                  f"min {best[1]:.3f} gm {best[2]:.3f} ({best[3]} layers)",
                  flush=True)
        else:
            detail = {c: ["%.2f" % g for g in gs] for c, gs in per_cfg.items()}
            print(f"KEEP   {model} npad={npad} BS{bs}: {detail}", flush=True)

    with open(HERE / f"v3_table_s{i}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["model", "npad", "BS", "cfg",
                               "min_gain", "gm_gain", "n_layers"])
        w.writeheader()
        w.writerows(rows)
    print(f"[confirm_v3] shard {i} done: {len(rows)} switches", flush=True)


if __name__ == "__main__":
    main()
