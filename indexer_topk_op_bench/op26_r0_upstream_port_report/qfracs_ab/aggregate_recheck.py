#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the worst-cell noise re-check: per-run per-cell ship/qr2 ratios
for the synth-worst K2048 fp32 seq-len batch (original run + 3 repeats)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/"
                   "TensorRT-LLM/indexer_topk_op_bench/report")
from parse_nsys_full import parse_rep  # noqa: E402

RUNS = [("orig", Path("/tmp/gvrqab/qab_results"))] + [
    (f"r{r}", Path(f"/tmp/gvrqab/worst_recheck_r{r}")) for r in (1, 2, 3)]
JN = "synth_worst_seqlen_K2048_fp32.jsonl"
REP = "nsys_reps/synth_seqlen_worst_K2048_fp32.nsys-rep"


def main():
    table = {}
    for name, d in RUNS:
        if not (d / JN).exists() or not (d / REP).exists():
            print(f"  !! {name}: missing artifacts, skip")
            continue
        us = parse_rep(d / REP)
        recs = [json.loads(x) for x in (d / JN).read_text().splitlines()]
        cells = {}
        for c in recs:
            if "error" in c:
                continue
            cells.setdefault(c["N"], {})[c["op"]] = us.get(c["range_cold"])
        for N, r in cells.items():
            if r.get("gvr_ship") and r.get("gvr_qr2"):
                table.setdefault(N, {})[name] = r["gvr_ship"] / r["gvr_qr2"]
    names = [n for n, _ in RUNS]
    print(f"{'N':>9} | " + " ".join(f"{n:>6}" for n in names) + " |   gm   min")
    gms = []
    for N in sorted(table):
        row = table[N]
        vals = [row.get(n) for n in names]
        ok = [v for v in vals if v]
        g = math.exp(sum(math.log(v) for v in ok) / len(ok))
        gms.append(g)
        print(f"{N:>9} | " + " ".join(f"{v:6.3f}" if v else "     ." for v in vals)
              + f" | {g:5.3f} {min(ok):5.3f}")
    print(f"\nbatch gm-of-gms: {math.exp(sum(map(math.log, gms))/len(gms)):.4f}")


if __name__ == "__main__":
    main()
