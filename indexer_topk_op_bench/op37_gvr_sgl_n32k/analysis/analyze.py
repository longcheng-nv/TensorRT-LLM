#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op37 analysis — parse an nsys results dir (via rival parse) and print the
loss map + composites vs a reference arm.

Usage: analyze.py <results_dir> [--ref sglang_v2] [--arm gvr_pr ...]
Ratios printed are t(ref)/t(arm) with us_span (>1 = arm FASTER than ref).
When --ref is an arm like gvr_pr, this compares GVR variants instead.
"""
import argparse
import collections
import json
import math
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPORT = _HERE.parents[1] / "op26_r0_upstream_port_report"
sys.path.insert(0, str(_REPORT / "rival_harness"))

BSS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def gm(v):
    v = [x for x in v if x and x == x]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--ref", default="sglang_v2")
    ap.add_argument("--arms", default=None, help="comma list; default = all non-ref")
    ap.add_argument("--nmin", type=int, default=32768)
    a = ap.parse_args()
    root = Path(a.results_dir)

    merged = root / "results.jsonl"
    subprocess.run([sys.executable, str(_REPORT / "rival_harness" / "parse_rival.py"),
                    str(root)], check=True)
    rows = [json.loads(l) for l in open(merged)]
    rows = [r for r in rows if r.get("us_span") and not r.get("error")]

    cells = collections.defaultdict(dict)
    for r in rows:
        cells[(r["model"], r["isl"], int(r["N"]), int(r["BS"]))][r["op"]] = r
    arms = (a.arms.split(",") if a.arms else
            sorted({r["op"] for r in rows} - {a.ref}))

    inex = [(k, r2["op"]) for k, d in cells.items() for r2 in d.values()
            if not r2.get("exact", True)]
    print(f"cells={len(cells)} arms={arms} ref={a.ref} inexact={inex if inex else 0}")

    for arm in arms:
        mat = collections.defaultdict(dict)
        for k, d in cells.items():
            if arm in d and a.ref in d and k[2] >= a.nmin:
                mat[(k[0], k[1], k[2])][k[3]] = (
                    float(d[a.ref]["us_span"]) / float(d[arm]["us_span"]))
        if not mat:
            continue
        print(f"\n== {a.ref}/{arm} (>1 = {arm} faster), N>={a.nmin}")
        allv = []
        for key in sorted(mat, key=lambda x: (x[0], x[2])):
            row = mat[key]
            print(f"{key[0]:5s} {key[1]:6s} N={key[2]:7d} | " +
                  " ".join(f"{row[b]:5.2f}" if b in row else "  .  " for b in BSS))
            allv += list(row.values())
        print(f"composite gm {gm(allv):.4f}  n={len(allv)}")
        for b in BSS:
            v = [mat[k][b] for k in mat if b in mat[k]]
            if v:
                print(f"  BS={b:4d} gm {gm(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
