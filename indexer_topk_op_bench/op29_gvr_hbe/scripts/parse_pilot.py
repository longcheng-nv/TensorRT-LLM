#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse pilot nsys reps -> paired ratio table (rival/hbe, rival/off).

Usage: env -u GITHUB_TOKEN -u HF_TOKEN python3 parse_pilot.py [pilot_dir]
"""
import csv
import io
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parents[1] / "results" / "pilot"


def parse_rep(rep):
    env = {k: v for k, v in os.environ.items()
           if k not in ("GITHUB_TOKEN", "HF_TOKEN")}
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_kern_sum", "--format", "csv",
         "--force-export=true", str(rep)],
        capture_output=True, text=True, env=env).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr = next((i for i, r in enumerate(rows)
                if r and r[0] == "NVTX Range"), None)
    tot, inst = defaultdict(float), {}
    for r in rows[hdr + 1:]:
        if not r or len(r) < 13 or "|" not in r[0]:
            continue
        rng = r[0].lstrip(":")
        try:
            ninst = int(r[4]); ns = float(r[6])
        except ValueError:
            continue
        if "distribution_elementwise" in ",".join(r[12:]).lower():
            continue
        tot[rng] += ns
        inst[rng] = ninst
    return {r: tot[r] / inst[r] / 1e3 for r in tot if inst.get(r)}


def geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main():
    allr = defaultdict(list)
    print(f"{'scen':>6} {'K':>5} {'N':>7} {'BS':>5} | {'rival':>8} "
          f"{'hbe':>8} {'off':>8} | {'hbe/rival':>9} {'off/rival':>9}")
    for scen in ("real", "best", "worst"):
        rep = PDIR / f"pilot_{scen}.nsys-rep"
        jl = PDIR / f"pilot_{scen}.jsonl"
        if not rep.exists() or not jl.exists():
            continue
        kern = parse_rep(rep)
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            if r["op"] != "sglang_v2":
                continue
            def c(op):
                return kern.get(f"c|{op}|{r['K']}|fp32|{r['N']}|{r['BS']}")
            rv, hb, off = c("sglang_v2"), c("gvr29_hbe"), c("gvr29_off")
            if rv and hb:
                allr[scen].append(rv / hb)
                print(f"{scen:>6} {r['K']:>5} {r['N']:>7} {r['BS']:>5} | "
                      f"{rv:8.2f} {hb:8.2f} {off or 0:8.2f} | "
                      f"{rv / hb:9.3f} {rv / off if off else 0:9.3f}")
    print("\ngeomean t(rival)/t(hbe):",
          {s: round(geo(v), 3) for s, v in allr.items()})


if __name__ == "__main__":
    main()
