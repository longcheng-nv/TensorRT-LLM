# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse bigbs_nsys run (jsonl + .nsys-rep) -> bigbs_triage.csv (wide, one row
per cell: frozen/runner/op26 nsys cold-L2 pure-kernel us + ratios + exactness).

Usage: python3 parse_bigbs.py [<jsonl> <rep>]   (defaults: /tmp/gvrval1/...)
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

jsonl = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gvrval1/bigbs_nsys.jsonl")
rep = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/gvrval1/bigbs_nsys.nsys-rep")

kern = parse_rep(rep)
cells = {}
for line in jsonl.read_text().splitlines():
    r = json.loads(line)
    key = (r["K"], r["dtype"], r["N"], r["scen"], r["BS"])
    c = cells.setdefault(key, {})
    us = kern.get(r["range_cold"])
    c[r["arm"]] = dict(us=us, exact=r["exact"],
                       cfg=r.get("cfg"), r0_arm=r.get("r0_arm"))

out = HERE / "bigbs_triage.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["K", "dtype", "N", "scen", "BS",
                "frozen_us", "runner_us", "op26_us",
                "frozen_vs_op26", "runner_vs_op26",
                "runner_cfg", "op26_arm", "exact_all"])
    rf, rr = [], []
    for key in sorted(cells):
        c = cells[key]
        uf, ur, u26 = (c[a]["us"] for a in ("pr_frozen", "pr_runner", "op26"))
        cfg = c["pr_runner"]["cfg"]
        rcfg = f"cs{cfg['cluster_size']}/T{cfg['num_threads']}/mb{cfg['min_blocks_per_mp']}"
        ex = all(c[a]["exact"] for a in c)
        w.writerow(list(key) + [f"{uf:.3f}", f"{ur:.3f}", f"{u26:.3f}",
                                f"{uf / u26:.3f}", f"{ur / u26:.3f}",
                                rcfg, c["op26"]["r0_arm"], ex])
        rf.append(uf / u26); rr.append(ur / u26)
    gf = math.exp(sum(math.log(x) for x in rf) / len(rf))
    gr = math.exp(sum(math.log(x) for x in rr) / len(rr))
print(f"wrote {out} ({len(cells)} cells)")
print(f"nsys geomean frozen/op26={gf:.3f} runner/op26={gr:.3f}")
