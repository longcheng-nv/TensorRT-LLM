#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the qfracs A/B: per-cell 3-arm cold-L2 table + grouped geomeans.

Reads /tmp/gvrqab/qab_results/*.jsonl + nsys_reps/*.nsys-rep (parse_nsys_full),
pairs gvr_ship / gvr_qr2 / gvr_qr1 per cell, prints per-cell ratios
(ship/qr* > 1 = new qfracs faster) and geomeans by (family, scenario|model, K),
writes qab.csv next to this script.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/"
                   "TensorRT-LLM/indexer_topk_op_bench/report")
from parse_nsys_full import parse_rep  # noqa: E402

OUT = Path("/tmp/gvrqab/qab_results")
HERE = Path(__file__).resolve().parent
ARMS = ("gvr_ship", "gvr_qr2", "gvr_qr1")


def gm(a):
    a = [x for x in a if x and x > 0]
    return math.exp(sum(math.log(x) for x in a) / len(a)) if a else float("nan")


def main():
    rows = {}
    for jf in sorted(OUT.glob("*.jsonl")):
        # rep tag: jsonl name pattern differs (synth_{scen}_{sweep}_... vs real_...)
        recs = [json.loads(x) for x in jf.read_text().splitlines()]
        if not recs:
            continue
        r0 = recs[0]
        if r0["family"] == "synth":
            tag = f"synth_{r0['sweep']}_{r0['scenario']}_K{r0['K']}_{r0['dtype']}"
        else:
            tag = f"real_{r0['sweep']}_{r0['model']}_{r0['dtype']}"
        rep = OUT / "nsys_reps" / f"{tag}.nsys-rep"
        if not rep.exists():
            print(f"  !! missing rep {rep.name}, skip {jf.name}")
            continue
        us = parse_rep(rep)
        for c in recs:
            if "error" in c:
                print(f"  !! cell error: {c.get('op')} {c.get('N')} {c.get('isl','')}: {c['error']}")
                continue
            key = (c["family"], c.get("scenario") or c.get("model"), c["K"],
                   c["N"], c["BS"], c.get("isl", ""))
            cold = us.get(c["range_cold"])
            rows.setdefault(key, {})[c["op"]] = (cold, c.get("exact"), c.get("qfracs"))
    csv = ["family,group,K,N,BS,isl,ship_us,qr2_us,qr1_us,ship_qr2,ship_qr1,"
           "exact_ship,exact_qr2,exact_qr1"]
    groups = {}
    print(f"{'cell':>34} | {'ship':>8} {'qr2':>8} {'qr1':>8} | {'s/qr2':>6} {'s/qr1':>6} | exact")
    for key in sorted(rows):
        r = rows[key]
        s, q2, q1 = (r.get(a, (None, None, None)) for a in ARMS)
        if not (s[0] and q2[0] and q1[0]):
            continue
        fam, grp, K, N, BS, isl = key
        rat2, rat1 = s[0] / q2[0], s[0] / q1[0]
        nm = f"{fam}/{grp}/K{K}/{isl or N}/BS{BS}"
        ex = "/".join("T" if x[1] else "F" for x in (s, q2, q1))
        print(f"{nm:>34} | {s[0]:8.2f} {q2[0]:8.2f} {q1[0]:8.2f} | {rat2:6.3f} {rat1:6.3f} | {ex}")
        csv.append(f"{fam},{grp},{K},{N},{BS},{isl},{s[0]:.2f},{q2[0]:.2f},{q1[0]:.2f},"
                   f"{rat2:.4f},{rat1:.4f},{s[1]},{q2[1]},{q1[1]}")
        groups.setdefault((fam, grp, K), {"r2": [], "r1": []})
        groups[(fam, grp, K)]["r2"].append(rat2)
        groups[(fam, grp, K)]["r1"].append(rat1)
    (HERE / "qab.csv").write_text("\n".join(csv) + "\n")
    print("\n=== geomeans (ship/qrX > 1 = new qfracs faster) ===")
    allr2, allr1 = [], []
    for k in sorted(groups):
        g = groups[k]
        print(f"  {k[0]:>5} {str(k[1]):>6} K{k[2]:<5} n={len(g['r2']):>2}  "
              f"ship/qr2 {gm(g['r2']):.3f}   ship/qr1 {gm(g['r1']):.3f}")
        allr2 += g["r2"]; allr1 += g["r1"]
    print(f"  {'ALL':>5} {'':>6} {'':<6} n={len(allr2):>2}  "
          f"ship/qr2 {gm(allr2):.3f}   ship/qr1 {gm(allr1):.3f}")
    print(f"\nwrote {HERE/'qab.csv'} ({len(csv)-1} cells)")


if __name__ == "__main__":
    main()
