#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the qfracs A/B: per-cell 3-arm cold-L2 table + grouped geomeans.

Reads /tmp/gvrqab/qab3_results/*.jsonl + nsys_reps/*.nsys-rep (parse_nsys_full),
pairs gvr_ship / gvr_qr2 / gvr_qr1 per cell, prints per-cell ratios
(ship/qr* > 1 = new qfracs faster) and geomeans by (family, scenario|model, K),
writes qab3.csv next to this script.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/"
                   "TensorRT-LLM/indexer_topk_op_bench/report")
from parse_nsys_full import parse_rep  # noqa: E402

OUT = Path("/tmp/gvrqab/qab3_results")
HERE = Path(__file__).resolve().parent
ARMS = ("gvr_ship", "gvr_h1", "gvr_kb", "gvr_full")


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
    csv = ["family,group,K,N,BS,isl,ship_us,h1_us,kb_us,full_us,"
           "ship_h1,ship_kb,ship_full,exact_all"]
    groups = {}
    print(f"{'cell':>34} | {'ship':>8} {'h1':>8} {'kb':>8} {'full':>8} | {'s/h1':>6} {'s/kb':>6} {'s/full':>6} | exact")
    for key in sorted(rows):
        r = rows[key]
        sh, h1, kb, fl = (r.get(a, (None, None, None)) for a in ARMS)
        if not (sh[0] and h1[0] and kb[0] and fl[0]):
            continue
        fam, grp, K, N, BS, isl = key
        r1, r2, r3 = sh[0]/h1[0], sh[0]/kb[0], sh[0]/fl[0]
        nm = f"{fam}/{grp}/K{K}/{isl or N}/BS{BS}"
        ex = "/".join("T" if x[1] else "F" for x in (sh, h1, kb, fl))
        print(f"{nm:>34} | {sh[0]:8.2f} {h1[0]:8.2f} {kb[0]:8.2f} {fl[0]:8.2f} | {r1:6.3f} {r2:6.3f} {r3:6.3f} | {ex}")
        csv.append(f"{fam},{grp},{K},{N},{BS},{isl},{sh[0]:.2f},{h1[0]:.2f},{kb[0]:.2f},{fl[0]:.2f},"
                   f"{r1:.4f},{r2:.4f},{r3:.4f},{ex}")
        groups.setdefault((fam, grp, K), {"h1": [], "kb": [], "full": []})
        groups[(fam, grp, K)]["h1"].append(r1)
        groups[(fam, grp, K)]["kb"].append(r2)
        groups[(fam, grp, K)]["full"].append(r3)
    (HERE / "qab3.csv").write_text("\n".join(csv) + "\n")
    print("\n=== geomeans (ship/X > 1 = flag faster) ===")
    alls = {"h1": [], "kb": [], "full": []}
    for k in sorted(groups):
        g = groups[k]
        print(f"  {k[0]:>5} {str(k[1]):>6} K{k[2]:<5} n={len(g['h1']):>2}  "
              f"s/h1 {gm(g['h1']):.3f}  s/kb {gm(g['kb']):.3f}  s/full {gm(g['full']):.3f}")
        for a in alls: alls[a] += g[a]
    print(f"  {'ALL':>5} {'':>6} {'':<6} n={len(alls['h1']):>2}  "
          f"s/h1 {gm(alls['h1']):.3f}  s/kb {gm(alls['kb']):.3f}  s/full {gm(alls['full']):.3f}")
    print(f"\nwrote {HERE/'qab3.csv'} ({len(csv)-1} cells)")


if __name__ == "__main__":
    main()
