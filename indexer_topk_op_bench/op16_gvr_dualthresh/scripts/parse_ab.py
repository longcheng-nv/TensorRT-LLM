# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Parse op16 nsys A/B reps (report-identical) → per-op cold/warm us + compare.

Reuses report/parse_nsys_full.parse_rep (nvtx_kern_sum / NVTX Inst, evict
filtered). Prints, per (K,dtype,N):
  - gvr_rs (anchor, this run) vs report gvr_cutedsl_rs_cold_us  → protocol check
  - gvr_dt (op16) cold us
  - op16 speedup vs radix_cutedsl (this run) and vs sglang (this run)
  - op16 vs baseline rank-scatter (pure kernel improvement)
Plus per-seqlen averages + win-rate vs BOTH baselines.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "report"))
from parse_nsys_full import parse_rep  # noqa: E402


def load_report(hw="B300"):
    rp = _HERE.parent.parent / "report" / "seqlen_data.csv"
    d = {}
    for r in csv.DictReader(open(rp)):
        if r["hw"] != hw or r["BS"] != "1":
            continue
        def g(k):
            try:
                return float(r[k])
            except Exception:
                return None
        d[(int(r["K"]), r["dtype"], int(r["N"]))] = {
            "rs": g("gvr_cutedsl_rs_cold_us"),
            "radix": g("radix_cutedsl_cold_us"),
            "sglang": g("sglang_streaming_cold_us"),
        }
    return d


def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE.parent / "results" / "nsys_ab"
    repdir = out_root / "nsys_reps"
    report = load_report()

    # cell -> {op: (cold_us, warm_us)}
    cells = defaultdict(dict)
    for jsonl in sorted(out_root.glob("ab_K*.jsonl")):
        stem = jsonl.stem  # ab_K512_fp32
        rep = repdir / f"{stem}.nsys-rep"
        kern = parse_rep(rep) if rep.exists() else {}
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" in rec:
                continue
            key = (rec["K"], rec["dtype"], rec["N"])
            cold = kern.get(rec["range_cold"])
            warm = kern.get(rec["range_warm"])
            cells[key][rec["op"]] = (cold, warm)

    print(f"{'K':>4} {'dt':>4} {'N':>7} | {'rs_now':>7} {'rs_rpt':>7} {'anch':>5} | "
          f"{'op16':>7} {'radix':>7} {'sglang':>7} | {'op16/rs':>7} {'vs_radix':>8} {'vs_sglang':>9} | beats")
    byN = defaultdict(list)
    win = 0; tot = 0
    for key in sorted(cells):
        K, dt, N = key
        c = cells[key]
        rs = c.get("gvr_cutedsl_rs", (None, None))[0]
        op16 = c.get("gvr_dt", (None, None))[0]
        rx = c.get("radix_cutedsl", (None, None))[0]
        sg = c.get("sglang_streaming", (None, None))[0]
        rpt = report.get(key, {})
        rs_rpt = rpt.get("rs")
        anchor = (rs / rs_rpt) if (rs and rs_rpt) else None
        sp_rs = (rs / op16) if (rs and op16) else None
        sp_rx = (rx / op16) if (rx and op16) else None
        sp_sg = (sg / op16) if (sg and op16) else None
        beats = None
        if op16:
            comps = []
            if rx is not None:
                comps.append(op16 < rx)
            if sg is not None:
                comps.append(op16 < sg)
            beats = all(comps) if comps else None
            tot += 1; win += 1 if beats else 0
            byN[N].append((sp_rx, sp_sg))
        fs = lambda x, p=".2f": (("%"+p) % x) if x is not None else "  n/a"
        print(f"{K:>4} {dt:>4} {N:>7} | {fs(rs):>7} {fs(rs_rpt):>7} {fs(anchor,'.3f'):>5} | "
              f"{fs(op16):>7} {fs(rx):>7} {fs(sg):>7} | {fs(sp_rs,'.3f'):>7} {fs(sp_rx,'.3f'):>8} {fs(sp_sg,'.3f'):>9} | {beats}")

    print(f"\nop16 beats BOTH available baselines: {win}/{tot} = {100*win/max(tot,1):.1f}%")
    print(f"\n{'N':>7} | {'avg op16/radix':>15} {'avg op16/sglang':>16} | cases")
    for N in sorted(byN):
        lst = byN[N]
        arx = [x[0] for x in lst if x[0]]
        asg = [x[1] for x in lst if x[1]]
        mrx = sum(arx)/len(arx) if arx else 0
        msg = sum(asg)/len(asg) if asg else 0
        print(f"{N:>7} | {mrx:>15.3f} {msg:>16.3f} | {len(lst)}")


if __name__ == "__main__":
    main()
