#!/usr/bin/env python3
# op26 iter7 预研 — wall-time phase decomposition from ncu warp-stall
# sampling: bucket all samples by SASS address into segments delimited by
# the EXECUTED cluster-barrier (UCGABAR_WAIT) sites. Sampling is uniform in
# time over all resident warps => segment sample share ~= wall-time share.
# Usage: python3 analyze_iter7_segments.py results_iter7_prof/<rep>.ncu-rep ...
import csv
import io
import os
import subprocess
import sys

ENV = {k: v for k, v in os.environ.items()
       if k not in ("GITHUB_TOKEN", "HF_TOKEN")}


def segments(rep):
    out = subprocess.run(
        ["ncu", "--import", rep, "--page", "source", "--csv"],
        capture_output=True, text=True, env=ENV).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hi = next((i for i, r in enumerate(rows) if r and r[0] == "Address"),
              None)
    if hi is None:
        print(f"{rep}: no source page")
        return
    hdr = rows[hi]
    isrc = hdr.index("Source")
    isamp = next(i for i, h in enumerate(hdr)
                 if "Stall Sampling (All" in h)
    iexe = hdr.index("Instructions Executed")
    # merge samples across repeated launches (same addresses)
    per_addr = {}
    for r in rows[hi + 1:]:
        if len(r) <= max(isamp, iexe) or not r[0].startswith("0x"):
            continue
        a = int(r[0], 16)
        try:
            s = int(r[isamp] or 0)
            e = int(r[iexe] or 0)
        except ValueError:
            continue
        if a in per_addr:
            per_addr[a][1] += s
        else:
            per_addr[a] = [r[isrc].strip(), s, e]
    insts = sorted((a, v[0], v[1], v[2]) for a, v in per_addr.items())
    bounds = [a for a, src, s, e in insts
              if "UCGABAR_WAIT" in src and e > 0]
    n_launch = 1  # execs are per-launch identical; samples merged
    print(f"\n=== {os.path.basename(rep)} ===")
    print(f"  executed cluster WAITs: {len(bounds)}")
    seg, agg, wait_s, lm = 0, {}, {}, {}
    for a, src, s, e in insts:
        agg[seg] = agg.get(seg, 0) + s
        if "UCGABAR_WAIT" in src and e > 0:
            wait_s[seg] = s
        op = src.split()[0].split(".")[0] if src else "?"
        if e > 0 and op in ("LDG", "STG", "ATOMS", "LDS", "STS", "BAR",
                            "REDUX", "CREDUX", "FSETP", "MATCH"):
            d = lm.setdefault(seg, {})
            d[op] = d.get(op, 0) + e
        if a in bounds:
            seg += 1
    tot = sum(agg.values()) or 1
    for k in sorted(agg):
        top = sorted(lm.get(k, {}).items(), key=lambda kv: -kv[1])[:4]
        w = wait_s.get(k, 0)
        print(f"  seg{k}: {100*agg[k]/tot:5.1f}%  samples={agg[k]:>4} "
              f"(atWAIT={w})  {top}")


if __name__ == "__main__":
    for rep in sys.argv[1:]:
        segments(rep)
