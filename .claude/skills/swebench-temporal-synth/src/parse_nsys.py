#!/usr/bin/env python3
"""Parse nvtx_gpu_proj_trace CSV -> median µs per (cfg, N, variant) -> speedup."""

import csv
import json
import re
import statistics
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else "nsys_synth_sweep_nvtx_gpu_proj_trace.csv"

# Tag form: :VARIANT|cfg_NN_bsB|repR
pat = re.compile(r":(GVR|RADIX)\|(beta_\w+?)_N(\d+)_bs(\d+)\|rep(\d+)")

# bucket[(cfg, N, variant)] = list of GPU-proj durations (ns)
bucket = {}
with open(CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        m = pat.match(row["Name"])
        if not m:
            continue
        variant, cfg, N, BS, rep = (
            m.group(1),
            m.group(2),
            int(m.group(3)),
            int(m.group(4)),
            int(m.group(5)),
        )
        dur_ns = int(row["Projected Duration (ns)"])
        bucket.setdefault((cfg, N, variant), []).append(dur_ns)

# Compute median per cell, then speedup per (cfg, N)
results = {}
cfgs_order = ["beta_shallow", "beta_moderate", "beta_deep"]
Ns_order = sorted(set(k[1] for k in bucket))

print(
    f"{'cfg':<14} {'N':>6} {'GVR µs':>9} {'Radix µs':>10} {'Speedup':>9} {'GVR n':>6} {'Radix n':>8}"
)
for N in Ns_order:
    for cfg in cfgs_order:
        gvr = bucket.get((cfg, N, "GVR"), [])
        rdx = bucket.get((cfg, N, "RADIX"), [])
        if not gvr or not rdx:
            continue
        gvr_us = statistics.median(gvr) / 1000.0
        rdx_us = statistics.median(rdx) / 1000.0
        sp = rdx_us / gvr_us
        results[f"{cfg}_N{N}"] = {
            "gvr_us_median": gvr_us,
            "radix_us_median": rdx_us,
            "speedup_radix_over_gvr": sp,
            "gvr_reps": len(gvr),
            "radix_reps": len(rdx),
            "gvr_us_min": min(gvr) / 1000.0,
            "gvr_us_max": max(gvr) / 1000.0,
            "radix_us_min": min(rdx) / 1000.0,
            "radix_us_max": max(rdx) / 1000.0,
        }
        print(
            f"{cfg:<14} {N:>6} {gvr_us:>9.2f} {rdx_us:>10.2f} {sp:>8.3f}x {len(gvr):>6} {len(rdx):>8}"
        )

with open("nsys_speedup_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSummary -> nsys_speedup_summary.json ({len(results)} cells)")
