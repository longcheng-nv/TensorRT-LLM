#!/usr/bin/env python3
"""Parse nvtx_gpu_proj_trace CSV -> median µs per (cfg, N, dtype, variant) -> speedup.

V4 variant tags from bench_nsys.py:
  GVR_<dtype>|<cfg>_N<N>_bs<BS>|rep<R>
  RADIX_fp32|<cfg>_N<N>_bs<BS>|rep<R>

Speedup is computed as median(Radix_fp32) / median(GVR_<dtype>) per
(cfg, N, BS, dtype) cell.
"""

import csv
import json
import re
import statistics
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else "nsys_synth_sweep_nvtx_gpu_proj_trace.csv"

pat = re.compile(
    r":(?P<variant>GVR_(?P<dtype>fp32|bf16|fp16)|RADIX_fp32)\|"
    r"(?P<cfg>beta_\w+?)_N(?P<N>\d+)_bs(?P<BS>\d+)\|rep(?P<rep>\d+)"
)

bucket = {}  # (cfg, N, BS, dtype, variant) -> list ns
with open(CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        m = pat.match(row["Name"])
        if not m:
            continue
        variant = m.group("variant")
        dt = m.group("dtype") or "fp32"
        kind = "GVR" if variant.startswith("GVR_") else "RADIX"
        cfg = m.group("cfg")
        N = int(m.group("N"))
        BS = int(m.group("BS"))
        dur_ns = int(row["Projected Duration (ns)"])
        bucket.setdefault((cfg, N, BS, dt, kind), []).append(dur_ns)

results = {}
cfgs_order = ["beta_shallow", "beta_moderate", "beta_deep"]
keys = sorted({(c, N, BS) for (c, N, BS, _, _) in bucket})

print(f"{'cfg':<14} {'N':>6} {'BS':>5} {'dtype':>5} "
      f"{'GVR µs':>9} {'Radix-fp32 µs':>15} {'Speedup':>9}")
for cfg, N, BS in keys:
    for dt in ("bf16", "fp16", "fp32"):
        gvr = bucket.get((cfg, N, BS, dt, "GVR"), [])
        rdx = bucket.get((cfg, N, BS, "fp32", "RADIX"), [])
        if not gvr or not rdx:
            continue
        gvr_us = statistics.median(gvr) / 1000.0
        rdx_us = statistics.median(rdx) / 1000.0
        sp = rdx_us / gvr_us
        results.setdefault(f"{cfg}_N{N}_bs{BS}", {})[dt] = dict(
            gvr_us_median=gvr_us, radix_fp32_us_median=rdx_us,
            speedup_radix_over_gvr=sp,
        )
        print(f"{cfg:<14} {N:>6} {BS:>5} {dt:>5} {gvr_us:>9.2f} {rdx_us:>15.2f} {sp:>8.3f}x")

with open("nsys_speedup_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSummary -> nsys_speedup_summary.json ({len(results)} (cfg,N,BS) cells)")
