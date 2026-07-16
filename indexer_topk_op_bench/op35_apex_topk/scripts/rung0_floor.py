# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""rung-0.1: floor calibration. Event timing (L1 screen) of empty kernel,
1-pass stream-reduce, and filter-append across (BS, N, ctas_per_row) with
cold-L2 eviction. Key nsys re-check runs separately on the chosen cells.

Usage: PYTHONNOUSERSITE=1 python3 scripts/rung0_floor.py [--gpu 0]
Writes results/rung0_floor.csv
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()
torch.cuda.set_device(a.gpu)

ext = load(name="floor_probe", sources=[os.path.join(HERE, "../src/floor_probe.cu")],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_100,code=sm_100"],
           build_directory=os.environ.get("BUILD_DIR", "/tmp/op35_build"), verbose=False)

EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

def cold_time(fn, reps):
    """median event time with cold L2 (evict outside timed window)."""
    ts = []
    for _ in range(reps):
        EVICT.uniform_()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)  # us
    ts.sort()
    return ts[len(ts) // 2]

rows_out = []
# 1) empty-kernel launch floor at several grids
for grid in (1, 8, 148, 1184, 9472):
    t = cold_time(lambda: ext.empty_launch(grid), a.reps)
    rows_out.append(dict(kind="empty", BS="", N="", cpr=grid, us=round(t, 2), bw=""))
    print(f"empty grid={grid:6d}: {t:7.2f} us")

# 2) stream-reduce: shapes x ctas_per_row
SHAPES = [(1, 16384), (1, 65536), (1, 131072), (1, 262144), (1, 1048576),
          (4, 131072), (32, 65536), (32, 262144), (256, 65536), (256, 262144),
          (1024, 16384), (1024, 65536), (1024, 262144)]
NSM = torch.cuda.get_device_properties(0).multi_processor_count
for BS, N in SHAPES:
    x = torch.rand(BS, N, device="cuda") + 1.0  # >0 so atomicMax-as-int works
    out = torch.zeros(BS, device="cuda")
    for cpr in sorted({1, 2, 8, 32, max(1, NSM // BS), max(1, 2 * NSM // BS)}):
        if BS * cpr > 16 * NSM:  # cap silly grids
            continue
        tickets = torch.zeros(BS, dtype=torch.int32, device="cuda")
        ext.stream_reduce(x, out, tickets, cpr)  # warm/JIT
        torch.cuda.synchronize()
        t = cold_time(lambda: ext.stream_reduce(x, out, tickets, cpr), a.reps)
        bw = BS * N * 4 / (t * 1e-6) / 1e12
        rows_out.append(dict(kind="reduce", BS=BS, N=N, cpr=cpr, us=round(t, 2), bw=round(bw, 2)))
        print(f"reduce BS={BS:5d} N={N:8d} cpr={cpr:4d}: {t:8.2f} us  {bw:5.2f} TB/s")

# 3) filter-append at ~3% admit (t_lo = 97th pct, t_hi = 99.6th) vs pure reduce
for BS, N in [(1, 131072), (1, 1048576), (32, 262144), (256, 262144), (1024, 65536)]:
    x = torch.rand(BS, N, device="cuda") + 1.0
    q = torch.quantile(x[0].float(), torch.tensor([0.97, 0.996], device="cuda"))
    t_lo, t_hi = q[0].item(), q[1].item()
    cap = max(4096, int(N * 0.05))
    cand_v = torch.zeros(BS, cap, device="cuda")
    cand_i = torch.zeros(BS, cap, dtype=torch.int32, device="cuda")
    for cpr in sorted({1, 8, max(1, NSM // BS)}):
        if BS * cpr > 16 * NSM:
            continue
        counts = torch.zeros(BS * 2, dtype=torch.int32, device="cuda")
        tickets = torch.zeros(BS, dtype=torch.int32, device="cuda")
        def run():
            ext.filter_append(x, t_hi, t_lo, cand_v, cand_i, counts, tickets, cpr)
        run(); torch.cuda.synchronize()
        admitted_once = counts.view(BS, 2)[:, 1].float().mean().item()
        counts.zero_()

        def run_timed():
            ext.filter_append(x, t_hi, t_lo, cand_v, cand_i, counts, tickets, cpr)
        t = cold_time(run_timed, a.reps)
        bw = BS * N * 4 / (t * 1e-6) / 1e12
        admitted = admitted_once
        rows_out.append(dict(kind="filter", BS=BS, N=N, cpr=cpr, us=round(t, 2), bw=round(bw, 2)))
        print(f"filter BS={BS:5d} N={N:8d} cpr={cpr:4d}: {t:8.2f} us  {bw:5.2f} TB/s  adm~{admitted:.0f}")

os.makedirs(os.path.join(HERE, "../results"), exist_ok=True)
with open(os.path.join(HERE, "../results/rung0_floor.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["kind", "BS", "N", "cpr", "us", "bw"])
    w.writeheader(); w.writerows(rows_out)
print("wrote results/rung0_floor.csv")
