# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op21 iter1 verdict: join gvr_ms nsys pure-kernel medians (results/nsys)
# with the per-cell BEST rival + best GVR-family op from the report nsys
# CSVs (B200 fp32 cold). Rival set = report.html ops the campaign must beat:
# radix single/multi CUDA, radix_cutedsl(+single/multi), sglang_streaming.
# GVR-family columns are shown for context (no-regression guard).
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_REPORT = _HERE.parents[1] / "report"
# OP21_NSYS_DIR: read reps from an archive dir (e.g. results/nsys/
# iter9_16bit_b200) instead of the live results/nsys — lets ship-review
# tables be regenerated without touching an in-flight sweep's dir.
_NSYS_DIR = Path(os.environ.get("OP21_NSYS_DIR", _ROOT / "results" / "nsys"))

RIVALS = ["radix_single_cuda", "radix_multi_cuda", "radix_cutedsl",
          "radix_cutedsl_single", "radix_cutedsl_multi", "sglang_streaming"]
GVR_BEST = ["gvr_cuda", "gvr_cutedsl", "gvr_cutedsl_rs", "gvr_multicta_cutedsl",
            "gvr_op8", "gvr_port", "gvr_mt", "gvr_sandwich"]

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "ms"
DTYPE = sys.argv[2] if len(sys.argv) > 2 else "fp32"  # iter8: bf16/fp16 too
HW = sys.argv[3] if len(sys.argv) > 3 else "B200"     # iter9: B300 cross-check

P0 = [(1024, N, BS) for N in (65536, 131072, 262144) for BS in (1, 4, 8, 16)]
if PREFIX == "msa":  # iter2 grid adds the cross-K largeN BS1 cells, no P1
    P0 = P0 + [(512, 131072, 1), (512, 262144, 1), (2048, 131072, 1),
               (2048, 262144, 1), (2048, 262144, 16)]
    P1 = []
else:
    P1 = [(K, N, BS) for K in (1024, 512, 2048) for N in (4096, 8192, 16384)
          for BS in (64, 256, 1024) if N > 2 * K]


def load_rivals():
    cells = {}
    for f in ("bs_data.csv", "seqlen_data.csv"):
        for r in csv.DictReader(open(_REPORT / f)):
            if r["hw"] != HW or r["dtype"] != DTYPE:
                continue
            key = (int(r["K"]), int(r["N"]), int(r["BS"]))
            if key in cells:
                continue
            best_r, best_rn = None, None
            for op in RIVALS:
                v = r.get(op + "_cold_us")
                if v:
                    if best_r is None or float(v) < best_r:
                        best_r, best_rn = float(v), op
            best_g, best_gn = None, None
            for op in GVR_BEST:
                v = r.get(op + "_cold_us")
                if v:
                    if best_g is None or float(v) < best_g:
                        best_g, best_gn = float(v), op
            cells[key] = (best_r, best_rn, best_g, best_gn)
    return cells


def ms_med_us(K, N, BS):
    rep = _NSYS_DIR / f"{PREFIX}_k{K}_{DTYPE}_n{N}_bs{BS}.nsys-rep"
    if not rep.exists():
        return None
    out = subprocess.run(
        ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
         "--force-export", "true", str(rep)],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        if re.search("gvr", line, re.I):
            parts = line.split(",")
            try:
                return float(parts[5]) / 1000.0  # med_ns -> us
            except (IndexError, ValueError):
                continue
    return None


def main():
    rivals = load_rivals()
    gm_r, gm_g = {"P0": [], "P1": []}, {"P0": [], "P1": []}
    print(f"{'K':>5} {'N':>7} {'BS':>5} | {'ms_us':>7} {'rival':>7} {'r/ms':>6}"
          f"  {'best_rival':<20} | {'gvrbest':>7} {'g/ms':>6}  best_gvr")
    for tag, cells in (("P0", P0), ("P1", P1)):
        for K, N, BS in cells:
            t = ms_med_us(K, N, BS)
            rv = rivals.get((K, N, BS))
            if t is None or rv is None:
                print(f"{K:>5} {N:>7} {BS:>5} | MISSING (ms={t} rival={rv})")
                continue
            br, brn, bg, bgn = rv
            gm_r[tag].append(br / t)
            gm_g[tag].append(bg / t)
            print(f"{K:>5} {N:>7} {BS:>5} | {t:7.2f} {br:7.2f} {br/t:6.3f}"
                  f"  {brn:<20} | {bg:7.2f} {bg/t:6.3f}  {bgn}")
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
    for tag in ("P0", "P1"):
        print(f"{tag}: gm rival/ms={gm(gm_r[tag]):.3f}  gvrbest/ms={gm(gm_g[tag]):.3f}"
              f"  win_vs_rival={sum(1 for x in gm_r[tag] if x >= 1.0)}/{len(gm_r[tag])}")


if __name__ == "__main__":
    main()
