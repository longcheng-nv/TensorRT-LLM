#!/usr/bin/env python3
"""L2 SHIP ARBITER: nsys pure-kernel timing, median of x3 independent batches,
with the anchor protocol (OmniKernel measurement ladder).

Why: CUDA-event/graph timing carries launch bias (0.76-0.95x) and the event
axis fabricated >=5 reproducible lies in the source record. Ship verdicts come
from nsys GPU kernel time only. A single batch has >=0.5us variance -> x3 median.

Anchor protocol: absolute us never transfer across nodes/sessions. Pass
--anchor-expected to assert the anchor cell reproduces within --anchor-tol
(default 3%) before any number from this run may be quoted.

Token hygiene: nsys artifacts embed the process env -> this wrapper always
launches nsys under `env -u GITHUB_TOKEN -u HF_TOKEN`. Never commit
*.sqlite / *.nsys-rep (gitignore them BEFORE the first results commit).

Impl module contract: kernel_fn(*args), get_inputs().

Usage:
    python scripts/nsys_verdict.py --impl src/candidate.py --baseline src/incumbent.py \
        [--batches 3] [--launches 30] [--kernel-regex '.*'] \
        [--anchor-impl src/incumbent.py --anchor-expected 17.8 --anchor-tol 0.03]
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

RUNNER = r'''
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("impl", sys.argv[1]); mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
n_launch = int(sys.argv[2])
inputs = mod.get_inputs()
evict = torch.empty(512 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
for _ in range(3):
    mod.kernel_fn(*inputs)              # compile warmup (outside profiled window)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(n_launch):
    evict.uniform_()                    # cold-L2; excluded from kernel_fn's own rows
    torch.cuda.synchronize()
    mod.kernel_fn(*inputs)
    torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
'''


def nsys_batch(impl, launches, kernel_regex, workdir, tag):
    """One independent nsys pass; returns median per-launch kernel us."""
    rep = os.path.join(workdir, f"{tag}.nsys-rep")
    runner = os.path.join(workdir, "runner.py")
    with open(runner, "w") as f:
        f.write(RUNNER)
    cmd = ["env", "-u", "GITHUB_TOKEN", "-u", "HF_TOKEN",
           "nsys", "profile", "--capture-range=cudaProfilerApi",
           "--capture-range-end=stop", "-o", rep, "--force-overwrite=true",
           sys.executable, runner, impl, str(launches)]
    subprocess.run(cmd, check=True, capture_output=True)
    stats = subprocess.run(
        ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", rep],
        check=True, capture_output=True, text=True).stdout
    # csv: Time(%), Total Time(ns), Instances, Avg, Med, Min, Max, StdDev, Name
    total_ns = 0.0
    pat = re.compile(kernel_regex)
    for line in stats.splitlines():
        cols = [c.strip().strip('"') for c in line.split(",")]
        if len(cols) < 9 or not cols[1].replace(".", "").isdigit():
            continue
        name = cols[-1]
        if "uniform" in name.lower() or "distribution" in name.lower():
            continue                     # the L2 evictor kernel
        if pat.search(name):
            total_ns += float(cols[1])
    return total_ns / launches / 1e3     # us per launch (all matched kernels)


def measure(impl, batches, launches, kernel_regex, workdir, tag):
    vals = [nsys_batch(impl, launches, kernel_regex, workdir, f"{tag}_b{i}")
            for i in range(batches)]
    return statistics.median(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--baseline", help="the TRUE incumbent (A/B against nothing else)")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--launches", type=int, default=30)
    ap.add_argument("--kernel-regex", default=".*")
    ap.add_argument("--anchor-impl", help="anchor-cell impl (usually the incumbent)")
    ap.add_argument("--anchor-expected", type=float, help="expected anchor us")
    ap.add_argument("--anchor-tol", type=float, default=0.03)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="nsys_verdict_") as wd:
        out = {}
        if args.anchor_impl and args.anchor_expected:
            a_med, _ = measure(args.anchor_impl, args.batches, args.launches,
                               args.kernel_regex, wd, "anchor")
            drift = a_med / args.anchor_expected
            out["anchor"] = {"us": round(a_med, 3), "expected": args.anchor_expected,
                             "drift": round(drift, 4)}
            if abs(drift - 1.0) > args.anchor_tol:
                print(json.dumps(out, indent=2))
                sys.exit("ANCHOR DRIFT > tol: re-baseline the whole grid on this "
                         "node; numbers from this run MUST NOT be quoted.")
        c_med, c_all = measure(args.impl, args.batches, args.launches,
                               args.kernel_regex, wd, "cand")
        out["candidate"] = {"impl": args.impl, "us_median_of_batches": round(c_med, 3),
                            "batches_us": [round(v, 3) for v in c_all]}
        if args.baseline:
            b_med, b_all = measure(args.baseline, args.batches, args.launches,
                                   args.kernel_regex, wd, "base")
            out["baseline"] = {"impl": args.baseline, "us_median_of_batches": round(b_med, 3),
                               "batches_us": [round(v, 3) for v in b_all]}
            out["speedup"] = round(b_med / c_med, 4)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
