# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""iter8c nsys PURE-KERNEL cold-L2 A/B: log-interp P2 variants vs baseline (+shipped p2c).

Per-K variant portfolio (iter8a host winners; kcc=None -> GvrParams default kC):
  K512 : base | p2c (iter7 ship ref, narrow-lin N<=65K) | logn (kcc=1024, kft=614)
  K1024: base | logn (kcc=2048, kft=1024) | logb (kcc=None, kft=1024)
  K2048: base | logb (kcc=None, kft=2048) | logn (kcc=4096, kft=2048)

log variants run at ALL N — the decisive new question is whether log-interp
makes the narrow window net-positive at N>=131072 (host replay: the 262K eval
tax fell from +1.75 to +1.0 while P3+P4 keeps its ~11us saving) and whether
K2048 logb's -0.58/-0.75 evals show up at 131K/262K.

Protocol identical to nsys_p2c_ab.py (eager+sync inside NVTX "c|<label>|<N>",
512MB evict outside, cudaProfilerApi window, nvtx_kern_sum/Inst, evict filtered).

Run (once per repeat r1/r2/r3):
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o results/nsys/p2clog_K512_fp32_r1 -f true \
    python3 scripts/nsys_p2clog_ab.py --K 512 --dt fp32
Parse median:
  python3 scripts/nsys_p2clog_ab.py --K 512 --parse-multi results/nsys/p2clog_K512_fp32_r{1,2,3}.nsys-rep
"""
import argparse
import csv
import io
import subprocess
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))

EVICT_SIG = ("distribution_elementwise",)
N_BY_K = {
    512: [4096, 8192, 16384, 32768, 65536, 131072, 262144],
    1024: [4096, 8192, 16384, 32768, 65536, 131072, 262144],
    2048: [8192, 16384, 32768, 65536, 131072, 262144],
}

# (kcc, kft) per log-variant label, per K.
LOG_VARIANTS = {
    512: {"logn": (1024, 614)},
    1024: {"logn": (2048, 1024), "logb": (None, 1024)},
    2048: {"logb": (None, 2048), "logn": (4096, 2048)},
}


def run_measure(K, dt, ns, reps):
    import torch
    import torch.cuda.profiler as prof
    from synth_data import get_bundle
    from kcc_walltime_ab import _EVICT, _exact
    from gvr_cutedsl_op import gvr_cutedsl
    from gvr_p2c_op import gvr_cutedsl_p2c
    from gvr_p2clog_op import gvr_cutedsl_p2clog

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dt]
    cr_val = 4 if K in (512, 1024) else 1

    variants = [("base", gvr_cutedsl)]
    if K == 512 and dt == "fp32":
        variants.append(("p2c", gvr_cutedsl_p2c))
    for label, (kcc, kft) in LOG_VARIANTS[K].items():
        variants.append((label, partial(gvr_cutedsl_p2clog, kcc=kcc, kft=kft)))

    built = []
    for N in ns:
        b = get_bundle(K, dtype, N, cfg="beta_moderate", seed=0)
        lg = b["logits"].to(dtype).contiguous(); pre = b["preIdx"].contiguous()
        sl = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
        for label, fn in variants:
            out = torch.empty(1, K, dtype=torch.int32, device="cuda")
            fn(lg, pre, sl, K, cr_val, out=out); torch.cuda.synchronize()
            if not _exact(out, lg[0], K):
                print(f"  !!! EXACTNESS FAIL K={K} {dt} N={N} {label}", flush=True)
            built.append((N, label, fn, lg, pre, sl, out))

    for (N, label, fn, lg, pre, sl, out) in built:
        for _ in range(10):
            fn(lg, pre, sl, K, cr_val, out=out)
    torch.cuda.synchronize()
    prof.start()
    for (N, label, fn, lg, pre, sl, out) in built:
        name = f"c|{label}|{N}"
        for _ in range(reps):
            _EVICT.uniform_(0, 1)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(name)
            fn(lg, pre, sl, K, cr_val, out=out)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    prof.stop()
    print("MEASURE DONE", flush=True)


def parse_cells(rep):
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_kern_sum", "--format", "csv",
         "--force-export=true", str(rep)], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr = next((i for i, r in enumerate(rows) if r and r[0] == "NVTX Range"), None)
    if hdr is None:
        return {}
    tot = defaultdict(float); inst = {}
    for r in rows[hdr + 1:]:
        if not r or len(r) < 13 or "|" not in r[0]:
            continue
        rng = r[0].lstrip(":")
        try:
            ninst = int(r[4]); total_ns = float(r[6])
        except ValueError:
            continue
        name = ",".join(r[12:]).lower()
        if any(s in name for s in EVICT_SIG):
            continue
        tot[rng] += total_ns; inst[rng] = ninst
    cells = defaultdict(dict)
    for rng in tot:
        if not inst.get(rng):
            continue
        parts = rng.split("|")
        if len(parts) == 3:
            _, label, N = parts
            cells[int(N)][label] = tot[rng] / inst[rng] / 1e3
    return cells


def parse_multi(K, reps):
    import statistics as st
    per = [parse_cells(r) for r in reps]
    Ns = sorted({N for p in per for N in p})
    labels = ["base"] + (["p2c"] if K == 512 else []) + list(LOG_VARIANTS[K])
    print(f"# K={K} log-interp A/B — median over {len(reps)} nsys batches (pure-kernel cold-L2 us)")
    print(f"{'N':>8} | " + " ".join(f"{l:>18s}" for l in labels))
    for N in Ns:
        med = {}
        for l in labels:
            vals = [p[N][l] for p in per if N in p and l in p[N]]
            med[l] = st.median(vals) if vals else float("nan")
        base = med.get("base", float("nan"))
        cells = [f"{base:8.2f}          " if l == "base" else
                 f"{med[l]:8.2f} ({100.0 * (med[l] - base) / base:+5.1f}%)" for l in labels]
        print(f"{N:>8} | " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--dt", default="fp32")
    ap.add_argument("--ns", default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--parse-multi", nargs="+", default=None)
    args = ap.parse_args()
    if args.parse_multi:
        parse_multi(args.K, args.parse_multi)
    else:
        ns = [int(x) for x in args.ns.split(",")] if args.ns else N_BY_K[args.K]
        run_measure(args.K, args.dt, ns, args.reps)
