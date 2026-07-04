# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Final nsys PURE-KERNEL cold-L2 A/B: the shipped p2c op vs the baseline op.

Unlike nsys_kcc_ab.py (which A/Bs kCC variants *within* one op via subclass
override), this compares the two ACTUAL shipped entry points across the full N
grid, exactly as production would call them:
  base : harness/gvr_cutedsl_op.gvr_cutedsl       (baseline GvrParams)
  p2c  : src/gvr_p2c_op.gvr_cutedsl_p2c           (N-dispatched kCC/kFTarget)

So at N<=65536 (fp32 K512/K1024) p2c uses narrow params; at N>=131072 p2c falls
back to baseline => the two ops should TIE there (validates the dispatch boundary,
i.e. p2c never loses at large N). This pins the 65K-131K crossover under nsys.

Protocol mirrors harness/sweep_nsys.py: eager launch + sync INSIDE NVTX range
"c|<variant>|<N>", 512MB L2 evict OUTSIDE, whole loop under cudaProfilerApi.
Pure-kernel us/call = nvtx_kern_sum kernel-Total / NVTX-Inst (evict filtered).

Run (under nsys), once per repeat r1/r2/r3 for a median:
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o results/nsys/p2c_K512_fp32_r1 -f true \
    python scripts/nsys_p2c_ab.py --K 512 --dt fp32
Then parse (median over repeats):
  python scripts/nsys_p2c_ab.py --parse-multi results/nsys/p2c_K512_fp32_r{1,2,3}.nsys-rep
"""
import argparse
import csv
import io
import subprocess
import sys
from collections import defaultdict
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


def run_measure(K, dt, ns, reps):
    import torch
    import torch.cuda.profiler as prof
    from synth_data import get_bundle
    from kcc_walltime_ab import _EVICT, _exact
    from gvr_cutedsl_op import gvr_cutedsl
    from gvr_p2c_op import gvr_cutedsl_p2c, dispatch_params

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dt]
    cr_val = 4 if K in (512, 1024) else 1
    VARIANTS = [("base", gvr_cutedsl), ("p2c", gvr_cutedsl_p2c)]

    built = []
    for N in ns:
        b = get_bundle(K, dtype, N, cfg="beta_moderate", seed=0)
        lg = b["logits"].to(dtype).contiguous(); pre = b["preIdx"].contiguous()
        sl = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
        kcc, _ = dispatch_params(dtype, K, N)
        for label, fn in VARIANTS:
            out = torch.empty(1, K, dtype=torch.int32, device="cuda")
            fn(lg, pre, sl, K, cr_val, out=out); torch.cuda.synchronize()
            ok = _exact(out, lg[0], K)
            note = f"(narrow kCC={kcc})" if (label == "p2c" and kcc) else "(baseline)"
            if not ok:
                print(f"  !!! EXACTNESS FAIL K={K} {dt} N={N} {label} {note}", flush=True)
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


def parse_multi(reps):
    import statistics as st
    per = [parse_cells(r) for r in reps]
    Ns = sorted({N for p in per for N in p})
    print(f"# p2c vs base — median over {len(reps)} nsys batches (pure-kernel cold-L2 us)")
    print(f"{'N':>8} | {'base':>8s} {'p2c':>18s} {'verdict':>10s}")
    for N in Ns:
        med = {}
        for l in ("base", "p2c"):
            vals = [p[N][l] for p in per if N in p and l in p[N]]
            med[l] = st.median(vals) if vals else float('nan')
        base = med["base"]; p2c = med["p2c"]
        d = p2c - base
        pct = 100.0 * d / base if base else 0.0
        tag = "WIN" if d < -0.2 else "loss" if d > 0.2 else "~tie"
        print(f"{N:>8} | {base:8.2f} {p2c:8.2f} ({d:+.2f}) {pct:+5.1f}% {tag:>6s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--dt", default="fp32")
    ap.add_argument("--ns", default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--parse", default=None)
    ap.add_argument("--parse-multi", nargs="+", default=None)
    args = ap.parse_args()
    if args.parse_multi:
        parse_multi(args.parse_multi)
    elif args.parse:
        parse_multi([args.parse])
    else:
        ns = [int(x) for x in args.ns.split(",")] if args.ns else N_BY_K[args.K]
        run_measure(args.K, args.dt, ns, args.reps)
