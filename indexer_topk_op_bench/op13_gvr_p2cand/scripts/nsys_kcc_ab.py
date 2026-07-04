# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""nsys PURE-KERNEL cold-L2 A/B for the modified-kCC snap kernel vs baseline.

Mirrors harness/sweep_nsys.py protocol: eager launch + sync INSIDE an NVTX range
"c|<variant>|<N>", 512MB L2 evict OUTSIDE the range, whole loop under
cudaProfilerApi. Pure-kernel us/call = nvtx_kern_sum kernel-Total / NVTX-Inst
(evict kernel filtered). Strips the ~16us CUDA-graph launch floor that made the
event-timed small-N A/B unresolvable.

Run (under nsys):
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o <rep> -f true python nsys_kcc_ab.py --K 512 --dt fp32 --ns 4096,8192,16384,32768,65536
Then parse:
  python nsys_kcc_ab.py --parse <rep>.nsys-rep
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

CONFIGS = {
    512:  [("base", None, None), ("kc2x", 1024, 1024), ("kc3x", 1536, 1280)],
    1024: [("base", None, None), ("kc2x", 2048, 2048), ("kc3x", 3072, 2560)],
    2048: [("base", None, None), ("kc2x", 4096, 3686), ("kc3x", 6144, 3686)],
}
EVICT_SIG = ("distribution_elementwise",)


def run_measure(K, dt, ns, reps):
    import torch
    import torch.cuda.profiler as prof
    from kcc_walltime_ab import _compile, _EVICT, _exact
    from synth_data import get_bundle
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dt]
    cr_val = 4 if K in (512, 1024) else 1
    # pre-build + exactness check (outside profiler)
    built = []
    for N in ns:
        b = get_bundle(K, dtype, N, cfg="beta_moderate", seed=0)
        lg = b["logits"].to(dtype).contiguous(); pre = b["preIdx"].contiguous()
        sl = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
        out = torch.empty(1, K, dtype=torch.int32, device="cuda")
        for label, kcc, kft in CONFIGS[K]:
            comp = _compile(dtype, 1, N, K, cr_val, kcc, kft)
            comp(lg, pre, sl, None, out); torch.cuda.synchronize()
            ok = _exact(out, lg[0], K)
            if not ok:
                print(f"  !!! EXACTNESS FAIL K={K} {dt} N={N} {label}", flush=True)
            built.append((N, label, comp, lg, pre, sl, out))
    # warmup
    for (N, label, comp, lg, pre, sl, out) in built:
        for _ in range(10):
            comp(lg, pre, sl, None, out)
    torch.cuda.synchronize()
    prof.start()
    for (N, label, comp, lg, pre, sl, out) in built:
        name = f"c|{label}|{N}"
        for _ in range(reps):
            _EVICT.uniform_(0, 1)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(name)
            comp(lg, pre, sl, None, out)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    prof.stop()
    print("MEASURE DONE", flush=True)


def parse(rep):
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_kern_sum", "--format", "csv",
         "--force-export=true", str(rep)], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    hdr = next((i for i, r in enumerate(rows) if r and r[0] == "NVTX Range"), None)
    if hdr is None:
        print("no nvtx_kern_sum table"); return {}
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
    us = {rng: tot[rng] / inst[rng] / 1e3 for rng in tot if inst.get(rng)}
    # group by N
    cells = defaultdict(dict)
    for rng, v in us.items():
        # c|<label>|<N>
        parts = rng.split("|")
        if len(parts) == 3:
            _, label, N = parts
            cells[int(N)][label] = v
    print(f"{'N':>8} | {'base':>7s} {'kc2x':>14s} {'kc3x':>14s}")
    for N in sorted(cells):
        c = cells[N]
        base = c.get("base")
        def fmt(lbl):
            if lbl not in c or base is None:
                return f"{c.get(lbl, float('nan')):7.2f}"
            d = c[lbl] - base
            tag = "WIN" if d < -0.2 else "loss" if d > 0.2 else "~"
            return f"{c[lbl]:7.2f}({d:+.2f}{tag})"
        print(f"{N:>8} | {base:7.2f} {fmt('kc2x'):>14s} {fmt('kc3x'):>14s}")
    return cells


def parse_cells(rep):
    """Like parse() but returns {N: {label: us}} without printing."""
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
    labels = ["base", "kc2x", "kc3x"]
    print(f"# median over {len(reps)} nsys batches")
    print(f"{'N':>8} | " + " ".join(f"{l:>16s}" for l in labels))
    for N in Ns:
        med = {}
        for l in labels:
            vals = [p[N][l] for p in per if N in p and l in p[N]]
            med[l] = st.median(vals) if vals else float('nan')
        base = med.get("base", float('nan'))
        cells = []
        for l in labels:
            v = med[l]
            if l == "base":
                cells.append(f"{v:7.2f}")
            else:
                d = v - base
                tag = "WIN" if d < -0.2 else "loss" if d > 0.2 else "~"
                cells.append(f"{v:7.2f}({d:+.2f}{tag})")
        print(f"{N:>8} | " + " ".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--dt", default="fp32")
    ap.add_argument("--ns", default="4096,8192,16384,32768,65536")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--parse", default=None)
    ap.add_argument("--parse-multi", nargs="+", default=None)
    args = ap.parse_args()
    if args.parse_multi:
        parse_multi(args.parse_multi)
    elif args.parse:
        parse(args.parse)
    else:
        ns = [int(x) for x in args.ns.split(",")]
        run_measure(args.K, args.dt, ns, args.reps)
