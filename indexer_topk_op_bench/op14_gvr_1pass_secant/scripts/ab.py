# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""op14 A/B: gvr_1pass vs gvr_rs_base (rank-scatter op#7) — exactness + nsys.

Data is byte-identical to report.html: synth_data.get_bundle(K,dtype,N,
cfg='beta_moderate', seed=42). Perf = nsys pure-kernel cold-L2 (harness/sweep_nsys
protocol): 512MB L2 evict OUTSIDE NVTX range, eager+sync INSIDE, whole loop under
cudaProfilerApi, nvtx_kern_sum Total/Inst, evict kernel filtered. ×3-median.

Modes:
  --exact            : value-equiv to torch.topk for both ops over cfgs×seeds×grid.
  (under nsys)       : measure -> writes NVTX ranges "c|<base|1pass>|<N>".
  --parse-multi *.rep: median over batches, print 1pass vs base.

Run (typical seqlen first, large-N regime where the win must appear):
  python scripts/ab.py --exact --K 512 --dt fp32 --ns 65536,131072,262144
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o results/nsys/ab_K512_fp32_r1 -f true \
    python scripts/ab.py --K 512 --dt fp32 --ns 65536,131072,262144
  python scripts/ab.py --parse-multi results/nsys/ab_K512_fp32_r{1,2,3}.nsys-rep
"""
import argparse
import csv
import io
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "harness"))
sys.path.insert(0, str(_HERE.parent / "src"))

EVICT_SIG = ("distribution_elementwise",)
CFGS = ["beta_shallow", "beta_moderate", "beta_deep"]
N_BY_K = {512: [65536, 131072, 262144], 1024: [65536, 131072, 262144]}


def _evict():
    import torch
    return torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")


def _exact_one(out, logits_row, K):
    import torch
    idx = out[0].clamp(min=0).long()
    if len(set(idx.tolist())) != K:
        return False, float("inf")
    v = logits_row.float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits_row.float(), K).values
    d = (v - ref).abs().max().item()
    return d < 1e-3, d


def run_exact(K, dt, ns, seeds):
    import torch
    from synth_data import get_bundle
    from gvr_1pass_op import gvr_rs_base, gvr_1pass
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dt]
    cr_val = 4 if K in (512, 1024) else 1
    nfail = 0; ncell = 0
    for N in ns:
        for cfg in CFGS:
            for seed in seeds:
                b = get_bundle(K, dtype, N, cfg=cfg, seed=seed)
                lg = b["logits"].to(dtype).contiguous(); pre = b["preIdx"].contiguous()
                sl = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
                for name, fn in (("base", gvr_rs_base), ("1pass", gvr_1pass)):
                    out = fn(lg, pre, sl, K, cr_val); torch.cuda.synchronize()
                    ok, d = _exact_one(out, lg[0], K)
                    ncell += 1
                    if not ok:
                        nfail += 1
                        print(f"  FAIL {name} K={K} {dt} N={N} {cfg} s{seed} d={d:.2e}", flush=True)
    print(f"EXACT {dt} K={K}: {ncell - nfail}/{ncell} pass", flush=True)
    return nfail


def run_measure(K, dt, ns, reps):
    import torch
    import torch.cuda.profiler as prof
    from synth_data import get_bundle
    from gvr_1pass_op import gvr_rs_base, gvr_1pass
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dt]
    cr_val = 4 if K in (512, 1024) else 1
    EVICT = _evict()
    VARIANTS = [("base", gvr_rs_base), ("1pass", gvr_1pass)]
    built = []
    for N in ns:
        b = get_bundle(K, dtype, N, cfg="beta_moderate", seed=42)
        lg = b["logits"].to(dtype).contiguous(); pre = b["preIdx"].contiguous()
        sl = torch.full((1,), N * cr_val, dtype=torch.int32, device="cuda")
        for label, fn in VARIANTS:
            out = torch.empty(1, K, dtype=torch.int32, device="cuda")
            fn(lg, pre, sl, K, cr_val, out=out); torch.cuda.synchronize()
            ok, d = _exact_one(out, lg[0], K)
            if not ok:
                print(f"  !!! EXACT FAIL K={K} {dt} N={N} {label} d={d:.2e}", flush=True)
            built.append((N, label, fn, lg, pre, sl, out))
    for (N, label, fn, lg, pre, sl, out) in built:
        for _ in range(10):
            fn(lg, pre, sl, K, cr_val, out=out)
    torch.cuda.synchronize()
    prof.start()
    for (N, label, fn, lg, pre, sl, out) in built:
        name = f"c|{label}|{N}"
        for _ in range(reps):
            EVICT.uniform_(0, 1)
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
    print(f"# op14 1pass vs base(rank-scatter) — median over {len(reps)} nsys batches (pure-kernel cold-L2 us)")
    print(f"{'N':>8} | {'base':>8s} {'1pass':>18s} {'verdict':>8s}")
    for N in Ns:
        med = {}
        for l in ("base", "1pass"):
            vals = [p[N][l] for p in per if N in p and l in p[N]]
            med[l] = st.median(vals) if vals else float('nan')
        base = med["base"]; m = med["1pass"]
        d = m - base; pct = 100.0 * d / base if base else 0.0
        tag = "WIN" if d < -0.2 else "loss" if d > 0.2 else "~tie"
        print(f"{N:>8} | {base:8.2f} {m:8.2f} ({d:+.2f}) {pct:+5.1f}% {tag:>6s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--dt", default="fp32")
    ap.add_argument("--ns", default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--parse-multi", nargs="+", default=None)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")] if args.ns else N_BY_K[args.K]
    if args.parse_multi:
        parse_multi(args.parse_multi)
    elif args.exact:
        nfail = run_exact(args.K, args.dt, ns, [int(x) for x in args.seeds.split(",")])
        sys.exit(1 if nfail else 0)
    else:
        run_measure(args.K, args.dt, ns, args.reps)
