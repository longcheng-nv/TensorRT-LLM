#!/usr/bin/env python3
"""L1 sweep timing: cold-L2 + CUDA-graph median (OmniKernel measurement ladder).

Canonical L1 numbers are COLD-L2 (warm understates memory-bound kernels 25-35%).
Warm-L2 is reported separately (models fused-producer L2-hot deployments) and
doubles as the warm-L2 veto: a traffic-saving variant that is slower even warm
is rejected without further tuning.

NOT a ship arbiter — graph-launch bias vs pure-kernel time is 0.76-0.95x.
Ship claims escalate to scripts/nsys_verdict.py (L2).

Impl module contract: kernel_fn(*args), get_inputs() -> list (fresh each call).

Usage:
    python scripts/bench_cold.py --impl src/candidate.py [--reps 30] [--evict-mb 512]
    python scripts/bench_cold.py --impl src/candidate.py --baseline src/incumbent.py
"""
import argparse
import importlib.util
import json
import statistics
import sys

import torch


def load_module(path):
    spec = importlib.util.spec_from_file_location("impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class L2Evictor:
    """Flush L2 by writing a buffer larger than L2 with a real kernel,
    OUTSIDE the timed window (pattern: harness/sweep.py::_time_both)."""

    def __init__(self, mb):
        self.buf = torch.empty(mb * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

    def evict(self):
        self.buf.uniform_()


def time_graph(fn, inputs, reps, evictor, warmup=5):
    """Capture one launch into a CUDA graph; time single replays."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(*inputs)                      # JIT/compile warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn(*inputs)
    torch.cuda.synchronize()

    def one(cold):
        if cold:
            evictor.evict()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        graph.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3       # us

    for _ in range(warmup):
        one(cold=True)
    cold = [one(cold=True) for _ in range(reps)]
    warm = [one(cold=False) for _ in range(reps)]
    return statistics.median(cold), statistics.median(warm)


def bench(path, reps, evictor):
    mod = load_module(path)
    inputs = mod.get_inputs()
    cold, warm = time_graph(mod.kernel_fn, inputs, reps, evictor)
    return {"impl": path, "cold_us": round(cold, 3), "warm_us": round(warm, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--baseline", help="incumbent for a paired same-process A/B")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--evict-mb", type=int, default=512, help="must exceed L2 (B200/B300: >=512)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    torch.cuda.init()
    ev = L2Evictor(args.evict_mb)
    out = {"label": args.label, "candidate": bench(args.impl, args.reps, ev)}
    if args.baseline:
        out["baseline"] = bench(args.baseline, args.reps, ev)
        c, b = out["candidate"], out["baseline"]
        out["speedup_cold"] = round(b["cold_us"] / c["cold_us"], 4)
        out["speedup_warm"] = round(b["warm_us"] / c["warm_us"], 4)
        # warm-L2 veto signal for traffic-saving levers
        out["warm_l2_veto"] = out["speedup_warm"] < 1.0
    print(json.dumps(out, indent=2))
    if args.baseline and out["warm_l2_veto"]:
        print("# WARM-L2 VETO: candidate loses even with a hot L2 — if this lever "
              "claims to save memory traffic, reject it now (see WALLS.md: L2 trap).",
              file=sys.stderr)


if __name__ == "__main__":
    main()
