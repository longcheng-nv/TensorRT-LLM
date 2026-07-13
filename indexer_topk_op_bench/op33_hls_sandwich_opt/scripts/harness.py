# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op33 harness: build/time/gate op27_hls (gvr_ms_auto @ op27 HEAD) and op33
candidate arms on the op22rr synth bundles, BS=1 fp32. Reusable by probes.

L1 = cold-L2 + CUDA-graph median (screening; NOISE at N<=16K per op32).
nsys = ship arbiter (separate driver). Exactness = tie-aware value-multiset.
"""
import os
import sys
from pathlib import Path

import torch

# op27_hls needs these env BEFORE importing the arm builders.
os.environ.setdefault("OP21_FB_LOGFALSI", "1")
os.environ.setdefault("OP27_K2048_TAIL", "1")

HERE = Path(__file__).resolve().parents[1]
BENCH = HERE.parents[0]
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
import bundle_data_rr  # noqa: E402
from sweep_nsys import build_call  # noqa: E402

DEV = "cuda"
DTYPES = {"fp32": torch.float32}
_FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


def flush_l2():
    _FLUSH.uniform_()


def get_inputs(scen, K, N, dt="fp32"):
    b = bundle_data_rr.get_bundle(scen, K, DTYPES[dt], N)
    return b["logits"], b["preIdx"], b["cr"]


def make_call(op, K, N, BS=1, scen="real", dt="fp32"):
    logits, preidx, cr = get_inputs(scen, K, N, dt)
    call, keep, extra = build_call(op, K, DTYPES[dt], N, BS, cr, logits, preidx)
    return call, keep, extra, logits


def exact_ok(out, logits, N, K):
    o = out[0]
    idx = o.long()
    if bool(((idx < 0) | (idx >= N)).any()):
        return False, "oob"
    if len(set(o.tolist())) != K:
        return False, f"dup{len(set(o.tolist()))}"
    sel = logits[0].gather(0, idx).float().sort().values
    ref = torch.topk(logits[0][:N].float(), K).values.sort().values
    return (torch.equal(sel, ref),
            "ok" if torch.equal(sel, ref) else f"d{(sel-ref).abs().max():.1e}")


def time_cold(call, reps=30, warmup=6):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for _ in range(warmup):
        flush_l2()
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        flush_l2()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    op = os.environ.get("OP", "gvr_ms_auto")
    Ks = [int(x) for x in os.environ.get("K_LIST", "512,1024,2048").split(",")]
    Ns = [int(x) for x in os.environ.get("N_LIST", "8192,16384,32768,65536,131072").split(",")]
    print(f"# op33 harness L1  op={op}  gpu={torch.cuda.get_device_name()}")
    print(f"{'scen':5}{'K':>5}{'N':>8}{'us_cold':>9}{'exact':>7}{'extra':>18}")
    for K in Ks:
        for N in Ns:
            for scen in ["best", "real", "worst"]:
                try:
                    call, keep, extra, logits = make_call(op, K, N, 1, scen)
                    call()
                    torch.cuda.synchronize()
                    out = keep[3] if len(keep) > 3 else keep[-1]
                    ok, msg = exact_ok(out, logits, N, K)
                    us = time_cold(call)
                    print(f"{scen:5}{K:>5}{N:>8}{us:>9.3f}{msg:>7}{str(extra):>18}")
                except Exception as e:
                    print(f"{scen:5}{K:>5}{N:>8}  ERR {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
