# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op32 rung-0/1 baseline: measure the incumbent gvr_cutedsl_op26 (op26_r0auto's
fp32 short-row route) on the op22rr synth bundles, cold-L2 + CUDA-graph median.

Establishes the A/B floor and prints register-feasibility for the register-
resident variant. Screening tool (L1) — nsys is the ship arbiter later.
"""
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
BENCH = HERE.parents[0]
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(BENCH / "op26_gvr_logfalsi_rs" / "src"))
import bundle_data_rr  # noqa: E402
from gvr_op26_op import gvr_cutedsl_op26  # noqa: E402

DEV = "cuda"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


def flush_l2():
    _FLUSH.uniform_()


def exact_ok(out, logits, N, K):
    o = out[0]
    idx = o.long()
    if bool(((idx < 0) | (idx >= N)).any()):
        return False, "oob"
    if len(set(o.tolist())) != K:
        return False, f"dup uniq={len(set(o.tolist()))}"
    sel = logits[0].gather(0, idx).float().sort().values
    ref = torch.topk(logits[0][:N].float(), K).values.sort().values
    if not torch.equal(sel, ref):
        return False, f"maxdiff={(sel - ref).abs().max().item():.2e}"
    return True, "ok"


def time_cold(fn, reps=20, warmup=5):
    # CUDA-graph capture of one call, replay under cold L2 (flush before each).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
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
        ts.append(e0.elapsed_time(e1) * 1e3)  # us
    ts.sort()
    return ts[len(ts) // 2]


def main():
    scen_list = ["best", "real", "worst"]
    Ns = [int(x) for x in os.environ.get("N_LIST", "4096,8192,16384").split(",")]
    Ks = [int(x) for x in os.environ.get("K_LIST", "512").split(",")]
    dt = os.environ.get("DT", "fp32")
    dtype = DTYPES[dt]
    print(f"# op32 baseline  dt={dt}  Ns={Ns}  Ks={Ks}  gpu={torch.cuda.get_device_name()}")
    print(f"{'scen':5s} {'K':>4s} {'N':>7s} {'us_cold':>9s} {'exact':>6s} "
          f"{'elem/thr(512)':>13s} {'row_regs':>8s}")
    for K in Ks:
        for N in Ns:
            for scen in scen_list:
                b = bundle_data_rr.get_bundle(scen, K, dtype, N)
                logits, pre, cr = b["logits"], b["preIdx"], b["cr"]
                seq = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
                out = gvr_cutedsl_op26(logits, pre, seq, K, compress_ratio=cr)
                torch.cuda.synchronize()
                ok, msg = exact_ok(out, logits, N, K)
                fn = lambda: gvr_cutedsl_op26(logits, pre, seq, K, compress_ratio=cr)
                us = time_cold(fn)
                ept = (N + 511) // 512
                print(f"{scen:5s} {K:>4d} {N:>7d} {us:>9.3f} {msg:>6s} "
                      f"{ept:>13d} {ept:>8d}")


if __name__ == "__main__":
    main()
