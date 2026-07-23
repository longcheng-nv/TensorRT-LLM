# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op40 Phase-1 characterization: per-phase breakdown of the e612 BASELINE
(gvrpkg40t [ptime] twin, default flags) over representative real cells
covering all (model, ISL-band, cs) rungs.

Fractions from clock64 medians over 20 cold-L2 launches (monotonicity checked
every launch); absolute us anchored to results/bl0/cells.csv (same node).

  CUDA_VISIBLE_DEVICES=<g> python3 measure_phases_40.py
Writes results/phase_v1_40.json + prints the table.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
BENCH = OP40.parent
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(HERE))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

from gvrpkg40v1t.top_k.gvr_topk_decode import GvrTopKKernel as TimedK  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from ab40 import launch_cfg  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
WARMUP, REPS = 10, 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

PHASES = [("p1_gather", 0, 1), ("smem_stage", 1, 2), ("p1b_rungs", 2, 3),
          ("p2_count_adm", 3, 4), ("p3_collect", 4, 5), ("p4_select", 5, 6),
          ("epilogue", 6, 7)]

# representative cells: every ISL rung x {flash L02/L42, pro L30, v32 L34}
CELLS = ([("flash", i, 2) for i in RV4.ISLS] + [("flash", "128k", 42)]
         + [("pro", i, 30) for i in RV4.ISLS]
         + [("v32", i, 34) for i in ("4k", "8k", "16k", "32k", "64k",
                                     "128k", "256k")])

_CACHE = {}


def timed_compile(K, cr, cfg):
    key = (K, cr) + tuple(sorted(cfg.items()))
    c = _CACHE.get(key)
    if c is not None:
        return c
    kobj = TimedK(dtype=cutlass.Float32, top_k=K, next_n=1, compress_ratio=cr,
                  return_output_values=False, p4_rs_rw_search=True,
                  p4_fine_skip=True, p4_peer_push=True, **cfg)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cutlass.Float32, (nr, nc),
                                        stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, K),
                                        stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, K),
                                        stride_order=(1, 0), assumed_align=16)
    ts_f = crt.make_fake_compact_tensor(cutlass.Int64, (nr, 8),
                                        stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, ts_f,
                     stream=fs, options="--enable-tvm-ffi")
    _CACHE[key] = c
    return c


def bl0_us():
    us = {}
    for r in csv.DictReader(open(OP40 / "results" / "ab_v1" / "cells.csv")):
        us[r["uuid"]] = float(r["v1_us"])
    return us


def main():
    anchor = bl0_us()
    results = {}
    for model, isl, layer in CELLS:
        uuid = f"{model}_{isl}_L{layer:02d}"
        RD = RV32 if model == "v32" else RV4
        b = RD.get_bundle(model, isl, layer, "fp32")
        logits, pre = b["logits"].contiguous(), b["preIdx"].contiguous()
        N, K, cr = b["N"], b["K"], b["cr"]
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        cfg = launch_cfg(logits, N)
        fn = timed_compile(K, cr, cfg)
        oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
        ts = torch.zeros(1, 8, dtype=torch.int64, device=DEV)
        for _ in range(WARMUP):
            fn(logits, pre, sl, None, oi, None, ts)
        torch.cuda.synchronize()
        samples, mono = [], True
        for _ in range(REPS):
            _EVICT.random_()
            torch.cuda.synchronize()
            fn(logits, pre, sl, None, oi, None, ts)
            torch.cuda.synchronize()
            t = ts[0].tolist()
            mono &= all(t[i] <= t[i + 1] for i in range(7))
            samples.append(t)
        med = [statistics.median(s[i] for s in samples) for i in range(8)]
        total = med[7] - med[0]
        frac = {n: (med[b1] - med[a0]) / total
                for n, a0, b1 in PHASES} if total > 0 else {}
        arm_us = anchor.get(uuid)
        results[uuid] = dict(cs=cfg["cluster_size"], N=N, K=K, mono=mono,
                             total_cyc=total, bl0_us=arm_us,
                             us={n: round(f * arm_us, 3) for n, f in frac.items()},
                             frac={n: round(f, 4) for n, f in frac.items()})
        print(f"{uuid:20s} cs={cfg['cluster_size']:2d} bl0={arm_us:6.2f}us "
              f"mono={mono} | "
              + " ".join(f"{n.split('_')[0]}={results[uuid]['us'].get(n, 0):5.2f}"
                         for n, _, _ in PHASES), flush=True)
        RV4._bundle_cache.clear()
        RV32._bundle_cache.clear()
    (OP40 / "results" / "phase_v1_40.json").write_text(
        json.dumps(results, indent=1))
    print("\nwritten results/phase_v1_40.json")


if __name__ == "__main__":
    main()
