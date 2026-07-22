# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op37 sub-stage DIFFERENTIAL (NOTES ship item 2): per-phase breakdown of
gvrpkg37t ([ptime] twin of the flag variant) with flags OFF (=head twin) vs
ALL flags ON, on the 26 representative real cells.

Expectation: only P4 (t5->t6, which contains everything d1a/d2a/d2b touch)
moves; P1/P1b/P2/P3/epilogue deltas stay within noise. Absolute us anchors to
the SAME-node pristine ship run (ship/ship_cells.csv base_us/all_us), so no
nsys pass is needed here; fractions come from clock64 medians over 20 cold-L2
launches per arm (monotonicity checked every launch).

  CUDA_VISIBLE_DEVICES=4 python3 measure_phases_37.py
Writes ship/phase_diff_37.json + prints the differential table.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent

sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE / "variant"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

from gvrpkg37t.top_k.gvr_topk_decode import GvrTopKKernel as TimedK  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from validate_op37 import launch_cfg  # noqa: E402
from ab_op37 import CELLS  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
WARMUP, REPS = 10, 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

PHASES = [("p1_gather", 0, 1), ("smem_stage", 1, 2), ("p1b_rungs", 2, 3),
          ("p2_count_adm", 3, 4), ("p3_collect", 4, 5), ("p4_select", 5, 6),
          ("epilogue", 6, 7)]
ARMS = [("base", {}),
        ("all", dict(p4_rs_rw_search=True, p4_fine_skip=True,
                     p4_peer_push=True))]

_CACHE = {}


def timed_compile(K, cr, cfg, flags):
    key = (K, cr) + tuple(sorted(cfg.items())) + tuple(sorted(flags.items()))
    c = _CACHE.get(key)
    if c is not None:
        return c
    kobj = TimedK(dtype=cutlass.Float32, top_k=K, next_n=1, compress_ratio=cr,
                  return_output_values=False, **cfg, **flags)
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


def ship_us():
    us = {}
    for r in csv.DictReader(open(HERE / "ship" / "ship_cells.csv")):
        us[r["uuid"]] = (float(r["base_us"]), float(r["all_us"]))
    return us


def main():
    anchor = ship_us()
    results = {}
    for model, isl, layer in CELLS:
        uuid = f"{model}_{isl}_L{layer:02d}"
        RD = RV32 if model == "v32" else RV4
        b = RD.get_bundle(model, isl, layer, "fp32")
        logits, pre = b["logits"].contiguous(), b["preIdx"].contiguous()
        N, K, cr = b["N"], b["K"], b["cr"]
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        cfg = launch_cfg(logits, N)
        cell = dict(cs=cfg["cluster_size"], N=N, K=K)
        for ai, (arm, flags) in enumerate(ARMS):
            fn = timed_compile(K, cr, cfg, flags)
            oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
            ts = torch.zeros(1, 8, dtype=torch.int64, device=DEV)
            for _ in range(WARMUP):
                fn(logits, pre, sl, None, oi, None, ts)
            torch.cuda.synchronize()
            samples = []
            mono = True
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
            arm_us = anchor[uuid][ai]
            cell[arm] = dict(mono=mono, total_cyc=total, arm_us=arm_us,
                             us={n: round(f * arm_us, 3)
                                 for n, f in frac.items()},
                             frac={n: round(f, 4) for n, f in frac.items()})
            print(f"  {uuid} {arm} mono={mono} "
                  + " ".join(f"{n}={cell[arm]['us'].get(n, 0):.2f}"
                             for n, _, _ in PHASES), flush=True)
        results[uuid] = cell
        RV4._bundle_cache.clear()
        RV32._bundle_cache.clear()

    # differential table: delta per phase (base - all, us)
    print(f"\n{'cell':22s} {'cs':>2s} {'d_total':>8s} {'d_P4':>7s} "
          f"{'d_other_max':>11s} worst_other")
    ok = True
    for uuid, c in results.items():
        du = {n: c["base"]["us"][n] - c["all"]["us"][n] for n, _, _ in PHASES}
        d_tot = c["base"]["arm_us"] - c["all"]["arm_us"]
        others = {n: v for n, v in du.items() if n != "p4_select"}
        wname, wval = max(others.items(), key=lambda kv: abs(kv[1]))
        flag = abs(wval) > max(0.4, 0.15 * abs(d_tot))
        ok &= not flag
        print(f"{uuid:22s} {c['cs']:2d} {d_tot:8.2f} {du['p4_select']:7.2f} "
              f"{wval:11.2f} {wname}{'  <-- CHECK' if flag else ''}")
    (HERE / "ship" / "phase_diff_37.json").write_text(
        json.dumps(results, indent=1))
    print(f"\n[phase-diff] {'CLEAN: only P4 moves' if ok else 'FLAGGED cells above'}")


if __name__ == "__main__":
    main()
