#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op37 — clock64 per-phase breakdown of the PR-head GVR kernel at BS>1.

Extends p4f1_harness/phase_breakdown_ptime/measure_phases_prod2.py to the
BS axis: rows are the same captured cell replicated (expand+contiguous, same
as the sweep harness), phase_ts is (BS, 8) — one timestamp row per logical
row (cluster leader writes for cs>1). Rows are content-identical so row 0 is
representative; we also report the max-over-rows window as the straggler
check. Qualitative-fraction use ONLY (clock64 instrumentation slows the
kernel path-dependently; absolute us anchored to the untimed prod arm).

Cells x BS chosen for the op37 BS<=8 loss region (see PLAN.md).
Output: phase_bs.csv + stdout table. GPU via CUDA_VISIBLE_DEVICES.
"""
import csv
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP37 = HERE.parent
OPBENCH = OP37.parent
P4F1 = OPBENCH / "op26_r0_upstream_port_report" / "p4f1_harness"
sys.path.insert(0, str(P4F1))
sys.path.insert(0, str(OPBENCH / "harness"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0"), f"need cutlass 4.5.0, got {cutlass.__version__}"

from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as ProdK  # noqa: E402
from gvrpkgtimed.top_k.gvr_topk_decode import GvrTopKKernel as TimedK  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
sys.path.insert(0, str(OPBENCH / "op26_r0_upstream_port_report" / "harness"))
import real_data_v32 as RV32  # noqa: E402

RV32.BENCH_LAYERS = list(getattr(RV32, "LAYERS_ALL", [14, 34, 54]))

DEV = "cuda"
WARMUP = 10
REPS = 30
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=DEV)

PHASES = [
    ("p1_gather_stats", 0, 1),
    ("smem_stage", 1, 2),
    ("p1b_rungs", 2, 3),
    ("p2_count_admission", 3, 4),
    ("p3_collect", 4, 5),
    ("p4_select", 5, 6),
    ("epilogue", 6, 7),
]

# (model, isl, layer, BS list) — op37 BS<=8 loss cells
import os as _os
_ONLY = _os.environ.get("CELLS_ONLY", "")
CELLS = [
    ("flash", "128k", 22, [1, 2, 8]),   # N=32771  cs1 all
    ("flash", "512k", 22, [2, 8]),      # N=131075 cs8/cs4
    ("pro", "512k", 30, [2, 8]),        # N=131075 cs8/cs4
    ("pro", "256k", 30, [2, 8]),        # N=65539  cs4/cs4
    ("v32", "64k", 34, [2, 8]),         # N=65551  cs4/cs4
    ("flash", "1024k", 22, [2, 8]),     # N=262127 cs8/cs4
]
if _ONLY:  # e.g. "flash:512k:22:256,1024"
    m, isl, ly, bss = _ONLY.split(":")
    CELLS = [(m, isl, int(ly), [int(b) for b in bss.split(",")])]


def make_kernel(cls, K, cr, cfg, timed):
    kobj = cls(dtype=cutlass.Float32, top_k=K, next_n=1, compress_ratio=cr,
               return_output_values=False, **cfg)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cutlass.Float32, (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    if timed:
        ts_f = crt.make_fake_compact_tensor(cutlass.Int64, (nr, 8), stride_order=(1, 0), assumed_align=16)
        return cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, ts_f, stream=fs, options="--enable-tvm-ffi")
    return cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, stream=fs, options="--enable-tvm-ffi")


def cold_launches(callf, ts=None):
    for _ in range(WARMUP):
        callf()
    torch.cuda.synchronize()
    walls, tss = [], []
    for _ in range(REPS):
        _EVICT.zero_()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        callf()
        e1.record()
        torch.cuda.synchronize()
        walls.append(e0.elapsed_time(e1) * 1e3)
        if ts is not None:
            tss.append(ts.cpu().tolist())
    return walls, tss


def value_set_exact(idx, logits_row, N, K, ref):
    idx = idx.to(torch.int64)
    if int((idx < 0).sum()) > 0 or torch.unique(idx).numel() != K:
        return False
    lg = logits_row[:N].float()
    return bool(torch.equal(lg[idx].sort().values,
                            lg[ref.to(torch.int64)].sort().values))


def run_cell(model, isl, layer, BS):
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, layer, "fp32")
    lg_row = b["logits"].to(torch.float32).contiguous()
    pre_row = b["preIdx"].contiguous()
    N, K, cr, ref = b["N"], b["K"], b["cr"], b["ref"]
    logits = lg_row.expand(BS, -1).contiguous()
    pre = pre_row.expand(BS, -1).contiguous()
    seq_lens = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)

    cfg = ProdK.pick_config(torch.float32, BS, N, max_seq_len=N * cr)
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    print(f"[{model}/{isl} L{layer} BS={BS}] K={K} N={N} cs={cfg['cluster_size']} "
          f"T{cfg['num_threads']} hit={b['hit_rate']:.3f}", flush=True)

    out_t = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    out_p = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ts = torch.zeros(BS, 8, dtype=torch.int64, device=DEV)

    prod = make_kernel(ProdK, K, cr, cfg, timed=False)
    timed = make_kernel(TimedK, K, cr, cfg, timed=True)
    call_p = lambda: prod(logits, pre, seq_lens, None, out_p, None)  # noqa: E731
    call_t = lambda: timed(logits, pre, seq_lens, None, out_t, None, ts)  # noqa: E731

    call_p(); call_t(); torch.cuda.synchronize()
    ex_p = value_set_exact(out_p[0], logits[0], N, K, ref)
    ex_t = value_set_exact(out_t[BS - 1], logits[0], N, K, ref)

    walls_p, _ = cold_launches(call_p)
    walls_t, tss = cold_launches(call_t, ts=ts)
    us_p = statistics.median(walls_p)
    us_t = statistics.median(walls_t)

    # row 0 phases; straggler = max over rows of (t7-t0)
    cyc = {n: statistics.median([r[0][b_] - r[0][a_] for r in tss]) for n, a_, b_ in PHASES}
    window0 = statistics.median([r[0][7] - r[0][0] for r in tss])
    window_max = statistics.median([max(row[7] - row[0] for row in r) for r in tss])
    tot = sum(cyc.values()) or 1.0
    frac = {k: v / tot for k, v in cyc.items()}
    return dict(cell=f"{model}/{isl}/L{layer}", BS=BS, N=N, K=K,
                cs=cfg["cluster_size"], hit=b["hit_rate"],
                exact=ex_p and ex_t, us_prod=us_p, us_timed=us_t,
                overhead=us_t / us_p - 1.0, straggle=window_max / window0,
                cyc=cyc, frac=frac)


def main():
    torch.manual_seed(0)
    rows = []
    for model, isl, layer, bss in CELLS:
        for BS in bss:
            r = run_cell(model, isl, layer, BS)
            rows.append(r)
            print(f"  wall prod={r['us_prod']:.2f}us timed(+{100 * r['overhead']:.0f}%) "
                  f"exact={r['exact']} straggle={r['straggle']:.3f}", flush=True)
            for n, _, _ in PHASES:
                print(f"    {n:<22s} {100 * r['frac'][n]:5.1f}%  "
                      f"{r['frac'][n] * r['us_prod']:6.2f} us", flush=True)
    with open(OP37 / "results" / "phase_bs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "BS", "N", "cs", "hit", "exact", "us_prod",
                    "straggle"] + [n for n, _, _ in PHASES])
        for r in rows:
            w.writerow([r["cell"], r["BS"], r["N"], r["cs"], f"{r['hit']:.3f}",
                        r["exact"], f"{r['us_prod']:.2f}", f"{r['straggle']:.3f}"] +
                       [f"{r['frac'][n]:.4f}" for n, _, _ in PHASES])
    print("CSV ->", OP37 / "results" / "phase_bs.csv", flush=True)


if __name__ == "__main__":
    main()
