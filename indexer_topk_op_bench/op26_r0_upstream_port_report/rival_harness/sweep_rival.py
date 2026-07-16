#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op26 report §9 — nsys batch sweep of 4 arms (op26_r0auto GVR anchor +
radix_cutedsl + sglang_v2 + flashinfer_topk) on the SAME inputs as report
§3 (op22-§env synthetic) and §4 (V4 Flash/Pro + V3.2 real decode-capture),
across BS x seqlen x dtype x K.

Timing protocol == harness/sweep_nsys.measure_cell (NVTX c|/w| ranges, cold-L2
512MB evict OUTSIDE the range, 20 cold + 50 warm reps); one nsys-rep per
(family, sweep, key, dtype) batch, resumable at cell granularity via the jsonl.
Correctness is folded in: each cell records exact = (unique==K AND gathered
value-set == torch.topk value-set) on row0 & row BS-1 (tie-robust, order-free).

Unsupported (op,dtype)/(op,K) or missing real captures raise -> recorded as
error and OMITTED downstream (never faked). Run UNDER nsys via
drive_rival_shard.sh; kernel us filled by parse_rival.py.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]            # indexer_topk_op_bench/ (rival_harness is 2 levels down)
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_BENCH / "op22_temporal_fixed_hr_bench"))

from sweep_nsys import measure_cell                         # noqa: E402
from ops_rival import build_call_rival, ops_for_rival, GVR_ANCHOR  # noqa: E402
import bundle_data_env as SYNTH                             # noqa: E402
import real_data_v4cap as RV4                               # noqa: E402
import real_data_v32 as RV32                                # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DEV = "cuda"

N_SEQ = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_BS_REP = [16384, 65536, 131072, 262144]      # synth BS-sweep representative Ns
REAL_LAYER = {"flash": 22, "pro": 30, "v32": 34}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
REAL_BS_ISL = {"flash": "128k", "pro": "128k", "v32": "128k"}   # BS-sweep rung


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["op"], r["N"], r["BS"], r.get("isl", "")))
            except Exception:
                pass
    return done


def _exact(out_idx_getter, logits_full, N, K, BS):
    """out_idx_getter()->[BS,K] i32. True iff row0 & row BS-1 select the correct
    top-K value-set (unique count K, gathered values == torch.topk, sorted)."""
    idx_t = out_idx_getter()
    ref = torch.topk(logits_full[:N].float(), K).values.sort().values
    rows = (0, BS - 1) if BS > 1 else (0,)
    for r in rows:
        idx = idx_t[r].long()
        if idx.numel() != K or idx.min() < 0 or idx.max() >= N or idx.unique().numel() != K:
            return False
        got = logits_full[idx].float().sort().values
        if not torch.equal(got, ref):
            return False
    return True


def synth_bundle(scen, K, dtype, N):
    b = SYNTH.get_bundle(scen, K, torch.float32, N)      # values dtype-invariant
    return dict(logits=b["logits"].to(dtype).contiguous(), preIdx=b["preIdx"].contiguous(),
                cr=b["cr"], N=N, K=K, hit=b.get("kernel_hit_rate"),
                data_src=b.get("cfg"), isl="")


def real_bundle(model, isl, dtype_name):
    L = REAL_LAYER[model]
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, L, dtype_name)
    return dict(logits=b["logits"].contiguous(), preIdx=b["preIdx"].contiguous(),
                cr=b["cr"], N=b["N"], K=b["K"], hit=b["hit_rate"],
                data_src=f"{model}/{isl}/L{L}", isl=isl)


def cells_synth(sweep, K, dt):
    if sweep == "seqlen":
        return [(N, 1) for N in N_SEQ if N > 2 * K]
    return [(N, BS) for N in N_BS_REP if N > 2 * K for BS in BS_GRID]


def cells_real(sweep, model, isl_override=None):
    if sweep == "seqlen":
        return [(isl, 1) for isl in REAL_ISLS[model]]
    isl = isl_override or REAL_BS_ISL[model]
    return [(isl, BS) for BS in BS_GRID]


def run_synth(sweep, scen, K, dt_name, out_path, rc, rw):
    dtype = DTYPES[dt_name]
    done = _load_done(out_path)
    cells = cells_synth(sweep, K, dt_name)
    ops = ops_for_rival(dt_name, K)
    print(f"# synth {scen} {sweep} K={K} {dt_name} cells={len(cells)} ops={ops}", flush=True)
    f = open(out_path, "a")
    prof.start()
    try:
        for i, (N, BS) in enumerate(cells):
            bd = synth_bundle(scen, K, dtype, N)
            lg_full = bd["logits"][0, :N]
            for op in ops:
                if (op, N, BS, "") in done:
                    continue
                base = f"{op}|{K}|{dt_name}|{N}|{BS}"
                rec = dict(family="synth", sweep=sweep, scenario=scen, op=op, K=K,
                           dtype=dt_name, N=N, BS=BS, cr=bd["cr"], hit=bd["hit"],
                           data_src=bd["data_src"], isl="",
                           range_cold=f"c|{base}", range_warm=f"w|{base}",
                           reps_cold=rc, reps_warm=rw)
                try:
                    call, keep, extra, getter = build_call_rival(
                        op, K, dtype, N, BS, bd["cr"], bd["logits"], bd["preIdx"])
                    rec.update(extra)
                    if getter is not None:
                        rec["exact"] = bool(_exact(getter, lg_full, N, K, BS))
                    measure_cell(call, base, rc, rw)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
                f.write(json.dumps(rec) + "\n"); f.flush()
                gc.collect(); torch.cuda.empty_cache()
            if (i + 1) % 2 == 0 or i + 1 == len(cells):
                print(f"[synth {scen}/{sweep} K{K} {dt_name}] {i+1}/{len(cells)} (N={N} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()


def run_real(sweep, model, dt_name, out_path, rc, rw, isl_override=None, ops_filter=None):
    dtype = DTYPES[dt_name]
    done = _load_done(out_path)
    cells = cells_real(sweep, model, isl_override)
    print(f"# real {model} {sweep} {dt_name} cells={len(cells)}", flush=True)
    f = open(out_path, "a")
    prof.start()
    try:
        for i, (isl, BS) in enumerate(cells):
            try:
                bd = real_bundle(model, isl, dt_name)
            except Exception as e:                       # missing capture -> omit rung
                print(f"  SKIP real {model} {isl}: {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
            K, N, cr = bd["K"], bd["N"], bd["cr"]
            lg_full = bd["logits"][0, :N]
            ops = ops_for_rival(dt_name, K)
            if ops_filter is not None:
                ops = [o for o in ops if o in ops_filter]
            for op in ops:
                if (op, N, BS, isl) in done:
                    continue
                base = f"{op}|{model}|{isl}|{dt_name}|{N}|{BS}"
                rec = dict(family="real", sweep=sweep, model=model, op=op, K=K,
                           dtype=dt_name, N=N, BS=BS, cr=cr, hit=bd["hit"], isl=isl,
                           data_src=bd["data_src"],
                           range_cold=f"c|{base}", range_warm=f"w|{base}",
                           reps_cold=rc, reps_warm=rw)
                try:
                    call, keep, extra, getter = build_call_rival(
                        op, K, dtype, N, BS, cr, bd["logits"], bd["preIdx"])
                    rec.update(extra)
                    if getter is not None:
                        rec["exact"] = bool(_exact(getter, lg_full, N, K, BS))
                    measure_cell(call, base, rc, rw)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
                f.write(json.dumps(rec) + "\n"); f.flush()
                gc.collect(); torch.cuda.empty_cache()
            if (i + 1) % 2 == 0 or i + 1 == len(cells):
                print(f"[real {model}/{sweep} {dt_name}] {i+1}/{len(cells)} (isl={isl} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["synth", "real"], required=True)
    ap.add_argument("--sweep", choices=["seqlen", "bs"], required=True)
    ap.add_argument("--dtype", choices=list(DTYPES), required=True)
    ap.add_argument("--K", type=int, choices=[512, 1024, 2048])
    ap.add_argument("--scenario", choices=["best", "worst"], default="best")
    ap.add_argument("--model", choices=["flash", "pro", "v32"])
    ap.add_argument("--isl", default=None,
                    help="real bs-sweep ISL override (backfill: one batch per ISL)")
    ap.add_argument("--ops", default=None,
                    help="comma list; intersected with ops_for_rival (dtype contract kept)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    a = ap.parse_args()
    root = Path(a.out_root); root.mkdir(parents=True, exist_ok=True)
    ops_filter = set(a.ops.split(",")) if a.ops else None
    if a.family == "synth":
        assert a.K, "--K required for synth"
        out = root / f"synth_{a.scenario}_{a.sweep}_K{a.K}_{a.dtype}.jsonl"
        run_synth(a.sweep, a.scenario, a.K, a.dtype, out, a.reps, a.reps_warm)
    else:
        assert a.model, "--model required for real"
        suffix = f"_{a.isl}" if a.isl else ""
        out = root / f"real_{a.model}_{a.sweep}_{a.dtype}{suffix}.jsonl"
        run_real(a.sweep, a.model, a.dtype, out, a.reps, a.reps_warm,
                 isl_override=a.isl, ops_filter=ops_filter)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
