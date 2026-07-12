# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nsys-based PURE-KERNEL sweep — same grid / inputs / ops as the CUDA-event
sweep (sweep.py + sweep_cluster.py + sweep_op8.py), but the per-op time is the
kernel's GPU time measured by Nsight Systems (NVTX-range per cell), NOT a CUDA
event (which also charges the cuda-graph single-replay launch/queue ~0.76-0.95x
bias seen in report/nsys_vs_event.csv).

Covers ALL 11 report ops (so the nsys re-test fully replaces the cuda-event
report.html, including the cluster + op8 columns):
  gvr_cuda, gvr_cutedsl, gvr_cutedsl_rs, gvr_multicta_cutedsl, gvr_op8,
  radix_single_cuda, radix_multi_cuda, radix_cutedsl, radix_cutedsl_single,
  radix_cutedsl_multi, sglang_streaming (fp32-only).

Inputs are byte-identical to the cuda-event sweep: synth_data.get_bundle(seed=42,
unified preIdx hit-rate), so K=512->V4 Flash, K=1024->V4 Pro, K=2048->DSv3.2.

Both L2 regimes are measured per cell (matching the report's cold+warm columns):
  - warm-L2 : reps with logits HOT in L2 (no evict between reps); range "w|...".
  - cold-L2 : evict 512MB L2 (OUTSIDE the range) before each single timed call;
              range "c|...". cold-L2 is the canonical memory-bound metric.

Each timed call is EAGER with a sync INSIDE the NVTX range, so the op's kernels
execute within the host range window and nsys's NVTX->GPU projection attributes
them to that range. (cuda-graph replay does NOT work here: the graph launch is
async and its kernels run after range_pop, so the projection window is empty.
nsys measures pure kernel GPU time, identical eager vs graph.)

The whole loop runs between torch.cuda.profiler.start()/stop(), so nsys (launched
with --capture-range=cudaProfilerApi) records only it. Per-cell metadata + the two
NVTX range names are appended to a jsonl; the kernel us is filled in afterwards by
report/parse_nsys_full.py from the .nsys-rep.

Run UNDER nsys (see drive_nsys_full.sh), e.g.:
  CUDA_VISIBLE_DEVICES=1 nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o <rep> -f true \
    python sweep_nsys.py --sweep bs --K 512 --dtype fp32 --out-root ../results_b200_nsys

Resumable: cells already present in the batch jsonl are skipped, but note nsys
overwrites the .nsys-rep per run, so a resumed batch must re-measure the whole
batch in ONE nsys run (the driver enforces this at batch granularity).
"""
import argparse
import gc
import json
from pathlib import Path

import torch
import torch.cuda.profiler as prof

from sweep import (_build_inputs, _EVICT, DTYPES, KS, N_SEQ, BS_GRID, OPS,
                   ALL_OPS, get_bundle, valid_N_for_K)
from sweep_cluster import _build_cluster_call
from sweep_op8 import _build_op8_call
from sweep_op21 import _build_op21_call
from sweep_op26 import (_build_op26_1cta_call, _build_op26_mc_call,
                        _build_op26_r0_call)

DEV = "cuda"

# Full report op set (11). sglang is fp32-only and added per-dtype below.
FULL_OPS = ["gvr_cuda", "gvr_cutedsl", "gvr_cutedsl_rs",
            "gvr_multicta_cutedsl", "gvr_op8",
            "radix_single_cuda", "radix_multi_cuda", "radix_cutedsl",
            "radix_cutedsl_single", "radix_cutedsl_multi"]
KNOWN_OPS = (set(FULL_OPS) | set(ALL_OPS)
             | {"gvr_ms_auto", "op26_1cta", "op26_mc", "op26_r0"})


def build_call(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    """Return (call, keep, extra) where extra has e.g. cluster_size, or {}."""
    if op == "gvr_multicta_cutedsl":
        call, keep, cs = _build_cluster_call(K, dtype, N, BS, cr,
                                             logits_row, preidx_row)
        return call, keep, {"cluster_size": cs}
    if op == "gvr_op8":
        call, keep, cs = _build_op8_call(K, dtype, N, BS, cr,
                                         logits_row, preidx_row)
        return call, keep, {"cluster_size": cs}
    if op == "gvr_ms_auto":
        call, keep, path = _build_op21_call(K, dtype, N, BS, cr,
                                            logits_row, preidx_row)
        return call, keep, {"ms_path": path}
    if op == "op26_1cta":
        call, keep = _build_op26_1cta_call(K, dtype, N, BS, cr,
                                           logits_row, preidx_row)
        return call, keep, {}
    if op == "op26_r0":
        call, keep = _build_op26_r0_call(K, dtype, N, BS, cr,
                                         logits_row, preidx_row)
        return call, keep, {}
    if op == "op26_mc":
        call, keep, cs = _build_op26_mc_call(K, dtype, N, BS, cr,
                                             logits_row, preidx_row)
        return call, keep, {"cluster_size": cs}
    call, keep = _build_inputs(op, K, dtype, N, BS, cr, logits_row, preidx_row)
    return call, keep, {}


def measure_cell(call, base, reps_cold, reps_warm, warmup=10):
    """Measure warm-L2 then cold-L2 pure-kernel time for one cell.

    warm: reps with no evict (logits stay hot in L2 across reps); range "w|base".
    cold: evict 512MB (OUTSIDE the range) before each single timed call; "c|base".
    A sync INSIDE each range keeps it open until the kernels finish, so nsys's
    NVTX->GPU projection attributes the op's kernels to the range."""
    warm_name, cold_name = f"w|{base}", f"c|{base}"
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    # warm-L2 (no evict -> logits hot in L2)
    for _ in range(reps_warm):
        torch.cuda.nvtx.range_push(warm_name)
        call()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    # cold-L2 (evict before each timed call)
    for _ in range(reps_cold):
        _EVICT.uniform_(0, 1)            # cold-L2 flush (OUTSIDE the range)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(cold_name)
        call()
        torch.cuda.synchronize()         # keep range open until kernels finish
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["sweep"], r["op"], r["K"], r["dtype"], r["N"], r["BS"]))
            except Exception:
                pass
    return done


def run_batch(sweep, cells, ops, out_path, cfg, reps_cold, reps_warm):
    done = _load_done(out_path)
    f = open(out_path, "a")
    total = len(cells)
    prof.start()
    try:
        for i, (K, dt_name, N, BS) in enumerate(cells):
            dtype = DTYPES[dt_name]
            b = get_bundle(K, dtype, N, cfg=cfg)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            for op in ops:
                key = (sweep, op, K, dt_name, N, BS)
                if key in done:
                    continue
                base = f"{op}|{K}|{dt_name}|{N}|{BS}"
                rec = {"sweep": sweep, "op": op, "K": K, "dtype": dt_name,
                       "N": N, "BS": BS, "cr": cr,
                       "data_src": b.get("cfg", cfg),
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "reps_cold": reps_cold, "reps_warm": reps_warm}
                try:
                    call, keep, extra = build_call(op, K, dtype, N, BS, cr,
                                                   logits_row, preidx_row)
                    rec.update(extra)
                    measure_cell(call, base, reps_cold, reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            if (i + 1) % 2 == 0 or i + 1 == total:
                print(f"[{sweep} K={cells[0][0]} {cells[0][1]}] {i+1}/{total} "
                      f"(N={N} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()


def default_ops(dtype):
    ops = list(FULL_OPS)
    if dtype == "fp32":
        ops = ops + ["sglang_streaming"]   # fp32-only kernel
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["seqlen", "bs"], required=True)
    ap.add_argument("--K", type=int, required=True, choices=KS)
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    ap.add_argument("--cfg", default="beta_moderate")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ops", default=None,
                    help="comma op subset (default: all 11 report ops; sglang fp32-only)")
    ap.add_argument("--reps", type=int, default=20, help="cold-L2 reps")
    ap.add_argument("--reps-warm", type=int, default=50, help="warm-L2 reps")
    args = ap.parse_args()

    if args.ops:
        ops = [o.strip() for o in args.ops.split(",") if o.strip()]
        bad = [o for o in ops if o not in KNOWN_OPS]
        if bad:
            ap.error(f"unknown op(s): {bad}")
    else:
        ops = default_ops(args.dtype)

    K, dt = args.K, args.dtype
    results = Path(args.out_root)
    sub = "seqlen_sweep" if args.sweep == "seqlen" else "bs_scaling"
    (results / sub).mkdir(parents=True, exist_ok=True)
    out_path = results / sub / f"results_K{K}_{dt}.jsonl"

    if args.sweep == "seqlen":
        cells = [(K, dt, N, 1) for N in valid_N_for_K(K)]
    else:
        cells = [(K, dt, N, BS) for N in [n for n in N_SEQ if n > 2 * K]
                 for BS in BS_GRID]
    print(f"# nsys sweep batch: sweep={args.sweep} K={K} dt={dt} "
          f"cells={len(cells)} ops={ops} reps_cold={args.reps} "
          f"reps_warm={args.reps_warm}", flush=True)
    run_batch(args.sweep, cells, ops, out_path, args.cfg, args.reps, args.reps_warm)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
