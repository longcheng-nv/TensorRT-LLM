# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op15 SMEM-resident vs baseline rank-scatter — nsys pure-kernel cold-L2 A/B.

Reuses the EXACT report methodology (measure_cell + _EVICT + get_bundle(seed=42)
from the shared harness): warm-L2 then cold-L2, sync inside each NVTX range, whole
loop under cudaProfilerApi. cold-L2 is canonical. Two ops per cell:
  - gvr_cutedsl_rs : baseline op#7 (harness/gvr_cutedsl_rs_op.py, gmem streaming P2/P3)
  - gvr_smem       : op15 (op15.../harness/gvr_smem_op.py, SMEM-resident P1-P3)
Identical inputs, identical launch config; only the logits read source differs.

Run UNDER nsys (see scripts/run_ab.sh). Per-cell metadata + NVTX range names go to
<out_root>/seqlen_sweep/results_K<K>_<dt>.jsonl; report/parse_nsys_full.py fills us.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path

import torch
import torch.cuda.profiler as prof

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent.parent
import sys
sys.path.insert(0, str(_BENCH / "harness"))
from sweep import _EVICT, DTYPES, get_bundle  # noqa: E402
from sweep_nsys import measure_cell, _load_done  # noqa: E402
from gvr_cutedsl_rs_op import gvr_cutedsl_rs as baseline_rs  # noqa: E402

# Load op15 wrapper as an isolated module (its own kernel src on sys.path).
_spec = importlib.util.spec_from_file_location(
    "gvr_smem_op", str(_BENCH / "op15_gvr_smem_resident" / "harness" / "gvr_smem_op.py"))
_smem_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smem_mod)
smem_rs = _smem_mod.gvr_cutedsl_rs
resident_n = _smem_mod._resident_n

DEV = "cuda"
OPS = {"gvr_cutedsl_rs": baseline_rs, "gvr_smem": smem_rs}
# Small-N envelope (fits SMEM) + boundary control N=65536 (fp32 falls back to gmem).
N_SMALL = [4096, 8192, 16384, 32768, 65536]


def build_call(op_fn, K, dtype, N, BS, cr, logits_row, preidx_row):
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()
    seq_div = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    pre = preidx_row.expand(BS, -1).contiguous()
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    op_fn(logits, pre, seq_div, K, compress_ratio=cr, out=out)  # warm compile
    return (lambda: op_fn(logits, pre, seq_div, K, compress_ratio=cr, out=out)), [logits, seq_div, pre, out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=50)
    ap.add_argument("--cfg", default="beta_moderate")
    args = ap.parse_args()

    K, dt = args.K, args.dtype
    dtype = DTYPES[dt]
    out_dir = Path(args.out_root) / "seqlen_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_K{K}_{dt}.jsonl"
    done = _load_done(out_path)
    cells = [N for N in N_SMALL if N > 2 * K]
    print(f"# AB nsys: K={K} dt={dt} cells={len(cells)} ops={list(OPS)} "
          f"reps_cold={args.reps} reps_warm={args.reps_warm}", flush=True)

    f = open(out_path, "a")
    prof.start()
    try:
        for i, N in enumerate(cells):
            b = get_bundle(K, dtype, N, cfg=args.cfg)
            logits_row, preidx_row, cr = b["logits"], b["preIdx"], b["cr"]
            rn = resident_n(dtype, N, K, cr)
            for op, fn in OPS.items():
                key = ("seqlen", op, K, dt, N, 1)
                if key in done:
                    continue
                base = f"{op}|{K}|{dt}|{N}|1"
                rec = {"sweep": "seqlen", "op": op, "K": K, "dtype": dt, "N": N,
                       "BS": 1, "cr": cr, "range_cold": f"c|{base}",
                       "range_warm": f"w|{base}", "reps_cold": args.reps,
                       "reps_warm": args.reps_warm,
                       "resident_n": (rn if op == "gvr_smem" else 0)}
                try:
                    call, keep = build_call(fn, K, dtype, N, 1, cr, logits_row, preidx_row)
                    measure_cell(call, base, args.reps, args.reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:140]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            print(f"[K={K} {dt}] {i+1}/{len(cells)} (N={N})", flush=True)
    finally:
        prof.stop()
    f.close()
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
