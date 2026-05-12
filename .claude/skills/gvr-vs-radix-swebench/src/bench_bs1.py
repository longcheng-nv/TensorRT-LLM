#!/usr/bin/env python3
"""BS=1 GVR-vs-Radix Top-K benchmark on SWE-Bench-64K (production op).

This script runs through the 9-layer SWE-Bench-64K decode logits dataset and
goes through the production op `torch.ops.trtllm.indexer_topk_decode`.

For each layer L in {0,1,20,21,22,40,41,42,60} and each sampled row r:

  * N_valid(r) = mc - (tr - 1 - r)              # mc=70690, tr=2025
  * logits[r, N_valid:] is masked to -inf       # mask the invalid tail
  * preIdx = topk(prev_row, K, sorted=True)     # forces preIdx[0] = argmax
  * Production op call:
      torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, 1, K)
        -> Radix kernel
      torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, 1, K,
                                           preIdx, scratch)
        -> Scheme X dispatcher -> heuristicTopKMultiRowKernel[Dtype]
           (kernel internally applies preIdxOffset = +1, V3.2 decode)

Two sub-commands:

  profile       Run the GPU workload under nsys profile --trace=cuda,nvtx.
  parse         Parse the nsys-stats nvtx_gpu_proj_trace CSV into per-row CSV
                + per-(layer, variant) summary CSV.

The summary CSV layout:

    layer,variant,n_rows,median_us,p50_us,p90_us,p99_us

The `layer == ALL` row gives the cross-layer pooled median.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Backend loaders. Radix always lives in TRT-LLM (no standalone Radix in this
# repo). GVR can be either:
#   * production  : torch.ops.trtllm.indexer_topk_decode (preIdxOffset=+1
#                   baked into heuristicTopKMultiRowKernel{,Dtype})
#   * standalone  : heuristic_topk_cuda from auto_optimization_v1/topk_cuda.py
#                   (local JIT; kernel uses preIdxOffset=0, so the bench
#                   shifts preIdx by +1 in Python before the call -- net
#                   semantics match TRT-LLM's internal +1 offset).
# ---------------------------------------------------------------------------
_TRTLLM_REPO = os.environ.get(
    "TRTLLM_REPO", "/home/scratch.loncheng_gpu/workspace/perf/TensorRT-LLM"
)
_GVR_STANDALONE_REPO = os.environ.get(
    "GVR_STANDALONE_REPO", "/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1"
)
_LIBTH_COMMON = os.path.join(_TRTLLM_REPO, "tensorrt_llm/libs/libth_common.so")

_STANDALONE_GVR_FN = None


def _load_trtllm_op():
    """Load libth_common.so so torch.ops.trtllm.indexer_topk_decode resolves.

    Required for Radix (always) and for production-backend GVR.
    """
    if not os.path.isfile(_LIBTH_COMMON):
        raise ImportError(
            f"libth_common.so not found at {_LIBTH_COMMON}. "
            "Set $TRTLLM_REPO to a built TensorRT-LLM checkout."
        )
    torch.ops.load_library(_LIBTH_COMMON)
    if not hasattr(torch.ops.trtllm, "indexer_topk_decode"):
        raise ImportError(
            "torch.ops.trtllm.indexer_topk_decode missing after loading "
            f"{_LIBTH_COMMON} -- rebuild TensorRT-LLM with the Scheme X dispatcher."
        )


def _load_standalone_gvr():
    """Import heuristic_topk_cuda from auto_optimization_v1/topk_cuda.py.

    Returns the callable. First call triggers JIT compile of cuda_ext/.
    """
    global _STANDALONE_GVR_FN
    if _STANDALONE_GVR_FN is not None:
        return _STANDALONE_GVR_FN
    path = os.path.join(_GVR_STANDALONE_REPO, "topk_cuda.py")
    if not os.path.isfile(path):
        raise ImportError(f"topk_cuda.py not found at {path}. Set $GVR_STANDALONE_REPO.")
    spec = importlib.util.spec_from_file_location("topk_cuda_local", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, _GVR_STANDALONE_REPO)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    _STANDALONE_GVR_FN = mod.heuristic_topk_cuda
    return _STANDALONE_GVR_FN


L2_FLUSH_BYTES = 128 * 1024 * 1024  # > B200 L2 (126.5 MiB)
DEFAULT_DATA_DIR = os.environ.get(
    "SWE_BENCH_PATH",
    "/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/"
    "data_distri/deepseek-v3.2-logging/notebooks/SWE_Bench_64K_decode_logits",
)
DEFAULT_LAYERS = [0, 1, 20, 21, 22, 40, 41, 42, 60]


@dataclass(frozen=True)
class Variant:
    tag: str
    kind: str  # "radix" | "gvr"
    K: int
    dtype: torch.dtype


def make_default_variants() -> list[Variant]:
    """Apples-to-apples: Radix(fp32 K=2048) vs GVR(fp32 K=2048).

    Both kernels see the same K and the same dtype, so the wall delta is
    pure algorithmic. Extra K x dtype combos can be added on the CLI.
    """
    return [
        Variant("radix", "radix", 2048, torch.float32),
        Variant("gvr_K2048_fp32", "gvr", 2048, torch.float32),
    ]


ALL_VARIANTS = {
    "radix": Variant("radix", "radix", 2048, torch.float32),
    "gvr_K2048_fp32": Variant("gvr_K2048_fp32", "gvr", 2048, torch.float32),
    "gvr_K2048_bf16": Variant("gvr_K2048_bf16", "gvr", 2048, torch.bfloat16),
    "gvr_K2048_fp16": Variant("gvr_K2048_fp16", "gvr", 2048, torch.float16),
    "gvr_K1024_fp32": Variant("gvr_K1024_fp32", "gvr", 1024, torch.float32),
    "gvr_K1024_bf16": Variant("gvr_K1024_bf16", "gvr", 1024, torch.bfloat16),
    "gvr_K1024_fp16": Variant("gvr_K1024_fp16", "gvr", 1024, torch.float16),
    "gvr_K512_fp32": Variant("gvr_K512_fp32", "gvr", 512, torch.float32),
    "gvr_K512_bf16": Variant("gvr_K512_bf16", "gvr", 512, torch.bfloat16),
    "gvr_K512_fp16": Variant("gvr_K512_fp16", "gvr", 512, torch.float16),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_flush_buf():
    return torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")


def flush_l2(buf):
    buf.zero_()
    torch.cuda.synchronize()


def load_layer(data_dir: str, L: int) -> torch.Tensor:
    path = f"{data_dir}/Layer_{L}_pd.npy"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing layer file: {path}. Set --data_dir / $SWE_BENCH_PATH.")
    return torch.from_numpy(np.load(path)).to(device="cuda", dtype=torch.float32)


def _pad_align(t: torch.Tensor, align: int) -> torch.Tensor:
    """Right-pad columns to a multiple of `align` with -inf for ldg.128.

    Vector-load alignment: fp32 -> 4, bf16/fp16 -> 8. seq_lens still gates
    the valid range so padding does not affect correctness.
    """
    c = t.shape[1]
    r = c % align
    if r == 0:
        return t.contiguous()
    pad = torch.full((t.shape[0], align - r), -float("inf"), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1).contiguous()


def prep_inputs(lscore_dt: torch.Tensor, row_id: int, K: int):
    """Build per-cell input tensors for one BS=1 row.

    Returns (logits[1, S], preIdx[1, K], seq_lens[1], indices[1, K],
    scratch[K], N_valid). preIdx = exact Top-K of the previous row, sorted
    descending so preIdx[0] is the argmax.
    """
    tr, mc = lscore_dt.shape  # (2025, 70690)
    prev = row_id - 1

    # ---- previous row Top-K (preIdx) ----
    N_prev = mc - (tr - 1 - prev)
    lp = lscore_dt[prev].unsqueeze(0).clone()
    lp[:, N_prev:] = float("-inf")
    pre_idx = torch.topk(lp, K, largest=True, sorted=True).indices.to(torch.int32)

    # ---- current row logits, masked beyond N_valid ----
    N_valid = mc - (tr - 1 - row_id)
    lc = lscore_dt[row_id].unsqueeze(0).clone()
    lc[:, N_valid:] = float("-inf")

    align = 4 if lscore_dt.dtype is torch.float32 else 8
    lc = _pad_align(lc, align)

    seq_lens = torch.full((1,), N_valid, dtype=torch.int32, device="cuda")
    indices = torch.empty((1, K), dtype=torch.int32, device="cuda")
    scratch = torch.empty((K,), dtype=lscore_dt.dtype, device="cuda")
    return lc.contiguous(), pre_idx.contiguous(), seq_lens, indices, scratch, N_valid


def make_call(
    v: Variant, logits, preIdx, seq_lens, indices, scratch, gvr_backend: str = "production"
):
    """Build a closure that runs one variant once.

    For `gvr_backend == 'standalone'`, GVR routes to the local JIT
    `heuristic_topk_cuda(logits, preIdx_plus_1, K)`; preIdx_plus_1 is
    pre-shifted by +1 in Python so the standalone kernel (which uses
    preIdxOffset=0 internally) produces the same effective column shift as
    the TRT-LLM production kernel (which bakes +1). Radix always stays on
    TRT-LLM (no standalone Radix exists in this repo).
    """
    K = v.K
    if v.kind == "radix":

        def _call():
            torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, 1, K)

        return _call

    if gvr_backend == "production":

        def _call():
            torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices, 1, K, preIdx, scratch)

        return _call

    # standalone backend: BS=1, single-row Python loop. The bench preserves
    # +1 semantics by shifting preIdx in Python once per cell.
    gvr_fn = _load_standalone_gvr()
    # preIdx is already the prev-row Top-K; shift by +1 to mimic TRT-LLM.
    preIdx_p1 = (preIdx + 1).contiguous()
    # Clamp to N_valid - 1 just in case (shouldn't trigger for SWE-Bench:
    # prev_row Top-K indices are in [0, N_prev-1] = [0, N_curr-2], +1 stays
    # within [1, N_curr-1]). The clamp is cheap insurance against pathological
    # callers.
    N_valid = int(seq_lens[0].item())
    preIdx_p1.clamp_(max=N_valid - 1)

    def _call():
        # heuristic_topk_cuda returns indices; we don't need to write into
        # the supplied `indices` tensor for timing purposes.
        gvr_fn(logits, preIdx_p1, K)

    return _call


def launch_tagged(fn, flush_buf, tag):
    flush_l2(flush_buf)
    torch.cuda.nvtx.range_push(tag)
    fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def launch_untagged(fn, flush_buf):
    flush_l2(flush_buf)
    fn()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------
def run_profile(args):
    variants = [ALL_VARIANTS[t] for t in args.variants.split(",") if t.strip()]
    row_ids = list(range(args.row_start, args.row_end + 1, args.row_stride))

    # Backend setup. Radix always needs TRT-LLM; production-GVR also needs it.
    needs_trtllm = any(v.kind == "radix" for v in variants) or (
        args.gvr_backend == "production" and any(v.kind == "gvr" for v in variants)
    )
    if needs_trtllm:
        _load_trtllm_op()
    if args.gvr_backend == "standalone":
        if any(v.kind == "gvr" for v in variants):
            _load_standalone_gvr()  # eager-load to fail fast
        print(
            "[bench_bs1] GVR backend = standalone "
            "(local JIT; bench shifts preIdx by +1 in Python to match "
            "TRT-LLM's internal preIdxOffset=+1)",
            flush=True,
        )

    cells = []
    flush_buf = make_flush_buf()
    t0 = time.time()

    for L in args.layers:
        print(f"\n=== Layer {L} ({len(row_ids)} rows x {len(variants)} variants) ===", flush=True)
        lscore_fp32 = load_layer(args.data_dir, L)
        lscore_by_dt = {
            dt: (lscore_fp32 if dt is torch.float32 else lscore_fp32.to(dt))
            for dt in {v.dtype for v in variants}
        }

        for row_id in row_ids:
            try:
                inputs = {}
                N_valid = None
                for v in variants:
                    lc, pi, sl, idx, sc, N = prep_inputs(lscore_by_dt[v.dtype], row_id, v.K)
                    inputs[v.tag] = (lc, pi, sl, idx, sc)
                    N_valid = N
            except Exception as ex:
                print(f"  [skip] L{L} row={row_id}: {ex}", flush=True)
                continue

            fn_for = {
                v.tag: make_call(v, *inputs[v.tag], gvr_backend=args.gvr_backend) for v in variants
            }

            for _ in range(args.warmup):
                for v in variants:
                    launch_untagged(fn_for[v.tag], flush_buf)
            for r in range(args.repeats):
                for v in variants:
                    tag = f"L{L}/row{row_id}/{v.tag}/rep{r}"
                    launch_tagged(fn_for[v.tag], flush_buf, tag)

            cells.append({"layer": L, "row_id": row_id, "N": N_valid})
            del inputs, fn_for

        del lscore_fp32, lscore_by_dt
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nProfile pass done: {len(cells)} cells in {elapsed:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "variants": [v.tag for v in variants],
                "row_start": args.row_start,
                "row_end": args.row_end,
                "row_stride": args.row_stride,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "layers": args.layers,
                "cells": cells,
                "harness": "production_op"
                if args.gvr_backend == "production"
                else "production_radix_standalone_gvr",
                "gvr_backend": args.gvr_backend,
                "bench": "bs1",
            },
            f,
        )
    print(f"Index -> {args.output}", flush=True)


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def parse_nvtx_csv(csv_path: str):
    per_tag: dict = {}
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            nvtx = (row.get("Name", "") or row.get("NVTX Range", "")).lstrip(":")
            if not nvtx or "/" not in nvtx:
                continue
            parts = nvtx.split("/")
            if len(parts) != 4:
                continue
            try:
                L = int(parts[0].replace("L", ""))
                rw = int(parts[1].replace("row", ""))
                variant = parts[2]
            except ValueError:
                continue
            dur = (
                row.get("Projected Duration (ns)")
                or row.get("Duration (ns)")
                or row.get("Duration", 0)
            )
            try:
                dur_ns = float(dur)
            except (TypeError, ValueError):
                continue
            per_tag.setdefault((L, rw, variant), []).append(dur_ns)
    if not per_tag:
        raise RuntimeError(
            f"No NVTX rows of the expected 'L*/row*/variant/rep*' shape "
            f"found in {csv_path}. Did nsys-stats produce the projection trace?"
        )
    return per_tag


def run_parse(args):
    per_tag = parse_nvtx_csv(args.nsys_csv)
    with open(args.index_json) as f:
        idx = json.load(f)
    variants = idx["variants"]

    raw = []
    for cell in idx["cells"]:
        L, rw, N = cell["layer"], cell["row_id"], cell["N"]
        for v in variants:
            samples_ns = per_tag.get((L, rw, v), [])
            if not samples_ns:
                continue
            arr = np.array(samples_ns, dtype=np.float64) / 1000.0  # ns -> us
            raw.append(
                {
                    "layer": L,
                    "row_id": rw,
                    "N": N,
                    "variant": v,
                    "n_samples": len(arr),
                    "median_us": float(np.median(arr)),
                    "min_us": float(np.min(arr)),
                    "p50_us": float(np.percentile(arr, 50)),
                    "p90_us": float(np.percentile(arr, 90)),
                    "p99_us": float(np.percentile(arr, 99)),
                }
            )

    os.makedirs(os.path.dirname(args.output_raw) or ".", exist_ok=True)
    with open(args.output_raw, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)
    print(f"Raw CSV: {args.output_raw}  ({len(raw)} rows)")

    summary_rows = []
    L_set = sorted({r["layer"] for r in raw})
    for L in L_set:
        for v in variants:
            cells = [r for r in raw if r["layer"] == L and r["variant"] == v]
            if not cells:
                continue
            meds = np.array([c["median_us"] for c in cells])
            summary_rows.append(
                {
                    "layer": L,
                    "variant": v,
                    "n_rows": len(cells),
                    "median_us": float(np.median(meds)),
                    "p50_us": float(np.percentile(meds, 50)),
                    "p90_us": float(np.percentile(meds, 90)),
                    "p99_us": float(np.percentile(meds, 99)),
                }
            )
    for v in variants:
        cells = [r for r in raw if r["variant"] == v]
        if not cells:
            continue
        meds = np.array([c["median_us"] for c in cells])
        summary_rows.append(
            {
                "layer": "ALL",
                "variant": v,
                "n_rows": len(cells),
                "median_us": float(np.median(meds)),
                "p50_us": float(np.percentile(meds, 50)),
                "p90_us": float(np.percentile(meds, 90)),
                "p99_us": float(np.percentile(meds, 99)),
            }
        )

    with open(args.output_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary CSV: {args.output_summary}  ({len(summary_rows)} rows)")


def main():
    p = argparse.ArgumentParser(
        description="BS=1 GVR-vs-Radix Top-K on SWE-Bench-64K (production op)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("profile")
    pp.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    pp.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    pp.add_argument("--row_start", type=int, default=1)
    pp.add_argument("--row_end", type=int, default=2024)
    pp.add_argument("--row_stride", type=int, default=10)
    pp.add_argument("--warmup", type=int, default=3)
    pp.add_argument("--repeats", type=int, default=5)
    pp.add_argument(
        "--variants",
        default=",".join(v.tag for v in make_default_variants()),
        help="Comma-separated variant tags. Defaults to 'radix,gvr_K2048_fp32' (apples-to-apples).",
    )
    pp.add_argument(
        "--gvr_backend",
        choices=("production", "standalone"),
        default="production",
        help="Which GVR implementation to time. 'production' = "
        "torch.ops.trtllm.indexer_topk_decode (preIdxOffset=+1 "
        "baked into the kernel). 'standalone' = local JIT "
        "from auto_optimization_v1/topk_cuda.py "
        "(preIdxOffset=0 in the kernel; bench shifts preIdx "
        "by +1 in Python so net semantics match production). "
        "Radix always stays on TRT-LLM (no standalone Radix "
        "in this repo).",
    )
    pp.add_argument("--output", required=True)
    pp.set_defaults(func=run_profile)

    pr = sub.add_parser("parse")
    pr.add_argument("--nsys_csv", required=True)
    pr.add_argument("--index_json", required=True)
    pr.add_argument("--output_raw", required=True)
    pr.add_argument("--output_summary", required=True)
    pr.set_defaults(func=run_parse)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
