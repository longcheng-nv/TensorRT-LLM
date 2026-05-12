#!/usr/bin/env python3
"""BS-scaling GVR-vs-Radix Top-K benchmark on SWE-Bench-64K (production op).

Goes through `torch.ops.trtllm.indexer_topk_decode` for both Radix and
production-backend GVR.

For each layer L in {0,1,20,21,22,40,41,42,60} and each BS:

  * Pick row_id = lscore.shape[0] - 1 = 2024  (the last decode step)
    -> N_valid = 70690 - (2024 - 2024) = 70690  (full row, no mask)
  * preIdx = topk(prev_row=2023, K, sorted=True).indices.int()  # [1, K]
  * logits = lscore[2024:2025].expand(BS, -1).contiguous()   # [BS, S]
  * preIdx = preIdx.expand(BS, -1).contiguous()              # [BS, K]
  * seq_lens = full((BS,), N_valid, int32)
  * indices = empty((BS, K), int32)
  * scratch = empty((BS*K,), dtype)

Replicating the last row matches the user-requested workload: every CTA in
the batch sees the same full-N (70690-element) input. This stresses both
the dispatcher's per-row work and the GPU launch / occupancy ramp.
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

# Backend loaders -- see bench_bs1.py for the full design notes.
_TRTLLM_REPO = os.environ.get(
    "TRTLLM_REPO", "/home/scratch.loncheng_gpu/workspace/perf/TensorRT-LLM"
)
_GVR_STANDALONE_REPO = os.environ.get(
    "GVR_STANDALONE_REPO", "/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1"
)
_LIBTH_COMMON = os.path.join(_TRTLLM_REPO, "tensorrt_llm/libs/libth_common.so")
_STANDALONE_GVR_FN = None


def _load_trtllm_op():
    if not os.path.isfile(_LIBTH_COMMON):
        raise ImportError(f"libth_common.so not found at {_LIBTH_COMMON}. Set $TRTLLM_REPO.")
    torch.ops.load_library(_LIBTH_COMMON)
    if not hasattr(torch.ops.trtllm, "indexer_topk_decode"):
        raise ImportError(
            f"torch.ops.trtllm.indexer_topk_decode missing after loading {_LIBTH_COMMON}."
        )


def _load_standalone_gvr():
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


L2_FLUSH_BYTES = 128 * 1024 * 1024
DEFAULT_DATA_DIR = os.environ.get(
    "SWE_BENCH_PATH",
    "/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/"
    "data_distri/deepseek-v3.2-logging/notebooks/SWE_Bench_64K_decode_logits",
)
DEFAULT_LAYERS = [0, 1, 20, 21, 22, 40, 41, 42, 60]
DEFAULT_BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


@dataclass(frozen=True)
class Variant:
    tag: str
    kind: str
    K: int
    dtype: torch.dtype


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


def default_variant_tags() -> list[str]:
    return ["radix", "gvr_K2048_fp32"]


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


def _pad_align(t, align):
    c = t.shape[1]
    r = c % align
    if r == 0:
        return t.contiguous()
    pad = torch.full((t.shape[0], align - r), -float("inf"), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1).contiguous()


def prep_preidx(lscore_dt: torch.Tensor, last_row: int, K: int):
    tr, mc = lscore_dt.shape
    prev = last_row - 1
    N_prev = mc - (tr - 1 - prev)
    lp = lscore_dt[prev].unsqueeze(0).clone()
    lp[:, N_prev:] = float("-inf")
    pi = torch.topk(lp, K, largest=True, sorted=True).indices.to(torch.int32)
    return pi.contiguous(), mc - (tr - 1 - last_row)


def build_inputs(lscore_dt, last_row: int, BS: int, K: int):
    pi_1, N_valid = prep_preidx(lscore_dt, last_row, K)

    lc = lscore_dt[last_row].unsqueeze(0).clone()
    lc[:, N_valid:] = float("-inf")
    align = 4 if lscore_dt.dtype is torch.float32 else 8
    lc = _pad_align(lc, align)

    logits = lc.expand(BS, -1).contiguous()
    preIdx = pi_1.expand(BS, -1).contiguous()
    seq_lens = torch.full((BS,), N_valid, dtype=torch.int32, device="cuda")
    indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
    scratch = torch.empty((BS * K,), dtype=lscore_dt.dtype, device="cuda")
    return logits, preIdx, seq_lens, indices, scratch, N_valid


def make_call(
    v: Variant, logits, preIdx, seq_lens, indices, scratch, gvr_backend: str = "production"
):
    """Build a closure that runs one variant once.

    Standalone GVR is single-row only -- BS>1 is served by looping in Python
    across the BS rows. This loops INSIDE the timed region so the resulting
    wall captures the realistic "no batched kernel" cost. Compare with
    production GVR which fuses BS rows into one heuristicTopKMultiRowKernel
    launch.
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

    # standalone backend: shift preIdx by +1 once per cell so the standalone
    # kernel (preIdxOffset=0) produces the same effective +1 column shift as
    # the TRT-LLM production kernel.
    gvr_fn = _load_standalone_gvr()
    preIdx_p1 = (preIdx + 1).contiguous()
    N_valid = int(seq_lens[0].item())
    preIdx_p1.clamp_(max=N_valid - 1)
    BS = logits.shape[0]
    if BS == 1:

        def _call():
            gvr_fn(logits, preIdx_p1, K)
    else:
        # Python loop -- timed wall captures BS launches + BS sync points.
        # This is intentionally apples-to-oranges vs production multi-row
        # kernel; the report flags it.
        row_views_logits = [logits[i : i + 1] for i in range(BS)]
        row_views_pi = [preIdx_p1[i : i + 1] for i in range(BS)]

        def _call():
            for i in range(BS):
                gvr_fn(row_views_logits[i], row_views_pi[i], K)

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
def run_profile(args):
    variants = [ALL_VARIANTS[t] for t in args.variants.split(",") if t.strip()]
    bs_list = [int(b) for b in args.bs_list.split(",")]

    needs_trtllm = any(v.kind == "radix" for v in variants) or (
        args.gvr_backend == "production" and any(v.kind == "gvr" for v in variants)
    )
    if needs_trtllm:
        _load_trtllm_op()
    if args.gvr_backend == "standalone":
        if any(v.kind == "gvr" for v in variants):
            _load_standalone_gvr()
        print(
            "[bench_bs_scaling] GVR backend = standalone "
            "(single-row JIT; BS>1 loops in Python; bench shifts preIdx by "
            "+1 in Python to match TRT-LLM preIdxOffset=+1)",
            flush=True,
        )

    cells = []
    flush_buf = make_flush_buf()
    t0 = time.time()

    for L in args.layers:
        print(
            f"\n=== Layer {L} (BS sweep over {len(bs_list)} values x {len(variants)} variants) ===",
            flush=True,
        )
        lscore_fp32 = load_layer(args.data_dir, L)
        tr = lscore_fp32.shape[0]
        last_row = tr - 1  # 2024 for SWE-Bench-64K
        dt_used = {v.dtype for v in variants}
        lscore_by_dt = {
            dt: (lscore_fp32 if dt is torch.float32 else lscore_fp32.to(dt)) for dt in dt_used
        }

        for BS in bs_list:
            for v in variants:
                lscore_dt = lscore_by_dt[v.dtype]
                logits, preIdx, sl, idx, sc, N_valid = build_inputs(lscore_dt, last_row, BS, v.K)
                fn = make_call(v, logits, preIdx, sl, idx, sc, gvr_backend=args.gvr_backend)

                for _ in range(args.warmup):
                    launch_untagged(fn, flush_buf)
                for r in range(args.repeats):
                    tag = f"L{L}/bs{BS}/{v.tag}/rep{r}"
                    launch_tagged(fn, flush_buf, tag)

                cells.append(
                    {
                        "layer": L,
                        "row_id": last_row,
                        "bs": BS,
                        "variant": v.tag,
                        "N": N_valid,
                    }
                )
                del logits, preIdx, sl, idx, sc
                torch.cuda.empty_cache()

        del lscore_fp32, lscore_by_dt
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\nProfile pass done: {len(cells)} cells in {elapsed:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "variants": [v.tag for v in variants],
                "bs_list": bs_list,
                "layers": args.layers,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "cells": cells,
                "harness": "production_op"
                if args.gvr_backend == "production"
                else "production_radix_standalone_gvr",
                "gvr_backend": args.gvr_backend,
                "bench": "bs_scaling",
            },
            f,
        )
    print(f"Index -> {args.output}", flush=True)


def parse_nvtx_csv(csv_path):
    per_tag = {}
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
                bs = int(parts[1].replace("bs", ""))
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
            per_tag.setdefault((L, bs, variant), []).append(dur_ns)
    if not per_tag:
        raise RuntimeError(
            f"No NVTX rows of the expected 'L*/bs*/variant/rep*' shape found in {csv_path}."
        )
    return per_tag


def run_parse(args):
    per_tag = parse_nvtx_csv(args.nsys_csv)
    with open(args.index_json) as f:
        idx = json.load(f)
    variants = idx["variants"]
    bs_list = idx["bs_list"]

    raw = []
    for cell in idx["cells"]:
        L, bs, v = cell["layer"], cell["bs"], cell["variant"]
        samples_ns = per_tag.get((L, bs, v), [])
        if not samples_ns:
            continue
        arr = np.array(samples_ns, dtype=np.float64) / 1000.0
        raw.append(
            {
                "layer": L,
                "bs": bs,
                "variant": v,
                "N": cell["N"],
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

    summary = []
    for v in variants:
        for bs in bs_list:
            cells = [r for r in raw if r["variant"] == v and r["bs"] == bs]
            if not cells:
                continue
            meds = np.array([c["median_us"] for c in cells])
            summary.append(
                {
                    "variant": v,
                    "bs": bs,
                    "n_layers": len(cells),
                    "median_us": float(np.median(meds)),
                    "p50_us": float(np.percentile(meds, 50)),
                    "p90_us": float(np.percentile(meds, 90)),
                    "p99_us": float(np.percentile(meds, 99)),
                }
            )

    with open(args.output_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Summary CSV: {args.output_summary}  ({len(summary)} rows)")


def main():
    p = argparse.ArgumentParser(
        description="BS-scaling GVR-vs-Radix Top-K on SWE-Bench-64K (production op)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("profile")
    pp.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    pp.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    pp.add_argument("--bs_list", default=",".join(str(b) for b in DEFAULT_BS))
    pp.add_argument("--warmup", type=int, default=3)
    pp.add_argument("--repeats", type=int, default=5)
    pp.add_argument("--variants", default=",".join(default_variant_tags()))
    pp.add_argument(
        "--gvr_backend",
        choices=("production", "standalone"),
        default="production",
        help="See bench_bs1.py --help. Standalone backend loops "
        "across BS rows in Python (single-row JIT kernel); "
        "the resulting wall is not apples-to-apples vs the "
        "production multi-row kernel at BS>1.",
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
