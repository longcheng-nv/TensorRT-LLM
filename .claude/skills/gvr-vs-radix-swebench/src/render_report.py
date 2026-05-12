#!/usr/bin/env python3
"""Render an English Markdown REPORT.md from the BS=1 / BS-scaling CSVs.

Consumes outputs of bench_bs1.py and bench_bs_scaling.py.

Inputs:
  --bs1_raw         per-(layer, row, variant) CSV
  --bs1_summary     per-(layer, variant) CSV
  --bs_scaling_raw  per-(layer, BS, variant) CSV
  --preidx_check    text file from verify_preidx_offset.py
  --output          REPORT.md path

Aggregations:
  BS=1:
    - per-layer Radix/GVR median us -> per-layer speedup
    - across 9 layers: min / max / mean speedup
  BS-scaling:
    - per-(layer, BS) Radix/GVR -> per-layer speedup
    - per BS: min / max / mean speedup over the 9 layers

The GVR variant compared against Radix is the first non-radix variant in
the summary CSV's `variant` column (default `gvr_K2048_fp32` -- the
apples-to-apples baseline).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import statistics
from collections import defaultdict


def _read_index(path: str | None) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _pivot_layer_variant(rows: list[dict]) -> dict[tuple[int, str], float]:
    """Return {(layer, variant): median_us} from a per-layer summary CSV."""
    out = {}
    for r in rows:
        if r["layer"] == "ALL":
            continue
        out[(int(r["layer"]), r["variant"])] = float(r["median_us"])
    return out


def _pivot_layer_bs_variant(rows: list[dict]) -> dict[tuple[int, int, str], float]:
    """Return {(layer, bs, variant): median_us} from raw per-cell CSV."""
    bucket: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for r in rows:
        key = (int(r["layer"]), int(r["bs"]), r["variant"])
        bucket[key].append(float(r["median_us"]))
    return {k: statistics.median(v) for k, v in bucket.items()}


def _gvr_variant(variants: list[str]) -> str | None:
    for v in variants:
        if v != "radix":
            return v
    return None


def _fmt_us(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


def _fmt_speedup(x: float | None) -> str:
    return f"{x:.2f}x" if x is not None else "n/a"


def render(args):
    bs1_summary = _read_csv(args.bs1_summary)
    bs1_pivot = _pivot_layer_variant(bs1_summary)
    bs1_variants = sorted({r["variant"] for r in bs1_summary})
    gvr_v = _gvr_variant(bs1_variants)
    if gvr_v is None:
        raise RuntimeError("No GVR variant in BS=1 summary CSV.")

    bs1_idx = _read_index(args.bs1_index)
    bs_idx = _read_index(args.bs_scaling_index)
    gvr_backend = bs1_idx.get("gvr_backend") or bs_idx.get("gvr_backend") or "production"
    if gvr_backend == "production":
        backend_line = (
            "`torch.ops.trtllm.indexer_topk_decode` "
            "(Scheme X v1.2 dispatcher, preIdxOffset=+1 baked in)"
        )
    else:
        backend_line = (
            "standalone JIT `heuristic_topk_cuda` from "
            "`auto_optimization_v1/topk_cuda.py` (kernel uses "
            "preIdxOffset=0; bench shifts preIdx by +1 in Python "
            "to match TRT-LLM production semantics). Radix still "
            "via `torch.ops.trtllm.indexer_topk_decode`."
        )

    layers = sorted({k[0] for k in bs1_pivot.keys()})

    # ---- BS=1 per-layer table + footer ----
    bs1_table_lines = [
        "| Layer | Radix us | GVR us (`" + gvr_v + "`) | Speedup (Radix / GVR) |",
        "|---:|---:|---:|---:|",
    ]
    speedups_bs1 = []
    for L in layers:
        rx = bs1_pivot.get((L, "radix"))
        gv = bs1_pivot.get((L, gvr_v))
        sp = (rx / gv) if (rx and gv) else None
        if sp is not None:
            speedups_bs1.append(sp)
        bs1_table_lines.append(f"| {L} | {_fmt_us(rx)} | {_fmt_us(gv)} | {_fmt_speedup(sp)} |")
    if speedups_bs1:
        s_min = min(speedups_bs1)
        s_max = max(speedups_bs1)
        s_mean = sum(speedups_bs1) / len(speedups_bs1)
    else:
        s_min = s_max = s_mean = None

    # ---- BS-scaling table ----
    bs_scaling_lines = []
    if args.bs_scaling_raw and os.path.isfile(args.bs_scaling_raw):
        bs_raw = _read_csv(args.bs_scaling_raw)
        bs_pivot = _pivot_layer_bs_variant(bs_raw)
        bs_set = sorted({k[1] for k in bs_pivot.keys()})
        layers_bs = sorted({k[0] for k in bs_pivot.keys()})
        bs_variants = sorted({k[2] for k in bs_pivot.keys()})
        gvr_v_bs = _gvr_variant(bs_variants)

        bs_scaling_lines.append(
            "| BS | Min speedup | Max speedup | Mean speedup | Median Radix us | Median GVR us |"
        )
        bs_scaling_lines.append("|---:|---:|---:|---:|---:|---:|")
        bs_min_mean_max: list[tuple[int, float, float, float, float, float]] = []
        for bs in bs_set:
            sps = []
            rx_vals = []
            gv_vals = []
            for L in layers_bs:
                rx = bs_pivot.get((L, bs, "radix"))
                gv = bs_pivot.get((L, bs, gvr_v_bs))
                if rx and gv:
                    sps.append(rx / gv)
                    rx_vals.append(rx)
                    gv_vals.append(gv)
            if not sps:
                continue
            s_min_bs = min(sps)
            s_max_bs = max(sps)
            s_mean_bs = sum(sps) / len(sps)
            rx_med = statistics.median(rx_vals)
            gv_med = statistics.median(gv_vals)
            bs_scaling_lines.append(
                f"| {bs} | {_fmt_speedup(s_min_bs)} | {_fmt_speedup(s_max_bs)} | "
                f"{_fmt_speedup(s_mean_bs)} | {_fmt_us(rx_med)} | {_fmt_us(gv_med)} |"
            )
            bs_min_mean_max.append((bs, s_min_bs, s_max_bs, s_mean_bs, rx_med, gv_med))
    else:
        bs_min_mean_max = []

    # ---- Preidx audit text ----
    preidx_text = ""
    if args.preidx_check and os.path.isfile(args.preidx_check):
        with open(args.preidx_check) as f:
            preidx_text = f.read().strip()

    timestamp = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    out_lines = [
        "# GVR Top-K vs Radix Top-K on SWE-Bench-64K -- Performance Report",
        "",
        f"_Generated: {timestamp}_",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        "| GPU | NVIDIA B200 (sm_100 / Blackwell) |",
        f"| GVR backend | `{gvr_backend}` -- {backend_line} |",
        "| Radix variant | `radix` (fp32, K=2048) |",
        f"| GVR variant   | `{gvr_v}` |",
        "| Dataset | SWE-Bench-64K decode logits, 9 layers x 2025 rows x ~70690 fp32 |",
        "| Layers profiled | 0, 1, 20, 21, 22, 40, 41, 42, 60 |",
        "| Timing | nsys `nvtx_gpu_proj_trace` (per-kernel projected GPU duration) |",
        "| Per-iter L2 flush | 128 MiB > B200 L2 (126.5 MiB) -- cold-cache |",
        "",
        "## preIdxOffset audit (V3.2 decode semantics)",
        "",
        "GVR's contract on decode: the kernel reads `preIdx[i]` (exact Top-K",
        "indices of the previous row) and internally shifts the column index",
        "by `+1` before scanning current-row logits. The benchmark provides",
        "the unshifted prev-row Top-K; the kernel adds `+1` itself.",
        "",
        "```",
        preidx_text or "(no audit log)",
        "```",
        "",
        "## BS = 1 (real variable-length rows, per-row valid-N masked)",
        "",
        "Each layer holds 2025 rows; for row `r` only the first",
        "`70690 - (2024 - r)` elements are valid -- the tail is masked to",
        "`-inf` before the kernel call. `preIdx` is the **exact** Top-K of",
        "the previous row (sorted descending so `preIdx[0]` is the argmax).",
        "Per-layer numbers are the median across all sampled rows x reps.",
        "",
        *bs1_table_lines,
        "",
        f"**Speedup across the 9 layers (BS=1):** min = {_fmt_speedup(s_min)}, "
        f"max = {_fmt_speedup(s_max)}, mean = {_fmt_speedup(s_mean)}.",
        "",
        "## BS-scaling (BS in 1..512, last row N=70690 replicated)",
        "",
        "For each layer the **last row** (`row_id = 2024`, `N_valid = 70690`,",
        "i.e. no per-row mask) is replicated across BS rows. preIdx is the",
        "Top-K of row 2023 (also broadcast). Per-layer numbers are the median",
        "across the repetitions; aggregation across the 9 layers is min /",
        "max / mean of the per-layer speedups.",
        "",
        *(bs_scaling_lines or ["(no BS-scaling data)"]),
        "",
        "## Interpretation",
        "",
    ]

    if s_mean and s_mean >= 1.0:
        out_lines.append(
            f"At BS=1, GVR is on average **{_fmt_speedup(s_mean)}** faster than "
            f"the Radix-based Top-K across the 9 SWE-Bench layers, with the "
            f"per-layer speedup ranging from {_fmt_speedup(s_min)} to "
            f"{_fmt_speedup(s_max)}."
        )
    elif s_mean:
        out_lines.append(
            f"At BS=1, GVR is on average **{_fmt_speedup(s_mean)}** vs Radix "
            f"(below 1x indicates a regression on this dataset). Per-layer "
            f"spread: {_fmt_speedup(s_min)} -> {_fmt_speedup(s_max)}."
        )

    if bs_min_mean_max:
        # Find the BS that gives the highest mean speedup.
        best_bs = max(bs_min_mean_max, key=lambda x: x[3])
        worst_bs = min(bs_min_mean_max, key=lambda x: x[3])
        out_lines.append(
            f"Across BS in 1..512, GVR's best mean speedup is "
            f"{_fmt_speedup(best_bs[3])} at BS={best_bs[0]}, and the worst "
            f"mean speedup is {_fmt_speedup(worst_bs[3])} at BS={worst_bs[0]}."
        )

    out_lines.append("")
    if gvr_backend == "standalone":
        out_lines.append(
            "**Standalone-backend caveat:** the standalone JIT kernel is "
            "single-row only. At BS>1 the bench loops in Python, so the "
            "BS-scaling row reflects `BS x (kernel launch + sync)` and is "
            "not apples-to-apples vs the production multi-row kernel "
            "(`heuristicTopKMultiRowKernel*`). BS=1 numbers ARE "
            "apples-to-apples on algorithm level; expect them to be ~20-34 % "
            "above production walls due to the launch surface difference "
            "(F006)."
        )
        out_lines.append("")
    out_lines.append(
        "Caveats: numbers measured under nsys with per-iteration L2 flush. "
        "Real inference under cudaGraph + warm L2 will show a slightly "
        "different floor (launch + sync overhead drops out under cudaGraph)."
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"REPORT.md -> {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs1_summary", required=True)
    p.add_argument("--bs1_raw", required=False, default=None)
    p.add_argument("--bs_scaling_raw", required=False, default=None)
    p.add_argument(
        "--bs1_index",
        required=False,
        default=None,
        help="bs1_index.json (used to read gvr_backend)",
    )
    p.add_argument(
        "--bs_scaling_index",
        required=False,
        default=None,
        help="bs_scaling_index.json (used to read gvr_backend)",
    )
    p.add_argument("--preidx_check", required=False, default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    render(args)


if __name__ == "__main__":
    main()
