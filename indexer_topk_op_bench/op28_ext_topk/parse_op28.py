#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op28 — parse the nsys sweeps into merged results.jsonl per (scenario,
sweep). Reuses report/parse_nsys_full.parse_rep (nvtx_kern_sum, evict kernel
filtered) for the canonical `us` (kernel-time SUM — comparable to every prior
report number), and ADDS `us_span` from nvtx_gpu_proj_sum (projected NVTX
range = first-kernel-start .. last-kernel-end): for the sglang_v2
persistent-cluster path (2 PDL-overlapped kernels) the sum double-counts the
overlap, span is the honest wall-clock. Single-kernel cells: span ~= sum.

Usage: python3 parse_op28.py [<out_root>]   default ../results_b200_op28
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

SUBS = [("seqlen", "seqlen_sweep"), ("bs", "bs_scaling"),
        ("bs_hugeN", "bs_hugeN")]
KEEP = ("sweep", "op", "K", "dtype", "N", "BS", "cr",
        "scenario", "data_src", "hit_rate", "layer", "seed")


def parse_rep_span(rep):
    """{range_name: span_us_per_call} via nvtx_gpu_proj_sum (projected NVTX)."""
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_gpu_proj_sum", "--format", "csv",
         "--force-export=true", str(rep)],
        capture_output=True, text=True).stdout
    rdr = csv.reader(io.StringIO(out))
    rows = list(rdr)
    hdr = next((i for i, r in enumerate(rows)
                if r and r[0] in ("Range", "NVTX Range", "Name")), None)
    if hdr is None:
        return {}
    cols = rows[hdr]
    try:
        i_inst = next(i for i, c in enumerate(cols) if "Instances" in c)
        i_tot = next(i for i, c in enumerate(cols) if "Total" in c)
    except StopIteration:
        return {}
    res = {}
    for r in rows[hdr + 1:]:
        if not r or "|" not in r[0]:
            continue
        rng = r[0].lstrip(":")
        try:
            ninst = int(r[i_inst]); total_ns = float(r[i_tot])
        except (ValueError, IndexError):
            continue
        if ninst:
            res[rng] = total_ns / ninst / 1e3
    return res


def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE.parents[0] / "results_b200_op28"
    for scen_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        repdir = scen_dir / "nsys_reps"
        for sweep, sub in SUBS:
            if not (scen_dir / sub).exists():
                continue
            merged = []
            for batch in sorted((scen_dir / sub).glob("results_K*.jsonl")):
                kd = batch.stem[len("results_K"):]      # e.g. 512_fp32
                rep = repdir / f"{sweep}_K{kd}.nsys-rep"
                kern = parse_rep(rep) if rep.exists() else {}
                span = parse_rep_span(rep) if rep.exists() else {}
                n_ok = 0
                for line in batch.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    out = {k: rec[k] for k in KEEP if k in rec}
                    if "error" in rec:
                        out["error"] = rec["error"]
                    else:
                        us_cold = kern.get(rec.get("range_cold"))
                        us_warm = kern.get(rec.get("range_warm"))
                        if us_cold is None and us_warm is None:
                            out["error"] = "no_nvtx_range_in_rep"
                        else:
                            canon = us_cold if us_cold is not None else us_warm
                            out["us"] = canon           # canonical = cold-L2
                            if us_cold is not None:
                                out["us_cold"] = us_cold
                            if us_warm is not None:
                                out["us_warm"] = us_warm
                            sp_c = span.get(rec.get("range_cold"))
                            sp_w = span.get(rec.get("range_warm"))
                            if sp_c is not None:
                                out["us_cold_span"] = sp_c
                            if sp_w is not None:
                                out["us_warm_span"] = sp_w
                            n_ok += 1
                    merged.append(out)
                print(f"  {scen_dir.name}/{sweep} {kd}: "
                      f"rep={'yes' if rep.exists() else 'MISSING'} "
                      f"ranges={len(kern)} span_ranges={len(span)} "
                      f"cells_ok={n_ok}")
            if merged:
                dst = scen_dir / sub / "results.jsonl"
                with open(dst, "w") as f:
                    for r in merged:
                        f.write(json.dumps(r) + "\n")
                print(f"wrote {dst} ({len(merged)} recs)")


if __name__ == "__main__":
    main()
