#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parse the §9 rival nsys-reps into one merged results.jsonl.

Canonical `us` = cold-L2 kernel-SUM within the NVTX range (nvtx_kern_sum, evict
kernel filtered) — comparable to every prior report number. `us_span` = the
projected NVTX GPU range (nvtx_gpu_proj_sum): honest wall-clock for the
sglang_v2 2-kernel PDL-overlap path where the sum double-counts.

Usage: python3 parse_rival.py [<results_dir>]   default ./results
"""
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]            # indexer_topk_op_bench/ (rival_harness is 2 levels down)
sys.path.insert(0, str(_BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402  (nvtx_kern_sum, evict-filtered)

KEEP = ("family", "sweep", "scenario", "model", "op", "K", "dtype", "N", "BS",
        "cr", "hit", "isl", "data_src", "exact", "r0_arm", "ms_path")


def parse_rep_span(rep):
    out = subprocess.run(
        ["nsys", "stats", "--report", "nvtx_gpu_proj_sum", "--format", "csv",
         "--force-export=true", str(rep)],
        capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
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
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "results"
    repdir = root / "nsys_reps"
    merged = []
    for batch in sorted(root.glob("*.jsonl")):
        if batch.name == "results.jsonl":
            continue
        tag = batch.stem
        # The nsys-rep is named by drive_rival_shard.sh's tag, which orders the
        # fields differently from sweep_rival's jsonl stem. Reconstruct the rep
        # name from the first record's fields (matches the driver's convention).
        first = next((json.loads(l) for l in batch.read_text().splitlines() if l.strip()), None)
        if (repdir / f"{tag}.nsys-rep").exists():
            reptag = tag                      # backfill driver names rep == jsonl stem
        elif first and first.get("family") == "synth":
            reptag = f"synth_{first['sweep']}_{first['scenario']}_K{first['K']}_{first['dtype']}"
        elif first and first.get("family") == "real":
            reptag = f"real_{first['sweep']}_{first['model']}_{first['dtype']}"
        else:
            reptag = tag
        rep = repdir / f"{reptag}.nsys-rep"
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
                uc = kern.get(rec.get("range_cold"))
                uw = kern.get(rec.get("range_warm"))
                if uc is None and uw is None:
                    out["error"] = "no_nvtx_range_in_rep"
                else:
                    out["us"] = uc if uc is not None else uw
                    if uc is not None:
                        out["us_cold"] = uc
                    if uw is not None:
                        out["us_warm"] = uw
                    sc = span.get(rec.get("range_cold"))
                    if sc is not None:
                        out["us_span"] = sc          # projected NVTX range (honest for overlapped kernels)
                    n_ok += 1
            merged.append(out)
        print(f"  {tag}: rep={'yes' if rep.exists() else 'MISSING'} "
              f"ranges={len(kern)} cells_ok={n_ok}")
    dst = root / "results.jsonl"
    with open(dst, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {dst} ({len(merged)} recs)")


if __name__ == "__main__":
    main()
