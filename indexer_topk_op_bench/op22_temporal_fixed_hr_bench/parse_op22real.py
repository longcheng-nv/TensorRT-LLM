#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 REAL-capture chapter — parse the per-(model,dtype) nsys batches into
one merged results.jsonl. Reuses report/parse_nsys_full.parse_rep
(NVTX->GPU projection, evict kernel filtered) with the per-rep kern cache
from parse_op22_cached.py (keyed on rep mtime+size).

Usage: python3 parse_op22real.py [<out_root>]  default ../results_b200_op22real
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep as _parse_rep_raw  # noqa: E402

KEEP = ("sweep", "op", "harness_op", "model", "K", "dtype", "N", "Npad",
        "BS", "cr", "layer", "s_last", "hit_rate")
OPT = ("cluster_size", "ms_path", "r0_arm", "vdiff", "recall", "n_neg")


def parse_rep_cached(rep):
    rep = Path(rep)
    if not rep.exists():
        return {}
    st = rep.stat()
    key = [int(st.st_mtime), st.st_size]
    cache = rep.with_suffix(rep.suffix + ".kern.json")
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                return c["kern"]
        except (json.JSONDecodeError, KeyError):
            pass
    kern = _parse_rep_raw(rep)
    if kern:
        cache.write_text(json.dumps({"key": key, "kern": kern}))
    return kern


def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE.parents[0] / "results_b200_op22real"
    repdir = out_root / "nsys_reps"
    merged = []
    for batch in sorted((out_root / "realcap_sweep").glob("results_*.jsonl")):
        md = batch.stem[len("results_"):]           # e.g. flash_fp32
        rep = repdir / f"realcap_{md}.nsys-rep"
        kern = parse_rep_cached(rep)
        n_ok = n_err = 0
        for line in batch.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            out = {k: rec[k] for k in KEEP if k in rec}
            out["batch"] = md
            for k in OPT:
                if k in rec:
                    out[k] = rec[k]
            if "error" in rec:
                out["error"] = rec["error"]
                n_err += 1
            else:
                us_cold = kern.get(rec.get("range_cold"))
                us_warm = kern.get(rec.get("range_warm"))
                if us_cold is None and us_warm is None:
                    out["error"] = "no_nvtx_range_in_rep"
                    n_err += 1
                else:
                    canon = us_cold if us_cold is not None else us_warm
                    out["us"] = canon               # canonical = cold-L2
                    if us_cold is not None:
                        out["us_cold"] = us_cold
                    if us_warm is not None:
                        out["us_warm"] = us_warm
                    n_ok += 1
            merged.append(out)
        print(f"  {md}: rep={'yes' if rep.exists() else 'MISSING'} "
              f"ranges={len(kern)} cells_ok={n_ok} err={n_err}")
    if merged:
        dst = out_root / "realcap_sweep" / "results.jsonl"
        with open(dst, "w") as f:
            for r in merged:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {dst} ({len(merged)} recs)")


if __name__ == "__main__":
    main()
