#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op22 — parse the nsys pure-kernel sweeps into merged results.jsonl per
(scenario, sweep). Reuses report/parse_nsys_full.parse_rep (nvtx_kern_sum,
evict kernel filtered) and preserves the op22 metadata fields (scenario,
hit_rate, layer, seed, ms_path, cluster_size).

Usage: python3 parse_op22.py [<out_root>]   default ../results_b200_op22
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

SUBS = [("seqlen", "seqlen_sweep"), ("bs", "bs_scaling"),
        ("bs_hugeN", "bs_hugeN")]
KEEP = ("sweep", "op", "K", "dtype", "N", "BS", "cr",
        "scenario", "data_src", "hit_rate", "layer", "seed")
OPT = ("cluster_size", "ms_path")


def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE.parents[0] / "results_b200_op22"
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
                n_ok = 0
                for line in batch.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    out = {k: rec[k] for k in KEEP if k in rec}
                    for k in OPT:
                        if k in rec:
                            out[k] = rec[k]
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
                            n_ok += 1
                    merged.append(out)
                print(f"  {scen_dir.name}/{sweep} {kd}: "
                      f"rep={'yes' if rep.exists() else 'MISSING'} "
                      f"ranges={len(kern)} cells_ok={n_ok}")
            if merged:
                dst = scen_dir / sub / "results.jsonl"
                with open(dst, "w") as f:
                    for r in merged:
                        f.write(json.dumps(r) + "\n")
                print(f"wrote {dst} ({len(merged)} recs)")


if __name__ == "__main__":
    main()
