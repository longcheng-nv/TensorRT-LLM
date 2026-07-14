#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op34 — parse per-(model,ISL) nsys batches (tag dir) into merged results.jsonl.
Reuses report/parse_nsys_full.parse_rep (NVTX->GPU projection). Usage:
  python3 parse_op34.py <out_tag_dir> <nsys_reps_dir> <tag>
e.g. python3 parse_op34.py ../results/probe ../results/nsys_reps probe
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "report"))
from parse_nsys_full import parse_rep as _parse_rep_raw  # noqa: E402

KEEP = ("sweep", "arm", "model", "isl", "K", "N", "layer", "BS", "hit_rate",
        "vdiff", "recall", "n_neg")


def parse_cached(rep):
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
    tagdir = Path(sys.argv[1])
    repdir = Path(sys.argv[2])
    tag = sys.argv[3]
    merged = []
    for batch in sorted(tagdir.glob("results_*.jsonl")):
        md = batch.stem[len("results_"):]        # model_isl
        rep = repdir / f"{tag}_{md}.nsys-rep"
        kern = parse_cached(rep)
        nok = nerr = 0
        for line in batch.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            out = {k: rec[k] for k in KEEP if k in rec}
            if "error" in rec:
                out["error"] = rec["error"]
                nerr += 1
            else:
                uc = kern.get(rec.get("range_cold"))
                uw = kern.get(rec.get("range_warm"))
                if uc is None and uw is None:
                    out["error"] = "no_nvtx_range"
                    nerr += 1
                else:
                    out["us"] = uc if uc is not None else uw
                    if uc is not None:
                        out["us_cold"] = uc
                    if uw is not None:
                        out["us_warm"] = uw
                    nok += 1
            merged.append(out)
        print(f"  {md}: rep={'yes' if rep.exists() else 'MISS'} "
              f"ranges={len(kern)} ok={nok} err={nerr}")
    if merged:
        dst = tagdir / "results.jsonl"
        dst.write_text("\n".join(json.dumps(r) for r in merged) + "\n")
        print(f"wrote {dst} ({len(merged)} recs)")


if __name__ == "__main__":
    main()
