# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Merge the §8 rival full-ISL BS backfill (2026-07-16, umbriel-b200-081) into
rival_long.csv.

- The backfill re-ran the 3 EXTERNAL arms (radix_cutedsl / sglang_v2 /
  flashinfer_topk) + the op26_r0auto ANCHOR over the FULL real ISL x BS grid.
- Existing real/bs rows of the 3 external arms (07-15 single-rung 128k run,
  b200-044) are REPLACED by the backfill rows so the whole real BS view is
  single-node consistent. op26 backfill rows are NOT merged — they exist only
  to gate cross-node comparability vs the GVR rows (07-16 refresh, b200-094).
- Drift gates (memory lesson: per-batch p95, not just aggregate median):
    anchor  : backfill op26 vs rival_long op26, per (model,dtype,isl) batch,
              p95(ratio in [x, 1/x] sense) <= 1.15
    rival128: backfill rivals vs old 044 rival rows at the 128k overlap.

Usage: python3 aggregate_backfill.py [<results.jsonl>]
"""
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
SRC = _args[0] if _args else "/tmp/gvrval1/rival_results_bf/results.jsonl"
CSV = os.path.join(HERE, "rival_long.csv")

COLS = ["family", "sweep", "scenario", "model", "op", "K", "dtype", "N", "BS",
        "isl", "cr", "hit", "us", "us_span", "exact"]
RIVALS = {"radix_cutedsl", "sglang_v2", "flashinfer_topk"}
ANCHOR = "op26_r0auto"
GATE = 1.15

# ---- load existing csv ------------------------------------------------------
old = []
with open(CSV) as f:
    hdr = f.readline().strip().split(",")
    assert hdr == COLS, hdr
    for line in f:
        old.append(dict(zip(COLS, line.rstrip("\n").split(","))))

# ---- load backfill (+ optional re-run overlays) ------------------------------
# A batch that failed the drift gate is re-run into rival_results_bf2/ (rm of
# the original artifacts is not permitted); its rows REPLACE the same
# (model,dtype,isl) batch from the primary run.
OVERLAYS = [p for p in ["/tmp/gvrval1/rival_results_bf2/results.jsonl"]
            if os.path.exists(p)]


def _load(path):
    out, nerr = [], 0
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or "us" not in r:
            nerr += 1
            continue
        out.append(r)
    return out, nerr


bf, n_err = _load(SRC)
for ov in OVERLAYS:
    ov_rows, ov_err = _load(ov)
    n_err += ov_err
    batches = {(r["model"], r["dtype"], r["isl"]) for r in ov_rows}
    bf = [r for r in bf if (r["model"], r["dtype"], r["isl"]) not in batches]
    bf += ov_rows
    print(f"overlay {ov}: {len(ov_rows)} rows replace batches {sorted(batches)}")
print(f"backfill: {len(bf)} rows kept, {n_err} omitted (error/no_us)")


def key(r):
    return (r.get("model", ""), r.get("isl", ""), r["dtype"], int(r["N"]), int(r["BS"]))


# ---- gate 1: anchor drift vs refresh GVR rows (094) -------------------------
old_op26 = {key(r): float(r["us"]) for r in old
            if r["family"] == "real" and r["sweep"] == "bs" and r["op"] == ANCHOR and r["us"]}
per_batch = defaultdict(list)
for r in bf:
    if r["op"] != ANCHOR:
        continue
    k = key(r)
    if k in old_op26:
        per_batch[(r["model"], r["dtype"], r["isl"])].append(float(r["us"]) / old_op26[k])
all_ratios, bad = [], []
for b, rs in sorted(per_batch.items()):
    rs2 = sorted(max(x, 1 / x) for x in rs)
    p95 = rs2[min(len(rs2) - 1, int(round(0.95 * (len(rs2) - 1))))]
    med = statistics.median(rs)
    all_ratios += rs
    flag = "  <-- FAIL" if p95 > GATE else ""
    if p95 > GATE:
        bad.append((b, med, p95))
    print(f"  anchor {'/'.join(b):24s} n={len(rs):3d} med={med:.3f} p95(sym)={p95:.3f}{flag}")
if all_ratios:
    rs2 = sorted(max(x, 1 / x) for x in all_ratios)
    print(f"ANCHOR DRIFT overall: n={len(all_ratios)} median={statistics.median(all_ratios):.3f} "
          f"p95(sym)={rs2[int(round(0.95 * (len(rs2) - 1)))]:.3f} (gate {GATE}) "
          f"failing_batches={len(bad)}")

# ---- gate 2: rival 128k overlap vs old 044 rows -----------------------------
old_riv = {(r["op"],) + key(r): float(r["us"]) for r in old
           if r["family"] == "real" and r["sweep"] == "bs" and r["op"] in RIVALS and r["us"]}
per_op = defaultdict(list)
for r in bf:
    if r["op"] not in RIVALS:
        continue
    k = (r["op"],) + key(r)
    if k in old_riv:
        per_op[r["op"]].append(float(r["us"]) / old_riv[k])
for op, rs in sorted(per_op.items()):
    rs2 = sorted(max(x, 1 / x) for x in rs)
    print(f"  rival128 {op:16s} n={len(rs):3d} med={statistics.median(rs):.3f} "
          f"p95(sym)={rs2[int(round(0.95 * (len(rs2) - 1)))]:.3f}")

if bad and "--force" not in sys.argv:
    print("\nDRIFT GATE FAILED — rival_long.csv NOT rewritten. Inspect the failing "
          "batches (external GPU contention?) and re-run those batches, or pass --force.")
    sys.exit(2)

# ---- merge ------------------------------------------------------------------
kept = [r for r in old if not (r["family"] == "real" and r["sweep"] == "bs"
                               and r["op"] in RIVALS)]
n_dropped = len(old) - len(kept)
new_rows = [{c: ("" if r.get(c) is None else r.get(c, "")) for c in COLS}
            for r in bf if r["op"] in RIVALS]
merged = kept + new_rows


def _sortkey(r):
    return (r["family"], r["sweep"], r.get("scenario", ""), r.get("model", ""),
            r["dtype"], int(r["K"]), int(r["N"]), int(r["BS"]),
            r.get("isl", ""), r["op"])


merged.sort(key=_sortkey)
with open(CSV, "w") as f:
    f.write(",".join(COLS) + "\n")
    for r in merged:
        f.write(",".join(str(r[c]) for c in COLS) + "\n")
print(f"\nrival_long.csv rewritten: {len(old)} -> {len(merged)} rows "
      f"(dropped {n_dropped} old 128k-only rival rows, added {len(new_rows)} full-ISL rival rows)")
ex = [r for r in new_rows if r.get("exact") in (True, "True", False, "False")]
n_true = sum(1 for r in ex if r["exact"] in (True, "True"))
print(f"backfill rival exactness: {n_true}/{len(ex)}")
