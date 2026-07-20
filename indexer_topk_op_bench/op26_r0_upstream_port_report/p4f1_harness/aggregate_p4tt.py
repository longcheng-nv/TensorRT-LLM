#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the p4tt gated-shape validation sweeps (chains A+B):
synth seqlen/bs best+worst x K{1024,2048} + real bs pro/v32, fp32.
Per-cell ratio = t(p4tt_off)/t(p4tt_on) cold-L2 (>1 = fast wins).
Writes p4tt_sweep.csv next to this script and prints group summaries."""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gvrp4tt_sw")

JSONL_TO_REP = {
    "synth_best_seqlen_K1024_fp32": "synth_seq_best_K1024",
    "synth_best_seqlen_K2048_fp32": "synth_seq_best_K2048",
    "synth_worst_seqlen_K1024_fp32": "synth_seq_worst_K1024",
    "synth_worst_seqlen_K2048_fp32": "synth_seq_worst_K2048",
    "synth_best_bs_K1024_fp32": "synth_bs_best_K1024",
    "synth_best_bs_K2048_fp32": "synth_bs_best_K2048",
    "synth_worst_bs_K1024_fp32": "synth_bs_worst_K1024",
    "synth_worst_bs_K2048_fp32": "synth_bs_worst_K2048",
    "real_pro_bs_fp32": "real_bs_pro",
    "real_v32_bs_fp32": "real_bs_v32",
}


def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


rows, inexact, errors = [], [], []
for stem, rep in JSONL_TO_REP.items():
    jf = ROOT / f"{stem}.jsonl"
    if not jf.exists():
        print(f"MISSING {jf}", file=sys.stderr)
        continue
    kern = parse_rep(ROOT / "nsys_reps" / f"{rep}.nsys-rep")
    cells = defaultdict(dict)
    meta = {}
    for line in jf.read_text().splitlines():
        r = json.loads(line)
        if r.get("error"):
            errors.append((stem, r.get("op"), r.get("N"), r.get("BS"), r["error"]))
            continue
        if r.get("exact") is False:
            inexact.append((stem, r["op"], r.get("isl") or r["N"], r["BS"]))
        key = (r.get("isl") or "", r["N"], r["BS"])
        us = kern.get(r.get("range_cold") or f"c|{r['op']}|{r['K']}|fp32|{r['N']}|{r['BS']}")
        if us:
            cells[key][r["op"]] = us
            meta[key] = r
    for key, d in sorted(cells.items()):
        if "p4tt_off" in d and "p4tt_on" in d:
            r = meta[key]
            rows.append(dict(batch=stem, family=r["family"],
                             scen=r.get("scenario", r.get("model", "")),
                             K=r["K"], isl=key[0], N=key[1], BS=key[2],
                             off_us=round(d["p4tt_off"], 3),
                             on_us=round(d["p4tt_on"], 3),
                             ratio=round(d["p4tt_off"] / d["p4tt_on"], 4)))

out = _HERE / "p4tt_sweep.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} paired cells -> {out}")
print(f"errors: {len(errors)}", *errors[:5], sep="\n  ")
print(f"inexact: {len(inexact)}", *inexact[:10], sep="\n  ")

print("\n== group geomeans (off/on, >1 = fast wins) ==")
groups = defaultdict(list)
for r in rows:
    groups[(r["family"], r["scen"], r["K"])].append(r["ratio"])
for k in sorted(groups):
    xs = groups[k]
    print(f"  {k[0]}/{k[1]}/K{k[2]}: gm {gm(xs):.4f}  min {min(xs):.3f}  "
          f"max {max(xs):.3f}  n {len(xs)}  <0.95: {sum(x < 0.95 for x in xs)}")

print("\n== by BS (all cells) ==")
bybs = defaultdict(list)
for r in rows:
    bybs[r["BS"]].append(r["ratio"])
for bs in sorted(bybs):
    xs = bybs[bs]
    print(f"  BS={bs:5d}: gm {gm(xs):.4f}  min {min(xs):.3f}  n {len(xs)}")

allr = [r["ratio"] for r in rows]
print(f"\nALL: gm {gm(allr):.4f}  min {min(allr):.3f}  max {max(allr):.3f}  "
      f"n {len(allr)}  cells<0.95: {sum(x < 0.95 for x in allr)}")
worst = sorted(rows, key=lambda r: r["ratio"])[:10]
print("worst 10:")
for r in worst:
    print(f"  {r['batch']} {r['isl'] or r['N']} BS{r['BS']}: {r['ratio']}")
