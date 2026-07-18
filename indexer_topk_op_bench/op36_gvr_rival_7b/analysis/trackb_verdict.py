#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op36 Track B verdict analysis (works on partial results — anchor checkpoint
mid-sweep, full verdict when all 25 batches land).

Axes (PLAN measurement discipline):
  - sglang_v2 and sgl_bx (both PDL multi-kernel): us_span (nvtx_gpu_proj cold).
  - gvr_pr (single kernel): us (nvtx_kern_sum cold) == us_span for 1 kernel.
  - Anchor drift vs results/baseline_real_bs.csv (07-16 b200-081 grid): per
    batch median/p95 of here/baseline for gvr_pr and sglang_v2. This sweep runs
    on umbriel-b200-093 (NOT 047 of iter2/3) — composites are same-node only.

Outputs:
  1. anchor drift table per batch
  2. epsilon table: sgl_bx / sglang_v2 per band (the guard must be a wash)
  3. per-cell pr vs bx vs sglang + best shape-keyed N-threshold dispatch per
     (model=K): route bx below threshold, pr above; composite gm vs sglang
     for pr-only / dispatch / oracle.

Usage: python3 trackb_verdict.py [results_dir] [--no-parse]
"""
import csv
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from math import exp, log
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OP36 = _HERE.parent
_RH = _OP36.parents[0] / "op26_r0_upstream_port_report" / "rival_harness"
sys.path.insert(0, str(_RH))

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else _OP36 / "results" / "b_screen"
NO_PARSE = "--no-parse" in sys.argv

ISLS = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]
BAND = {"4k": "4-16k", "8k": "4-16k", "16k": "4-16k",
        "32k": "32-128k", "64k": "32-128k", "128k": "32-128k",
        "256k": "256k-1M", "512k": "256k-1M", "1024k": "256k-1M"}


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return exp(sum(log(x) for x in xs) / len(xs)) if xs else float("nan")


def parse_results():
    """Run parse_rival on ROOT (writes ROOT/results.jsonl), then load."""
    if not NO_PARSE:
        subprocess.run([sys.executable, str(_RH / "parse_rival.py"), str(ROOT)],
                       check=True)
    rows = [json.loads(l) for l in (ROOT / "results.jsonl").read_text().splitlines()
            if l.strip()]
    return rows


def cell_us(r):
    """Canonical per-arm time: us_span for PDL multi-kernel arms, us else."""
    if r["op"] in ("sglang_v2", "sgl_bx"):
        return r.get("us_span") or r.get("us")
    return r.get("us")


def main():
    rows = parse_results()
    cells = defaultdict(dict)   # (model, isl, BS) -> {op: us}
    exact_bad = []
    for r in rows:
        if r.get("dtype") != "fp32":
            continue
        key = (r["model"], r["isl"], r["BS"])
        u = cell_us(r)
        if u:
            cells[key][r["op"]] = u
        if r.get("exact") is False:
            exact_bad.append((key, r["op"]))

    print(f"== loaded {len(rows)} rows -> {len(cells)} fp32 cells "
          f"(batches present: {len(set((m, i) for m, i, _ in cells))}/25)")
    if exact_bad:
        print(f"!! EXACT FAILURES: {exact_bad}")

    # ---- 1. anchor drift vs baseline CSV --------------------------------------
    base = {}
    with open(_OP36 / "results" / "baseline_real_bs.csv") as f:
        for r in csv.DictReader(f):
            if r["dtype"] != "fp32":
                continue
            base[(r["model"], r["isl"], int(r["BS"]))] = r
    print("\n== 1. anchor drift (this-node / b200-081 baseline), per batch ==")
    print(f"{'batch':22s} {'pr med':>7s} {'pr p95':>7s} {'sgl med':>8s} {'sgl p95':>8s}")
    for m in ("flash", "pro", "v32"):
        for isl in ISLS:
            dpr, dsg = [], []
            for (mm, ii, bs), ops in cells.items():
                if (mm, ii) != (m, isl):
                    continue
                b = base.get((mm, ii, bs))
                if not b:
                    continue
                if ops.get("gvr_pr") and b.get("gvr_pr"):
                    dpr.append(ops["gvr_pr"] / float(b["gvr_pr"]))
                if ops.get("sglang_v2") and b.get("sglang_v2"):
                    dsg.append(ops["sglang_v2"] / float(b["sglang_v2"]))
            if dpr or dsg:
                def mp(d):
                    if not d:
                        return "   -", "   -"
                    d = sorted(d)
                    p95 = d[min(len(d) - 1, int(0.95 * len(d)))]
                    return f"{st.median(d):7.3f}", f"{p95:7.3f}"
                a, b_ = mp(dpr); c, d_ = mp(dsg)
                print(f"{m + '_' + isl:22s} {a} {b_} {c:>8s} {d_:>8s}")

    # ---- 2. epsilon: sgl_bx / sglang_v2 ---------------------------------------
    print("\n== 2. guard epsilon (sgl_bx / sglang_v2) by band ==")
    eps_band = defaultdict(list)
    eps_all = []
    for (m, isl, bs), ops in cells.items():
        if ops.get("sgl_bx") and ops.get("sglang_v2"):
            e = ops["sgl_bx"] / ops["sglang_v2"]
            eps_band[BAND[isl]].append(e)
            eps_all.append(((m, isl, bs), e))
    for band in ("4-16k", "32-128k", "256k-1M"):
        es = eps_band[band]
        if es:
            print(f"  {band:9s}: gm {gm(es):.3f}  med {st.median(es):.3f}  "
                  f"min {min(es):.3f}  max {max(es):.3f}  n={len(es)}")
    if eps_all:
        print(f"  ALL      : gm {gm([e for _, e in eps_all]):.3f}")
        worst = sorted(eps_all, key=lambda t: -t[1])[:5]
        print("  worst 5  :", [(f"{m}/{i}/BS{b}", round(e, 3))
                               for (m, i, b), e in worst])

    # ---- 3. dispatch: bx below N-threshold, pr above --------------------------
    print("\n== 3. shape-keyed dispatch (per model: bx @ N < T, pr @ N >= T) ==")
    per_model = defaultdict(list)   # model -> [(N, isl, BS, pr, bx, sgl)]
    for (m, isl, bs), ops in cells.items():
        if all(ops.get(o) for o in ("gvr_pr", "sgl_bx", "sglang_v2")):
            n = next((r["N"] for r in rows
                      if (r["model"], r["isl"], r["BS"]) == (m, isl, bs)), None)
            per_model[m].append((n, isl, bs, ops["gvr_pr"], ops["sgl_bx"],
                                 ops["sglang_v2"]))
    thresholds = [0, 8192, 16384, 32768, 65536, 131072, 1 << 40]
    all_disp, all_pr, all_oracle = [], [], []
    chosen = {}
    for m, lst in per_model.items():
        best_t, best_g = None, -1
        for t in thresholds:
            g = gm([(sgl / (bx if n < t else pr))
                    for n, _, _, pr, bx, sgl in lst])
            if g > best_g:
                best_t, best_g = t, g
        pr_g = gm([sgl / pr for n, _, _, pr, bx, sgl in lst])
        or_g = gm([sgl / min(pr, bx) for n, _, _, pr, bx, sgl in lst])
        chosen[m] = best_t
        all_pr += [sgl / pr for n, _, _, pr, bx, sgl in lst]
        all_disp += [(sgl / (bx if n < best_t else pr))
                     for n, _, _, pr, bx, sgl in lst]
        all_oracle += [sgl / min(pr, bx) for n, _, _, pr, bx, sgl in lst]
        print(f"  {m:6s}: pr-only {pr_g:.3f} -> dispatch@N<{best_t} {best_g:.3f} "
              f"(oracle {or_g:.3f}, n={len(lst)})")
    if all_disp:
        print(f"  COMPOSITE vs sglang: pr-only {gm(all_pr):.3f} -> "
              f"dispatch {gm(all_disp):.3f} (oracle {gm(all_oracle):.3f})")
        print(f"  thresholds: {chosen}")

    # ---- 3b. (N, BS)-keyed rules — pr's residual wins are the mid-BS valley
    # at large N (N>=65536, BS 32-256), not an N-band. Both keys are
    # inference-known shapes (NOT hit-rate — red line respected).
    print("\n== 3b. (N, BS)-keyed dispatch rules ==")
    full = [((m, isl, bs, n), o) for (m, isl, bs), o0 in cells.items()
            for n, o in [(next((r["N"] for r in rows
                                if (r["model"], r["isl"], r["BS"]) == (m, isl, bs)),
                          None), o0)]
            if all(o.get(x) for x in ("gvr_pr", "sgl_bx", "sglang_v2"))]
    rules = {
        "always_bx": lambda n, bs: False,
        "R1 N>=64k,BS32-128": lambda n, bs: n >= 65536 and 32 <= bs <= 128,
        "R2 N>=64k,BS32-256": lambda n, bs: n >= 65536 and 32 <= bs <= 256,
    }
    for name, rule in rules.items():
        vals = [o["sglang_v2"] / (o["gvr_pr"] if rule(n, bs) else o["sgl_bx"])
                for (m, isl, bs, n), o in full]
        reg = sum(1 for (m, isl, bs, n), o in full
                  if rule(n, bs) and o["gvr_pr"] > 1.02 * o["sgl_bx"])
        print(f"  {name:20s} gm {gm(vals):.3f}  pr-routed regressions {reg}")
    vals = [o["sglang_v2"] / min(o["gvr_pr"], o["sgl_bx"]) for _, o in full]
    print(f"  {'oracle':20s} gm {gm(vals):.3f}")

    # ---- 4. the 99-cell hole close-up -----------------------------------------
    print("\n== 4. ISL 4-16k hole (pr vs bx vs sglang) ==")
    hole = [(m, isl, bs, ops) for (m, isl, bs), ops in cells.items()
            if BAND[isl] == "4-16k"
            and all(ops.get(o) for o in ("gvr_pr", "sgl_bx", "sglang_v2"))]
    if hole:
        print(f"  pr/sgl gm {gm([o['sglang_v2'] / o['gvr_pr'] for *_, o in hole]):.3f}  "
              f"bx/sgl gm {gm([o['sglang_v2'] / o['sgl_bx'] for *_, o in hole]):.3f}  "
              f"n={len(hole)}")


if __name__ == "__main__":
    main()
