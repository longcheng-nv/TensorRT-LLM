# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanity: realcap arm/base cold ratios vs the op22 report's REAL-scenario
synthetic ratios at the nearest N (§2 bs sweep, node-local op22rr data +
backfill CSVs are NOT re-read here — we read the report's const D blob so
the comparison is against exactly what the report shows).

Real N per model -> nearest report N:
  flash 25154 -> 32768 | pro 14478 -> 16384 | v32 70690 -> 65536

Expect ROUGH agreement (same ordering, ratios within ~2x): real rows have
per-layer hit-rate spread and a different N, so this is a smell test, not
a gate.  Usage: python3 sanity_realcap_vs_rr.py [<out_root>]
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op22real"
NEAREST = {"flash": 32768, "pro": 16384, "v32": 65536}
KOF = {"flash": 512, "pro": 1024, "v32": 2048}


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# --- realcap layer-gm ratio per (model, arm, BS grouped) ------------------
rc = {}
for line in (OUT_ROOT / "realcap_sweep" / "results.jsonl").read_text() \
        .splitlines():
    r = json.loads(line)
    if "error" in r or r["dtype"] != "fp32" or "us_cold" not in r:
        continue
    rc.setdefault((r["model"], r["op"], r["BS"]), {})[r["layer"]] = \
        r["us_cold"]

# --- report D blob (synthetic §1-2 re-test, real scenario) ----------------
t = (HERE / "REPORT.html").read_text(encoding="utf-8")
m = re.search(r"const D=(\[.*?\]);", t, re.S)
D = json.loads(m.group(1))
syn = {}
for r in D:
    if r["s"] == "real" and r["w"] == "bs" and r["d"] == "fp32":
        syn[(r["K"], r["o"], r["N"], r["B"])] = r["c"]

print(f"{'model':6} {'arm':22} {'BS':>5} {'real':>7} {'synth':>7} {'Δ%':>7}")
for model, N in NEAREST.items():
    K = KOF[model]
    base_rc = {BS: gm(list(rc[(model, "gvr_cutedsl", BS)].values()))
               for BS in (1, 16, 256) if (model, "gvr_cutedsl", BS) in rc}
    for arm in ("op27_hls", "radix_cutedsl", "sglang_v2",
                "gvr_multicta_cutedsl", "op26_r0auto"):
        for BS in (1, 16, 256):
            k = (model, arm, BS)
            if k not in rc or BS not in base_rc or not base_rc[BS]:
                continue
            r_real = gm(list(rc[k].values())) / base_rc[BS]
            # synth ratio: op27_hls maps to itself in the report blob
            a = syn.get((K, arm, N, BS))
            b = syn.get((K, "gvr_cutedsl", N, BS))
            r_syn = (a / b) if (a and b) else None
            d = (f"{100 * (r_real / r_syn - 1):+6.1f}%"
                 if r_syn else "    n/a")
            print(f"{model:6} {arm:22} {BS:>5} {r_real:7.3f} "
                  f"{r_syn if r_syn is None else round(r_syn, 3)!s:>7} {d}")
