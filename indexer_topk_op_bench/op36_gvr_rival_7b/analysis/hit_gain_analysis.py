#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op36: relationship between preIdx hit rate and GVR gain vs sglang_v2.

hit varies only per capture batch (model, isl) and is jointly confounded with
N (and K across models), so this script separates the two:
  1. per-batch table: N, hit, gvr_pr/sglang gm (all-BS and valley BS 32-128)
  2. within-model residual analysis: fit log(pr/sgl) ~ a + b*log(N) per model
     (LSQ over batches), then correlate the residual with hit — the N-trend
     is removed, what remains is the hit-linked (plus data-shape) effect.
Controlled-experiment context (fixed shape, hit swept): op24 RESULTS.md and
op30 REPORT — cited in the conclusions, not recomputed here.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

_OP36 = Path(__file__).resolve().parent.parent
RES = _OP36 / "results" / "b_screen"


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


cells, meta = defaultdict(dict), {}
for l in (RES / "results.jsonl").read_text().splitlines():
    r = json.loads(l)
    if r.get("dtype") != "fp32":
        continue
    u = (r.get("us_span") or r.get("us")) \
        if r["op"] in ("sglang_v2", "sgl_bx") else r.get("us")
    k = (r["model"], r["isl"], r["BS"])
    if u:
        cells[k][r["op"]] = u
    meta[(r["model"], r["isl"])] = (r["N"], r["hit"], r["K"])

batches = []
for (m, isl), (n, hit, K) in sorted(meta.items(), key=lambda t: (t[0][0], t[1][0])):
    sub = [o for k, o in cells.items() if (k[0], k[1]) == (m, isl)
           and all(o.get(x) for x in ("gvr_pr", "sglang_v2"))]
    val = [o for k, o in cells.items() if (k[0], k[1]) == (m, isl)
           and 32 <= k[2] <= 128
           and all(o.get(x) for x in ("gvr_pr", "sglang_v2"))]
    batches.append(dict(m=m, isl=isl, N=n, hit=hit, K=K,
                        g_all=gm([o["sglang_v2"] / o["gvr_pr"] for o in sub]),
                        g_val=gm([o["sglang_v2"] / o["gvr_pr"] for o in val])))

print(f"{'batch':14s} {'K':>5s} {'N':>8s} {'hit':>6s} {'pr/sgl all':>10s} "
      f"{'valley':>7s}")
for b in batches:
    print(f"{b['m'] + '/' + b['isl']:14s} {b['K']:5d} {b['N']:8d} "
          f"{b['hit']:6.3f} {b['g_all']:10.3f} {b['g_val']:7.3f}")

# within-model N-detrended residual vs hit
print("\n== within-model: residual of log(pr/sgl) after removing the "
      "log(N) trend, vs hit ==")
for m in ("flash", "pro", "v32"):
    bs = [b for b in batches if b["m"] == m]
    for key in ("g_all", "g_val"):
        xs = [math.log(b["N"]) for b in bs]
        ys = [math.log(b[key]) for b in bs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
            sum((x - mx) ** 2 for x in xs)
        resid = [(b["hit"], y - my - beta * (x - mx))
                 for b, x, y in zip(bs, xs, ys)]
        hs = [h for h, _ in resid]
        rs = [r for _, r in resid]
        mh, mr = sum(hs) / n, sum(rs) / n
        num = sum((h - mh) * (r - mr) for h, r in resid)
        den = math.sqrt(sum((h - mh) ** 2 for h in hs) *
                        sum((r - mr) ** 2 for r in rs))
        corr = num / den if den else float("nan")
        # gain per +0.1 hit (slope of resid on hit)
        slope = num / sum((h - mh) ** 2 for h in hs)
        print(f"  {m:6s} {key:5s}: N-slope beta={beta:+.3f}  "
              f"corr(resid, hit)={corr:+.2f}  "
              f"effect per +0.1 hit = {math.exp(slope * 0.1) - 1:+.1%}  (n={n})")
