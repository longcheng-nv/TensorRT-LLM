# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op25 S1a — per-model / per-K breakdown of the qfracs screen + ship-table
recommendation. Reads results/screen_qfracs.jsonl (round-2, incl. wide/cap
arms)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
recs = [json.loads(l) for l in
        open(HERE / "results" / "screen_qfracs.jsonl")]
recs = [r for r in recs if "error" not in r]
arms = list(recs[0]["arms"].keys())


def fast(rows, a):
    return sum(r["arms"][a]["mode"] == "fast" for r in rows) / len(rows)


def fbp(rows, a):
    return sum(r["arms"][a]["fbp"] for r in rows) / len(rows)


def table(title, groups):
    print(f"\n== {title} ==")
    hdr = f"{'group':28s}" + "".join(f"{a:>11s}" for a in arms)
    print(hdr)
    for gname, rows in groups:
        if not rows:
            continue
        print(f"{gname:28s}" + "".join(f"{fast(rows, a):11.3f}" for a in arms)
              + f"   n={len(rows)}")


# ---- axis A op22rr per (scen, model) ----
g = defaultdict(list)
for r in recs:
    if r["axis"] == "op22rr":
        g[(r["scen"], r["model"])].append(r)
table("op22rr fast-rate per (scen, model)",
      [(f"{s}/{m}", g[(s, m)]) for s in ("best", "worst", "real")
       for m in ("v4flash", "v4pro", "v32")])

# ---- axis C pro (K1024 real) split by h bucket ----
pro = [r for r in recs if r["axis"] == "pro"]
bux = [(f"pro h<{hi}" if lo == 0 else f"pro {lo}<=h<{hi}",
        [r for r in pro if lo <= r["h_true"] < hi])
       for lo, hi in ((0, 0.5), (0.5, 0.65), (0.65, 0.75), (0.75, 0.85),
                      (0.85, 0.95), (0.95, 1.01))]
table("pro real (K1024) fast-rate per h bucket", bux)
table("pro real pooled", [("pro ALL", pro)])

# ---- axis B op24 per model x hr ----
g24 = defaultdict(list)
for r in recs:
    if r["axis"] == "op24":
        s = r["scen"]
        hr = ("samp" if "hrsamp" in s or "_st4" in s else
              s.split("hr")[-1].split("_")[0] if "hr" in s else "?")
        g24[(r["model"], hr)].append(r)
table("op24 fast-rate per (model, target_hr)",
      [(f"{m}/hr={hr}", g24[(m, hr)]) for m in ("v4flash", "v4pro", "v32")
       for hr in ("0.05", "0.3", "0.55", "0.75", "0.9", "samp")])

# ---- regression cells: candidate loses fast where base3 was fast ----
if len(sys.argv) > 1:
    cand = sys.argv[1]
    print(f"\n== cells where {cand} regresses vs base3 (fast->miss) ==")
    for r in recs:
        b, c = r["arms"]["base3"], r["arms"][cand]
        if b["mode"] == "fast" and c["mode"] != "fast" \
                and r["axis"] != "pro":
            print(f"  {r['axis']} {r['scen']} {r['model']} N{r['N']} "
                  f"h={r['h_true']} -> {c['mode']}")
    n_reg = sum(1 for r in pro
                if r["arms"]["base3"]["mode"] == "fast"
                and r["arms"][cand]["mode"] != "fast")
    print(f"  pro axis fast->miss count: {n_reg}/{len(pro)}")
