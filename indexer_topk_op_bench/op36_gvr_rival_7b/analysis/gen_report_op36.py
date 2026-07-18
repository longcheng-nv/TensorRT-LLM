#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op36 campaign REPORT generator (idempotent last-writer of REPORT.html).

Recomputes every table from the committed results jsonls / baseline CSV:
  - baseline arithmetic (results/baseline_real_bs.csv, 07-16 b200-081 grid)
  - Track A0 screening (results/a0_screen)
  - Track B screening + verdict (results/b_screen, results/b_verdict)
  - Track A2 verdict (results/a2_verdict)
  - ship-table composites + pivot-gate ceiling arithmetic (b_screen grid)

House style: bilingual zh/en, CSS-only language toggle, NO <script>.
Usage: python3 gen_report_op36.py   (writes ../REPORT.html)
"""
import csv
import json
import statistics as st
from collections import defaultdict
from math import exp, log
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OP36 = _HERE.parent
RES = _OP36 / "results"

ISLS = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]
BAND = {"4k": "4-16k", "8k": "4-16k", "16k": "4-16k",
        "32k": "32-128k", "64k": "32-128k", "128k": "32-128k",
        "256k": "256k-1M", "512k": "256k-1M", "1024k": "256k-1M"}
BANDS = ["4-16k", "32-128k", "256k-1M"]

# A2 ship shapes (ITERATIONS iter6): K512@N>=262127 (flash/1024k class),
# K2048@N>=163775 (v32/256k class); routed region = N>=65536 & 32<=BS<=128.
A2_SHIP = {("flash", "1024k"), ("v32", "256k")}


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return exp(sum(log(x) for x in xs) / len(xs)) if xs else float("nan")


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else float("nan")


def load_jsonl(root):
    """(model, isl, BS) -> {op: us}; plus meta (op -> full row)."""
    cells, meta = defaultdict(dict), defaultdict(dict)
    for l in (root / "results.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("dtype") != "fp32":
            continue
        u = (r.get("us_span") or r.get("us")) \
            if r["op"] in ("sglang_v2", "sgl_bx") else r.get("us")
        k = (r["model"], r["isl"], r["BS"])
        if u:
            cells[k][r["op"]] = u
        meta[k][r["op"]] = r
    return cells, meta


def routed(k, meta_b):
    m, isl, bs = k
    n = meta_b[k].get("gvr_pr", {}).get("N") or meta_b[k].get("sglang_v2", {}).get("N")
    return n is not None and n >= 65536 and 32 <= bs <= 128


# ---------------------------------------------------------------- load all
bcells, bmeta = load_jsonl(RES / "b_screen")
vcells, _ = load_jsonl(RES / "b_verdict")
a2cells, a2meta = load_jsonl(RES / "a2_verdict")
a0cells, a0meta = load_jsonl(RES / "a0_screen")

# 'flags'/'launch_cfg' live in the per-batch spec jsonls (pre-parse), not in
# the merged results.jsonl — merge them into a2meta.
for f in (RES / "a2_verdict").glob("real_*.jsonl"):
    for l in f.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("dtype") != "fp32":
            continue
        a2meta[(r["model"], r["isl"], r["BS"])][r["op"]].update(
            {k: r[k] for k in ("flags", "launch_cfg", "cluster_size") if k in r})

base = {}
with open(RES / "baseline_real_bs.csv") as f:
    for r in csv.DictReader(f):
        if r["dtype"] == "fp32":
            base[(r["model"], r["isl"], int(r["BS"]))] = r

# ---------------------------------------------------------------- §1 baseline
base_ratios = [(k, float(v["sglang_v2"]) / float(v["gvr_pr"]))
               for k, v in base.items()
               if v.get("sglang_v2") and v.get("gvr_pr")]
base_gm_all = gm([x for _, x in base_ratios])
base_band = {b: gm([x for (m, i, _), x in base_ratios if BAND[i] == b])
             for b in BANDS}
n_base = len(base_ratios)

# ---------------------------------------------------------------- §2 A0
a0_att = []          # (model, isl) -> a0/pr gm over BS
for (m, isl) in sorted({(m, i) for (m, i, _) in a0cells}):
    rs = [a0cells[k]["gvr_pr"] / a0cells[k]["gvr_a0"]
          for k in a0cells if (k[0], k[1]) == (m, isl)
          and all(a0cells[k].get(x) for x in ("gvr_pr", "gvr_a0"))]
    if rs:
        a0_att.append((m, isl, gm(rs), min(rs), max(rs), len(rs)))
a0_overall = gm([a0cells[k]["gvr_pr"] / a0cells[k]["gvr_a0"] for k in a0cells
                 if all(a0cells[k].get(x) for x in ("gvr_pr", "gvr_a0"))])

# ---------------------------------------------------------------- §3 Track B
eps_band = defaultdict(list)
eps_all = []
for k, o in bcells.items():
    if o.get("sgl_bx") and o.get("sglang_v2"):
        e = o["sgl_bx"] / o["sglang_v2"]
        eps_band[BAND[k[1]]].append(e)
        eps_all.append((k, e))
eps_worst = max(eps_all, key=lambda t: t[1])

full = [(k, o) for k, o in bcells.items()
        if all(o.get(x) for x in ("gvr_pr", "sgl_bx", "sglang_v2"))]
hole = [(k, o) for k, o in full if BAND[k[1]] == "4-16k"]
hole_pr = gm([o["sglang_v2"] / o["gvr_pr"] for _, o in hole])
hole_bx = gm([o["sglang_v2"] / o["sgl_bx"] for _, o in hole])

comp_pr = gm([o["sglang_v2"] / o["gvr_pr"] for _, o in full])
comp_bx = gm([o["sglang_v2"] / o["sgl_bx"] for _, o in full])
comp_r1 = gm([o["sglang_v2"] / (o["gvr_pr"] if routed(k, bmeta) else o["sgl_bx"])
              for k, o in full])
comp_oracle = gm([o["sglang_v2"] / min(o["gvr_pr"], o["sgl_bx"]) for _, o in full])
n_routed = sum(1 for k, _ in full if routed(k, bmeta))
pr_wins = [(k, o["sgl_bx"] / o["gvr_pr"]) for k, o in full
           if routed(k, bmeta) and o["gvr_pr"] < o["sgl_bx"]]

# per-(model,isl) grid table: pr/sgl, bx/sgl, ship/sgl (a2 folded below)
grid_rows = []
for m in ("flash", "pro", "v32"):
    for isl in ISLS:
        sub = [(k, o) for k, o in full if (k[0], k[1]) == (m, isl)]
        if not sub:
            continue
        pr_g = gm([o["sglang_v2"] / o["gvr_pr"] for _, o in sub])
        bx_g = gm([o["sglang_v2"] / o["sgl_bx"] for _, o in sub])
        ship_vals = []
        for k, o in sub:
            if routed(k, bmeta):
                if (m, isl) in A2_SHIP and k in a2cells and \
                        all(a2cells[k].get(x) for x in ("gvr_a2", "sglang_v2")):
                    ship_vals.append(a2cells[k]["sglang_v2"] / a2cells[k]["gvr_a2"])
                else:
                    ship_vals.append(o["sglang_v2"] / o["gvr_pr"])
            else:
                ship_vals.append(o["sglang_v2"] / o["sgl_bx"])
        grid_rows.append((m, isl, len(sub), pr_g, bx_g, gm(ship_vals)))

# verdict consistency (b_verdict vs b_screen shared cells)
drift, ratio_diff, eps2 = [], [], []
for k, o in vcells.items():
    b = bcells.get(k)
    if not b:
        continue
    for op in ("gvr_pr", "sgl_bx", "sglang_v2"):
        if o.get(op) and b.get(op):
            drift.append(o[op] / b[op])
    if all(o.get(x) for x in ("sgl_bx", "sglang_v2")) and \
       all(b.get(x) for x in ("sgl_bx", "sglang_v2")):
        ratio_diff.append(abs(o["sgl_bx"] / o["sglang_v2"] -
                              b["sgl_bx"] / b["sglang_v2"]))
    if all(o.get(x) for x in ("sgl_bx", "sglang_v2")):
        eps2.append(((k), o["sgl_bx"] / o["sglang_v2"]))
vhole = [(k, o) for k, o in vcells.items() if BAND[k[1]] == "4-16k"
         and all(o.get(x) for x in ("sgl_bx", "sglang_v2"))]
vhole_bx = gm([o["sglang_v2"] / o["sgl_bx"] for _, o in vhole])
eps2_gm = gm([e for _, e in eps2])
eps2_worst = max(eps2, key=lambda t: t[1]) if eps2 else (None, float("nan"))

# ---------------------------------------------------------------- §4 A2
a2_att = []   # all measured a2 cells: (k, N, pr/a2, a2 vs sgl, ship?, routed?)
for k in sorted(a2cells):
    o = a2cells[k]
    if not all(o.get(x) for x in ("gvr_pr", "gvr_a2", "sglang_v2")):
        continue
    n = a2meta[k].get("gvr_a2", {}).get("N")
    a2_att.append((k, n, o["gvr_pr"] / o["gvr_a2"],
                   o["sglang_v2"] / o["gvr_a2"],
                   (k[0], k[1]) in A2_SHIP, 32 <= k[2] <= 128))
a2_win_max = max(s for (_, _, s, _, ship, _) in a2_att if ship)
a2_loss_min = min(s for (_, _, s, _, ship, ro) in a2_att if not ship and ro)

# ship composite with a2 folded on its 6 ship cells. a2's speedup is anchor-
# transferred onto the b_screen grid (pr scaled by the within-a2-run pr/a2) so
# the composite stays a single-grid number — the canonical 1.017.
ship_vals, n_a2_cells = [], 0
for k, o in full:
    m, isl, bs = k
    if routed(k, bmeta):
        if (m, isl) in A2_SHIP and k in a2cells and \
                all(a2cells[k].get(x) for x in ("gvr_a2", "gvr_pr")):
            ak = a2cells[k]
            ship_vals.append(o["sglang_v2"] /
                             (o["gvr_pr"] * ak["gvr_a2"] / ak["gvr_pr"]))
            n_a2_cells += 1
        else:
            ship_vals.append(o["sglang_v2"] / o["gvr_pr"])
    else:
        ship_vals.append(o["sglang_v2"] / o["sgl_bx"])
comp_ship = gm(ship_vals)
# full 3-arm oracle
oracle3 = []
for k, o in full:
    cands = [o["gvr_pr"], o["sgl_bx"]]
    if k in a2cells and a2cells[k].get("gvr_a2") and a2cells[k].get("sglang_v2"):
        # translate a2 into b_screen scale via within-run ratio
        cands.append(o["sglang_v2"] * a2cells[k]["gvr_a2"] / a2cells[k]["sglang_v2"])
    oracle3.append(o["sglang_v2"] / min(cands))
comp_oracle3 = gm(oracle3)

# ---------------------------------------------------------------- §5 ceiling
ceil = {}
for f in (1.10, 1.25, 1.578):
    ceil[f] = gm([o["sglang_v2"] / min(o["gvr_pr"] / f, o["sgl_bx"])
                  for _, o in full])

# ---------------------------------------------------------------- format
def f3(x):
    return f"{x:.3f}"


def band_table():
    hdr = ("<table><tr><th>ISL band</th><th>n</th>"
           "<th>gvr_pr / sglang_v2 (gm)</th></tr>")
    rows = "".join(
        f"<tr><td>{b}</td><td>{sum(1 for (m, i, _), _ in base_ratios if BAND[i] == b)}</td>"
        f"<td>{f3(base_band[b])}</td></tr>" for b in BANDS)
    return hdr + rows + (f"<tr><td><b>ALL</b></td><td>{n_base}</td>"
                         f"<td><b>{f3(base_gm_all)}</b></td></tr></table>")


def a0_table():
    hdr = ("<table><tr><th>model / ISL</th><th>a0 vs pr gm</th><th>min</th>"
           "<th>max</th><th>n</th></tr>")
    rows = "".join(f"<tr><td>{m}/{i}</td><td>{f3(g)}</td><td>{f3(lo)}</td>"
                   f"<td>{f3(hi)}</td><td>{n}</td></tr>"
                   for m, i, g, lo, hi, n in a0_att
                   if abs(g - 1.0) > 0.02)
    return hdr + rows + "</table>"


def grid_table():
    hdr = ("<table><tr><th>model / ISL</th><th>n</th><th>pr / sgl</th>"
           "<th>bx / sgl (guard&nbsp;ε)</th><th>SHIP / sgl</th></tr>")
    rows = ""
    for m, isl, n, prg, bxg, shg in grid_rows:
        hl = ' style="background:#eef7ee"' if shg >= 1.0 else ""
        rows += (f"<tr{hl}><td>{m}/{isl}</td><td>{n}</td><td>{f3(prg)}</td>"
                 f"<td>{f3(bxg)}</td><td><b>{f3(shg)}</b></td></tr>")
    return hdr + rows + "</table>"


def a2_table():
    hdr = ("<table><tr><th>cell</th><th>N</th><th>pr / a2</th>"
           "<th>a2 vs sglang</th><th>routed</th><th>SHIP</th></tr>")
    rows = ""
    for (m, i, bs), n, sp, vs, ship, ro in a2_att:
        if abs(sp - 1.0) < 0.03 and not (ship and ro):
            continue
        hl = ' style="background:#eef7ee"' if ship and ro else \
             (' style="background:#fbecec"' if sp < 0.97 and ro else "")
        rows += (f"<tr{hl}><td>{m}/{i}/BS{bs}</td><td>{n}</td><td>{f3(sp)}</td>"
                 f"<td>{f3(vs)}</td><td>{'Y' if ro else ''}</td>"
                 f"<td>{'Y' if ship and ro else ''}</td></tr>")
    return hdr + rows + "</table>"


# ------------------------------------------------------------ §6 charts
# Static SVG (viewer strips <script>): categorical slots 1-3 of the validated
# dataviz reference palette, fixed assignment order: ship=blue, sglang=green,
# gvr_pr=magenta. Light surface only (report is fixed-light house style).
C_SHIP, C_SGL, C_PR = "#2a78d6", "#008300", "#e87ba4"
BSS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def ship_time(k, o):
    """Per-cell ship-arm time on the b_screen scale (a2 anchor-transferred)."""
    m, isl, bs = k
    if routed(k, bmeta):
        if (m, isl) in A2_SHIP and k in a2cells and \
                all(a2cells[k].get(x) for x in ("gvr_a2", "gvr_pr")):
            ak = a2cells[k]
            return o["gvr_pr"] * ak["gvr_a2"] / ak["gvr_pr"]
        return o["gvr_pr"]
    return o["sgl_bx"]


def svg_panel(title, xlabels, series, ylab, ref=None, w=340, h=230, logy=False,
              yzero=True):
    """series = [(name, color, [y or None per x]), ...]; returns one SVG."""
    import math as _m
    ml, mr, mt, mb = 44, 8, 22, 34
    pw, ph = w - ml - mr, h - mt - mb
    ys = [v for _, _, vs in series for v in vs if v]
    lo = 0.0 if (yzero and not logy) else min(ys) * 0.92
    hi = max(ys) * 1.06
    if ref is not None:
        lo, hi = min(lo, ref * 0.92), max(hi, ref * 1.08)
    tf = (lambda v: _m.log(v)) if logy else (lambda v: v)
    Y = lambda v: mt + ph * (1 - (tf(v) - tf(lo if lo > 0 or not logy else min(ys))) /
                             (tf(hi) - tf(lo if lo > 0 or not logy else min(ys))))
    X = lambda i: ml + pw * (i / max(1, len(xlabels) - 1))
    p = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'style="font-family:system-ui;background:#fff">']
    p.append(f'<text x="{ml}" y="14" font-size="11" fill="#24435f" '
             f'font-weight="600">{title}</text>')
    # y grid: 4 ticks
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        y = Y(v)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" '
                 f'stroke="#e4ebf2" stroke-width="1"/>')
        p.append(f'<text x="{ml - 5}" y="{y + 3.5:.1f}" font-size="9" '
                 f'fill="#667" text-anchor="end">{v:.2f}</text>' if hi < 10 else
                 f'<text x="{ml - 5}" y="{y + 3.5:.1f}" font-size="9" '
                 f'fill="#667" text-anchor="end">{v:.0f}</text>')
    if ref is not None:
        y = Y(ref)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" '
                 f'stroke="#8a99a8" stroke-width="1.2" stroke-dasharray="5,4"/>')
        p.append(f'<text x="{w - mr - 2}" y="{y - 3:.1f}" font-size="8.5" '
                 f'fill="#8a99a8" text-anchor="end">sglang_v2 = 1.0</text>')
    for i, xl in enumerate(xlabels):
        p.append(f'<text x="{X(i):.1f}" y="{h - mb + 14}" font-size="8.5" '
                 f'fill="#667" text-anchor="end" '
                 f'transform="rotate(-38 {X(i):.1f} {h - mb + 14})">{xl}</text>')
    p.append(f'<text x="12" y="{mt + ph / 2:.0f}" font-size="9" fill="#667" '
             f'text-anchor="middle" transform="rotate(-90 12 {mt + ph / 2:.0f})">'
             f'{ylab}</text>')
    for name, col, vs in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(vs) if v]
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        # sglang dashed: in the BS=1 panels the ship line hugs it (guard-eps
        # apart) — the dash keeps the overlapping pair distinguishable.
        dash = ' stroke-dasharray="6,4"' if name == "sglang_v2" else ""
        p.append(f'<path d="{d}" fill="none" stroke="{col}" '
                 f'stroke-width="2"{dash}/>')
        for x, y in pts:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{col}" '
                     f'stroke="#fff" stroke-width="1.4"/>')
    p.append("</svg>")
    return "".join(p)


def legend3():
    it = [("op36 ship", C_SHIP), ("sglang_v2", C_SGL), ("gvr_pr (PR#16457)", C_PR)]
    s = '<div style="font-size:0.8em;margin:2px 0 4px">'
    for n, c in it:
        s += (f'<span style="margin-right:16px"><span style="display:inline-block;'
              f'width:14px;height:3px;background:{c};vertical-align:middle;'
              f'border-radius:2px"></span> <span style="color:#333">{n}</span></span>')
    return s + "</div>"


def chart_seqlen():
    panels, tbl = [], []
    for m in ("flash", "pro", "v32"):
        xls, s_sgl, s_pr, s_ship = [], [], [], []
        for isl in ISLS:
            k = (m, isl, 1)
            o = bcells.get(k, {})
            if not all(o.get(x) for x in ("gvr_pr", "sgl_bx", "sglang_v2")):
                continue
            xls.append(isl)
            s_sgl.append(o["sglang_v2"])
            s_pr.append(o["gvr_pr"])
            s_ship.append(ship_time(k, o))
            tbl.append((m, isl, o["sglang_v2"], o["gvr_pr"], ship_time(k, o)))
        panels.append(svg_panel(
            f"{m} · BS=1", xls,
            [("gvr_pr", C_PR, s_pr), ("sglang_v2", C_SGL, s_sgl),
             ("op36 ship", C_SHIP, s_ship)], "µs (nsys cold-L2)"))
    rows = "".join(f"<tr><td>{m}/{i}</td><td>{a:.2f}</td><td>{b:.2f}</td>"
                   f"<td>{c:.2f}</td><td>{a / c:.3f}</td></tr>"
                   for m, i, a, b, c in tbl)
    return ('<div style="display:flex;flex-wrap:wrap;gap:8px">'
            + "".join(panels) + "</div>"
            + '<details><summary><span class="zh">数据表</span>'
              '<span class="en">data table</span></summary>'
              "<table><tr><th>cell</th><th>sglang_v2 µs</th><th>gvr_pr µs</th>"
              "<th>ship µs</th><th>ship speedup vs sgl</th></tr>"
            + rows + "</table></details>")


def chart_bs():
    panels, tbl = [], []
    for m in ("flash", "pro", "v32"):
        xls, r_pr, r_ship = [], [], []
        for bs in BSS:
            cs = [(k, o) for k, o in full if k[0] == m and k[2] == bs]
            if not cs:
                continue
            xls.append(str(bs))
            r_pr.append(gm([o["sglang_v2"] / o["gvr_pr"] for _, o in cs]))
            r_ship.append(gm([o["sglang_v2"] / ship_time(k, o) for k, o in cs]))
            tbl.append((m, bs, r_pr[-1], r_ship[-1], len(cs)))
        panels.append(svg_panel(
            f"{m} · gm over ISL 4k-1M", xls,
            [("gvr_pr", C_PR, r_pr), ("op36 ship", C_SHIP, r_ship)],
            "speedup vs sglang_v2 (gm)", ref=1.0, yzero=False))
    rows = "".join(f"<tr><td>{m}</td><td>{bs}</td><td>{a:.3f}</td>"
                   f"<td>{b:.3f}</td><td>{n}</td></tr>"
                   for m, bs, a, b, n in tbl)
    return ('<div style="display:flex;flex-wrap:wrap;gap:8px">'
            + "".join(panels) + "</div>"
            + '<details><summary><span class="zh">数据表</span>'
              '<span class="en">data table</span></summary>'
              "<table><tr><th>model</th><th>BS</th><th>gvr_pr / sgl</th>"
              "<th>ship / sgl</th><th>n ISLs</th></tr>"
            + rows + "</table></details>")


# ------------------------------------------------------------ §6c hit split
# Post-hoc statistics only: hit is a property of the capture batch (model,isl),
# identical across BS, and the ship table never reads it (red line).
def cell_hit(k):
    r = bmeta[k].get("gvr_pr") or bmeta[k].get("sglang_v2")
    return r["hit"]


hit_split = {}
for name, cond in (("hit>0.4", lambda h: h > 0.4),
                   ("hit<=0.4", lambda h: h <= 0.4)):
    sub = [(k, o) for k, o in full if cond(cell_hit(k))]
    ship = [o["sglang_v2"] / ship_time(k, o) for k, o in sub]
    ro = [(k, o) for k, o in sub if routed(k, bmeta)]
    hit_split[name] = {
        "n": len(sub),
        "nb": len({(k[0], k[1]) for k, _ in sub}),
        "ship": (gm(ship), min(ship), max(ship)),
        "pr": gm([o["sglang_v2"] / o["gvr_pr"] for _, o in sub]),
        "bx": gm([o["sglang_v2"] / o["sgl_bx"] for _, o in sub]),
        "per_model": {m: gm([o["sglang_v2"] / ship_time(k, o)
                             for k, o in sub if k[0] == m] or [float("nan")])
                      for m in ("flash", "pro", "v32")},
        "routed": (len(ro),
                   gm([o["sglang_v2"] / ship_time(k, o) for k, o in ro]),
                   max([o["sglang_v2"] / ship_time(k, o) for k, o in ro])),
    }


def hit_table():
    hdr = ("<table><tr><th></th>"
           "<th>n (cells / batches)</th><th>ship / sgl gm</th><th>min</th>"
           "<th>max</th><th>pr / sgl</th><th>bx / sgl</th>"
           "<th>flash</th><th>pro</th><th>v32</th>"
           "<th>routed cells (n / gm / max)</th></tr>")
    rows = ""
    for name in ("hit>0.4", "hit<=0.4"):
        s = hit_split[name]
        g, lo, hi = s["ship"]
        rn, rg, rm = s["routed"]
        pm = s["per_model"]
        v32 = f3(pm['v32']) if pm['v32'] == pm['v32'] else "—"
        label = "hit&gt;0.4" if name == "hit>0.4" else "hit&le;0.4"
        rows += (f"<tr><td>{label}</td><td>{s['n']} / {s['nb']}</td>"
                 f"<td><b>{f3(g)}</b></td><td>{f3(lo)}</td><td>{f3(hi)}</td>"
                 f"<td>{f3(s['pr'])}</td><td>{f3(s['bx'])}</td>"
                 f"<td>{f3(pm['flash'])}</td><td>{f3(pm['pro'])}</td>"
                 f"<td>{v32}</td>"
                 f"<td>{rn} / {f3(rg)} / {f3(rm)}</td></tr>")
    return hdr + rows + "</table>"


def hit_batch_table():
    hdr = ("<table><tr><th>batch</th><th>N</th><th>hit</th>"
           "<th>domain</th></tr>")
    rows = ""
    for (m, isl) in sorted({(k[0], k[1]) for k, _ in full}):
        k = next(k for k, _ in full if (k[0], k[1]) == (m, isl))
        r = bmeta[k].get("gvr_pr") or bmeta[k].get("sglang_v2")
        h = r["hit"]
        rows += (f"<tr><td>{m}/{isl}</td><td>{r['N']}</td><td>{h:.3f}</td>"
                 f"<td>{'&gt;0.4' if h > 0.4 else '&le;0.4'}</td></tr>")
    return hdr + rows + "</table>"


# ------------------------------------------------------------ §7 algorithm
# Flow diagram (pure CSS boxes, no <script>) + pseudocode for the ship scheme.
# Kernel structure cross-checked against variant/gvrpkg36/top_k/gvr_topk_decode.py
# (GvrTopKKernel docstring + dist_p4 sites), TRACKA2_DESIGN.md (SYNC1-6) and
# src/trackb/topk_v2_exact_standalone.cu (impl families, plan routing).
FLOW_HTML = """
<div style="font-size:0.82em;line-height:1.35">
<style>
.fn{border:1.5px solid #3b6ea5;border-radius:6px;padding:6px 10px;background:#f4f8fb;
    margin:0 auto;max-width:640px;text-align:center}
.fd{border:1.5px solid #b8860b;border-radius:6px;padding:6px 10px;background:#fdf6e3;
    margin:0 auto;max-width:640px;text-align:center}
.fa{text-align:center;color:#3b6ea5;font-weight:bold;padding:1px 0}
.fcols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:4px}
.fcol{border:1.5px solid #3b6ea5;border-radius:6px;padding:8px}
.fcol h4{margin:2px 0 6px;font-size:1em;color:#24435f;text-align:center}
.fcond{text-align:center;color:#885;font-size:0.92em;min-height:2.6em}
.fstep{border:1px solid #c5d3e0;border-radius:4px;padding:3px 6px;margin:4px 0;background:#fff}
.fgreen{background:#eef7ee}.fesc{border-style:dashed;background:#fdf0f0}
</style>
<div class="fn"><span class="zh">每个 decode step 输入:indexer logits[BS, N] (fp32) · K∈{512,1024,2048} · 上一步 top-K 索引 preIdx</span><span class="en">Per decode step: indexer logits[BS, N] (fp32) · K∈{512,1024,2048} · prev-step top-K indices preIdx</span></div>
<div class="fa">▼</div>
<div class="fd"><span class="zh"><b>host 端 shape 路由</b>(仅用推理期可知的 N / BS / K;命中率永不参与 — 红线)</span><span class="en"><b>Host-side shape dispatch</b> (inference-known N / BS / K only; hit-rate never participates — red line)</span></div>
<div class="fcols">
<div class="fcol fgreen">
<h4><span class="zh">规则 3(默认):sgl_bx</span><span class="en">Rule 3 (default): sgl_bx</span></h4>
<div class="fcond"><span class="zh">N&lt;65536 或 BS∉[32,128]<br>(248/275 cells)</span><span class="en">N&lt;65536 or BS∉[32,128]<br>(248/275 cells)</span></div>
<div class="fstep"><span class="zh"><b>plan kernel</b>(非关键路径):按 seq_len 分桶 Register2 (≤8k) / Register4 (≤16k) / Streaming / 8-CTA cluster 池;<b>清零 per-row 溢出旗标</b></span><span class="en"><b>plan kernel</b> (off critical path): bucket rows → Register2 (≤8k) / Register4 (≤16k) / Streaming / 8-CTA cluster pool; <b>zero per-row overflow flags</b></span></div>
<div class="fa">▼</div>
<div class="fstep"><span class="zh"><b>transform kernel</b>:fp16 4096-bin 直方图 → 阈值 bin b* → 双 fp32 边界收集(&gt;hi 直写,[lo,hi] 进 tie 缓冲, cap 2048)→ P4 补齐</span><span class="en"><b>transform kernel</b>: fp16 4096-bin histogram → threshold bin b* → collect by two fp32 boundaries (&gt;hi direct, [lo,hi] → tie buffer, cap 2048) → P4 fill</span></div>
<div class="fa">▼</div>
<div class="fstep"><span class="zh"><b>[op36 守卫]</b> 4 个截断点置 overflow_flag</span><span class="en"><b>[op36 guard]</b> overflow_flag set at all 4 truncation sites</span></div>
<div class="fa">▼</div>
<div class="fstep fesc"><span class="zh"><b>escape</b>(数据罕见,实测仅 V3.2 L52 类行):被标行经 radix_cutedsl 精确重跑 → <b>无条件精确</b></span><span class="en"><b>escape</b> (data-rare; only V3.2 L52-class rows in real captures): flagged rows re-run through radix_cutedsl → <b>unconditional exactness</b></span></div>
<div class="fcond"><span class="zh">洞 0.60→0.99,守卫税 ε 0.4-0.8%</span><span class="en">hole 0.60→0.99, guard tax ε 0.4-0.8%</span></div>
</div>
<div class="fcol">
<h4><span class="zh">规则 2:gvr_pr + A0 旗标</span><span class="en">Rule 2: gvr_pr + A0 flags</span></h4>
<div class="fcond"><span class="zh">N≥65536 且 BS∈[32,128](中批量谷)</span><span class="en">N≥65536 &amp; BS∈[32,128] (mid-BS valley)</span></div>
<div class="fstep"><span class="zh"><b>P1</b> 按 preIdx gather 上一步 top-K 值 → min/max/mean 种子</span><span class="en"><b>P1</b> gather prev top-K values at preIdx → min/max/mean seed</span></div>
<div class="fstep"><span class="zh"><b>P1b</b> gather 值上建 256-bin 直方图 → M 级 rung 阈值梯 (R0)</span><span class="en"><b>P1b</b> 256-bin histogram over gathered values → M rung thresholds (R0)</span></div>
<div class="fstep"><span class="zh"><b>P2</b> 单遍全行读:多阈值 rung-ladder 计数准入;全 miss → secant 回退。A0: skip_h1 删 P2 末冗余 cluster 握手 (per-K 门), kNumBins 512@K2048</span><span class="en"><b>P2</b> ONE full-row read: multi-threshold rung-ladder admission count; all-rungs-miss → secant fallback. A0: skip_h1 drops the redundant end-of-P2 cluster handshake (per-K gated); kNumBins 512@K2048</span></div>
<div class="fstep"><span class="zh"><b>P3</b> 准入区间第二遍读:ballot-free 收集候选进 smem</span><span class="en"><b>P3</b> second read of admitted range: ballot-free candidate collect into smem</span></div>
<div class="fstep"><span class="zh"><b>P4</b>(leader 路径)handoff2 值汇聚 → leader 256-bin 直方图 → rank-and-scatter 精确 top-K → 写回</span><span class="en"><b>P4</b> (leader path) handoff2 value-gather → leader 256-bin histogram → rank-and-scatter exact top-K → writeback</span></div>
<div class="fcond"><span class="zh">26 谷区 cell 1.05-1.57× vs bx</span><span class="en">26 valley cells 1.05-1.57× vs bx</span></div>
</div>
<div class="fcol">
<h4><span class="zh">规则 1:gvr dist_p4 (+A0)</span><span class="en">Rule 1: gvr dist_p4 (+A0)</span></h4>
<div class="fcond"><span class="zh">路由区内 K512@N≥262k 或 K2048@N≥164k(cs&gt;1)</span><span class="en">routed ∧ (K512@N≥262k or K2048@N≥164k) (cs&gt;1)</span></div>
<div class="fstep"><span class="zh">P1-P3 与规则 2 相同;P4 换分布式:</span><span class="en">P1-P3 as rule 2; P4 goes distributed:</span></div>
<div class="fstep"><span class="zh">杀 handoff2 值搬运;各 CTA 把 (min,max) f32-bits 写自家 DSMEM 槽 <b>S1</b> → red.relaxed.cluster 全簇建直方图 <b>S2-3</b> → leader 标量搜 b* 并广播 <b>S4-5</b> → 各 CTA 就地 scatter 写回;罕见 b* 歧义 → gather 回退保精确 <b>S6</b>(共 6 次 release-sync)</span><span class="en">kill the handoff2 value-ship; each CTA publishes (min,max) f32-bits in its own DSMEM slot <b>S1</b> → cluster-wide histogram via red.relaxed.cluster <b>S2-3</b> → leader scalar-searches b*, broadcasts <b>S4-5</b> → every CTA scatters its own candidates in place; rare b*-tie ambiguity → gather fallback, exact <b>S6</b> (6 release-syncs total)</span></div>
<div class="fcond"><span class="zh">6 cells 同时胜 pr 与 bx(至 1.36 vs sglang);N≤131k 被 sync 税吃掉 → 只在大 N 开</span><span class="en">6 cells beat BOTH pr and bx (up to 1.36 vs sglang); sync tax dominates at N≤131k → large-N only</span></div>
</div>
</div>
<div class="fa">▼</div>
<div class="fn"><span class="zh">输出:每行精确 top-K 索引(三臂全部无条件精确)→ 稀疏注意力 gather</span><span class="en">Output: exact per-row top-K indices (all three arms unconditionally exact) → sparse-attention gather</span></div>
</div>
"""

PSEUDO_HTML = """
<pre style="background:#f7f9fb;border:1px solid #c5d3e0;border-radius:6px;padding:10px 14px;font-size:0.78em;overflow-x:auto">
# ---------- host dispatch (ship table; shape keys only, hit-rate never used) ----------
def indexer_topk_decode(logits, K, N, BS, preIdx):
    routed = (N >= 65536) and (32 <= BS <= 128)              # GVR mid-BS valley
    if not routed:
        return sgl_bx(logits, K)                             # rule 3 (default)
    if (K == 512 and N >= 262144) or (K == 2048 and N >= 163775):
        return gvr(logits, K, preIdx, A0_FLAGS | DIST_P4)    # rule 1 (large-N, cs&gt;1)
    return gvr(logits, K, preIdx, A0_FLAGS)                  # rule 2
    # A0_FLAGS: skip_h1 ON {K512@N&gt;=262144, K2048}, OFF K1024; kNumBins=512 @K2048 only

# ---------- sgl_bx = vendored sglang v2 + unconditional-exactness guard ----------
plan_kernel:                                   # untimed, once per decode step
    bucket rows by N: Register2(&lt;=8192) | Register4(&lt;=16384) | Streaming | cluster pool
    overflow_flag[0:BS] = 0                                  # [op36 guard]

transform_kernel(row):                         # per impl family, same skeleton
    P1  hist[4096] += popcount per fp16 coarse bin           # one full-row read
    P2  b* = bin where suffix_count crosses K
    P3  emit idx where x &gt; hi(b*); candidates in [lo(b*), hi(b*)] -&gt; tie buf (cap 2048)
    P4  fill the remaining K-slots from tie buf
        if ties_needed &gt; ties_collected:                     # cap truncated
            overflow_flag[row] = 1     # [op36: 4 sites — Register/Streaming P4,
                                       #  cluster rank-0 P4, per-rank local cap]
escape:                                        # data-rare (14/2245 classes; real: V3.2 L52)
    for row where overflow_flag[row]: out[row] = radix_topk_exact(logits[row], K)

# ---------- gvr = Guess-Verify-Refine, 1 CTA (or cs-CTA cluster) per row ----------
gvr_kernel(row):
    P1   seed = min/max/mean over logits[preIdx]             # prev-step guess
    P1b  rungs[M] = thresholds from 256-bin hist over the gathered values   # R0
    P2   one full-row pass: count admissions per rung        # verify
         r* = first rung with count &gt;= K;  all-miss -&gt; secant refine loop
         # skip_h1 (A0): drop redundant end-of-P2 cluster handshake (cs&gt;1)
    P3   second pass over admitted range: ballot-free collect -&gt; smem keys/vals
    P4   exact top-K among candidates:
      if not DIST_P4:                                        # leader path (pr)
          handoff2: peer CTAs ship candidate values to leader
          leader: 256-bin hist -&gt; rank-and-scatter -&gt; writeback
      else:                                                  # dist_p4 (A2), cs&gt;1
          each CTA writes own (min,max) f32-bits to own DSMEM slot;   SYNC1
          cluster-wide hist build via red.relaxed.cluster.add;        SYNC2-3
          leader scalar-searches split bin b*, broadcasts;            SYNC4-5
          distributed scatter: every CTA writes its own candidates;
          rare b*-tie ambiguity -&gt; gather fallback (exact);           SYNC6
          # relaxed atomics ordered by release cluster_arrive()/wait()
          # (cluster_arrive_relaxed has NO release -- stale-DSMEM hazard)
</pre>
"""

ZH, EN = "zh", "en"


def bi(zh, en, tag="p"):
    return f'<{tag} class="zh">{zh}</{tag}><{tag} class="en">{en}</{tag}>'


def h2(zh, en):
    return f'<h2><span class="zh">{zh}</span><span class="en">{en}</span></h2>'


def h3(zh, en):
    return f'<h3><span class="zh">{zh}</span><span class="en">{en}</span></h3>'


HTML = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>op36 GVR vs sglang real-§7b campaign report</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;margin:24px auto;max-width:1100px;
     color:#1a1a2e;line-height:1.55;padding:0 16px}}
h1{{font-size:1.4em;border-bottom:3px solid #3b6ea5;padding-bottom:8px}}
h2{{font-size:1.15em;color:#24435f;margin-top:28px;border-left:4px solid #3b6ea5;padding-left:10px}}
h3{{font-size:1.0em;color:#24435f}}
table{{border-collapse:collapse;margin:12px 0;font-size:0.85em}}
th,td{{border:1px solid #c5d3e0;padding:4px 8px;text-align:right}}
th{{background:#eaf1f7}}td:first-child,th:first-child{{text-align:left}}
.meta{{color:#667;font-size:0.85em}}
.big{{font-size:1.05em;background:#f4f8fb;border:1px solid #c5d3e0;padding:10px 14px;border-radius:6px}}
#lang-en:checked ~ .content .zh{{display:none}}
#lang-zh:checked ~ .content .en{{display:none}}
.lang-toggle{{position:sticky;top:0;background:#fff;padding:8px 0;z-index:5}}
.lang-toggle label{{border:1px solid #3b6ea5;padding:4px 14px;cursor:pointer;border-radius:4px;margin-right:6px}}
#lang-zh:checked ~ .lang-toggle label[for=lang-zh],
#lang-en:checked ~ .lang-toggle label[for=lang-en]{{background:#3b6ea5;color:#fff}}
details{{margin:10px 0}}summary{{cursor:pointer;color:#24435f}}
code{{background:#f0f3f7;padding:1px 4px;border-radius:3px;font-size:0.9em}}
</style></head><body>
<input type="radio" name="lang" id="lang-zh" checked hidden>
<input type="radio" name="lang" id="lang-en" hidden>
<div class="lang-toggle"><label for="lang-zh">中文</label><label for="lang-en">English</label></div>
<div class="content">
<h1><span class="zh">op36 — GVR vs sglang_v2 真实 §7b 轴战役报告</span><span class="en">op36 — GVR vs sglang_v2 on the real §7b axis: campaign report</span></h1>
<p class="meta">2026-07-18 · umbriel-b200-047 → -093 (8×B200) · base = PR#16457 HEAD eae374554c
(worktree TensorRT-LLM-gvr-r0) · axis = op26 REPORT §8 real decode-capture fp32 grid
(V4 Flash / V4 Pro / V3.2 · BS 1-1024 × ISL 4k-1M · {len(full)} sglang-comparable cells ·
us_span · nsys cold-L2) · screening 8-way, ship verdicts ≤2-concurrent nsys ·
harness = op26 rival_harness clone (src/) · 16 commits, single-day campaign</p>

{h2("0 · 执行摘要", "0 · Executive summary")}
{bi("用户目标 = 复合 geomean ≥ <b>1.10× vs sglang_v2</b>。起点 gvr_pr 单臂 0.745 "
    "(本节点 0.722)。战役以 <b>可行性 pivot 门</b> 运行:先做算术定界,再上硅逐轨证实/证伪。"
    "最终 ship = 三臂 shape 路由(全部 key 推理期可知,无 hit-rate 红线触碰):"
    "<b>sgl_bx</b>(精确化 sglang 移植)为默认;<b>gvr_pr(+A0 旗标)</b> 只在 N≥65536 且 "
    "BS∈[32,128] 的中批量谷;<b>dist_p4</b> 只在其中 K512@N≈262k / K2048@N≈164k 大 N 子集。",
    "User target = composite geomean ≥ <b>1.10× vs sglang_v2</b>. Starting point: gvr_pr "
    "alone 0.745 (0.722 on this node). The campaign ran under a <b>feasibility pivot gate</b>: "
    "bound arithmetic first, then silicon per track. Final ship = a 3-arm shape-keyed dispatch "
    "(all keys inference-known; the hit-rate red line untouched): <b>sgl_bx</b> (exactness port "
    "of sglang) as default; <b>gvr_pr (+A0 flags)</b> only in the mid-BS valley N≥65536 & "
    "BS∈[32,128]; <b>dist_p4</b> only on its K512@N≈262k / K2048@N≈164k large-N subset.")}
<div class="big">{bi(
    f"最终复合值 <b>{f3(comp_ship)}× vs sglang_v2</b>(单臂基线 0.722,<b>+41% 同节点</b>;"
    f"对 07-16 基线网格 0.745 计 +37%)。三臂 oracle {f3(comp_oracle3)} — 路由已捕获几乎全部可得空间。"
    f"1.10 目标经算术 + 实测双重判定<b>不可达</b>(§5)。全程无条件精确:275 折叠检查 ×2 + "
    "电池 93/93 + 2233/2233 + 29/29;vendored sglang_v2 在真实 V3.2 L52 行上被证明不精确,"
    "我方 escape 路径恰好全部接住。",
    f"Final composite <b>{f3(comp_ship)}× vs sglang_v2</b> (single-arm baseline 0.722, "
    f"<b>+41% same-node</b>; +37% vs the 07-16 baseline-grid 0.745). Three-arm oracle "
    f"{f3(comp_oracle3)} — the dispatch captures essentially all available headroom. The 1.10 "
    "target is judged <b>unreachable</b> by arithmetic and confirmed on silicon (§5). "
    "Unconditional exactness throughout: 275-cell folded checks ×2 + batteries 93/93 + "
    "2233/2233 + 29/29; vendored sglang_v2 is provably inexact on real V3.2 L52 rows, all of "
    "which our escape path catches.")}</div>

{h2("1 · 基线算术与可行性定界 (iter0)", "1 · Baseline arithmetic & feasibility bounds (iter0)")}
{bi("判定轴 = op26 REPORT §8 真实 decode-capture fp32 网格(07-16 b200-081 回填,"
    "analysis/baseline_7b.py 提取)。gvr_pr 单臂对 sglang_v2:",
    "Judgment axis = the op26 REPORT §8 real decode-capture fp32 grid (07-16 b200-081 "
    "backfill, extracted by analysis/baseline_7b.py). gvr_pr alone vs sglang_v2:")}
{band_table()}
{bi("上界表(开工前算术,全部后被实测证实):杠杆 ×1.25 全铺 → 0.931;完美 exact dispatch → "
    "0.827;+ 每个失地 sglang-parity 路径 → 1.030;+ 胜地 25% 杠杆 → 1.069。结论先行:1.10 "
    "需要在 sglang 主场逐 cell 胜出 — 已被 op34/op35/apex 三重证伪 — 故设 pivot 门而非盲追。",
    "Bound table (pre-silicon arithmetic, all later confirmed by measurement): levers ×1.25 "
    "everywhere → 0.931; perfect exact dispatch → 0.827; + an sglang-parity path at every lost "
    "cell → 1.030; + 25% levers on won cells → 1.069. Stated up front: 1.10 requires beating "
    "sglang per-cell on its home turf — triple-falsified by op34/op35/apex — hence a pivot "
    "gate instead of blind iteration.")}

{h2("2 · Track A0 — bundle-v2 收割 (iter1-3)", "2 · Track A0 — bundle-v2 harvest (iter1-3)")}
{bi("iter1 教训:GVR 臂最初按冻结 shape 构建,anchor 漂移 med 1.14 / p95 1.93(sglang med "
    "1.000 证明是 build 而非节点)→ 数据作废,全部 GVR 臂改用生产 launch 契约 "
    "(launch/pick_config)。<b>规程收获:任何 screening 在 ~1/3 处必须做 anchor 校验。</b>",
    "iter1 lesson: GVR arms were first built at frozen shapes — anchor drift med 1.14 / p95 "
    "1.93 (sglang med 1.000 proved it was the build, not the node) → data invalidated, all GVR "
    "arms moved to the production launch contract (launch/pick_config). <b>Process win: "
    "validate anchors at ~1/3 of any screening sweep.</b>")}
{bi(f"A0 (skip_h1 + kNumBins@K2048→512) 对 pr 全网格 gm {f3(a0_overall)} — 战役轴上 WASH,"
    "但形状可门。判决级复测(≤2 并发,6 极点批)逐位确认 screening:pro/512k 0.813 损失为真 "
    "(K1024 大 N 暖命中),flash/1024k 1.21 增益为真(冷命中)。op35 的 BS=1 \"0 lost\" 判决"
    "被限定范围而非推翻 — 记为 LEARNING。A0 ship 旗标表:skip_h1 ON {{K512@N≥262144, "
    "K2048(+kb512)}},K1024 OFF;复合 0.726→0.738,零回退。",
    f"A0 (skip_h1 + kNumBins@K2048→512) vs pr full-grid gm {f3(a0_overall)} — a WASH on the "
    "campaign axis, but shape-gateable. Verdict-grade re-runs (≤2-concurrent, 6 pole batches) "
    "confirmed screening bit-for-bit: the pro/512k 0.813 loss is real (K1024 large-N warm-hit), "
    "the flash/1024k 1.21 win is real (cold-hit). op35's BS=1 \"0 lost\" verdict is scoped, not "
    "contradicted — recorded as a LEARNING. A0 ship flag table: skip_h1 ON {{K512@N≥262144, "
    "K2048(+kb512)}}, OFF K1024; composite 0.726→0.738, zero regressions.")}
<details><summary>{bi("A0 形状归因表 (|Δ|>2% 的批)", "A0 shape attribution (batches with |Δ|>2%)", "span")}</summary>
{a0_table()}</details>

{h2("3 · Track B — sgl_bx 精确化移植 + (N,BS) 路由 (iter4)",
    "3 · Track B — the sgl_bx exactness port + (N,BS) dispatch (iter4)")}
{bi("结构判定(op32/op34):ISL 4-16k 的 99 cell 洞(pr gm 0.599)源于 GVR 1-CTA 骨架 "
    "~9.7µs 启动地板 vs sglang 8-CTA MLP 4.7-6.7µs — 骨架内无解。唯一到 parity+ 的路 = 移植 "
    "sglang v2 并修复其精确性缺陷(kMaxNumTie=2048 截断)。实现 = vendored 源逐字 + 全部 4 个"
    "截断点的 per-row overflow flag(Register/Streaming P4、cluster rank-0 P4、cluster 非主 "
    "rank 本地 cap)+ flag 在未计时 plan kernel 中清零 + 溢出行走树内 radix_cutedsl escape。",
    "Structural finding (op32/op34): the 99-cell ISL 4-16k hole (pr gm 0.599) comes from the "
    "GVR 1-CTA skeleton's ~9.7µs launch floor vs sglang's 8-CTA MLP at 4.7-6.7µs — unsolvable "
    "inside the skeleton. The only route to parity+ = port sglang v2 and FIX its exactness "
    "defect (the kMaxNumTie=2048 truncation). Implementation = vendored source verbatim + a "
    "per-row overflow flag at all 4 truncation sites (Register/Streaming phase-4, cluster "
    "rank-0 phase-4, cluster non-primary local cap) + flags zeroed in the untimed plan kernel "
    "+ an in-tree radix_cutedsl escape for flagged rows.")}
{h3("3a · 精确性护城河", "3a · The exactness moat")}
{bi("电池 93/93(强制溢出:all-tie / near-tie(单 fp16 粗 bin 内 >2048 个不同 fp32)/ "
    "cluster 单 rank 边角 / 混合批只重跑被标行);<b>TEETH 检查:vendored sglang_v2 在同一批行上"
    "被证明确实不精确</b>。ship 闸 = op26 全网格 2245 级电池:2233/2233 PASS,14 次 escape "
    "恰好全部落在越限行 — 含 4 个真实 V3.2 L52 行(sglang 实败)。护城河端到端成立。",
    "Battery 93/93 (forced overflow: all-tie / near-tie (>2048 distinct fp32 inside one fp16 "
    "coarse bin) / cluster single-rank edge / mixed batches re-running only flagged rows); "
    "<b>TEETH check: vendored sglang_v2 provably inexact on the same rows</b>. Ship gate = the "
    "op26 full-grid 2245-class battery: 2233/2233 PASS with exactly 14 escapes, all landing on "
    "over-cap rows — including the 4 real V3.2 L52 rows where sglang measurably fails. The "
    "moat holds end-to-end.")}
{h3("3b · 测量与路由判决", "3b · Measurement & dispatch verdict")}
{bi(f"守卫税 ε (sgl_bx/sglang_v2):全网格 gm {f3(gm([e for _, e in eps_all]))}"
    f"(分带 {'/'.join(f3(gm(eps_band[b])) for b in BANDS)}),8 并发下最差 "
    f"{f3(eps_worst[1])} ({eps_worst[0][0]}/{eps_worst[0][1]}/BS{eps_worst[0][2]});"
    f"≤2 并发判决下 gm {f3(eps2_gm)}、最差 {f3(eps2_worst[1])} — 1.09 离群是 8 并发伪影。"
    f"洞(4-16k, {len(hole)} cells):pr {f3(hole_pr)} → bx {f3(hole_bx)}(判决批 "
    f"{f3(vhole_bx)})。<b>PARITY 交付。</b>",
    f"Guard tax ε (sgl_bx/sglang_v2): full-grid gm {f3(gm([e for _, e in eps_all]))} (bands "
    f"{'/'.join(f3(gm(eps_band[b])) for b in BANDS)}), worst under 8-way "
    f"{f3(eps_worst[1])} ({eps_worst[0][0]}/{eps_worst[0][1]}/BS{eps_worst[0][2]}); at the "
    f"≤2-way verdict gm {f3(eps2_gm)}, worst {f3(eps2_worst[1])} — the 1.09 outlier was an "
    f"8-concurrency artifact. The hole (4-16k, {len(hole)} cells): pr {f3(hole_pr)} → bx "
    f"{f3(hole_bx)} (verdict batches {f3(vhole_bx)}). <b>PARITY DELIVERED.</b>")}
{bi(f"<b>真发现:纯 N 阈值 dispatch 退化为 always-bx({f3(comp_bx)})— pr 在任何 N 带上"
    f"都不净胜 sglang。</b> pr 的残余优势是一个连贯的 (N,BS) 区域:N≥65536 且 BS∈[32,128]"
    f"(GVR 中批量谷,{n_routed} 个路由 cell,其中 {len(pr_wins)} 个 pr 实胜 bx,幅度至 "
    f"{f3(max(x for _, x in pr_wins))}×)。规则对比(复合 gm vs sglang):always_bx "
    f"{f3(comp_bx)} | R1 {f3(comp_r1)} | 双臂 oracle {f3(comp_oracle)}。判决级复测 6/6 批:"
    "漂移 med 1.000 / p95 1.028,screening 确认。",
    f"<b>The real discovery: a pure N-threshold dispatch DEGENERATES to always-bx "
    f"({f3(comp_bx)}) — pr beats sglang on no N-band geomean.</b> pr's residual advantage is a "
    f"coherent (N,BS) REGION: N≥65536 & BS∈[32,128] (the GVR mid-BS valley; {n_routed} routed "
    f"cells, {len(pr_wins)} of them genuine pr-over-bx wins up to "
    f"{f3(max(x for _, x in pr_wins))}×). Rule comparison (composite gm vs sglang): always_bx "
    f"{f3(comp_bx)} | R1 {f3(comp_r1)} | two-arm oracle {f3(comp_oracle)}. Verdict re-run 6/6 "
    "batches: drift med 1.000 / p95 1.028 — screening confirmed.")}

{h2("4 · Track A2 — distP4 (iter6)", "4 · Track A2 — distP4 (iter6)")}
{bi("op35 归因把 P4blk(握手#2+P4+写回)判为中位 37% 的最大战场;A2 = 分布式 P4:杀 "
    "handoff2 值搬运,DSMEM red/atom 原语,直方图构建 + scatter 全 cluster 化,leader 标量"
    "搜索,罕见歧义回退 gather 保精确(TRACKA2_DESIGN.md 钉死到行号后由子代理实现,gvrpkg36 "
    "+758 行,battery_a2 29/29 一次编译通过)。",
    "op35 attribution named P4blk (handoff2+P4+writeback, median 37%) the biggest "
    "battleground; A2 = distributed P4: kill the handoff2 value-ship, DSMEM red/atom "
    "primitives, cluster-wide histogram build + scatter, leader scalar searches, and a rare-"
    "ambiguity gather fallback to preserve exactness (TRACKA2_DESIGN.md pinned to line level, "
    "agent-implemented; gvrpkg36 +758 lines, battery_a2 29/29 on first compile).")}
{bi(f"判决级测量(9 路由区批,≤2 并发,{len(a2_att)} cells 直接判决级):效果<b>形状连贯</b> — "
    f"大 N 赢(flash/1024k pr/a2 至 {f3(a2_win_max)},BS≥128 达 1.30-1.41;v32/256k 至 "
    f"1.19),N≤131k 输(路由区最差 {f3(a2_loss_min)}):6 次 cluster release-sync 的固定税在 "
    "~160k 候选规模以下压过分摊收益 — 与 A0 skip_h1 的门形状同构(K512 大 N + K2048 赢,"
    "K1024 关)。ship 追加:dist_p4 仅在路由 cell 的 {K512@N≈262k, K2048@N≈164k} 类形状开启"
    "(6 cells,全部同时胜 pr 与 bx,如 flash/1024k/BS128 1.047→1.361)。",
    f"Verdict-grade measurement (9 routed-region batches, ≤2-way, {len(a2_att)} cells at "
    f"verdict grade directly): the effect is <b>shape-coherent</b> — wins at large N "
    f"(flash/1024k pr/a2 up to {f3(a2_win_max)}, BS≥128 at 1.30-1.41; v32/256k up to 1.19), "
    f"loses at N≤131k (worst routed {f3(a2_loss_min)}): the fixed tax of 6 cluster "
    "release-syncs beats the distributed-work win below ~160k candidate scale — isomorphic to "
    "A0's skip_h1 gate shape (K512-large + K2048 win, K1024 off). Ship addition: dist_p4 ON "
    "only at routed cells of the {K512@N≈262k, K2048@N≈164k} shape classes (6 cells, each "
    "beating BOTH pr and bx, e.g. flash/1024k/BS128 1.047→1.361).")}
<details><summary>{bi("dp4-active 逐 cell 表", "dp4-active per-cell table", "span")}</summary>
{a2_table()}</details>

{h2("5 · Pivot 门:1.10 不可达的双重锁定", "5 · The pivot gate: 1.10 double-locked unreachable")}
{bi(f"门在 A2 上硅前触发(iter5,算术):Track B 后所有 A 杠杆只作用于 pr,而 pr 只路由 "
    f"{n_routed}/{len(full)} cells。在实测 b_screen 网格上做天花板重算(oracle 双臂 + 杠杆"
    f"全铺,含已证伪区域 — 刻意宽松):ship {f3(comp_r1)} | pr×1.10 → {f3(ceil[1.10])} | "
    f"pr×1.25 → {f3(ceil[1.25])} | pr×1.578(P4blk 归零,物理不可能)→ {f3(ceil[1.578])}。"
    "跨 1.10 需 pr 全域 ×~1.4 — 落在\"现实\"与\"物理不可能\"之间,且逐 cell 胜 sglang 已被三重"
    "证伪。用户裁决:跑完 A2 再收口。",
    f"The gate fired before A2 silicon (iter5, arithmetic): after Track B every A-track lever "
    f"acts only on pr, which the ship table routes at {n_routed}/{len(full)} cells. Ceiling "
    "recompute on the measured b_screen grid (two-arm oracle + levers applied everywhere, "
    f"including falsified regions — deliberately generous): ship {f3(comp_r1)} | pr×1.10 → "
    f"{f3(ceil[1.10])} | pr×1.25 → {f3(ceil[1.25])} | pr×1.578 (zero-P4blk, physically "
    f"impossible) → {f3(ceil[1.578])}. Crossing 1.10 needs pr ×~1.4 uniform — between "
    "realistic and physically impossible, and beating sglang per-cell is triple-falsified. "
    "User decision: run A2, then close.")}
{bi(f"<b>门被实测确认:</b>最大已知杠杆 distP4 全量落地后复合仅 +0.2%({f3(comp_r1)} → "
    f"{f3(comp_ship)});1.03-1.07 天花板成立,1.10 不可达。A1(admission escape)与 A3"
    "(C>8 多 CTA)按门算术放弃 — 两者都只作用于 pr 路由区,上限已被 ×1.25 行覆盖。",
    f"<b>The gate is empirically confirmed:</b> the biggest known lever, distP4, fully landed, "
    f"moved the composite +0.2% ({f3(comp_r1)} → {f3(comp_ship)}); the 1.03-1.07 ceiling "
    "stands and 1.10 is unreachable. A1 (admission escape) and A3 (multi-CTA C>8) dropped per "
    "the gate arithmetic — both act only on the pr-routed region, whose limit the ×1.25 row "
    "already covers.")}

{h2("6 · 最终 ship 表与全网格结果", "6 · Final ship table & full-grid results")}
{bi("三臂 shape 路由(全部 key 为推理期可知形状;命中率永不参与 dispatch):",
    "Three-arm shape-keyed dispatch (all keys are inference-known shapes; hit-rate never "
    "participates):")}
<table><tr><th>{'/'.join(['route'])}</th><th>condition (shape keys)</th><th>arm</th></tr>
<tr><td>1</td><td>N ≥ 65536 AND 32 ≤ BS ≤ 128 AND (K512@N≥262k or K2048@N≥164k)</td><td><b>gvr_a2 (dist_p4 + A0 flags)</b></td></tr>
<tr><td>2</td><td>N ≥ 65536 AND 32 ≤ BS ≤ 128 (else)</td><td><b>gvr_pr + A0 flags</b> (skip_h1 per §2 table, kb512@K2048)</td></tr>
<tr><td>3</td><td>everything else (default)</td><td><b>sgl_bx</b> (exactness-ported sglang v2 + overflow escape)</td></tr></table>
{bi(f"复合演进(同节点 b200-093,{len(full)} cells):pr-only {f3(comp_pr)} → +Track B 路由 "
    f"{f3(comp_r1)} → +A2 {f3(comp_ship)}(三臂 oracle {f3(comp_oracle3)})。逐 (model,ISL) "
    "网格如下;绿底 = ship ≥ 1.0。",
    f"Composite evolution (same-node b200-093, {len(full)} cells): pr-only {f3(comp_pr)} → "
    f"+Track B dispatch {f3(comp_r1)} → +A2 {f3(comp_ship)} (3-arm oracle {f3(comp_oracle3)}). "
    "Per-(model, ISL) grid below; green = ship ≥ 1.0.")}
{grid_table()}

{h3("6a · Seq-len sweep (BS=1) — 绝对时延对比", "6a · Seq-len sweep (BS=1) — absolute latency")}
{bi("BS=1 时 ship 表全部路由到 sgl_bx(BS∉[32,128]),因此 ship 线贴着 sglang_v2 —— 只差 "
    "0.4-0.8% 守卫税;gvr_pr 单臂在 BS=1 全线更慢(1-CTA 骨架启动地板),这正是 Track B "
    "存在的理由。",
    "At BS=1 the ship table routes everything to sgl_bx (BS∉[32,128]), so the ship line "
    "hugs sglang_v2 — separated only by the 0.4-0.8% guard tax; gvr_pr alone is slower across "
    "the whole sweep at BS=1 (the 1-CTA skeleton launch floor), which is exactly why Track B "
    "exists.")}
{legend3()}
{chart_seqlen()}

{h3("6b · BS-scaling — 对 sglang_v2 的加速比 (每 BS 沿 ISL 取 gm)",
    "6b · BS-scaling — speedup vs sglang_v2 (gm over ISLs per BS)")}
{bi("中批量谷清晰可见:gvr_pr 线在 BS 32-128 明显上抬(flash/pro 0.80-0.82,v32 0.89-0.93 "
    "—— sglang cluster 池在此饱和退化),但因混入小 N cell 整体仍 <1.0;ship 线只在 N≥65536 "
    "路由 pr(+A2),把谷区推过 1.0(flash 1.04-1.08 / pro 1.07-1.09 / v32 1.10-1.13),"
    "其余 BS 段经 sgl_bx 贴平 1.0,全轴不低于 0.986。这就是 §3b \"(N,BS) 区域而非 N 带\" "
    "的图形形态:凸起属于 (大 N × 中 BS) 交集,不属于任何单轴。",
    "The mid-BS valley is plainly visible: the gvr_pr line lifts sharply at BS 32-128 "
    "(flash/pro 0.80-0.82, v32 0.89-0.93 — where sglang's cluster pool saturates and "
    "degrades) yet stays <1.0 overall because small-N cells drag the geomean; the ship line "
    "routes pr (+A2) only at N≥65536, pushing the valley over 1.0 (flash 1.04-1.08 / pro "
    "1.07-1.09 / v32 1.10-1.13) while hugging 1.0 via sgl_bx at every other BS — never below "
    "0.986 across the axis. This is the graphical form of §3b's \"an (N,BS) region, not an "
    "N-band\": the bump lives in the (large-N × mid-BS) intersection, not on either axis "
    "alone.")}
{legend3()}
{chart_bs()}

{h3("6c · hit 分域统计 (hit>0.4 vs hit≤0.4)", "6c · Hit-domain split (hit>0.4 vs hit≤0.4)")}
{bi("事后统计视角(hit 是 capture 批 (model,ISL) 的属性,同批各 BS 相同;ship 表从不读 "
    "hit — 红线)。两个 hit 域的 ship 复合都 ≥1.0:不存在按 hit 分域后被翻盘的隐藏区域。",
    "Post-hoc view only (hit is a property of the capture batch (model,ISL), identical "
    "across BS; the ship table never reads hit — red line). The ship composite is ≥1.0 in "
    "BOTH hit domains: there is no hidden region that flips once you condition on hit.")}
{hit_table()}
{bi("解读注意:pr 单臂在 hit>0.4 域更差(0.684 vs 0.783)主要是 <b>N 结构混杂</b> — 高 hit "
    "批显著偏向小 N(flash 4k/32k/128k、pro 4k-32k 都在此域,而 flash 512k hit=0.057 这类"
    "大 N 低 hit 批在另一域),小 N 恰是 4-16k 洞。这不能读成 \"hit 高对 GVR 不利\" 的因果 — "
    "op30 对固定形状的结论方向相反(GVR 快在低 hr、慢在高 hr 是控制形状后的效应)。v32 全部 "
    "7 批 hit 均 >0.4,故 hit≤0.4 域无 v32 切片。",
    "Interpretation caveat: pr-alone being worse in the hit>0.4 domain (0.684 vs 0.783) is "
    "mostly an <b>N-structure confound</b> — high-hit batches skew strongly toward small N "
    "(flash 4k/32k/128k and pro 4k-32k sit in this domain, while large-N low-hit batches like "
    "flash 512k hit=0.057 sit in the other), and small N is exactly the 4-16k hole. Do NOT "
    "read it as causal \"high hit hurts GVR\" — op30's fixed-shape finding points the other "
    "way (GVR fast at low hr, slow at high hr, once shape is controlled). All 7 v32 batches "
    "have hit>0.4, so the hit≤0.4 domain has no v32 slice.")}
<details><summary><span class="zh">逐批 hit 值</span><span class="en">per-batch hit values</span></summary>
{hit_batch_table()}</details>

{h2("7 · Ship 方案算法:流程图与伪代码", "7 · The ship scheme: flow diagram & pseudocode")}
{bi("下图为最终推荐方案(§6 三臂路由)的端到端算法流程;三臂输出契约一致(每行精确 top-K "
    "索引),差异只在 kernel 内部。结构与 gvrpkg36 内核实现逐相位对齐"
    "(GvrTopKKernel docstring / TRACKA2_DESIGN.md SYNC1-6 / "
    "topk_v2_exact_standalone.cu impl 族)。",
    "The end-to-end algorithm of the final recommendation (the §6 three-arm dispatch); all "
    "three arms share the output contract (exact per-row top-K indices) and differ only "
    "inside the kernel. The structure is phase-aligned with the gvrpkg36 implementation "
    "(GvrTopKKernel docstring / TRACKA2_DESIGN.md SYNC1-6 / topk_v2_exact_standalone.cu "
    "impl families).")}
{FLOW_HTML}
{h3("伪代码", "Pseudocode")}
{PSEUDO_HTML}
{bi("为何这套组合胜过 sglang_v2:小 N / 极端 BS 段用的就是 sglang 自己的骨架(8-CTA MLP,"
    "启动地板 4.7-6.7µs,GVR 1-CTA 骨架的 ~9.7µs 地板在此无解),只付 0.4-0.8% 守卫税换回"
    "无条件精确;中批量谷(N≥65536, BS 32-128)sglang 的 cluster 池饱和退化,而 GVR 借 "
    "preIdx 时间先验把全行读降到 2 遍 + 单遍多阈值准入,快 1.05-1.57×;大 N 大 K 处 P4 "
    "串行 leader 成为瓶颈(op35 归因中位 37%),dist_p4 把直方图与 scatter 摊到全 cluster,"
    "再赢至 1.36×。",
    "Why this combination beats sglang_v2: at small N / extreme BS the dispatch uses sglang's "
    "OWN skeleton (8-CTA MLP, 4.7-6.7µs launch floor — unbeatable by the GVR 1-CTA skeleton's "
    "~9.7µs floor), paying only the 0.4-0.8% guard tax to buy back unconditional exactness; "
    "in the mid-BS valley (N≥65536, BS 32-128) sglang's cluster pool saturates and degrades, "
    "while GVR exploits the preIdx temporal prior to get away with 2 full-row reads + a "
    "single-pass multi-threshold admission — 1.05-1.57× faster; at large N·K the serial "
    "leader P4 becomes the bottleneck (op35 attribution: median 37%), and dist_p4 spreads "
    "the histogram + scatter across the whole cluster for a further win up to 1.36×.")}

{h2("8 · 证伪与学习台账", "8 · Falsification & learnings ledger")}
<ul>
<li>{bi("纯 N 阈值 dispatch:退化为 always-bx — pr 无任何净胜 N 带;残余优势是 (N,BS) 区域。",
        "Pure N-threshold dispatch: degenerates to always-bx — pr wins no N-band; the residual "
        "advantage is an (N,BS) region.", "span")}</li>
<li>{bi("skip_h1@K1024:pro/512k 0.72-0.91 单向回退(暖命中大 N)— 必须 per-K 门;op35 BS=1 "
        "\"0 lost\" 是范围限定教训,判决必须覆盖全 BS 轴。",
        "skip_h1@K1024: pro/512k 0.72-0.91 one-signed regression (warm-hit large-N) — must be "
        "per-K gated; op35's BS=1 \"0 lost\" is a scoping lesson: verdicts must cover the full "
        "BS axis.", "span")}</li>
<li>{bi("distP4@N≤131k:6 次 cluster sync 固定税 > 分摊收益,边界 ≈160k 候选规模;17 个路由"
        "回退至 1.40× — 只能 shape 门收窄至大 N。",
        "distP4@N≤131k: the 6-cluster-sync fixed tax beats the distributed-work win; boundary "
        "≈160k candidate scale; 17 routed regressions up to 1.40× — shipped only shape-gated "
        "to large N.", "span")}</li>
<li>{bi("8 并发 screening 伪造 ±10-15% 离群(pro/16k/BS1 ε 1.091 → ≤2 并发下 0.93-1.02)— "
        "反模式 #16 第三次实证;screening 永不作 ship 判决。",
        "8-concurrent screening fabricates ±10-15% outliers (pro/16k/BS1 ε 1.091 → 0.93-1.02 "
        "at ≤2-way) — anti-pattern #16 confirmed a third time; screening is never a ship "
        "verdict.", "span")}</li>
<li>{bi("冻结 shape 的 GVR 构建 anchor 漂移 p95 1.93 — 所有 GVR 臂必须走生产 launch 契约;"
        "anchor 校验在任何 sweep 的 ~1/3 处强制执行。",
        "Frozen-shape GVR builds drift anchors to p95 1.93 — every GVR arm must use the "
        "production launch contract; anchor validation is mandatory at ~1/3 of any sweep.",
        "span")}</li>
<li>{bi("1.10 目标:算术 + 最大杠杆实测双重锁死不可达(§5);A1/A3 按门放弃,未烧硅。",
        "The 1.10 target: double-locked unreachable by arithmetic + the biggest lever measured "
        "(§5); A1/A3 dropped at the gate without burning silicon.", "span")}</li>
</ul>

{h2("9 · 流程台账", "9 · Process ledger")}
<ol>
<li>{bi("iter0 骨架 + 算术定界(PLAN.md 红线/pivot 门/测量纪律先行)",
        "iter0 skeleton + bound arithmetic (PLAN.md red lines / pivot gate / measurement "
        "discipline first)", "span")}</li>
<li>{bi("iter1-3 A0:冻结 shape 作废 → launch 契约重跑 → 25 批 8-way screening → 6 极点批 "
        "≤2-way 判决 → shape 门旗标表",
        "iter1-3 A0: frozen-shape invalidated → launch-contract re-run → 25-batch 8-way "
        "screening → 6-pole-batch ≤2-way verdict → shape-gated flag table", "span")}</li>
<li>{bi("iter4 Track B:设计钉死(TRACKB_DESIGN.md)→ 移植 + 4 截断点守卫 → 电池 93/93 + "
        "TEETH → 25 批 screening + 6 批判决 → 2245 级 ship 闸 → (N,BS) 路由",
        "iter4 Track B: design pinned (TRACKB_DESIGN.md) → port + 4-site guard → battery 93/93 "
        "+ TEETH → 25-batch screening + 6-batch verdict → 2245-class ship gate → (N,BS) "
        "dispatch", "span")}</li>
<li>{bi("iter5 pivot 门(纯算术,零硅耗)→ 用户裁决 A2-then-close",
        "iter5 pivot gate (pure arithmetic, zero silicon) → user decision A2-then-close",
        "span")}</li>
<li>{bi("iter6 A2:设计钉死(TRACKA2_DESIGN.md)→ 子代理实现 → battery 29/29 → 9 批直接"
        "判决级测量 → shape 门 micro-ship → 战役收口",
        "iter6 A2: design pinned (TRACKA2_DESIGN.md) → agent implementation → battery 29/29 → "
        "9 batches measured at verdict grade directly → shape-gated micro-ship → campaign "
        "close", "span")}</li>
</ol>
{bi("节点:umbriel-b200-047(iter0-3)→ b200-093(iter4-6,047 失联);复合值只做同节点比,"
    "跨节点仅经 per-batch anchor 转移(pr/sgl med 全绿)。16 commits,单日战役,全部结果 "
    "cell-resumable。",
    "Nodes: umbriel-b200-047 (iter0-3) → b200-093 (iter4-6 after 047 went away); composites "
    "are same-node only, cross-node claims only via per-batch anchor transfer (pr/sgl medians "
    "all green). 16 commits, single-day campaign, all results cell-resumable.")}

{h2("10 · 生产移植方案(摘要)", "10 · Production-port plan (summary)")}
{bi("完整方案见 <code>PROD_PORT_PLAN.md</code>。要点:① sgl_bx 作为新 CUDA op 进 "
    "gvrpkg(vendored sglang v2 内核 + 4 点溢出守卫 + plan 融合清零),escape 走既有 "
    "radix_cutedsl;② shape 路由进 GVR host 端 pick_config 层(3 条规则,全部编译期/启动期"
    "可知 key);③ A0 旗标 + dist_p4 已在 gvrpkg36 以默认关旗标存在,按 §6 表在 dispatch 内"
    "定值;④ upstream PR 建议拆两个:PR-A(sgl_bx op + dispatch,复合 +41%,含精确性电池)"
    "与 PR-B(dist_p4,+0.2%,可选)。License 注意:vendored sglang 源为 Apache-2.0,"
    "保留原始头 + NVIDIA 修改注记。",
    "Full plan in <code>PROD_PORT_PLAN.md</code>. Key points: ① sgl_bx enters gvrpkg as a new "
    "CUDA op (vendored sglang v2 kernels + 4-site overflow guard + plan-fused flag zeroing); "
    "escape reuses the existing radix_cutedsl; ② the shape dispatch lands in the GVR host-side "
    "pick_config layer (3 rules, all compile/launch-time-known keys); ③ A0 flags + dist_p4 "
    "already exist in gvrpkg36 as default-off flags, pinned per the §6 table inside the "
    "dispatch; ④ upstream as two PRs: PR-A (sgl_bx op + dispatch, +41% composite, with the "
    "exactness batteries) and PR-B (dist_p4, +0.2%, optional). License note: the vendored "
    "sglang source is Apache-2.0 — keep original headers + NVIDIA modification notices.")}

<p class="meta">generated by analysis/gen_report_op36.py · sources: results/{{baseline_real_bs.csv,
a0_screen, b_screen, b_verdict, a2_verdict}} · batteries: src/trackb/{{battery_bx.py,
bx_topk_correctness.py, bx_2245_battery.log}}, variant/battery_a2.py · design docs:
TRACKB_DESIGN.md, TRACKA2_DESIGN.md · canonical numbers: ITERATIONS.md</p>
</div></body></html>
"""

out = _OP36 / "REPORT.html"
out.write_text(HTML)
print(f"wrote {out} ({len(HTML)} bytes)")
print(f"  baseline gm {base_gm_all:.3f} bands "
      + " ".join(f"{b}={base_band[b]:.3f}" for b in BANDS))
print(f"  a0 overall pr/a0 gm {a0_overall:.3f}")
print(f"  eps gm {gm([e for _, e in eps_all]):.3f} eps2 gm {eps2_gm:.3f} "
      f"worst2 {eps2_worst[1]:.3f}")
print(f"  hole pr {hole_pr:.3f} -> bx {hole_bx:.3f} (verdict {vhole_bx:.3f})")
print(f"  composites: pr {comp_pr:.3f} bx {comp_bx:.3f} R1 {comp_r1:.3f} "
      f"oracle2 {comp_oracle:.3f} SHIP {comp_ship:.3f} oracle3 {comp_oracle3:.3f}")
print(f"  routed {n_routed} cells, pr-wins {len(pr_wins)}, a2 cells folded {n_a2_cells}")
print(f"  ceiling: x1.10 {ceil[1.10]:.3f} x1.25 {ceil[1.25]:.3f} x1.578 {ceil[1.578]:.3f}")
print(f"  a2 cells n={len(a2_att)} win-max {a2_win_max:.3f} "
      f"routed-loss-min {a2_loss_min:.3f}")
