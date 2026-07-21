# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotent §6 Final-verdict injector for KF_PROCESS_LOG.html.

Computes the ship kernel's (c74f_sbx, grid_c74fsbx.csv) per-ISL geomeans vs
PR head and vs the three REPORT rival arms (PR-arm-normalized), then renders:
  - KPI hero row
  - Chart E1: geomean speedup by ISL, all models pooled (4 toggleable series)
  - Chart E2: per-model small multiples (model + series checkboxes)
  - summary table (25 model x ISL groups) in <details>
CSS-only interactivity (checkbox ~ sibling); native SVG <title> tooltips;
no <script>. Palette = dataviz reference categorical slots 1-4.
"""
import collections
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
HTML = HERE / "KF_PROCESS_LOG.html"
MARK_S, MARK_E = "<!-- KF-FINAL:START -->", "<!-- KF-FINAL:END -->"

SERIES = [  # key, css class, label, color(light)
    ("xPR",  "sPR",  "vs GVR PR head",   "#2a78d6"),
    ("xSGL", "sSGL", "vs sglang v2",     "#008300"),
    ("xRDX", "sRDX", "vs radix_cutedsl", "#e87ba4"),
    ("xFI",  "sFI",  "vs flashinfer",    "#eda100"),
]
ISLS = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]
MODELS = ["flash", "pro", "v32"]


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v else None


def load():
    riv = collections.defaultdict(dict)
    for r in csv.DictReader(open(REPORT / "rival_layers_full.csv")):
        riv[r["op"]][f"{r['model']}_{r['isl']}_L{int(r['L']):02d}"] = float(r["us"])
    rep_pr = {f"{r['model']}_{r['isl']}_L{int(r['layer']):02d}": float(r["pr"])
              for r in csv.DictReader(open(REPORT / "real_3arm_layers_full.csv"))}
    cells = {}
    for r in csv.DictReader(open(HERE / "grid_c74fsbx.csv")):
        u = r["uuid"]
        m, isl, _ = u.split("_")
        pk, ck = float(r["pr_cold"]), float(r["cand_cold"])
        d = {"model": m, "isl": isl, "xPR": float(r["speedup_cold"])}
        for op, key in [("sglang_v2", "xSGL"), ("radix_cutedsl", "xRDX"),
                        ("flashinfer_topk", "xFI")]:
            if u in riv[op] and u in rep_pr:
                d[key] = (riv[op][u] / ck) * (pk / rep_pr[u])
        cells[u] = d
    return cells


def series_points(cells, model=None):
    """-> {seriesKey: [(isl_index, gm), ...]}"""
    out = {}
    for key, *_ in SERIES:
        pts = []
        for i, isl in enumerate(ISLS):
            v = [c[key] for c in cells.values()
                 if c["isl"] == isl and key in c and (model is None or c["model"] == model)]
            if v:
                pts.append((i, gm(v)))
        out[key] = pts
    return out


def chart(pts_by_series, cid, w=760, h=330, ymin=0.5, ymax=2.7, title=""):
    lpad, rpad, tpad, bpad = 52, 16, 26, 40
    pw, ph = w - lpad - rpad, h - tpad - bpad
    def X(i): return lpad + pw * i / (len(ISLS) - 1)
    def Y(v): return tpad + ph * (1 - (v - ymin) / (ymax - ymin))
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}" '
         f'style="max-width:100%;background:var(--vz-surface);border:1px solid var(--vz-grid);border-radius:8px">']
    # grid + y labels
    for gv in [0.5, 1.0, 1.5, 2.0, 2.5]:
        y = Y(gv)
        em = ' stroke-width="1.6"' if gv == 1.0 else ' stroke-width="1"'
        col = "var(--vz-ref)" if gv == 1.0 else "var(--vz-grid)"
        s.append(f'<line x1="{lpad}" y1="{y:.1f}" x2="{w-rpad}" y2="{y:.1f}" stroke="{col}"{em}/>')
        s.append(f'<text x="{lpad-8}" y="{y+4:.1f}" font-size="11" text-anchor="end" '
                 f'fill="var(--vz-text2)" font-family="sans-serif">{gv:.1f}×</text>')
    for i, isl in enumerate(ISLS):
        s.append(f'<text x="{X(i):.1f}" y="{h-bpad+18}" font-size="11" text-anchor="middle" '
                 f'fill="var(--vz-text2)" font-family="sans-serif">{isl}</text>')
    s.append(f'<text x="{lpad+pw/2}" y="{h-6}" font-size="11" text-anchor="middle" '
             f'fill="var(--vz-text2)" font-family="sans-serif">ISL (sequence length)</text>')
    # series
    for key, cls, label, col in SERIES:
        pts = pts_by_series.get(key) or []
        if not pts:
            continue
        poly = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in pts)
        g = [f'<g class="{cls}">',
             f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2"/>']
        for i, v in pts:
            g.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{col}" '
                     f'stroke="var(--vz-surface)" stroke-width="2">'
                     f'<title>{label} @ {ISLS[i]}: {v:.3f}×</title></circle>')
        li, lv = pts[-1]
        g.append(f'<text x="{X(li)+8:.1f}" y="{Y(lv)+4:.1f}" font-size="11" '
                 f'fill="var(--vz-text1)" font-family="sans-serif">{label.replace("vs ","")}</text>')
        g.append("</g>")
        s.append("".join(g))
    s.append("</svg>")
    return "".join(s)


def main():
    cells = load()
    n = len(cells)
    xpr = [c["xPR"] for c in cells.values()]
    kpi_pr = gm(xpr)
    kpi = {k: gm([c[k] for c in cells.values() if k in c]) for k, *_ in SERIES}

    # ---- controls CSS ----
    css = ["<style>.vizE{--vz-surface:#fcfcfb;--vz-grid:#e4e3df;--vz-ref:#8b8a85;"
           "--vz-text1:#0b0b0b;--vz-text2:#52514e;margin:14px 0}",
           ".vizE .chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}",
           ".vizE .chips label{border:1.5px solid #c9c8c2;border-radius:16px;padding:3px 12px;"
           "cursor:pointer;font-size:0.85em;color:#52514e;user-select:none}",
           ".vizE .chips label b{font-weight:600}",
           ".vizE input{position:absolute;opacity:0;pointer-events:none}"]
    for pfx in ("ck", "ck2"):
        for key, cls, _, col in SERIES:
            css.append(f".vizE #{pfx}-{cls}:not(:checked) ~ * .{cls}{{display:none}}")
            css.append(f".vizE #{pfx}-{cls}:checked ~ .chips label[for={pfx}-{cls}]"
                       f"{{border-color:{col};color:#0b0b0b;box-shadow:inset 0 0 0 1px {col}}}")
            css.append(f".vizE .chips label[for={pfx}-{cls}] i{{display:inline-block;width:10px;height:10px;"
                       f"border-radius:5px;background:{col};margin-right:6px}}")
    for m in MODELS:
        css.append(f".vizE #ck-m-{m}:not(:checked) ~ * .sm-{m}{{display:none}}")
        css.append(f".vizE #ck-m-{m}:checked ~ .chips label[for=ck-m-{m}]"
                   f"{{border-color:#555;color:#0b0b0b}}")
    css.append(".vizE .smrow{display:flex;gap:10px;flex-wrap:wrap}"
               ".vizE .smrow figure{flex:1 1 340px;margin:0}</style>")
    css = "".join(css)

    inputs = "".join(f'<input type="checkbox" id="ck-{cls}" checked>' for _, cls, _, _ in SERIES)
    inputs2 = "".join(f'<input type="checkbox" id="ck2-{cls}" checked>' for _, cls, _, _ in SERIES)
    minputs = "".join(f'<input type="checkbox" id="ck-m-{m}" checked>' for m in MODELS)
    chips = ('<div class="chips">'
             + "".join(f'<label for="ck-{cls}"><i></i><b>{lbl}</b></label>'
                       for _, cls, lbl, _ in SERIES) + "</div>")
    mchips = ('<div class="chips">series ▸ '
              + "".join(f'<label for="ck2-{cls}"><i></i><b>{lbl}</b></label>'
                        for _, cls, lbl, _ in SERIES)
              + " &nbsp;model ▸ "
              + "".join(f'<label for="ck-m-{m}"><b>{m}</b></label>' for m in MODELS)
              + "</div>")

    e1 = chart(series_points(cells), "e1",
               title="c74f_sbx geomean speedup by ISL, all models")
    sm = []
    for m in MODELS:
        sm.append(f'<figure class="sm-{m}">'
                  + chart(series_points(cells, m), f"sm{m}", w=430, h=280,
                          title=f"{m} geomean speedup by ISL")
                  + f'<figcaption style="font-size:0.85em;color:#52514e;text-align:center">{m}</figcaption></figure>')

    # ---- summary table ----
    groups = collections.defaultdict(list)
    for c in cells.values():
        groups[(c["model"], c["isl"])].append(c)
    rows = []
    for m in MODELS:
        for isl in ISLS:
            g = groups.get((m, isl))
            if not g:
                continue
            cols = [gm([c[k] for c in g if k in c]) for k, *_ in SERIES]
            mn = min(c["xPR"] for c in g)
            rows.append(f"<tr><td>{m}_{isl}</td><td>{len(g)}</td>"
                        + "".join(f"<td>{v:.3f}</td>" if v else "<td>-</td>" for v in cols)
                        + f"<td>{mn:.3f}</td></tr>")
    table = ('<details style="margin:10px 0"><summary style="cursor:pointer">'
             'Table view — 25 model×ISL groups / 表格视图</summary>'
             '<table><tr><th>group</th><th>layers</th>'
             + "".join(f"<th>{lbl}</th>" for _, _, lbl, _ in SERIES)
             + "<th>min ×PR</th></tr>" + "".join(rows) + "</table></details>")

    kpis = (f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0">'
            + "".join(
        f'<div style="border:1px solid #ddd;border-radius:8px;padding:10px 18px;text-align:center">'
        f'<div style="font-size:1.5em;font-weight:700">{v}</div>'
        f'<div style="font-size:0.8em;color:#52514e">{k}</div></div>'
        for k, v in [
            ("geomean vs PR head", f"{kpi_pr:.4f}×"),
            ("vs sglang v2", f"{kpi['xSGL']:.3f}×"),
            ("vs radix_cutedsl", f"{kpi['xRDX']:.3f}×"),
            ("vs flashinfer", f"{kpi['xFI']:.3f}×"),
            ("exact", f"{n}/865"), ("cold regressions", "0"),
            ("campaign cost", "$690.81"), ("orchestrator cost", "~$60"),
            ("total LLM cost", "~$751"), ("wall", "~7.5 h")])
            + "</div>")

    body = f"""{MARK_S}{css}
<p><b>SHIP: <code>c74f_sbx</code></b> — campaign round-2 winner <code>c74fb3c0</code> (agent r002-a003)
+ engineer dispatch graft (<code>topk_small&lt;17&gt;&lt;&lt;&lt;1,1024&gt;&gt;&gt;</code> rung, 8448&lt;n≤16896).
Verdict grid: 865 real decode cells (BS=1), nsys cold-L2, paired same-GPU vs PR head @e6fdbfac3d; borderline cells adjudicated at 60 reps;
per-rung <code>pr_cold</code> anchors clean. Rival ratios are PR-arm-normalized joins against the REPORT rival sweep (calibration med 1.010).
Code pushed to <code>github.com/longcheng-nv/TensorRT-LLM</code> branch <code>kf/gvr-topk-c74fsbx</code>.
两个基准口径:campaign 内部 1.3385(含 ~15µs 评测地板)≠ 本地 nsys 核时间比 1.6828。</p>
{kpis}
<div class="vizE">{inputs}{chips}
<figure style="margin:8px 0">{e1}
<figcaption style="font-size:0.9em;color:#444"><b>Fig. E1 — c74f_sbx geomean speedup by ISL (865 cells, all models pooled).</b>
Toggle series with the chips; hover points for values. 1.0× line = parity.<br>
图 E1 — 按 ISL 的 geomean 加速比(勾选切换系列;悬停看数值;1.0× 为持平线)。</figcaption></figure></div>
<div class="vizE">{minputs}{inputs2}{mchips}
<div class="smrow">{''.join(sm)}</div>
<figcaption style="font-size:0.9em;color:#444"><b>Fig. E2 — per-model small multiples (flash K=512 · pro K=1024 · v32 K=2048).</b>
Model and series chips both filter.<br>图 E2 — 分模型小倍图(模型与系列复选框皆可过滤)。</figcaption></div>
{table}
<p><b>Verdict / 终审:</b> geomean <b>1.6828×</b> vs PR head (bar ≥1.20 ✅), <b>zero cold regressions</b> ✅, 865/865 exact ✅;
vs sglang v2 <b>{kpi['xSGL']:.3f}×</b> (first in-tree-family win on the full real envelope; residual sglang strongholds at 32k ISL are now shallow),
vs radix_cutedsl <b>{kpi['xRDX']:.3f}×</b>, vs flashinfer <b>{kpi['xFI']:.3f}×</b>.
Production-port caveat: worst/best synthetic axes per house ship discipline remain to be run; warm-axis is secondary for BS=1 decode (cold-L2 canonical).
Campaign cancelled 07-21 09:1x UTC after round-2 plateau; 2 rounds, 13 agents, ~30 candidates.</p>
<p><b>Cost accounting / 成本口径:</b> two independent meters. <b>Campaign side $690.81</b> (KF platform billing,
<code>kf campaign cost</code>: round 1 ≈ $458.24 / 13.26 agent-hours; round 2 ≈ $232.6, cancelled mid-round; 48% cache hit;
6 agents/round = 2×Fable-5(max) + 2×GPT-5.6-sol(xhigh) + 2×Opus-4.8, single agents reaching 30-47M input tokens).
<b>Orchestrator side ≈ $60</b> (this Claude Code session: harness authoring, harvest/verify loops, grids, analysis, report).
<b>Full-stack LLM cost ≈ $751</b> for the shipped 1.6828×/zero-regression kernel — 指挥部 ~$60 + 13 个云端 agent $690.81,两套账互相独立。</p>
{MARK_E}"""

    html = HTML.read_text()
    if MARK_S in html:
        html = html[:html.index(MARK_S)] + html[html.index(MARK_E) + len(MARK_E):]
    anchor = "<h2>6 · Final verdict</h2>"
    pend_start = html.index(anchor) + len(anchor)
    pend_end = html.index("</p>", pend_start) + 4
    html = html[:pend_start] + "\n" + body + html[pend_end:]
    HTML.write_text(html)
    print("final section injected,", len(body), "bytes")


if __name__ == "__main__":
    main()
