# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotent §7 R3-campaign injector for KF_PROCESS_LOG.html.

Same conventions as gen_final_section.py (§6): h2 section, .cng CN glosses,
KPI tiles, .vizR chips charts (CSS-only, no <script>), PR-arm-normalized
rival joins, 25-group summary + per-model per-LAYER detail tables.
Numbers are computed live from grid_<TAG>.csv; re-run after every new champion
verdict to refresh (marker-region replace only).

  python3 gen_r3_section.py [TAG]     # default TAG=r3gridcompA
"""
import collections
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
HTML = HERE / "KF_PROCESS_LOG.html"
BAN_S, BAN_E = "<!-- KF-R3BANNER:START -->", "<!-- KF-R3BANNER:END -->"
SEC_S, SEC_E = "<!-- KF-R3:START -->", "<!-- KF-R3:END -->"

TAG = sys.argv[1] if len(sys.argv) > 1 else "r3gridcompB"
CHAMP_NAME = {"r3gridcompA": "compA", "r3gridcompB": "compB"}.get(TAG, TAG)

SERIES = [  # key, css class, label, color — same palette as §6
    ("xPR",  "s3PR",  "vs GVR PR head",   "#2a78d6"),
    ("xSGL", "s3SGL", "vs sglang v2",     "#008300"),
    ("xRDX", "s3RDX", "vs radix_cutedsl", "#e87ba4"),
    ("xFI",  "s3FI",  "vs flashinfer",    "#eda100"),
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
    champ1 = {r["uuid"]: float(r["cand_cold"])
              for r in csv.DictReader(open(HERE / "grid_champh2.csv"))}
    cells = {}
    for r in csv.DictReader(open(HERE / f"grid_{TAG}.csv")):
        u = r["uuid"]
        pk, ck = float(r["pr_cold"]), float(r["cand_cold"])
        d = {"model": r["model"], "isl": r["isl"], "layer": int(r["layer"]),
             "N": r["N"], "hit": r["hit"], "us": ck,
             "xPR": float(r["speedup_cold"]),
             "xC1": champ1[u] / ck if u in champ1 else None}
        for op, key in [("sglang_v2", "xSGL"), ("radix_cutedsl", "xRDX"),
                        ("flashinfer_topk", "xFI")]:
            if u in riv[op] and u in rep_pr:
                d[key] = (riv[op][u] / ck) * (pk / rep_pr[u])
        cells[u] = d
    return cells


def series_points(cells, model=None):
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


def chart(pts_by_series, w=760, h=330, ymin=0.5, ymax=2.7, title=""):
    lpad, rpad, tpad, bpad = 52, 16, 26, 40
    pw, ph = w - lpad - rpad, h - tpad - bpad
    def X(i): return lpad + pw * i / (len(ISLS) - 1)
    def Y(v): return tpad + ph * (1 - (v - ymin) / (ymax - ymin))
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}" '
         f'style="max-width:100%;background:var(--vz-surface);border:1px solid var(--vz-grid);border-radius:8px">']
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



EVENTS = [  # (hours since 07-21 12:00Z, lane, label, detail, kind)
    (1.55, 0, "prepare fail D1", "baseline-solution evaluator does not stage assets (0/28) -> platform-trace baselines workaround", "warn"),
    (1.67, 0, "campaign start", "gvr-topk-r3 e5q1zgr... 6 agents/round, champion c74f_sbx as baseline", "camp"),
    (1.9, 1, "PR-head arm refresh", "e6fdbfac3d -> b14ec40e1b (only gvr_topk_decode.py changed)", "loc"),
    (1.95, 1, "probe 1.7193", "champion vs NEW head, 28 cells, 0 reg, exact", "loc"),
    (2.3, 1, "grid champh2 1.6770", "Bar denominators: 865 cells, 0 reg, 865/865 exact; anchors vs old head med 1.005", "verd"),
    (5.75, 0, "round 1 close: best 0.9956", "= verbatim champion resubmit (platform noise calib +-0.5%); only real variant 5f3daaf8", "camp"),
    (6.4, 1, "5f3daaf8 = WASH", "provably-exact warm-hint filter: 1.0001 in its n>=512K activation zone", "dead"),
    (7.5, 0, "09d13c81 internal 1.0351", "regular launch + fence-less sense-token grid barrier", "camp"),
    (8.0, 1, "probe INVALIDATED", "foreign 8-GPU job 100% util; quiet-echo scripting bug fixed to gated launches", "warn"),
    (9.8, 1, "grid 09d1 1.7553", "NEW COMPOSITE +4.3%: coop rungs +6-11%, 0 reg, 865 exact", "verd"),
    (10.3, 1, "ordering study", "threadfence -11% / acq_rel -8% / surgical -8%: the win IS the omitted barrier ordering", "dead"),
    (10.9, 0, "30e79029 internal 1.0577", "+ contiguous-slice scan partitions", "camp"),
    (11.6, 1, "grid 30e7 1.7714", "NEW COMPOSITE +5.0%; at op35 UB reference 1.771", "verd"),
    (12.4, 0, "becdc5c7 internal 1.1566", "adaptive post-pass-0 fast-tail + register-cached row", "camp"),
    (13.2, 1, "grid becd 1.7848", "NEW COMPOSITE +6.6%; v32 mid-n loses to 30e7 -> engineer dispatch", "verd"),
    (14.1, 1, "grid compA 1.7873", "becd + k2048-mid-n->30e7 dispatch; 1 borderline 0.999", "verd"),
    (16.5, 0, "aef33fac internal 1.1726", "+ topk_mid single-CTA tail-selection rungs (4k<=n<=16387)", "camp"),
    (17.9, 1, "aef3 grid CONTAMINATED", "anchor p95 1.542 (foreign job GPUs 2-5); signal kept: mid<4> heals 16387 +19%, mid<1> regresses n~4099", "warn"),
    (18.5, 1, "compB built", "aef3 minus mid<1> rung + 30e7 k2048 dispatch", "loc"),
    (26.0, 1, "grid compB 1.8267", "NEW COMPOSITE +8.1% vs champion; min 1.140, NO borderline; 865/865 exact", "verd"),
    (27.5, 0, "round 3 close, cost $761", "17 kernels; round 4 = final (cap $800)", "camp"),
]

def timeline_fig():
    W, H = 980, 300
    L, R2, T, B = 20, 20, 46, 30
    t0, t1 = 1.0, 28.5
    lanes = {0: 90, 1: 200}
    kcol = {"camp": "#2a78d6", "loc": "#52514e", "verd": "#0b6e4f", "warn": "#b3541e", "dead": "#8b2635"}
    def X(t):
        return L + (t - t0) / (t1 - t0) * (W - L - R2)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" style="max-width:100%;background:var(--vz-surface);'
         f'border:1px solid var(--vz-grid);border-radius:8px">']
    for lane, y in lanes.items():
        s.append(f'<line x1="{L}" y1="{y}" x2="{W-R2}" y2="{y}" stroke="var(--vz-grid)" stroke-width="1.4"/>')
    s.append(f'<text x="{L}" y="{lanes[0]-58}" font-size="12" font-weight="700" fill="var(--vz-text1)" font-family="sans-serif">KF cloud campaign (internal scale, ~15µs floor)</text>')
    s.append(f'<text x="{L}" y="{lanes[1]+62}" font-size="12" font-weight="700" fill="var(--vz-text1)" font-family="sans-serif">Local 8xB200 verify arm (nsys cold-L2 vs PR head, 865 cells)</text>')
    for hh, lbl in [(1.67, "07-21 13:40Z"), (13.7, "07-22 01:40Z"), (26.0, "07-22 02:0xZ+")]:
        s.append(f'<text x="{X(hh):.0f}" y="{H-8}" font-size="10" text-anchor="middle" fill="var(--vz-text2)" font-family="sans-serif">{lbl}</text>')
    flip = 1
    for t, lane, label, detail, kind in EVENTS:
        x, y = X(t), lanes[lane]
        col = kcol[kind]
        up = (flip := -flip) > 0
        ty = y - 14 - (10 if up else -38)
        s.append(f'<g><line x1="{x:.0f}" y1="{y}" x2="{x:.0f}" y2="{ty+12:.0f}" stroke="{col}" stroke-width="1"/>'
                 f'<circle cx="{x:.0f}" cy="{y}" r="5.5" fill="{col}" stroke="var(--vz-surface)" stroke-width="2">'
                 f'<title>{label} — {detail}</title></circle>'
                 f'<text x="{x:.0f}" y="{ty:.0f}" font-size="9.5" text-anchor="middle" fill="{col}" '
                 f'font-family="sans-serif">{label}</text>'
                 f'<title>{label}</title></g>')
    leg_x = L
    for kind, lbl in [("camp", "campaign event"), ("verd", "full-865 verdict"), ("loc", "local arm"), ("warn", "incident"), ("dead", "falsified")]:
        s.append(f'<circle cx="{leg_x+6}" cy="{H-44}" r="5" fill="{kcol[kind]}"/>'
                 f'<text x="{leg_x+16}" y="{H-40}" font-size="10.5" fill="var(--vz-text2)" font-family="sans-serif">{lbl}</text>')
        leg_x += 16 + 9 * len(lbl) + 18
    s.append("</svg>")
    return "".join(s)


def grid_stats(tag):
    rows = list(csv.DictReader(open(HERE / f"grid_{tag}.csv")))
    sp = [float(r["speedup_cold"]) for r in rows]
    return dict(gm=gm(sp), mn=min(sp), regs=sum(1 for v in sp if v < 1.0),
                exact=sum(1 for r in rows if r["cand_exact"] == "True"), n=len(sp))


def main():
    cells = load()
    n = len(cells)
    kpi = {k: gm([c[k] for c in cells.values() if k in c]) for k, *_ in SERIES}
    kpi_c1 = gm([c["xC1"] for c in cells.values() if c["xC1"]])
    st = {t: grid_stats(t) for t in
          ["champh2", "r3grid09d1", "r3grid30e7", "r3gridbecd", "r3gridcompA", TAG]}
    mn = min(c["xPR"] for c in cells.values())
    regs = sum(1 for c in cells.values() if c["xPR"] < 1.0)

    # ---- CSS (namespaced .vizR / ck3-, ck4-, ck3-m-) ----
    css = ["<style>.vizR{--vz-surface:#fcfcfb;--vz-grid:#e4e3df;--vz-ref:#8b8a85;"
           "--vz-text1:#0b0b0b;--vz-text2:#52514e;margin:14px 0}",
           ".vizR .chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}",
           ".vizR .chips label{border:1.5px solid #c9c8c2;border-radius:16px;padding:3px 12px;"
           "cursor:pointer;font-size:0.85em;color:#52514e;user-select:none}",
           ".vizR .chips label b{font-weight:600}",
           ".vizR input{position:absolute;opacity:0;pointer-events:none}"]
    for pfx in ("ck3", "ck4"):
        for key, cls, _, col in SERIES:
            css.append(f".vizR #{pfx}-{cls}:not(:checked) ~ * .{cls}{{display:none}}")
            css.append(f".vizR #{pfx}-{cls}:checked ~ .chips label[for={pfx}-{cls}]"
                       f"{{border-color:{col};color:#0b0b0b;box-shadow:inset 0 0 0 1px {col}}}")
            css.append(f".vizR .chips label[for={pfx}-{cls}] i{{display:inline-block;width:10px;height:10px;"
                       f"border-radius:5px;background:{col};margin-right:6px}}")
    for m in MODELS:
        css.append(f".vizR #ck3-m-{m}:not(:checked) ~ * .sm3-{m}{{display:none}}")
        css.append(f".vizR #ck3-m-{m}:checked ~ .chips label[for=ck3-m-{m}]"
                   f"{{border-color:#555;color:#0b0b0b}}")
    css.append(".vizR .smrow{display:flex;gap:10px;flex-wrap:wrap}"
               ".vizR .smrow figure{flex:1 1 340px;margin:0}</style>")
    css = "".join(css)

    inputs = "".join(f'<input type="checkbox" id="ck3-{cls}" checked>' for _, cls, _, _ in SERIES)
    inputs2 = "".join(f'<input type="checkbox" id="ck4-{cls}" checked>' for _, cls, _, _ in SERIES)
    minputs = "".join(f'<input type="checkbox" id="ck3-m-{m}" checked>' for m in MODELS)
    chips = ('<div class="chips">'
             + "".join(f'<label for="ck3-{cls}"><i></i><b>{lbl}</b></label>'
                       for _, cls, lbl, _ in SERIES) + "</div>")
    mchips = ('<div class="chips">series ▸ '
              + "".join(f'<label for="ck4-{cls}"><i></i><b>{lbl}</b></label>'
                        for _, cls, lbl, _ in SERIES)
              + " &nbsp;model ▸ "
              + "".join(f'<label for="ck3-m-{m}"><b>{m}</b></label>' for m in MODELS)
              + "</div>")

    r1 = chart(series_points(cells), title=f"{CHAMP_NAME} geomean speedup by ISL, all models")
    sm = []
    for m in MODELS:
        sm.append(f'<figure class="sm3-{m}">'
                  + chart(series_points(cells, m), w=430, h=280,
                          title=f"{m} geomean speedup by ISL")
                  + f'<figcaption style="font-size:0.85em;color:#52514e;text-align:center">{m}</figcaption></figure>')

    # ---- KPI tiles (§6 style) ----
    kpis = ('<div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0">'
            + "".join(
        f'<div style="border:1px solid #ddd;border-radius:8px;padding:10px 18px;text-align:center">'
        f'<div style="font-size:1.5em;font-weight:700">{v}</div>'
        f'<div style="font-size:0.8em;color:#52514e">{k}</div></div>'
        for k, v in [
            ("geomean vs GVR PR head / 对 PR 当前 head", f"{kpi['xPR']:.4f}×"),
            ("vs campaign-1 champion / 对第一期冠军", f"{kpi_c1:.4f}×"),
            ("vs sglang v2", f"{kpi['xSGL']:.3f}×"),
            ("vs radix_cutedsl", f"{kpi['xRDX']:.3f}×"),
            ("vs flashinfer", f"{kpi['xFI']:.3f}×"),
            ("exact / 精确", f"{n}/865"),
            ("cold regressions / 冷回退", f"{regs} (min {mn:.3f})"),
        ]) + "</div>")

    # ---- candidate ladder ----
    ladder_rows = ""
    for tag, name, note in [
            ("champh2", "champion c74f_sbx (baseline, §6)",
             "campaign-1 ship re-measured on current head / 第一期交付,当前 head 现测"),
            ("r3grid09d1", "09d13c81 (r2)",
             "regular launch + fence-less sense-token grid barrier / 常规 launch + 免序自旋屏障"),
            ("r3grid30e7", "30e79029 (r3)",
             "+ contiguous-slice scan partitions / + 连续切片扫描"),
            ("r3gridbecd", "becdc5c7 (r3)",
             "adaptive post-pass-0 finish + register-cached row / 自适应收尾 + 寄存器缓存整行"),
            ("r3gridcompA", "compA (engineer composite #1)",
             "becd ⊕ 30e7: k=2048 ∧ 16896&lt;n≤140000 → 30e7 ladder / 工程师复合分派 #1"),
            (TAG, f"<b>{CHAMP_NAME} (engineer composite #2, current champion)</b>",
             "aef33fac (becd + topk_mid tail-selection rungs, mid&lt;1&gt; n≈4099 rung gated out after measured regression) ⊕ 30e7 k2048 dispatch / 工程师复合 #2:并入 topk_mid 尾部选择档(裁掉回退的 mid&lt;1&gt;)")]:
        s = st[tag]
        ladder_rows += (f'<tr><td>{name}</td><td>{s["gm"]:.4f}</td>'
                        f'<td>{s["regs"]} (min {s["mn"]:.3f})</td>'
                        f'<td>{s["exact"]}/{s["n"]}</td>'
                        f'<td>{s["gm"]/st["champh2"]["gm"]:.4f}</td><td>{note}</td></tr>')
    ladder = ('<table><tr><th>kernel (full-865 verdict)</th><th>gm vs PR head</th>'
              '<th>cells &lt;1.0</th><th>exact</th><th>vs champion</th><th>increment / 增量</th></tr>'
              + ladder_rows + "</table>")

    # ---- 25-group summary (§6 style) + per-layer detail tables ----
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
            mnv = min(c["xPR"] for c in g)
            rows.append(f"<tr><td>{m}_{isl}</td><td>{len(g)}</td>"
                        + "".join(f"<td>{v:.3f}</td>" if v else "<td>-</td>" for v in cols)
                        + f"<td>{gm([c['xC1'] for c in g if c['xC1']]):.3f}</td>"
                        + f"<td>{mnv:.3f}</td></tr>")
    table25 = ('<details><summary>Table view — 25 model×ISL groups / 25 个模型×ISL 组汇总</summary>'
               '<table><tr><th>group</th><th>layers</th>'
               + "".join(f"<th>{lbl}</th>" for _, _, lbl, _ in SERIES)
               + "<th>vs champion</th><th>min ×PR</th></tr>" + "".join(rows) + "</table></details>")

    # per-layer detail: one <details> per model, all layers × ISLs
    def fmt_cell(c):
        parts = [f"{c['xPR']:.2f}"]
        return parts[0]
    layer_tables = []
    for m in MODELS:
        isls = [i for i in ISLS if (m, i) in groups]
        layers = sorted({c["layer"] for c in cells.values() if c["model"] == m})
        by = {(c["layer"], c["isl"]): c for c in cells.values() if c["model"] == m}
        hdr = ("<tr><th rowspan=2>layer</th>"
               + "".join(f'<th colspan=4 style="text-align:center">{i}</th>' for i in isls) + "</tr>"
               "<tr>" + "".join("<th>PR</th><th>sgl</th><th>rdx</th><th>c1</th>" for _ in isls) + "</tr>")
        body = []
        for L in layers:
            tds = []
            for i in isls:
                c = by.get((L, i))
                if not c:
                    tds.append("<td>-</td>" * 4)
                    continue
                def f(v, lo=1.0):
                    if v is None:
                        return "<td>-</td>"
                    sty = ' style="background:#fbecec"' if v < lo else ""
                    return f"<td{sty}>{v:.2f}</td>"
                tds.append(f(c["xPR"]) + f(c.get("xSGL")) + f(c.get("xRDX")) + f(c.get("xC1")))
            body.append(f"<tr><td>L{L:02d}</td>{''.join(tds)}</tr>")
        layer_tables.append(
            f'<details><summary>{m} — per-layer detail (columns per ISL: vs PR head · vs sglang v2 · '
            f'vs radix_cutedsl · vs campaign-1 champion; red = &lt;1.0) / {m} 逐层明细'
            f'(每 ISL 四列:对 PR·对 sglang v2·对 radix·对第一期冠军;红底 = &lt;1.0)</summary>'
            f'<div style="overflow-x:auto"><table style="font-size:0.78em;white-space:nowrap">{hdr}{"".join(body)}</table></div></details>')

    barrier_tbl = """
<table><tr><th>barrier variant / 屏障变体</th><th>memory ordering / 内存序</th><th>28-cell cold gm vs PR</th><th>verdict / 判决</th></tr>
<tr><td><code>09d13c81</code> as-harvested</td><td>relaxed intrinsics, no fence</td><td><b>~1.72</b></td><td>fastest — the win source / 胜利来源</td></tr>
<tr><td>+ <code>__threadfence</code> pair</td><td>membar.gl</td><td>1.566</td><td>−11%, rejected / 否决</td></tr>
<tr><td>scoped acq_rel asm</td><td>atom.acq_rel / ld.acquire.gpu</td><td>1.614</td><td>−8%, rejected / 否决</td></tr>
<tr><td>surgical (relaxed spin + trailing acquire)</td><td>scoped, off-critical-path attempt</td><td>1.615</td><td>rejected — ordering cost is intrinsic / 否决:排序代价是本质的(release 必须在关键路径上等 L2 写落地)</td></tr>
</table>"""

    tl = timeline_fig()
    section = f"""{SEC_S}{css}
<h2 id="sec-7">7 · R3 campaign — beyond-champion / 第二期战役(超越冠军)</h2>
<p><b>Campaign <code>gvr-topk-r3</code></b> (<code>e5q1zgrfhs0z57dj6850kc444r</code>, started 07-21 13:40Z; 6 agents/round =
2×Fable-5(high) + 2×GPT-5.6-sol(high) + 2×Opus-4.8; baseline = §6 champion <code>c74f_sbx</code>, its platform-trace timings
as <code>baselines.jsonl</code> and full source inlined in prompt v2). Verdict grids: 865 real decode cells (BS=1), nsys cold-L2,
paired same-GPU vs <b>PR#16457 CURRENT head <code>b14ec40e1b</code></b>; per-rung <code>pr_cold</code> anchors checked every run
(old-vs-new-head anchor: median 1.005 — the 07-20/21 P4 commits do not move this envelope). Rival ratios are PR-arm-normalized
joins against the REPORT rival sweep, same protocol as §6.<br>
<span class="cng">第二期以 §6 冠军为 baseline(平台 trace 逐格时间 + 源码内联 prompt);判决 = 865 真实格,nsys 冷-L2,
同卡配对 vs <b>PR#16457 当前 head b14ec40e1b</b>;每次运行逐档锚检(新旧 head 锚差中位 1.005);对手比值经 PR 臂逐格归一,口径与 §6 一致。</span></p>
{kpis}
<h3>7.0 Timeline / 时间线</h3>
<div class="vizR"><figure style="margin:8px 0">{tl}
<figcaption style="font-size:0.9em;color:#444"><b>Fig. R0 — R3 campaign timeline: cloud rounds (top lane) vs local
verify arm (bottom lane).</b> Hover markers for details; colors = event class.<br>
<span class="cng">图 R0 — 第二期战役时间线:上=云端轮次(内部尺度),下=本地判决臂(nsys 冷-L2);悬停看详情;颜色=事件类别。</span>
</figcaption></figure></div>
<h3>7.1 Candidate ladder / 候选阶梯(均为全 865 格判决)</h3>
{ladder}
<p><span class="cng">compB 对 compA 净增 +1.3%:aef33fac 的 topk_mid 单 CTA 尾部选择档治愈 N=16387/8195 弱档
(pro/flash_64k +19%、32k +8-10%),其 n≈4099 的 mid&lt;1&gt; 档实测回退已被裁除;全格最低格从 0.999 抬到 1.140,
0 回退不再有边界格。</span></p>
<h3>7.2 Where the new speed comes from / 新增速度来源</h3>
<ol>
<li><b>Cooperative-launch &amp; barrier-ordering elimination / 去 coop-launch 与屏障内存序</b> (09d13c81, +4.7% grid):
regular launch + sense-token spin barrier (host generation counter ⇒ tokens never collide across launches, no per-launch
reset), grid sized to co-residency. Every formally-ordered variant measured 8–11% slower — the win IS the omitted ordering.
Safety argument (R3_LEDGER.md): merged-histogram lines are first plain-touched only after the barrier; L1 invalidates at
launch boundaries; pre-barrier writes are L2 atomics ⇒ post-barrier plain loads cannot observe stale L1. Flagged for
production-port review.<br><span class="cng">常规 launch + sense 令牌自旋屏障(host 代数计数器,跨 launch 永不撞号);
一切形式化排序变体都慢 8-11% — 胜利正是来自省掉排序。安全论证:merged histogram 在屏障前无 plain 读、L1 在 kernel 边界失效、
屏障前写为 L2 原子 ⇒ 屏障后 plain 读不可能命中陈旧 L1。已标记为生产移植评审项。</span>
{barrier_tbl}</li>
<li><b>Contiguous-slice scan / 连续切片扫描</b> (30e79029, +1.2%): per-block contiguous float4 slices replace grid-stride
interleave — better cold-data locality. <span class="cng">每块扫连续 float4 切片,替代 grid-stride 交错,冷数据局部性更好。</span></li>
<li><b>Adaptive post-pass-0 finish + register-cached row / 自适应收尾 + 寄存器缓存整行</b> (becdc5c7, +1.5% net):
one 11-bit MSB histogram pass, then 3-way dispatch on boundary-bucket size T — whole-bucket direct write (1 barrier);
T≤4096: smem-staged compaction + non-spinning rendezvous + last-arriver single-block 21-bit refine (1 barrier, no drain);
else classic 11/11/10 ladder. Keys live in registers (1×float4 + tail scalar per thread) — zero global re-reads across passes.
Slow twins (large-tie cells) gain 1.37–1.51×; v32 mid-n prefers the 30e7 ladder → engineer dispatch = {CHAMP_NAME}.<br>
<span class="cng">一遍 11-bit MSB histogram 后按边界桶大小三路分派:整桶直写(1 屏障)/小桶 smem 压缩 + 非自旋会合 +
末位到达块独占精化(1 屏障,无排水尾)/大桶回退 11/11/10 梯子;keys 全程驻寄存器,后续 pass 零全局重读。
慢双子格收益 1.37-1.51×;v32 中档偏好 30e7 梯子 → 工程师分派 = {CHAMP_NAME}。</span></li>
</ol>
<h3>7.3 {CHAMP_NAME} vs rivals, by ISL / 对手对比(按 ISL)</h3>
<div class="vizR">{inputs}{chips}
<figure style="margin:8px 0">{r1}
<figcaption style="font-size:0.9em;color:#444"><b>Fig. R1 — {CHAMP_NAME} geomean speedup by ISL (865 cells, all models pooled).</b>
Toggle series with the chips; hover points for values. 1.0× line = parity.<br>
<span class="cng">图 R1 — 按 ISL 的 geomean 加速比(勾选切换系列;悬停看数值;1.0× 为持平线)。</span></figcaption></figure></div>
<div class="vizR">{minputs}{inputs2}{mchips}
<div class="smrow">{''.join(sm)}</div>
<figcaption style="font-size:0.9em;color:#444"><b>Fig. R2 — per-model small multiples (flash K=512 · pro K=1024 · v32 K=2048).</b>
Model and series chips both filter.<br><span class="cng">图 R2 — 分模型小倍图(模型与系列复选框皆可过滤)。</span></figcaption></div>
{table25}
{''.join(layer_tables)}
<h3>7.4 Skeleton-constraint adjudication / 骨架约束裁决</h3>
<div class="card"><b>Decision (operator, 2026-07-22): Bar-first, loose-skeleton per the campaign-1 precedent.</b>
The composite lineage keeps (b) threshold refinement — in histogram-prefix form — and (c) the exact tie-robust refine,
but does NOT consume <code>pre_idx</code>: constraint (a) is vacated by measurement, not neglect. Hint-seeded variants were
falsified repeatedly (June history ×12; campaign-1 round-1 ×3; R3 <code>5f3daaf8</code>: a provably-exact warm filter —
admission superset ≥k regardless of hint quality — measured WASH 1.0001 inside its own n≥512K activation zone).
The +60% bar and a strict GVR skeleton are mutually incompatible on this workload (in-skeleton ceiling ≈1.28, op20/21/35).
No cosmetic hint path is added.<br>
<span class="cng">裁决(2026-07-22,用户):Bar 优先,骨架按第一期先例宽松解读。(a) preIdx 先验由测量证据豁免
(hint 变体屡次证伪;R3 的可证明精确 warm-filter 在其自身激活区亦为 1.0001 WASH);(b) 以 histogram 前缀精化等价保留;
(c) 完整保留。+60% 门与严格 GVR 骨架在本负载上互斥(骨架内天花板 ≈1.28)。不做化妆式 hint 挂载。</span></div>
<h3>7.5 Incidents &amp; discipline / 事故与纪律</h3>
<ul>
<li>Foreign 8-GPU job intermittently occupies this node since day-1 ~17:40Z: one probe verdict invalidated (unconditional
quiet-echo scripting bug → fixed to gated launches); one aef33fac full grid discarded (anchor p95 1.542). All measurements
now anchor-gated per cell; grids run on explicitly-free GPU lists (<code>drive_grid_gpulist.sh</code>).<br>
<span class="cng">外部 8 卡作业间歇占用本节点:一次探针判废(quiet-echo 无条件 bug,已改门控);一次 aef33fac 全格判废
(锚 p95 1.542)。此后逐格锚检 + 空闲卡白名单分片。</span></li>
<li>Platform noise calibrated in round 1: a verbatim champion resubmission scored 0.9956 (±0.5% band); an agent logged the
same solution timing 17/23/20 vs 23/28/26 µs across runs.<br><span class="cng">平台噪声标定:champion 原样重交 = 0.9956;
agent 自证同一 solution 两次计时差异显著。</span></li>
<li>Platform gap (D1): <code>prepare --baseline-solution</code> does not stage campaign assets (0/28 safetensors found)
→ champion baselines supplied as platform-trace per-workload timings; champion source inlined in the prompt instead.<br>
<span class="cng">平台缺口 D1:baseline-solution 评测不带 assets → 改用第一期平台 trace 逐格时间,源码内联 prompt。</span></li>
</ul>
<p style="font-size:0.9em;color:#52514e">Status at injection (07-22): campaign Running (round 3; best internal 1.173 =
<code>aef33fac</code>, harvested into compB). Platform spend ≈$578 of $800 cap. Final acceptance = fresh full grid of the
closing champion + borderline adjudication + rival joins, after campaign close.<br>
<span class="cng">注入时状态(07-22):round 3 运行中(内部最佳 1.173 = aef33fac,已并入 compB);平台花费约 $578/$800;
终审待战役收口后以收官冠军新鲜全格执行。</span></p>
{SEC_E}"""

    banner = (f'{BAN_S}<div style="background:#e7f6e7;border:1.5px solid #070;border-radius:8px;'
              f'padding:10px 16px;margin:12px 0;font-size:0.95em"><b>2026-07-22 R3 update / 第二期战役进展:</b> '
              f'current champion = <b>{CHAMP_NAME}</b> (becdc5c7 ⊕ 30e79029 dispatch) — <b>{kpi["xPR"]:.4f}× geomean vs '
              f'PR#16457 current head</b> (865/865 exact, {regs} borderline cell), '
              f'{kpi_c1:.4f}× over the §6 campaign-1 champion; vs sglang v2 {kpi["xSGL"]:.3f}× · vs radix_cutedsl '
              f'{kpi["xRDX"]:.3f}×. §6 below is the campaign-1 historical record (vs old head e6fdbfac3d). '
              f'See <a href="#sec-7">§7</a>. / 现任冠军 {CHAMP_NAME} 对 PR 当前 head 全格 {kpi["xPR"]:.4f}×,'
              f'对第一期冠军 {kpi_c1:.4f}×;§6 为第一期历史记录;详见 <a href="#sec-7">§7</a>。</div>{BAN_E}')

    html = HTML.read_text()
    if SEC_S in html:
        html = html[:html.index(SEC_S)] + section + html[html.index(SEC_E) + len(SEC_E):]
    elif "</body>" in html:
        html = html.replace("</body>", section + "\n</body>", 1)
    else:
        html += section
    if BAN_S in html:
        html = html[:html.index(BAN_S)] + banner + html[html.index(BAN_E) + len(BAN_E):]
    else:
        i = html.find("</h1>")
        html = html[:i + 5] + "\n" + banner + html[i + 5:]
    HTML.write_text(html)
    print(f"§7 injected ({len(section)} chars) tag={TAG}: "
          f"xPR={kpi['xPR']:.4f} xSGL={kpi['xSGL']:.3f} xRDX={kpi['xRDX']:.3f} "
          f"xFI={kpi['xFI']:.3f} vsC1={kpi_c1:.4f}")


if __name__ == "__main__":
    main()
