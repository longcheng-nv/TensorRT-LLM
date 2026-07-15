# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the op34 bilingual (EN / 中文) HTML report — CSS-only language
toggle (no inline JS; the target viewer strips <script>), static SVG bars.

Reads the campaign's nsys result jsonls (results/{harvest_pro,decomp2,grid})
and renders the data-driven regime map + double-lock decomposition. Idempotent:
re-run after the full grid completes to refresh the regime table/chart.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
RES = CAMP / "results"


def load(tag):
    p = RES / tag / "results.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def cold_by_cell(rows):
    by = defaultdict(dict)
    for r in rows:
        if "us_cold" in r:
            by[(r["model"], r["isl"], r["layer"])][r["arm"]] = r["us_cold"]
    return by


ISL_ORDER = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]
ISL_N = {"4k": 1027, "8k": 2051, "16k": 4099, "32k": 8195, "64k": 16387,
         "128k": 32771, "256k": 65539, "512k": 131075, "1024k": 262127}


def svg_bars(series, labels, colors, title, ymax=None, unit="×"):
    """Grouped bar chart as inline SVG. series = {name:[vals]} aligned to labels."""
    W, H, padL, padB, padT = 760, 250, 44, 46, 30
    n = len(labels)
    names = list(series.keys())
    g = len(names)
    ymax = ymax or max((max(v for v in s if v == v) for s in series.values()), default=1) * 1.1
    gw = (W - padL - 10) / n
    bw = gw * 0.8 / g
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}">']
    out.append(f'<text x="{W/2}" y="16" text-anchor="middle" class="ct">{title}</text>')
    # y gridlines
    for gy in range(0, 5):
        yy = padT + (H - padT - padB) * gy / 4
        val = ymax * (1 - gy / 4)
        out.append(f'<line x1="{padL}" y1="{yy:.1f}" x2="{W-6}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{padL-4}" y="{yy+3:.1f}" text-anchor="end" class="axl">{val:.1f}</text>')
    # 1.0 reference line (parity) if unit is ×
    if unit == "×" and ymax > 1:
        y1 = padT + (H - padT - padB) * (1 - 1.0 / ymax)
        out.append(f'<line x1="{padL}" y1="{y1:.1f}" x2="{W-6}" y2="{y1:.1f}" class="ref"/>')
    for i, lab in enumerate(labels):
        x0 = padL + i * gw
        for j, nm in enumerate(names):
            v = series[nm][i]
            if v != v:
                continue
            bh = (H - padT - padB) * min(v, ymax) / ymax
            x = x0 + gw * 0.1 + j * bw
            y = H - padB - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                       f'fill="{colors[j]}"><title>{nm} {lab}: {v:.2f}{unit}</title></rect>')
        out.append(f'<text x="{x0+gw/2:.1f}" y="{H-padB+14}" text-anchor="middle" class="axl">{lab}</text>')
    # legend
    lx = padL + 6
    for j, nm in enumerate(names):
        out.append(f'<rect x="{lx}" y="{H-14}" width="10" height="10" fill="{colors[j]}"/>')
        out.append(f'<text x="{lx+13}" y="{H-5}" class="lg">{nm}</text>')
        lx += 40 + len(nm) * 7
    out.append("</svg>")
    return "\n".join(out)


def build_regime(grid, harvest):
    """Merge grid + harvest_pro; return per-(model,isl) geomean ratios vs sglang."""
    by = cold_by_cell(grid)
    by_h = cold_by_cell(harvest)
    for k, v in by_h.items():
        by.setdefault(k, {}).update(v)
    agg = defaultdict(lambda: defaultdict(list))   # (model,isl) -> arm -> [ratio]
    for (model, isl, L), d in by.items():
        s = d.get("sglang_v2")
        if not s:
            continue
        for arm in ("op26_r0auto", "op34_mcta"):
            if d.get(arm):
                agg[(model, isl)][arm].append(d[arm] / s)
    return agg


def regime_table_and_chart(agg):
    rows_html = []
    labels, r0s, mcs = [], [], []
    grand = defaultdict(list)
    for model in ("flash", "pro"):
        for isl in ISL_ORDER:
            k = (model, isl)
            if k not in agg:
                continue
            r0 = geomean(agg[k]["op26_r0auto"])
            mc = geomean(agg[k]["op34_mcta"])
            rows_html.append(
                f"<tr><td>{model}</td><td>{isl}</td><td>{ISL_N[isl]:,}</td>"
                f"<td>1.000</td><td>{r0:.2f}</td><td>{mc:.2f}</td></tr>")
            labels.append(f"{model[0]}/{isl}")
            r0s.append(r0)
            mcs.append(mc)
            grand["op26_r0auto"] += agg[k]["op26_r0auto"]
            grand["op34_mcta"] += agg[k]["op34_mcta"]
    gr0 = geomean(grand["op26_r0auto"]) if grand["op26_r0auto"] else float("nan")
    gmc = geomean(grand["op34_mcta"]) if grand["op34_mcta"] else float("nan")
    chart = svg_bars({"op26_r0auto": r0s, "op34_mcta": mcs}, labels,
                     ["#4c78a8", "#e45756"],
                     "ratio vs sglang_v2 (cold nsys, lower=faster; 1.0=parity, goal<0.77)")
    return "\n".join(rows_html), chart, gr0, gmc


def main():
    harvest = load("harvest_pro")
    decomp = load("decomp2")
    grid = load("grid")
    agg = build_regime(grid, harvest)
    regime_rows, regime_chart, gr0, gmc = regime_table_and_chart(agg)

    # decomposition table (decomp2)
    dby = cold_by_cell(decomp)
    dec_rows = []
    dec_labels, sgl_s, co_s, ch_s, mo_s = [], [], [], [], []
    for (model, isl, L) in sorted(dby):
        d = dby[(model, isl, L)]
        s = d.get("sglang_v2", float("nan"))
        ch = d.get("op34_collect_only", float("nan"))
        co = d.get("op34_collect_oracle", float("nan"))
        mo = d.get("op34_mcta_oracle", float("nan"))
        dec_rows.append(f"<tr><td>{isl}</td><td>{L}</td><td>{s:.1f}</td>"
                        f"<td>{co:.1f}</td><td>{ch:.1f}</td><td>{mo:.1f}</td>"
                        f"<td>{(mo/s if s==s else 0):.2f}×</td></tr>")
        dec_labels.append(f"{isl}/L{L}")
        sgl_s.append(s); co_s.append(co); ch_s.append(ch); mo_s.append(mo)
    dec_chart = svg_bars(
        {"sglang(full)": sgl_s, "col_oracle(UB)": co_s, "col_hint": ch_s,
         "mcta_oracle": mo_s}, dec_labels,
        ["#54a24b", "#4c78a8", "#f58518", "#e45756"],
        "pure-kernel µs (cold nsys) — oracle collect ALONE ≈ sglang FULL", unit="µs")

    tmpl = (HERE / "template.html").read_text()
    repl = {
        "%%REGIME_ROWS%%": regime_rows,
        "%%REGIME_CHART%%": regime_chart,
        "%%DEC_ROWS%%": "\n".join(dec_rows),
        "%%DEC_CHART%%": dec_chart,
        "%%GR0%%": f"{gr0:.2f}" if gr0 == gr0 else "—",
        "%%GMC%%": f"{gmc:.2f}" if gmc == gmc else "—",
    }
    for k, v in repl.items():
        tmpl = tmpl.replace(k, v)
    out = CAMP / "report" / "op34_report.html"
    out.write_text(tmpl)
    print(f"wrote {out}  (regime cells={len(regime_rows.splitlines())}, "
          f"grand r0={gr0:.2f} mcta={gmc:.2f})")


if __name__ == "__main__":
    main()
