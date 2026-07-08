#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op22 W6 — build REPORT.html from the parsed nsys sweeps.

Mirrors the table design and report organization of
``indexer_topk_op_bench/report/report.html`` (dark NVIDIA theme, header
card stack, latency+speedup chart pairs per section, Full-data section
with CSV export + scrollable table, right-aligned numeric tables, KPI
chips). Bilingual (zh default / en); language and cold/warm-L2 toggles
stay radio + `:checked ~` CSS (op19 pattern) for the prose/tables, and the
CHARTS are the reference report's Plotly/JS pattern: one embedded JSON of
all cells + checkbox/radio-driven redraw (scenario x op x K x dtype [x N]),
replacing the former flat pre-rendered matplotlib SVG stack (~7 MB -> two
panels). Section 7 integrates the op23 deterministic UB/LB bounds
scenarios (loaded from --op23-root when present, selectable in the panel).

Also emits op22_seqlen_data.csv / op22_bs_data.csv next to the report
(reference §3 convention: per-op cold/warm µs + op21 cold speedups;
first column = scenario instead of hardware).

Tolerates partial data (missing scenarios/sweeps are skipped).

Usage: python3 gen_report_op22.py [--out-root ../results_b200_op22]
"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

OPS = ["gvr_ms_auto", "gvr_cutedsl", "gvr_multicta_cutedsl",
       "radix_cutedsl", "sglang_streaming"]
MAIN = "gvr_ms_auto"
RIVALS = [o for o in OPS if o != MAIN]
OP_LABEL = {
    "gvr_ms_auto": "GVR op#21 ms_auto (production dispatch)",
    "gvr_cutedsl": "GVR single-CTA (cuteDSL #14602)",
    "gvr_multicta_cutedsl": "GVR multi-CTA (cuteDSL, PR#15198)",
    "radix_cutedsl": "Radix (cuteDSL)",
    "sglang_streaming": "SGLang StreamingTopK (fp32, K≤1024)",
}
OP_SHORT = {"gvr_ms_auto": "op21", "gvr_cutedsl": "GVR-1CTA",
            "gvr_multicta_cutedsl": "GVR-mCTA", "radix_cutedsl": "Radix",
            "sglang_streaming": "SGLang"}
# report/report.html palette for the same op names (op21 = gold there too)
COL = {"gvr_ms_auto": "#ffd700", "gvr_cutedsl": "#b3e05a",
       "gvr_multicta_cutedsl": "#2ec4b6", "radix_cutedsl": "#4ea8de",
       "sglang_streaming": "#d62728"}
SCENARIOS = ["best", "worst", "real"]
SCEN_LABEL = {"best": "BEST (beta_deep, hr=0.90)",
              "worst": "WORST (beta_shallow, hr=0.05)",
              "real": "REAL (aggregate, sampled hr)"}
# op23 deterministic bounds scenarios (seqlen sweep only); loaded from
# ../results_b200_op23 when present — see op23_op21_bounds/RESULTS_SUMMARY.md
BOUNDS_SCENARIOS = ["ub", "lb"]
ALL_SCEN = SCENARIOS + BOUNDS_SCENARIOS
SCEN_LABEL.update({"ub": "UB (op23 hint-optimal construction)",
                   "lb": "LB (op23 adversarial construction)"})
SCEN_DASH = {"real": "solid", "best": "dash", "worst": "dot",
             "ub": "dashdot", "lb": "longdash"}
KS = [512, 1024, 2048]
DTS = ["fp32", "bf16", "fp16"]
K_MODEL = {512: "V4-Flash", 1024: "V4-Pro", 2048: "V3.2"}
BS_ANCHOR_N = 131072  # latency-vs-BS panel anchor (mid-grid N)

# ---- dark theme tokens (shared with the Plotly layout in interactive_panels)
BG, CARD, LINE, INK, GRN, GRN2 = ("#0f1419", "#161b22", "#2a3340",
                                  "#e6e6e6", "#76b900", "#9ecb3a")


# ---------------- data ----------------

def load_scenario(root, scen):
    """{sweep: {(K,dt,N,BS): {op: rec}}} — rec keeps us_cold/us_warm/meta."""
    out = {}
    for sweep, sub in (("seqlen", "seqlen_sweep"), ("bs", "bs_scaling"),
                       ("bs_hugeN", "bs_hugeN")):
        p = root / scen / sub / "results.jsonl"
        if not p.exists():
            continue
        cells = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "us_cold" not in r and "us_warm" not in r:
                continue
            cells.setdefault((r["K"], r["dtype"], r["N"], r["BS"]), {})[r["op"]] = r
        if cells:
            out[sweep] = cells
    # fold the stretch grid (N 512K/1M x BS 2..64) into the bs view so every
    # figure/table integrates it; sweep-key provenance stays in the jsonl.
    if "bs_hugeN" in out:
        out.setdefault("bs", {}).update(out.pop("bs_hugeN"))
    return out


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def pooled_cells(data, scen):
    """{(sweep,K,dt,N,BS): {op: rec}} over seqlen+bs."""
    sd = data.get(scen, {})
    out = {}
    for sweep in ("seqlen", "bs"):
        out.update({(sweep,) + k: v for k, v in sd.get(sweep, {}).items()})
    return out


def ratio_cells(cells, rival, metric):
    """Per-cell rival/main time ratio (>1 => op21 faster)."""
    out = {}
    for key, ops in cells.items():
        a, b = ops.get(MAIN), ops.get(rival)
        if a and b and a.get(metric) and b.get(metric):
            out[key] = b[metric] / a[metric]
    return out


# (matplotlib figure emitters removed — superseded by the
#  checkbox-driven Plotly panels in interactive_panels())


def fmt_r(r):
    if r is None:
        return "<td>—</td>"
    cls = "good" if r >= 1.05 else ("bad" if r <= 0.95 else "")
    return f"<td class='{cls}'>{r:.3f}×</td>"


def geomean_table(data, metric):
    """Rows: (scenario, K, dtype); cols: op21 geomean ratio vs each rival
    (seqlen+bs cells pooled)."""
    head = ("<tr><th>scenario</th><th>K</th><th>dtype</th>"
            + "".join(f"<th>vs {OP_SHORT[r]}</th>" for r in RIVALS)
            + "<th>cells</th></tr>")
    rows = []
    for scen in SCENARIOS:
        pooled = pooled_cells(data, scen)
        if not pooled:
            continue
        for K in KS:
            for dt in DTS:
                cells = {k: v for k, v in pooled.items()
                         if k[1] == K and k[2] == dt}
                if not cells:
                    continue
                tds = []
                ncell = 0
                for rival in RIVALS:
                    rs = [v[rival][metric] / v[MAIN][metric]
                          for v in cells.values()
                          if MAIN in v and rival in v
                          and v[MAIN].get(metric) and v[rival].get(metric)]
                    ncell = max(ncell, len(rs))
                    tds.append(fmt_r(gm(rs)))
                rows.append(f"<tr><td>{scen}</td><td>{K}</td><td>{dt}</td>"
                            + "".join(tds) + f"<td>{ncell}</td></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def hr_sensitivity_table(data, metric):
    """Per op: geomean over common (K,dt,N,BS) cells of worst_us/best_us —
    how much slower each op gets when hit-rate drops 0.90 -> 0.05."""
    head = ("<tr><th>operator</th>"
            + "".join(f"<th>K={K} {dt}</th>" for K in KS for dt in DTS)
            + "<th>overall</th></tr>")
    rows = []
    bd, wd = data.get("best", {}), data.get("worst", {})
    for op in OPS:
        tds, alls = [], []
        for K in KS:
            for dt in DTS:
                rs = []
                for sweep in ("seqlen", "bs"):
                    bc, wc = bd.get(sweep, {}), wd.get(sweep, {})
                    for key, ops in bc.items():
                        if key[0] != K or key[1] != dt:
                            continue
                        w = wc.get(key, {})
                        if (op in ops and op in w and ops[op].get(metric)
                                and w[op].get(metric)):
                            rs.append(w[op][metric] / ops[op][metric])
                g = gm(rs)
                alls += rs
                tds.append(fmt_r(g))
        rows.append(f"<tr><td style='color:{COL[op]}'>{OP_LABEL[op]}</td>"
                    f"{''.join(tds)}" + fmt_r(gm(alls)) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def analysis_items(data):
    """Reference-style auto-analysis bullets: per scenario x rival, median/
    min/max cold ratio + op21-faster cell count."""
    out = []
    for scen in SCENARIOS:
        pooled = pooled_cells(data, scen)
        for rival in RIVALS:
            sp = [v[rival]["us_cold"] / v[MAIN]["us_cold"]
                  for v in pooled.values()
                  if MAIN in v and rival in v
                  and v[MAIN].get("us_cold") and v[rival].get("us_cold")]
            if not sp:
                continue
            med, mn, mx = st.median(sp), min(sp), max(sp)
            faster, total = sum(x > 1 for x in sp), len(sp)
            en = (f"[{scen.upper()}] op#21 vs {OP_LABEL[rival]} (cold-L2): "
                  f"median={med:.2f}× (min={mn:.2f}×, max={mx:.2f}"
                  f"×); op#21 faster in {faster}/{total} cells.")
            zh = (f"[{scen.upper()}] op#21 对 {OP_LABEL[rival]}"
                  f"（冷 L2）：中位 {med:.2f}×"
                  f"（最小 {mn:.2f}×，最大 "
                  f"{mx:.2f}×）；op#21 更快的 cell "
                  f"数 {faster}/{total}。")
            out.append((en, zh))
    return out


def write_csvs(data):
    """Reference §3 convention: flat CSV per sweep; first column = scenario;
    per-op cold/warm µs + op21 cold speedups vs each rival."""
    outs = []
    for sweep_key, fn in (("seqlen", "op22_seqlen_data.csv"),
                          ("bs", "op22_bs_data.csv")):
        path = HERE / fn
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            head = ["scenario", "K", "dtype", "N", "BS", "hr", "layer"]
            for o in OPS:
                head += [f"{o}_cold_us", f"{o}_warm_us"]
            head += [f"op21_cold_speedup_vs_{o}" for o in RIVALS]
            w.writerow(head)
            nrow = 0
            for scen in SCENARIOS:
                cells = data.get(scen, {}).get(sweep_key, {})
                for (K, dt, N, BS), d in sorted(cells.items()):
                    m = d.get(MAIN, {})
                    hr = m.get("hit_rate")
                    row = [scen, K, dt, N, BS,
                           round(hr, 4) if hr is not None else "",
                           m.get("layer", "")]
                    for o in OPS:
                        r = d.get(o, {})
                        row += [round(r["us_cold"], 3) if r.get("us_cold") else "",
                                round(r["us_warm"], 3) if r.get("us_warm") else ""]
                    base = m.get("us_cold")
                    for o in RIVALS:
                        r = d.get(o, {})
                        row.append(round(r["us_cold"] / base, 3)
                                   if (base and r.get("us_cold")) else "")
                    w.writerow(row)
                    nrow += 1
        outs.append((fn, nrow))
    return outs


def fulldata_table(data, metric):
    """Reference §3 scrollable table: sweep,K,dtype,N,BS,hr + per-op µs."""
    head = ("<tr><th>scen</th><th>sweep</th><th>K</th><th>dtype</th>"
            "<th>N</th><th>BS</th><th>hr</th>"
            + "".join(f"<th style='color:{COL[o]}'>{OP_SHORT[o]} µs</th>"
                      for o in OPS) + "</tr>")
    rows = []
    for scen in SCENARIOS:
        for sweep in ("seqlen", "bs"):
            cells = data.get(scen, {}).get(sweep, {})
            for (K, dt, N, BS), d in sorted(cells.items()):
                hr = d.get(MAIN, {}).get("hit_rate")
                tds = []
                for o in OPS:
                    v = d.get(o, {}).get(metric)
                    tds.append(f"<td>{v:.1f}</td>" if v else "<td>–</td>")
                rows.append(
                    f"<tr><td>{scen}</td><td>{sweep}</td><td>{K}</td>"
                    f"<td>{dt}</td><td>{N}</td><td>{BS}</td>"
                    f"<td>{hr:.2f}</td>" + "".join(tds) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


# ---------------- html ----------------

def bi(en, zh, tag="div"):
    return (f'<{tag} class="i18n-en">{en}</{tag}>'
            f'<{tag} class="i18n-zh">{zh}</{tag}>')


CSS = """
body{font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#0f1419;color:#e6e6e6;line-height:1.55}
.wrap{max-width:1300px;margin:0 auto;padding:24px}
h1{color:#76b900} h2{color:#76b900;border-bottom:1px solid #2a3340;padding-bottom:6px;margin-top:36px}
h3{color:#9ecb3a}
.card{background:#161b22;border:1px solid #2a3340;border-radius:10px;padding:18px;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{border:1px solid #2a3340;padding:5px 8px;text-align:right}
th{background:#1c2530;color:#9ecb3a} td:first-child,th:first-child{text-align:left}
.kpi{display:inline-block;background:#1c2530;border-radius:8px;padding:10px 16px;margin:6px}
.kpi b{color:#76b900;font-size:20px}
.row{display:flex;gap:16px;flex-wrap:wrap} .row>div{flex:1;min-width:480px}
li{margin:4px 0} .reg{font-weight:bold;color:#9ecb3a}
.good{color:#7ee787;font-weight:600} .bad{color:#ff6b6b;font-weight:600}
a{color:#76b900}
code,.mono{font-family:ui-monospace,Consolas,monospace;font-size:0.92em;background:#1c2530;padding:1px 4px;border-radius:3px}
pre{background:#1c2530;padding:10px 12px;border-radius:6px;overflow-x:auto;font-size:12.5px}
.small{font-size:12.5px;color:#9aa}
.meta{color:#9aa;font-size:13px;margin-bottom:6px}
.fig{background:#161b22;border:1px solid #2a3340;border-radius:10px;padding:10px;margin:16px 0;overflow-x:auto}
.fig svg{max-width:100%;height:auto}
.scrolltbl{max-height:520px;overflow:auto}
.plt{height:430px;min-width:480px}
.ctl{margin:2px 0 10px 0;line-height:2.1}
.ck{background:#1c2530;border:1px solid #2a3340;border-radius:6px;padding:3px 8px;margin-right:4px;cursor:pointer;font-size:12.5px;white-space:nowrap}
.ck input{accent-color:#76b900;vertical-align:-2px;margin-right:4px}
/* --- CSS-only toggles for prose/tables (radio + :checked ~) --- */
input[name=lang],input[name=metric]{position:absolute;left:-9999px}
.langbar{position:fixed;top:14px;right:16px;z-index:99}
.langbar label,.regbar label{display:inline-block;background:#1c2530;color:#e6e6e6;border:1px solid #2a3340;font-weight:bold;padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px;margin-left:6px;user-select:none}
#lang-zh:checked ~ .wrap .langbar label[for=lang-zh],#lang-en:checked ~ .wrap .langbar label[for=lang-en]{background:#76b900;color:#0f1419}
#m-cold:checked ~ .wrap label[for=m-cold],#m-warm:checked ~ .wrap label[for=m-warm]{background:#76b900;color:#0f1419}
#lang-zh:checked ~ .wrap .i18n-en{display:none}
#lang-en:checked ~ .wrap .i18n-zh{display:none}
#m-cold:checked ~ .wrap .warm{display:none}
#m-warm:checked ~ .wrap .cold{display:none}
"""


def scen_dt_gm(data, scen, rival, dt, metric="us_cold"):
    pooled = pooled_cells(data, scen)
    rs = [v[rival][metric] / v[MAIN][metric]
          for k, v in pooled.items() if k[2] == dt
          and MAIN in v and rival in v
          and v[MAIN].get(metric) and v[rival].get(metric)]
    return gm(rs), len(rs)


def tldr(data):
    """Verdict card: KPI chips (REAL headline) + per-scenario lines."""
    kpi_en, kpi_zh = [], []
    for scen in SCENARIOS:
        g, _ = scen_dt_gm(data, scen, "radix_cutedsl", "fp32")
        if g:
            kpi_en.append(f"<div class='kpi'>{scen.upper()} vs Radix fp32 "
                          f"<b>{g:.3f}×</b></div>")
            kpi_zh.append(f"<div class='kpi'>{scen.upper()} 对 Radix fp32 "
                          f"<b>{g:.3f}×</b></div>")
    lines_en, lines_zh = [], []
    for scen in SCENARIOS:
        seg_en, seg_zh = [], []
        for rival, tag in (("gvr_cutedsl", "GVR-1CTA"), ("radix_cutedsl", "Radix")):
            gs = []
            for dt in DTS:
                g, n = scen_dt_gm(data, scen, rival, dt)
                if g:
                    gs.append(f"{dt} <b>{g:.3f}×</b>")
            if gs:
                seg_en.append(f"vs {tag}: " + " / ".join(gs))
                seg_zh.append(f"对 {tag}：" + " / ".join(gs))
        if seg_en:
            lines_en.append(f"<b>{scen.upper()}</b> — " + "; ".join(seg_en))
            lines_zh.append(f"<b>{scen.upper()}</b> — " + "；".join(seg_zh))
    if not lines_en:
        return "<b>TL;DR</b> — data pending", "<b>TL;DR</b> — 数据未就绪"
    en = ("<h3>TL;DR — verdict</h3>" + "".join(kpi_en)
          + "<p>(cold-L2 geomean, time ratio rival/op21, &gt;1 ⇒ op21 "
            "faster)</p><ul><li>" + "</li><li>".join(lines_en) + "</li></ul>"
          + "<p class='small'>Sanity anchor: REAL vs gvr_cutedsl reproduces "
            "the op21-campaign B200 verdict (1.249/1.091/1.055 fp32/bf16/"
            "fp16) within −4.1%/+1.3%/+2.2%. op#21 wins ONLY on "
            "real-scenario data; each stress scenario trips one of its two "
            "msc fallback triggers (§6 Mechanism).</p>")
    zh = ("<h3>TL;DR — 结论</h3>" + "".join(kpi_zh)
          + "<p>（cold-L2 几何均值，时间比 "
            "rival/op21，&gt;1 ⇒ op21 更快）</p><ul><li>"
          + "</li><li>".join(lines_zh) + "</li></ul>"
          + "<p class='small'>Sanity 锚点：REAL 对 gvr_cutedsl "
            "复现 op21 战役 B200 收口值（fp32/"
            "bf16/fp16 = 1.249/1.091/1.055），偏差 −4.1%/"
            "+1.3%/+2.2%。op#21 只在 real 场景获胜"
            "；两个应力场景各触发其 "
            "msc 双 fallback 之一（§6 机制）。</p>")
    return en, zh


def findings_html(data, metric="us_cold"):
    """Extreme win/loss cells for op21 vs radix per scenario."""
    blocks = []
    for scen in SCENARIOS:
        pooled = pooled_cells(data, scen)
        rats = []
        for k, v in pooled.items():
            if (MAIN in v and "radix_cutedsl" in v and v[MAIN].get(metric)
                    and v["radix_cutedsl"].get(metric)):
                rats.append((v["radix_cutedsl"][metric] / v[MAIN][metric], k, v))
        if not rats:
            continue
        rats.sort()
        rows = []
        for tag, sel in (("worst-5", rats[:5]), ("best-5", rats[-5:][::-1])):
            for r, k, v in sel:
                sweep, K, dt, N, BS = k
                hr = v[MAIN].get("hit_rate")
                lay = v[MAIN].get("layer")
                path = v[MAIN].get("ms_path", "")
                cls = "good" if r >= 1.05 else ("bad" if r <= 0.95 else "")
                rows.append(
                    f"<tr><td>{tag}</td><td>{sweep}</td><td>{K}</td><td>{dt}</td>"
                    f"<td>{N}</td><td>{BS}</td>"
                    f"<td>{v[MAIN][metric]:.1f}</td>"
                    f"<td>{v['radix_cutedsl'][metric]:.1f}</td>"
                    f"<td class='{cls}'>{r:.3f}×</td>"
                    f"<td>{hr:.2f}</td><td>L{lay} {path}</td></tr>")
        blocks.append(
            f"<h3>{SCEN_LABEL[scen]}</h3><table><tr><th></th><th>sweep</th>"
            "<th>K</th><th>dtype</th><th>N</th><th>BS</th>"
            "<th>op21 µs</th><th>radix µs</th>"
            "<th>radix/op21</th><th>hr</th>"
            "<th>layer/path</th></tr>" + "".join(rows) + "</table>")
    return "".join(blocks)


def mech_summary_table():
    """Per-scenario summary of the count_gvr_iters host replay (162 cells)."""
    p = HERE / "mech_check_iters.jsonl"
    if not p.exists():
        return "<p class='small'>mech_check_iters.jsonl missing — run mech_check_iters.py</p>"
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = []
    for scen in SCENARIOS:
        rs = [r for r in recs if r["scenario"] == scen]
        if not rs:
            continue
        n = len(rs)
        p2 = sorted(r["p2_iters"] for r in rs)
        ev = sorted(r["p2_evals"] for r in rs)
        p4 = sorted(r["p4_snap_iters"] for r in rs)
        ncv = sum(not r["p2_converged"] for r in rs)
        rows.append(f"<tr><td>{SCEN_LABEL[scen]}</td><td>{n}</td>"
                    f"<td>{p2[n // 2]} / {p2[-1]}</td>"
                    f"<td>{ev[n // 2]} / {ev[-1]}</td>"
                    f"<td>{ncv}/{n}</td>"
                    f"<td>{p4[n // 2]} / {p4[-1]}</td></tr>")
    return ("<table><tr><th>scenario</th><th>cells</th>"
            "<th>P2 refine iters med/max</th><th>full-row count evals med/max</th>"
            "<th>P2 non-converged</th><th>P4 snap iters med/max</th></tr>"
            + "".join(rows) + "</table>")


def mech_worst_check(data):
    """Build-time check of the replay prediction: worst (ev1 everywhere)
    should run FAST — geomean t_worst/t_real per op on matched cells."""
    if "worst" not in data or not data["worst"] or "real" not in data:
        return None
    out = []
    for op in OPS:
        rs = []
        for sweep in ("seqlen", "bs"):
            cw = data["worst"].get(sweep, {})
            cr_ = data["real"].get(sweep, {})
            for key, ops_w in cw.items():
                ops_r = cr_.get(key)
                if not ops_r or op not in ops_w or op not in ops_r:
                    continue
                a, b = ops_w[op].get("us_cold"), ops_r[op].get("us_cold")
                if a and b:
                    rs.append(a / b)
        g = gm(rs)
        if g:
            out.append((op, g, len(rs)))
    return out


def mech_html(data):
    xover = """<table>
<tr><th>op / cell (N=1M fp32)</th><th>lg=best,pi=best</th><th>lg=best,pi=real</th>
<th>lg=real,pi=best</th><th>lg=real,pi=real</th></tr>
<tr><td>ms_auto K2048</td><td>247.8</td><td><b>40.7</b></td><td>229.1</td><td><b>40.5</b></td></tr>
<tr><td>cutedsl K2048</td><td>146.0</td><td><b>103.1</b></td><td>168.0</td><td><b>102.3</b></td></tr>
<tr><td>ms_auto K512</td><td><b>125.0</b></td><td>187.0</td><td><b>124.5</b></td><td>186.2</td></tr>
<tr><td>cutedsl K512</td><td>123.9</td><td>125.7</td><td>125.5</td><td>128.0</td></tr>
</table>"""
    wc = mech_worst_check(data)
    wc_en = wc_zh = ""
    if wc:
        seg = " · ".join(f"{OP_LABEL[o]} <b>{g:.3f}×</b> (n={n})" for o, g, n in wc)
        wc_en = ("<p><b>Prediction check (build-time)</b> — replay says WORST needs zero "
                 "refines (ev1 in 54/54 cells) ⇒ the refine trigger is absent; the "
                 "single-CTA/multi-CTA GVR should match REAL. op21's msc paths carry the "
                 "second trigger: worst's boundary-flat data yields wide candidate bands "
                 "(replay cand→kC, e.g. 4318 at K2048 N=1M) → slot-overflow fallback, so "
                 "ms_auto may stay slower than REAL at hugeN even with ev1. Geomean "
                 "t<sub>worst</sub>/t<sub>real</sub> on matched cells (&lt;1 ⇒ worst "
                 "faster): " + seg + ". <i>Caveat: worst ran on b200-049, real on "
                 "b200-040 — cross-node absolute-µs comparison, treat direction-only.</i></p>")
        wc_zh = ("<p><b>预言验证（构建时计算）</b> — 回放表明 WORST 零 refine（54/54 cell "
                 "ev1）⇒ refine 触发器缺席；单 CTA/多 CTA GVR 应与 REAL 持平。op21 的 msc "
                 "路径带第二触发器：worst 的『贴边界平坦』数据产生宽候选带（回放 cand→kC，"
                 "如 K2048 N=1M 时 4318）→ slot 溢出 fallback，因此 ms_auto 在 hugeN 即使 "
                 "ev1 也可能仍慢于 REAL。匹配 cell 上 t<sub>worst</sub>/t<sub>real</sub> "
                 "几何均值（&lt;1 ⇒ worst 更快）：" + seg +
                 "。<i>注意：worst 在 b200-049、real 在 b200-040 —— 跨节点绝对 µs 对比，"
                 "只看方向。</i></p>")
    en = f"""
<p><b>Verdict: the BEST-scenario slowdown is real and data-dependent, but the
tie-density hypothesis is falsified.</b> The cost driver is the number of
<b>extra full-row re-scans</b> during threshold search, and that number is a
<b>pure function of the preIdx hint</b> — not of the logits value distribution.
Chain of evidence (full detail: <code>MECH_FINDINGS.md</code>, artifacts
<code>mech_check_iters.py/.jsonl</code>, <code>mech_crossover.py</code>):</p>
<ol>
<li><b>Non-convergence never happens.</b> A verified host replay of the vendored
kernel's control flow (<code>harness/count_gvr_iters.py</code>) on the actual
bundles: P2 converges in 162/162 cells, all scenarios.
{mech_summary_table()}</li>
<li><b>GVR single-CTA obeys a linear law in eval count, scenario drops out</b>:
K2048 fp32 N=1M ev1→93.3 µs, ev3→136.7 µs (≈22 µs per extra full-row scan);
best/real N=524K both ev4 → 86.2/86.3 µs (identical).</li>
<li><b>Crossover experiment (decisive)</b>: swap logits and preIdx between the
best and real bundles at the headline cells (CUDA-event median, 50 cold reps,
b200-049 GPU1, screening-only). Cost follows the <b>preIdx column only</b>:
{xover}
K512 flips the sign (real-preIdx slower than best-preIdx) ⇒ NON-monotone
in hit rate.</li>
<li><b>Why hr→1 poisons the init</b>: GVR seeds the threshold with
mean(logits[preIdx]). A near-perfect hint makes that ≈ the top-K <b>median</b>
→ initial count ≈ K/2 &lt; K → guaranteed undershoot → 1–3 refine re-scans.
Boundary-clustered misses (REAL, and WORST whose miss-depth model sits at the
selection boundary) put the seed ≈ the K-th value → first count already lands
in the [K, kC] acceptance band.</li>
<li><b>op21's cliff is sharper</b>: on a ladder miss or slot overflow the msc
kernels take the leader-CTA full-row fallback (<code>gvr_msc_op.py:1096</code>,
~95 µs per pass at N=1M fp32) — 236 µs (ev3) vs 28.6 µs (ev1) at the headline
cell. K512 shows the overflow trigger alone: all crossover combos replay ev2,
but real-preIdx yields cand≈4500 (&gt; slot cap) vs 654 → fallback → 186 vs
124 µs.</li>
</ol>
{wc_en}
<p class="small">Methodology note: a synthetic FIXED hr≥0.9 preIdx is a stress
construction — real captures never place the hint mean at the top-K median,
because real misses concentrate at the selection boundary.</p>"""
    zh = f"""
<p><b>结论：BEST 场景的变慢是真实、数据依赖的，但 tie 密度假设被证伪。</b>
代价来自阈值搜索期间的<b>额外全行重扫</b>次数，而该次数是 <b>preIdx 提示的纯函数</b>
—— 与 logits 值分布无关。证据链（详见 <code>MECH_FINDINGS.md</code>，工件
<code>mech_check_iters.py/.jsonl</code>、<code>mech_crossover.py</code>）：</p>
<ol>
<li><b>非收敛从未发生。</b>用忠实复刻内核控制流的 host 回放
（<code>harness/count_gvr_iters.py</code>）在真实 bundle 上：162/162 cell P2 全收敛。
{mech_summary_table()}</li>
<li><b>GVR 单 CTA 的用时是 eval 次数的线性函数，场景变量消失</b>：K2048 fp32 N=1M
ev1→93.3 µs、ev3→136.7 µs（每次额外全行扫描 ≈22 µs）；best/real N=524K 同为 ev4 →
86.2/86.3 µs（完全一致）。</li>
<li><b>交叉实验（决定性）</b>：在头条 cell 上把 best 与 real 的 logits、preIdx 两两互换
（CUDA-event 中位数，50 次 cold，b200-049 GPU1，仅筛查用）。用时<b>只跟随 preIdx 列</b>：
{xover}
K512 方向反转（real-preIdx 反而比 best-preIdx 慢）⇒ 对 hit rate 非单调。</li>
<li><b>为什么 hr→1 会毒化初始化</b>：GVR 用 mean(logits[preIdx]) 作阈值种子。近乎完美的
提示使它 ≈ top-K 的<b>中位数</b> → 初始计数 ≈ K/2 &lt; K → 必然 undershoot → 1–3 次
refine 重扫。而边界聚集的 miss（REAL，以及 miss-depth 模型贴着选择边界的 WORST）使种子
≈ 第 K 个值 → 首次计数即落入 [K, kC] 接受带。</li>
<li><b>op21 的悬崖更陡</b>：ladder 未命中或 slot 溢出时，msc 内核走 leader-CTA 全行
fallback（<code>gvr_msc_op.py:1096</code>，N=1M fp32 每遍 ~95 µs）—— 头条 cell
236 µs（ev3）对 28.6 µs（ev1）。K512 单独展示溢出触发：交叉组合回放全为 ev2，但
real-preIdx 的 cand≈4500（超 slot 容量）对 654 → fallback → 186 对 124 µs。</li>
</ol>
{wc_zh}
<p class="small">方法论注记：合成的固定 hr≥0.9 preIdx 是一种应力构造 —— 真实捕获中提示
均值从不落在 top-K 中位数上，因为真实 miss 聚集在选择边界。</p>"""
    return f'<div class="card">{bi(en, zh)}</div>'


def gates_table():
    return """<table><tr><th>gate</th><th>v32</th><th>v4flash</th><th>v4pro</th><th>limit</th></tr>
<tr><td>G1 per-layer KS max</td><td>0.005</td><td>0.003</td><td>0.002</td><td>≤0.05</td></tr>
<tr><td>G2 aggregate KS</td><td>0.021</td><td>0.018</td><td>0.021</td><td>≤0.05</td></tr>
<tr><td>G3 boundary mass @16K/64K/256K</td><td>1.01/1.10/1.14</td><td>1.11/1.12/1.11</td><td>1.03/1.03/1.01</td><td>0.80–1.25</td></tr>
<tr><td>G4 retention-curve max err</td><td>0.030</td><td>0.015</td><td>0.021</td><td>≤0.05</td></tr>
<tr><td>G5 realised-vs-target hr err</td><td>0.000</td><td>0.000</td><td>0.000</td><td>≤0.03</td></tr></table>"""


def repro_html():
    cli = """SKILL=.claude/skills/indexer-topk-temporal-synth
# per cell (model, dtype, N): seed = 42 + crc32("{K}|{N}") % 1e6   (see seed policy)
python3 $SKILL/src/synth_temporal_data.py --model {v4flash|v4pro|v32} --N &lt;N&gt; \\
    --cfg beta_deep    --target_hr 0.90 --bs 1 --dtype &lt;dt&gt; --seed &lt;seed&gt; --outdir bundles/best/...   # BEST
python3 $SKILL/src/synth_temporal_data.py ... --cfg beta_shallow --target_hr 0.05 ...                  # WORST
python3 $SKILL/src/synth_temporal_data.py ... --cfg aggregate                    ...                  # REAL (hr sampled)
# every bundle's exact CLI incl. resolved seed: bundles/&lt;scen&gt;/&lt;model&gt;_&lt;dt&gt;_N&lt;N&gt;/*/meta.json "gen_cmd"
"""
    nl = ("使用 indexer-topk-temporal-synth skill 为 {V4-Flash|V4-Pro|V3.2} 生成 N=&lt;N&gt;、"
          "dtype=&lt;dtype&gt;、seed=&lt;cell_seed&gt; 的单行 decode logits + 时序相关 preIdx；"
          "GVR 最优场景用 deep-layer 分布 (--cfg beta_deep) 且固定 hit rate 0.90，"
          "GVR 最差场景用 shallow-layer 分布 (--cfg beta_shallow) 且固定 hit rate 0.05，"
          "realistic 锚点用 --cfg aggregate 并让 hit rate 按真实 per-step 分布采样。")
    en = f"""<p><b>Seed policy</b>: <code>synthesize()</code> draws the row's layer as the FIRST rng call,
so a constant seed would collapse the layer mixture across the grid. Per-cell
<code>seed(K,N) = 42 + crc32("{{K}}|{{N}}") % 1e6</code>, shared across dtypes and scenarios;
fully deterministic from base 42. Realised hr verified ±0.03 per bundle (G5).</p>
<p><b>Canonical CLI</b> (verbatim; recorded per bundle in <code>meta.json.gen_cmd</code>):</p>
<pre>{cli}</pre>
<p><b>Natural-language prompt equivalent</b>:</p><pre>{nl}</pre>"""
    zh = f"""<p><b>Seed 策略</b>：<code>synthesize()</code> 的第一次 rng 调用就是抽层，常量 seed 会让全网格
塌缩到同一层。每 cell <code>seed(K,N) = 42 + crc32("{{K}}|{{N}}") % 1e6</code>，跨 dtype 与场景共享；
从基 42 完全确定。每 bundle 实测 hr 校验 ±0.03（G5）。</p>
<p><b>规范 CLI</b>（逐字；每个 bundle 的 <code>meta.json.gen_cmd</code> 记录了带解析 seed 的等价命令）：</p>
<pre>{cli}</pre>
<p><b>自然语言 prompt 等价物</b>：</p><pre>{nl}</pre>"""
    return bi(en, zh)


# ---------------- op23 bounds section ----------------

def bounds_tables(data_all, metric):
    """Bounds verdict table on the like-for-like subset (seqlen BS=1)."""
    def seq_cells(scen):
        return {k: v for k, v in data_all.get(scen, {}).get("seqlen", {}).items()}

    cells = {s: seq_cells(s) for s in ("ub", "real", "lb")}

    def four_cols(rv, keep=None):
        """{ub, real, lb, eff} geomeans, optionally restricted to keep(key)."""
        cols = {}
        for s in ("ub", "real", "lb"):
            rc = ratio_cells(cells[s], rv, metric)
            cols[s] = gm([v for k, v in rc.items()
                          if keep is None or keep(k)])
        effs = []
        for key, ops in cells["lb"].items():
            if keep is not None and not keep(key):
                continue
            a_lb = ops.get(MAIN)
            r_ops = cells["real"].get(key, {})
            a_re = r_ops.get(MAIN)
            if not (a_lb and a_re and a_lb.get(metric) and a_re.get(metric)):
                continue
            # effective LB: per-cell worst op21 time of {lb, real}, ratio
            # from the same scenario as that worst cell
            src = ops if a_lb[metric] >= a_re[metric] else r_ops
            b = src.get(rv)
            if b and b.get(metric):
                effs.append(b[metric] / src[MAIN][metric])
        cols["eff"] = gm(effs)
        return cols

    hdr = ("<th>UB</th><th>real</th><th>LB constr.</th>"
           "<th>LB_eff (worst{lb,real})</th>")
    tbl = [f"<table><tr><th></th>{hdr}</tr>"]
    for rv in RIVALS:
        c = four_cols(rv)
        tds = "".join(fmt_r(c[s]) for s in ("ub", "real", "lb", "eff"))
        tbl.append(f"<tr><td>{OP_SHORT[rv]}</td>{tds}</tr>")
    tbl.append("</table>")

    # full per-(K, dtype, rival) breakdown
    det = [f"<table><tr><th>K</th><th>dtype</th><th>rival</th>{hdr}"
           "<th>cells</th></tr>"]
    for K in KS:
        for dt in DTS:
            def keep(key, K=K, dt=dt):
                return key[0] == K and key[1] == dt
            for rv in RIVALS:
                c = four_cols(rv, keep)
                if c["ub"] is None and c["real"] is None:
                    continue  # sglang outside fp32 x K<=1024
                n = len([1 for k in cells["ub"] if keep(k)])
                tds = "".join(fmt_r(c[s])
                              for s in ("ub", "real", "lb", "eff"))
                det.append(f"<tr><td>{K}</td><td>{dt}</td>"
                           f"<td>{OP_SHORT[rv]}</td>{tds}<td>{n}</td></tr>")
    det.append("</table>")
    tbl.append('<div class="scrolltbl" style="margin-top:10px">'
               + "".join(det) + "</div>")

    # absolute op21 envelope
    env = ["<table><tr><th>K</th><th>dtype</th><th>real/UB</th><th>LB/UB</th>"
           "<th>UB@1M µs</th><th>LB@1M µs</th><th>real@1M µs</th></tr>"]
    for K in KS:
        for dt in DTS:
            r_ru, r_lu, spot = [], [], {}
            for (k2, d2, N, BS), ops in cells["ub"].items():
                if k2 != K or d2 != dt:
                    continue
                a = ops.get(MAIN)
                al = cells["lb"].get((k2, d2, N, BS), {}).get(MAIN)
                ar = cells["real"].get((k2, d2, N, BS), {}).get(MAIN)
                if not (a and al and ar and a.get(metric)):
                    continue
                r_ru.append(ar[metric] / a[metric])
                r_lu.append(al[metric] / a[metric])
                if N == 1048576:
                    spot = {"ub": a[metric], "lb": al[metric],
                            "re": ar[metric]}
            if r_ru:
                env.append(
                    f"<tr><td>{K}</td><td>{dt}</td>"
                    f"<td>{gm(r_ru):.3f}</td><td>{gm(r_lu):.3f}</td>"
                    f"<td>{spot.get('ub', 0):.1f}</td>"
                    f"<td>{spot.get('lb', 0):.1f}</td>"
                    f"<td>{spot.get('re', 0):.1f}</td></tr>")
    env.append("</table>")
    return "".join(tbl), "".join(env)


def bounds_html(data_all):
    if not all(s in data_all for s in ("ub", "lb")):
        return ""
    t_c, e_c = bounds_tables(data_all, "us_cold")
    t_w, e_w = bounds_tables(data_all, "us_warm")
    ctor_tbl = (
        "<table><tr><th>scenario</th><th>stash (preIdx)</th><th>verified</th></tr>"
        "<tr><td><b>UB</b></td><td>4-block sandwich: K/4 top-spread (argmax in) + "
        "K/4 [0.7K,K) + K/4 [K,1.3K) + K/4 geomspace deep tail; depth d "
        "binary-searched for first-pass count ∈ [K+8, kC−64]</td>"
        "<td>replay 78/78 <b>ev1</b>; host msc ok-gate PASS 78/78 "
        "(band 0.14–0.66K, slots ≤ 8/8)</td></tr>"
        "<tr><td><b>LB</b></td><td>ranks 0..K−2 + rank N−1 — «near-perfect "
        "hint + one stale deep entry», hr=(K−1)/K</td>"
        "<td>pmean = max possible ⇒ ev 5–8; pmin = v_min ⇒ m1g = N &gt; kC "
        "defeats the msc fallback short-circuit</td></tr>"
        "<tr><td>lb2 (replay-only)</td><td>exact top-K (hr = 1.0)</td>"
        "<td>ev16 nonconverged 78/78 — adversarial for the RIVAL single-CTA "
        "GVRs but short-circuits op21's msc fallback ⇒ invalid as op21 LB</td>"
        "</tr></table>")
    return ('<div class="card">'
            + bi(
        "<h3>Geomean rival/op21, seqlen BS=1 subset (&gt;1 ⇒ op21 faster)</h3>"
        "<p class='small'>Like-for-like: ub/lb have no BS sweep, so the real "
        "column here is the seqlen-BS=1 subset — NOT comparable to the pooled "
        "table in §4 (op21's pooled radix win comes from high-BS cells). "
        "First table = pooled over all K × dtype; scrollable table below = "
        "full per-(K, dtype, rival) breakdown.</p>",
        "<h3>几何均值 rival/op21，seqlen BS=1 子集（&gt;1 ⇒ op21 更快）</h3>"
        "<p class='small'>同口径对比：ub/lb 无 BS 扫描，故本表 real 列为 "
        "seqlen-BS=1 子集 —— 与 §4 的池化表不可比（op21 对 radix 的池化"
        "胜利来自高 BS cell）。首表 = 全 K × dtype 池化；"
        "下方可滚动明细表 = 完整 per-(K, dtype, rival) 分解。</p>")
            + f'<div class="cold">{t_c}</div><div class="warm">{t_w}</div>'
            + "</div><div class='card'>"
            + bi("<h3>op21 absolute-time envelope (data sensitivity)</h3>",
                 "<h3>op21 绝对时间包络（数据敏感度）</h3>")
            + f'<div class="cold">{e_c}</div><div class="warm">{e_w}</div>'
            + "</div><div class='card'>"
            + bi("<h3>Deterministic constructions (no RNG; "
                 "op23_op21_bounds/build_bounds_bundles.py)</h3>",
                 "<h3>确定性构造（零随机；op23_op21_bounds/"
                 "build_bounds_bundles.py）</h3>")
            + ctor_tbl
            + bi(
        "<p><b>Reading.</b> UB pins the P2 seed pmean into the acceptance band "
        "(single pass, ev1) AND satisfies the msc cluster fast-path gate "
        "(<code>gvr_msc_op.py:993-996</code>) at every cell — op21's analytic "
        "optimum on this grid. LB maximizes the stash mean (deepest undershoot) "
        "while defeating the fallback done-short-circuit "
        "(<code>gvr_msc_op.py:1104-1116</code>) — the leader CTA re-scans the "
        "full row 5–8×. <b>LB is mechanism-specific, not a global input "
        "minimum</b>: at 19/78 cells (K512 N≥32K, some K1024/K2048) REAL data "
        "is slower than the LB construction via the band/slot-overflow "
        "mechanism (§6, real-K512 cand≈4500 fallback) — hence the LB_eff "
        "column (per-cell worst of {lb, real}). Exactness gate 456/456; "
        "logits are byte-identical to the REAL bundles (only preIdx differs, "
        "backed by the §6 crossover). Full dossier: "
        "<code>op23_op21_bounds/RESULTS_SUMMARY.md</code>.</p>",
        "<p><b>解读。</b>UB 把 P2 种子 pmean 精确钉入接受带（单遍扫描 ev1），"
        "且每个 cell 都满足 msc 集群快路径门（<code>gvr_msc_op.py:993-996"
        "</code>）—— 是 op21 在该网格上的解析最优。LB 最大化 stash 均值"
        "（最深 undershoot），同时击破 fallback 短路种子（<code>"
        "gvr_msc_op.py:1104-1116</code>）—— leader CTA 全行重扫 5–8 遍。"
        "<b>LB 是机理特定下界，不是全输入空间最小值</b>：19/78 个 cell"
        "（K512 N≥32K 及部分 K1024/K2048）上 REAL 数据经 band/slot-溢出"
        "机理（§6，real-K512 cand≈4500 fallback）比 LB 构造更慢 —— 因此"
        "增设 LB_eff 列（按 cell 取 {lb, real} 中更差者）。精确性门 "
        "456/456；logits 与 REAL bundle 字节一致（仅 preIdx 不同，依据 §6 "
        "crossover）。完整卷宗：<code>op23_op21_bounds/RESULTS_SUMMARY.md"
        "</code>。</p>") + "</div>")


# ---------------- interactive charts (report/report.html style) ----------

def chart_records(data_all):
    """Flatten every loaded cell into compact JS records.

    Fields: s scenario, w sweep ('seq'|'bs'), K, d dtype, N, B BS, o op,
    c cold-µs, h warm-µs. ~13k records ≈ 1 MB JSON — replaces ~7 MB of
    pre-rendered SVG.
    """
    recs = []
    for scen, sweeps in data_all.items():
        for sweep, cells in sweeps.items():
            w = "seq" if sweep == "seqlen" else "bs"
            for (K, dt, N, BS), ops in cells.items():
                for op, r in ops.items():
                    recs.append({"s": scen, "w": w, "K": K, "d": dt, "N": N,
                                 "B": BS, "o": op,
                                 "c": round(r.get("us_cold", 0), 3),
                                 "h": round(r.get("us_warm", 0), 3)})
    return recs


def checks(cls, items, checked):
    return " ".join(
        f'<label class="ck"><input type="checkbox" class="{cls}" '
        f'value="{v}"{" checked" if v in checked else ""}>{lab}</label>'
        for v, lab in items)


def radios(name, items, sel):
    return " ".join(
        f'<label class="ck"><input type="radio" name="{name}" '
        f'value="{v}"{" checked" if v == sel else ""}>{lab}</label>'
        for v, lab in items)


def interactive_panels(data_all):
    """Two checkbox-driven Plotly panels (seqlen + BS) + one <script>."""
    import json as _json
    scen_items = [(s, SCEN_LABEL[s].split(" ")[0]) for s in ALL_SCEN
                  if s in data_all]
    op_items = [(o, OP_SHORT[o]) for o in OPS]
    k_items = [(str(k), f"K={k} ({K_MODEL[k]})") for k in KS]
    dt_items = [(d, d) for d in DTS]
    bs_ns = sorted({N for s in data_all.values() if "bs" in s
                    for (_, _, N, _) in s["bs"]})
    n_items = [(str(n), f"{n // 1024}K") for n in bs_ns]

    ctrl_seq = ('<div class="ctl">'
                + bi("<b>scenarios</b>", "<b>场景</b>", "span") + " "
                + checks("sck1", scen_items, {"real", "ub", "lb"})
                + '<br>' + bi("<b>operators</b>", "<b>算子</b>", "span") + " "
                + checks("ock1", op_items, set(OPS))
                + '<br>' + radios("kk1", k_items, "2048") + " · "
                + radios("dd1", dt_items, "fp32") + "</div>")
    ctrl_bs = ('<div class="ctl">'
               + bi("<b>scenarios</b>", "<b>场景</b>", "span") + " "
               + checks("sck2", [i for i in scen_items
                                 if i[0] in SCENARIOS], {"real"})
               + bi("<span class='small'> (ub/lb: seqlen only)</span>",
                    "<span class='small'>（ub/lb 仅有 seqlen 数据）</span>",
                    "span")
               + '<br>' + bi("<b>operators</b>", "<b>算子</b>", "span") + " "
               + checks("ock2", op_items, set(OPS))
               + '<br>' + radios("kk2", k_items, "2048") + " · "
               + radios("dd2", dt_items, "fp32") + " · N: "
               + radios("nn2", n_items, "131072") + "</div>")

    js_data = _json.dumps(chart_records(data_all), separators=(",", ":"))
    js = """
<script>
const D=%s;
const COL=%s,DASH=%s,SHORT=%s,MAIN=%s;
const T={en:{lat:'Latency vs N',spd:'Speedup rival/op21 (>1 = op21 faster)',
latb:'Latency vs BS',spdb:'Speedup rival/op21 vs BS',us:'µs (log)',
n:'N (post-compress, log)',bs:'batch size (log)',r:'ratio ×'},
zh:{lat:'延迟 对 N',spd:'加速比 rival/op21（>1 = op21 更快）',
latb:'延迟 对 BS',spdb:'加速比 rival/op21 对 BS',us:'µs（对数）',
n:'N（压缩后，对数）',bs:'batch size（对数）',r:'比值 ×'}};
function lang(){return document.getElementById('lang-zh').checked?'zh':'en'}
function reg(){return document.getElementById('m-cold').checked?'c':'h'}
function vals(cls){return [...document.querySelectorAll('.'+cls+':checked')].map(x=>x.value)}
function rad(n){const e=document.querySelector('input[name='+n+']:checked');return e?e.value:null}
function LAY(t,xt,yt,xlog){return {title:{text:t,font:{size:15}},
paper_bgcolor:'#161b22',plot_bgcolor:'#0f1419',font:{color:'#e6e6e6',size:12},
margin:{t:42,r:10,b:48,l:56},showlegend:true,
legend:{orientation:'h',y:-0.22,font:{size:10.5}},
xaxis:{title:xt,type:'log',gridcolor:'#2a3340'},
yaxis:{title:yt,type:xlog?'log':'linear',gridcolor:'#2a3340'}}}
function pick(w,scens,K,d){const m={};
D.filter(r=>r.w==w&&scens.includes(r.s)&&r.K==K&&r.d==d).forEach(r=>{
(m[r.s]=m[r.s]||{});(m[r.s][r.o]=m[r.s][r.o]||{});m[r.s][r.o][w=='seq'?r.N:r.B+'|'+r.N]=r});
return m}
function seqDraw(){const L=T[lang()],rg=reg(),scens=vals('sck1'),ops=vals('ock1'),
K=rad('kk1'),d=rad('dd1'),m=pick('seq',scens,K,d),lat=[],spd=[];
for(const s of scens){if(!m[s])continue;
for(const o of ops){const c=m[s][o];if(!c)continue;
const xs=Object.keys(c).map(Number).sort((a,b)=>a-b);
lat.push({x:xs,y:xs.map(N=>c[N][rg]),name:s+'·'+SHORT[o],mode:'lines+markers',
marker:{size:5},line:{color:COL[o],dash:DASH[s]}});
if(o!=MAIN&&m[s][MAIN]){const t21=m[s][MAIN];
const xs2=xs.filter(N=>t21[N]);
spd.push({x:xs2,y:xs2.map(N=>c[N][rg]/t21[N][rg]),name:s+'·'+SHORT[o]+'/op21',
mode:'lines+markers',marker:{size:5},line:{color:COL[o],dash:DASH[s]}});}}}
spd.push({x:[4096,1048576],y:[1,1],mode:'lines',showlegend:false,
line:{color:'#888',width:1}});
Plotly.newPlot('p_slat',lat,LAY(L.lat+' (K='+K+', '+d+', BS=1, '+(rg=='c'?'cold':'warm')+'-L2)',L.n,L.us,true),{responsive:true});
Plotly.newPlot('p_sspd',spd,LAY(L.spd,L.n,L.r,false),{responsive:true});}
function bsDraw(){const L=T[lang()],rg=reg(),scens=vals('sck2'),ops=vals('ock2'),
K=rad('kk2'),d=rad('dd2'),N=rad('nn2'),m=pick('bs',scens,K,d),lat=[],spd=[];
for(const s of scens){if(!m[s])continue;
for(const o of ops){const c=m[s][o];if(!c)continue;
const bs=Object.keys(c).filter(k=>k.split('|')[1]==N).map(k=>Number(k.split('|')[0])).sort((a,b)=>a-b);
if(!bs.length)continue;
lat.push({x:bs,y:bs.map(B=>c[B+'|'+N][rg]),name:s+'·'+SHORT[o],mode:'lines+markers',
marker:{size:5},line:{color:COL[o],dash:DASH[s]}});
if(o!=MAIN&&m[s][MAIN]){const t21=m[s][MAIN];
const bs2=bs.filter(B=>t21[B+'|'+N]);
spd.push({x:bs2,y:bs2.map(B=>c[B+'|'+N][rg]/t21[B+'|'+N][rg]),
name:s+'·'+SHORT[o]+'/op21',mode:'lines+markers',marker:{size:5},
line:{color:COL[o],dash:DASH[s]}});}}}
spd.push({x:[1,2048],y:[1,1],mode:'lines',showlegend:false,line:{color:'#888',width:1}});
Plotly.newPlot('p_blat',lat,LAY(L.latb+' (K='+K+', '+d+', N='+(N/1024)+'K, '+(rg=='c'?'cold':'warm')+'-L2)',L.bs,L.us,true),{responsive:true});
Plotly.newPlot('p_bspd',spd,LAY(L.spdb,L.bs,L.r,false),{responsive:true});}
function drawAll(){seqDraw();bsDraw()}
document.querySelectorAll('.sck1,.ock1,input[name=kk1],input[name=dd1],.sck2,.ock2,input[name=kk2],input[name=dd2],input[name=nn2]')
.forEach(e=>e.onchange=drawAll);
['lang-zh','lang-en','m-cold','m-warm'].forEach(id=>{const e=document.getElementById(id);
if(e)e.addEventListener('change',()=>setTimeout(drawAll,0));});
drawAll();
</script>""" % (js_data,
                _json.dumps(COL), _json.dumps(SCEN_DASH),
                _json.dumps(OP_SHORT), _json.dumps(MAIN))
    return ctrl_seq, ctrl_bs, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(HERE.parent / "results_b200_op22"))
    ap.add_argument("--op23-root", default=str(HERE.parent / "results_b200_op23"))
    ap.add_argument("--out", default=str(HERE / "REPORT.html"))
    args = ap.parse_args()
    root = Path(args.out_root)

    data = {s: load_scenario(root, s) for s in SCENARIOS}
    have = {s: sorted(data[s]) for s in SCENARIOS}
    print("loaded:", {s: v for s, v in have.items()})

    # op23 bounds scenarios (seqlen only, optional)
    op23_root = Path(args.op23_root)
    data_all = dict(data)
    for s in BOUNDS_SCENARIOS:
        d = load_scenario(op23_root, s)
        if d:
            data_all[s] = d
    print("bounds loaded:", {s: sorted(data_all[s])
                             for s in BOUNDS_SCENARIOS if s in data_all})

    csv_outs = write_csvs(data)
    print("csv:", csv_outs)

    ctrl_seq, ctrl_bs, chart_js = interactive_panels(data_all)

    # ---- assemble ----
    B = []
    B.append(f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>op22 — GVR op21 vs rivals on temporal-synth fixed-hit-rate data</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head><body>
<input type="radio" name="lang" id="lang-zh" checked>
<input type="radio" name="lang" id="lang-en">
<input type="radio" name="metric" id="m-cold" checked>
<input type="radio" name="metric" id="m-warm">
<div class="wrap">
<div class="langbar"><label for="lang-zh">中文</label><label for="lang-en">English</label></div>
<h1>{bi("op22 — GVR op#21 vs 4 rivals · temporal-synth FIXED hit-rate data",
        "op22 — GVR op#21 对 4 个对手 · 固定 hit-rate 时序合成数据", "span")}</h1>""")
    B.append('<p class="meta">2026-07-07 · B200 (sm_100), GPU0 · '
             'branch <code>omni/op21-gvr-prod</code> · bucket <code>indexer_topk_op_bench/'
             'op22_temporal_fixed_hr_bench/</code></p>')

    # ---- intro (reference style: objective/timing/structure prose) ----
    B.append(bi(
        """<p>Apples-to-apples comparison of the production-dispatch GVR kernel <b>op#21</b>
        (<code>gvr_ms_auto</code>, iter12 @f51f50f4da) against 4 rivals, on temporally-coherent
        synthetic data whose <b>temporal-hint quality (hit rate) is controlled</b>: GVR seeds its
        threshold search from the previous step's top-K hint, so its cost depends on the hint,
        while the radix/streaming rivals are hint-blind (§6 shows the dependence is on
        threshold-init quality and is NON-monotone in hit rate). Timing = <b>nsys pure-kernel GPU
        time</b> (per-cell NVTX range projection; the L2-evict kernel is excluded). Two L2 regimes:
        <span class="reg">cold-L2</span> (512 MB flush before each launch — canonical, isolated
        worst case) and <span class="reg">warm-L2</span> (logits hot in L2 — models the fused
        indexer→top-K path). Three scenarios: <span class="reg">BEST</span> (deep-layer marginal,
        hr=0.90), <span class="reg">WORST</span> (shallow-layer marginal, hr=0.05) and
        <span class="reg">REAL</span> (aggregate layer mixture, hr sampled from the real per-step
        distribution — the anchor comparable to <code>report/report.html</code>).
        <b>Section 6</b> resolves the mechanism behind the stress-scenario losses;
        <b>Section 7</b> brackets op21 with deterministic best/worst-case constructions (op23).</p>""",
        """<p>对生产 dispatch 的 GVR 内核 <b>op#21</b>（<code>gvr_ms_auto</code>，
        iter12 @f51f50f4da）与 4 个对手做同口径对比，数据为
        <b>时序提示质量（hit rate）受控</b>的时序相关合成数据：
        GVR 以上一步 top-K 提示为阈值搜索种子，代价依赖提示质量，
        而 radix/streaming 对手对提示不敏感（§6 证明该依赖实为
        『阈值初始化质量』，且对 hit rate 非单调）。计时 =
        <b>nsys 纯 kernel GPU 时间</b>（按 cell 的 NVTX 区间投影；L2 清空 kernel
        已排除）。两种 L2 状态：<span class="reg">冷 L2</span>（每次启动前清空
        512 MB —— 规范化、隔离的最坏情况）与 <span class="reg">热 L2</span>
        （logits 驻留 L2 —— 模拟融合 indexer→top-K 路径）。三场景：
        <span class="reg">BEST</span>（深层 marginal，hr=0.90）、<span class="reg">WORST</span>
        （浅层 marginal，hr=0.05）、<span class="reg">REAL</span>（aggregate 层混合、
        hr 按真实 per-step 分布采样 —— 与 <code>report/report.html</code> 可比的锚点）。
        <b>第 6 节</b>收口了应力场景失利的机制；<b>第 7 节</b>用
        确定性最优/最差构造为 op21 定界（op23）。</p>"""))

    # ---- TL;DR verdict card ----
    tl_en, tl_zh = tldr(data)
    if all(s in data_all for s in BOUNDS_SCENARIOS):
        bnd = {}
        for s in ("ub", "real", "lb"):
            cells = data_all.get(s, {}).get("seqlen", {})
            bnd[s] = gm(list(ratio_cells(cells, "radix_cutedsl",
                                         "us_cold").values()))
        line = (f"UB {fmt_r(bnd['ub'])} / real {fmt_r(bnd['real'])} / "
                f"LB {fmt_r(bnd['lb'])}")
        tl_en += (f"<p><b>op23 bounds (§7):</b> vs Radix cold, seqlen-BS=1 "
                  f"subset — {line}; op21 absolute-time envelope LB/UB "
                  f"≈ 1.8–2.4×.</p>")
        tl_zh += (f"<p><b>op23 定界（§7）：</b>对 Radix 冷 L2，seqlen-BS=1 "
                  f"子集 — {line}；op21 绝对时间包络 LB/UB ≈ 1.8–2.4×。</p>")
    B.append(f'<div class="card" style="border-color:#76b900">{bi(tl_en, tl_zh)}</div>')

    # ---- deep-dive link card (reference pattern) ----
    B.append('<div class="card" style="border-color:#76b900">'
             + bi("<h3>&#128214; Companion material</h3><p>Synthetic-baseline &amp; real-data "
                  "anchor: <a href='../report/report.html'><b>report/report.html</b></a> · "
                  "algorithm deep-dive (flowcharts + pseudocode per operator family): "
                  "<a href='../report/algorithms.html'><b>algorithms.html</b></a> · "
                  "mechanism dossier for §6: <code>MECH_FINDINGS.md</code>.</p>",
                  "<h3>&#128214; 配套材料</h3><p>合成基线与真实数据锚点："
                  "<a href='../report/report.html'><b>report/report.html</b></a> · "
                  "算法详解（各算子族流程图 + 伪代码）："
                  "<a href='../report/algorithms.html'><b>algorithms.html</b></a> · "
                  "§6 机制卷宗：<code>MECH_FINDINGS.md</code>。</p>")
             + "</div>")

    # ---- test environment card (KPI chips) ----
    ncells = {s: sum(len(v) for v in data[s].values()) for s in SCENARIOS}
    kpis = ("<div class='kpi'>GPU <b>B200 SXM</b> (sm_100)</div>"
            "<div class='kpi'>torch <b>2.11</b></div>"
            "<div class='kpi'>cutlass-dsl <b>4.x</b></div>"
            "<div class='kpi'>kernel <b>op#21 @f51f50f4da</b></div>")
    kpi_t_en = ("<div class='kpi'>timing <b>nsys pure-kernel GPU time, cold+warm L2</b> "
                "(NVTX-range projection)</div>")
    kpi_t_zh = ("<div class='kpi'>计时 <b>nsys 纯 kernel GPU 时间，冷+热 L2</b>"
                "（NVTX 区间投影）</div>")
    cells_en = ", ".join(f"<b>{s}</b> {n}" for s, n in ncells.items())
    env_en = (kpis + kpi_t_en +
              "<p class='small' style='margin-top:8px'>Nodes: <b>umbriel-b200-040</b> (real 18/18 "
              "+ best 16/18 batches) / <b>umbriel-b200-049</b> (best bs-K2048-bf16/fp16 + worst "
              "+ all bs_hugeN). Absolute µs do not transfer across nodes; the canonical metric "
              "is the per-cell rival ratio, which is node-internal (both ops of a cell always ran "
              "in the same nsys batch on the same node). Identical bundles for every op within a "
              "cell (generated once, loaded from disk). Exactness pre-gate 456/456 (sorted "
              "value-multiset — GVR output order is atomicAdd-nondeterministic). Cells loaded: "
              + cells_en + ".</p>")
    env_zh = (kpis + kpi_t_zh +
              "<p class='small' style='margin-top:8px'>节点：<b>umbriel-b200-040</b>"
              "（real 18/18 + best 16/18 批）/ <b>umbriel-b200-049</b>（best "
              "bs-K2048-bf16/fp16 + worst + 全部 bs_hugeN）。绝对 µs 不可"
              "跨节点迁移；规范指标是按-cell 的对手比值，"
              "它是节点内部的（同一 cell 的两个 op 总在同节点"
              "同 nsys 批内运行）。cell 内所有 op 输入完全相同"
              "（生成一次、磁盘加载）。exactness 预门 456/456"
              "（排序值多重集判据 —— GVR 输出顺序 atomicAdd "
              "非确定）。已加载 cells：" + cells_en + "。</p>")
    B.append('<div class="card">'
             + bi("<h3>Test environment</h3>" + env_en,
                  "<h3>测试环境</h3>" + env_zh) + "</div>")

    # ---- operators & methodology card (reference table format) ----
    ops_rows_en = (
        f"<tr><td style='color:{COL['gvr_ms_auto']}'>{OP_LABEL['gvr_ms_auto']} — <b>object under test</b></td>"
        "<td>CUDA C++ (msc dispatch)</td><td>exact (heuristic pre_idx threshold-seed)</td><td>fp32/bf16/fp16 × 512/1024/2048</td></tr>"
        f"<tr><td style='color:{COL['gvr_cutedsl']}'>{OP_LABEL['gvr_cutedsl']}</td>"
        "<td>CuTe DSL</td><td>exact (heuristic seed)</td><td>full — op21's code start</td></tr>"
        f"<tr><td style='color:{COL['gvr_multicta_cutedsl']}'>{OP_LABEL['gvr_multicta_cutedsl']}</td>"
        "<td>CuTe DSL (DSMEM cluster)</td><td>exact (heuristic seed)</td><td>full</td></tr>"
        f"<tr><td style='color:{COL['radix_cutedsl']}'>{OP_LABEL['radix_cutedsl']}</td>"
        "<td>CuTe DSL</td><td>exact (hint-blind)</td><td>full — strongest hint-blind rival</td></tr>"
        f"<tr><td style='color:{COL['sglang_streaming']}'>{OP_LABEL['sglang_streaming']}</td>"
        "<td>CUDA C++ (external)</td><td>exact (hint-blind)</td><td>fp32 × 512/1024 only</td></tr>")
    scen_tbl = ("<table><tr><th>scenario</th><th>marginal cfg</th><th>hit rate</th><th>intent</th></tr>"
                "<tr><td>BEST</td><td><code>beta_deep</code> (strong temporal family)</td><td>fixed 0.90</td>"
                "<td>GVR best case — near-perfect hint</td></tr>"
                "<tr><td>WORST</td><td><code>beta_shallow</code> (weak temporal family)</td><td>fixed 0.05</td>"
                "<td>GVR worst case — hint nearly useless</td></tr>"
                "<tr><td>REAL</td><td><code>aggregate</code> (layer mixture)</td><td>sampled per-step</td>"
                "<td>anchor — comparable to report.html</td></tr>"
                "<tr><td>UB (op23)</td><td>REAL logits byte-identical</td><td>0.50 constructed</td>"
                "<td>op21 analytic optimum — ev1 + msc fast path (§7)</td></tr>"
                "<tr><td>LB (op23)</td><td>REAL logits byte-identical</td><td>(K−1)/K + 1 stale deep</td>"
                "<td>op21 adversarial — leader re-scans 5–8× (§7)</td></tr></table>")
    B.append('<div class="card">' + bi(
        "<h3>Operators &amp; methodology</h3>"
        "<table><tr><th>Operator</th><th>Kind</th><th>exact?</th><th>dtypes×K</th></tr>"
        + ops_rows_en + "</table>"
        "<p>Protocol (report.html convention): 1-row bundle replicated to BS (all rows share L2 "
        "lines at high BS — documented caveat), <code>seq_lens = N·cr</code>, eager+sync inside "
        "the NVTX range; 20 cold + 50 warm reps/cell. N∈{512K,1M} rows are iid draws from the "
        "same empirical CDF+GPD tail (marginals calibrated on 64K captures) — fine for kernel "
        "benchmarking, noted as caveat.</p><h3>Scenario design</h3>" + scen_tbl,
        "<h3>算子与方法学</h3>"
        "<table><tr><th>算子</th><th>类型</th><th>精确?</th><th>dtype×K</th></tr>"
        + ops_rows_en + "</table>"
        "<p>协议（report.html 惯例）：单行 bundle 复制到 BS（高 BS "
        "下各行共享 L2 —— 已记录的 caveat），<code>seq_lens = N·cr</code>，"
        "NVTX 区间内 eager+sync；每 cell 20 cold + 50 warm。N∈{512K,1M} 行是同一"
        "经验 CDF+GPD 尾部的 iid 抽样（marginal 在 64K captures 上标定）—— "
        "对内核基准足够，已注明为 caveat。</p>"
        "<h3>场景设计</h3>" + scen_tbl) + "</div>")

    # ---- synthetic input generation card ----
    B.append('<div class="card">' + bi(
        "<h3>Synthetic input generation (random data characteristics)</h3>"
        "<p>Generator: <code>.claude/skills/indexer-topk-temporal-synth</code> (empirical "
        "inverse-CDF + GPD tail marginal, rank-conditional temporal model; supersedes the legacy "
        "single-Beta skills — motivating falsification: <code>synth_vs_real_validation/</code>). "
        "Validation gates, all PASS (2026-07-06):</p>" + gates_table(),
        "<h3>合成输入生成（随机数据特征）</h3>"
        "<p>生成器：<code>.claude/skills/indexer-topk-temporal-synth</code>（经验逆 "
        "CDF + GPD 尾部 marginal、按秩条件的时序模型；取代旧单-Beta "
        "skills —— 证伪研究见 <code>synth_vs_real_validation/</code>）。"
        "验证门全部 PASS（2026-07-06）：</p>" + gates_table()) + "</div>")

    # ---- deterministic reproduction card ----
    B.append('<div class="card">' + bi(
        "<h3>Deterministic reproduction &amp; SKILL invocation</h3>",
        "<h3>确定性复现与 SKILL 调用</h3>") + repro_html() + "</div>")

    # ---- auto-analysis card ----
    items = analysis_items(data)
    B.append('<div class="card">' + bi(
        "<h3>Auto-analysis (cold-L2, per scenario)</h3><ul>"
        + "".join(f"<li>{en}</li>" for en, _ in items) + "</ul>",
        "<h3>自动分析（冷 L2，按场景）</h3><ul>"
        + "".join(f"<li>{zh}</li>" for _, zh in items) + "</ul>") + "</div>")

    # ---- operator legend + regime selector card (reference pattern) ----
    op_legend = " ".join(
        f"<span style='color:{COL[o]};font-weight:bold;margin-right:14px'>"
        f"● {OP_LABEL[o]}</span>" for o in OPS)
    B.append('<div class="card regbar">'
             + bi("<b>Operators (fixed colors on every chart):</b>",
                  "<b>算子（所有图表固定配色）：</b>", "span")
             + f"<div style='margin:8px 0'>{op_legend}</div>"
             + bi("<b>L2 regime:</b>", "<b>L2 状态：</b>", "span")
             + ' <label for="m-cold">cold-L2 (flushed)</label>'
             + '<label for="m-warm">warm-L2</label> '
             + bi("<span class='small'>(switches every table via CSS and every "
                  "Plotly chart via the panel redraw)</span>",
                  "<span class='small'>（表格经 CSS 切换，Plotly "
                  "图表由面板联动重绘）</span>", "span")
             + "</div>")

    # ---- 1. seqlen ----
    B.append("<h2>" + bi("1. Seq-len sweep (BS=1)",
                         "1. 序列长度扫描（BS=1）", "span") + "</h2>")
    B.append(bi(
        "<p>Per (K, dtype): latency (top) and op21-speedup (bottom) vs N, three scenarios "
        "side-by-side — the reference report's latency+speedup pairing. In REAL, each N cell "
        "draws its own layer + hr (by design), so N-trends are jagged — judge per-cell ratios, "
        "not curve smoothness.</p>",
        "<p>每 (K, dtype)：延迟（上）与 op21 加速比（下）"
        "对 N，三场景并排 —— 即参考报告的延迟+加"
        "速比成对版式。REAL 场景每个 N cell 独立抽层与 "
        "hr（设计使然），N 趋势有锯齿 —— 看每 cell "
        "的比值，不看曲线平滑度。</p>"))
    B.append('<div class="card">' + ctrl_seq
             + '<div class="row"><div id="p_slat" class="plt"></div>'
               '<div id="p_sspd" class="plt"></div></div></div>')

    # ---- 2. BS scaling ----
    B.append("<h2>" + bi("2. BS-scaling (BS 1→2048, N 4K→256K; stretch N 512K/1M at BS≤64)",
                         "2. BS 扩展性（BS 1→2048，N 4K→256K；补充档 "
                         "N 512K/1M 到 BS≤64）", "span") + "</h2>")
    B.append(bi(
        f"<p>Per (K, dtype): latency vs BS at the anchor N={BS_ANCHOR_N//1024}K (top) and speedup "
        "rival/op21 vs BS (&gt;1 ⇒ op21 faster; one line per N — the 512K/1M lines come from "
        "the bs_hugeN stretch grid, BS 2–64). Speedup rows: vs Radix (hint-blind rival), vs GVR "
        "single-CTA (op21's own code start).</p>",
        f"<p>每 (K, dtype)：锚点 N={BS_ANCHOR_N//1024}K 处延迟对 BS（上），"
        "及 rival/op21 加速比对 BS（&gt;1 ⇒ op21 更快；每条线一个 "
        "N —— 512K/1M 两条线来自 bs_hugeN 补充档，BS 2–64）。"
        "加速比两行：对 Radix（提示盲对手）、对 GVR 单 "
        "CTA（op21 的代码起点）。</p>"))
    B.append('<div class="card">' + ctrl_bs
             + '<div class="row"><div id="p_blat" class="plt"></div>'
               '<div id="p_bspd" class="plt"></div></div></div>')

    # ---- 3. full data (reference §3: CSV download + scrollable table) ----
    B.append("<h2>" + bi("3. Full data", "3. 完整数据", "span") + "</h2>")
    B.append('<div class="card">'
             + bi("<p>Download: <a href='op22_seqlen_data.csv'>op22_seqlen_data.csv</a> · "
                  "<a href='op22_bs_data.csv'>op22_bs_data.csv</a> (both regimes + op21 cold "
                  "speedups; first column = scenario). Table below: per-cell µs in the active "
                  "L2 regime.</p>",
                  "<p>下载：<a href='op22_seqlen_data.csv'>op22_seqlen_data.csv</a> · "
                  "<a href='op22_bs_data.csv'>op22_bs_data.csv</a>（两种 L2 状态 + "
                  "op21 冷加速比；第一列 = 场景）。下表："
                  "当前 L2 状态下的每-cell µs。</p>")
             + f'<div class="scrolltbl cold">{fulldata_table(data, "us_cold")}</div>'
             + f'<div class="scrolltbl warm">{fulldata_table(data, "us_warm")}</div>'
             + "</div>")

    # ---- 4. geomean + hr-sensitivity tables ----
    B.append("<h2>" + bi("4. Geomean ratio &amp; hit-rate sensitivity",
                         "4. 几何均值比值与 hit-rate 敏感度", "span") + "</h2>")
    B.append('<div class="card">'
             + bi("<h3>Geomean time ratio rival/op21 (&gt;1 ⇒ op21 faster; pooled seqlen+bs cells)</h3>",
                  "<h3>几何均值时间比 rival/op21（&gt;1 ⇒ op21 更快；"
                  "seqlen+bs cell 合并）</h3>")
             + f'<div class="cold">{geomean_table(data, "us_cold")}</div>'
             + f'<div class="warm">{geomean_table(data, "us_warm")}</div>'
             + "</div>")
    B.append('<div class="card">'
             + bi("<h3>Hit-rate sensitivity — worst/best time on the same cells</h3>"
                  "<p>How much slower each op runs when hr drops 0.90→0.05 (marginal also shifts "
                  "deep→shallow). ≈1.0 ⇒ hint-insensitive; the GVR family's values quantify its "
                  "worst-case exposure.</p>",
                  "<h3>hit-rate 敏感度 — 相同 cell 上 worst/best 用时比</h3>"
                  "<p>hr 从 0.90→0.05（marginal 同时 deep→shallow）时各 op "
                  "变慢多少。≈1.0 ⇒ 对提示不敏感；GVR 家族"
                  "的数值量化了其最坏暴露。</p>")
             + f'<div class="cold">{hr_sensitivity_table(data, "us_cold")}</div>'
             + f'<div class="warm">{hr_sensitivity_table(data, "us_warm")}</div>'
             + "</div>")

    # ---- 5. findings ----
    B.append("<h2>" + bi("5. Findings — extreme cells (op21 vs Radix)",
                         "5. Findings — 极端 cell（op21 对 Radix）", "span") + "</h2>")
    B.append('<div class="card">' + bi(
        "<p>Per scenario, the 5 worst and 5 best cells by radix/op21 time ratio "
        "(active L2 regime), with the cell's hit-rate, source layer and op21 dispatch path — "
        "the raw material for regime analysis. §6 identifies the mechanism behind "
        "the loss cells: extra full-row re-scans triggered by poisoned "
        "threshold-init, a pure function of the preIdx hint (NOT value "
        "tie-density, and NOT monotone in hr).</p>",
        "<p>每场景按 radix/op21 时间比（当前 L2 状态）列最差 "
        "5 个与最好 5 个 cell，附该 cell 的 hit-rate、来源层与 "
        "op21 dispatch 路径 —— regime 分析的原始素材。失利 cell "
        "背后的机制见 §6：阈值初始化被污染导致的"
        "额外全行重扫，纯粹由 preIdx 提示决定（与 value "
        "tie 密度无关，对 hr 也非单调）。</p>")
        + f'<div class="cold">{findings_html(data, "us_cold")}</div>'
        + f'<div class="warm">{findings_html(data, "us_warm")}</div>'
        + "</div>")

    # ---- 6. mechanism ----
    B.append("<h2>" + bi("6. Mechanism — why hr=0.90 slows the GVR family down",
                         "6. 机制 — 为什么 hr=0.90 反而拖慢整个 "
                         "GVR 家族", "span") + "</h2>")
    B.append(mech_html(data))

    # ---- 7. op23 deterministic bounds ----
    bh = bounds_html(data_all)
    if bh:
        B.append("<h2>" + bi(
            "7. op23 — deterministic UB/LB bounds on op21 (best/worst-case "
            "speedup)", "7. op23 — op21 的确定性上/下界（最优/最差加速比）",
            "span") + "</h2>")
        B.append(bi(
            "<p>Two DETERMINISTIC preIdx constructions derived from the §6 "
            "mechanism bracket op21's data sensitivity; logits stay "
            "byte-identical to REAL. Select the <b>ub</b>/<b>lb</b> scenarios "
            "in the §1 interactive panel to see the per-N curves. Measured "
            "2026-07-07 on umbriel-b200-037 GPU0 (ratios are node-internal).</p>",
            "<p>由 §6 机理推导的两个<b>确定性</b> preIdx 构造为 op21 的数据"
            "敏感度定界；logits 与 REAL 字节一致。在 §1 交互面板勾选 "
            "<b>ub</b>/<b>lb</b> 场景即可查看各 N 曲线。测量于 2026-07-07 "
            "umbriel-b200-037 GPU0（比值为节点内部口径）。</p>"))
        B.append(bh)

    B.append(chart_js)
    B.append("</div></body></html>")
    out = Path(args.out)
    html = "\n".join(B)
    out.write_text(html)
    n_script = html.count("<script")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB, <script> tags: "
          f"{n_script})")
    assert n_script == 2, n_script  # plotly CDN + chart driver


if __name__ == "__main__":
    main()
