#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op30 — build REPORT.html + CSVs from the parsed nsys sweeps.

op22-style report (dark NVIDIA theme, plotly-CDN chart pairs, KPI chips,
CSS-only radio toggles for lang / cold-warm, checkbox-driven Plotly redraw)
for the op30 campaign: the op22 10-arm grid re-run on the two NEW synthetic
scenarios defined relative to the GVR (cuteDSL) BASELINE itself —
BEST = data where GVR-base is fastest, WORST = where it is slowest
(absolute cold-L2 poles from the phase-1 calibration sweep, CALIBRATION.md
+ scen_op30.json). All speedups here are t(gvr_cutedsl)/t(op) — &gt;1 means
the op is FASTER than the baseline (opposite orientation to op22, whose
main object was op21/gvr_ms_auto).

Emits next to this script:
    REPORT.html
    op30_seqlen_data.csv / op30_bs_data.csv / op30_bs_hugeN_data.csv

Tolerates partial data (missing scenarios / sweeps / K / dtype are simply
absent from charts and tables; KPI chips show n/a). Rows with "error" are
skipped and counted for the data-quality footnote; any exact=="FAIL" row
aborts the build.

Usage: python3 gen_report_op30.py [<out_root>]   default ../results_b200_op30
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MAIN = "gvr_cutedsl"
OPS = ["gvr_cutedsl", "gvr_multicta_cutedsl", "radix_cutedsl",
       "radix_single_cuda", "radix_multi_cuda", "op25_hls", "op27_hls",
       "op26_r0auto", "sglang_v2", "flashinfer_topk"]
RIVALS = [o for o in OPS if o != MAIN]
OP_LABEL = {
    "gvr_cutedsl": "GVR (cuteDSL) — base",
    "gvr_multicta_cutedsl": "GVR multi-CTA (cuteDSL, PR#15198)",
    "radix_cutedsl": "Radix (cuteDSL)",
    "radix_single_cuda": "Radix single-CTA (CUDA)",
    "radix_multi_cuda": "Radix multi-CTA (CUDA)",
    "op25_hls": "GVR op#21 ms_auto (HLS-op25)",
    "op27_hls": "GVR op#21 ms_auto (HLS-op27)",
    "op26_r0auto": "GVR op#26 R0 (auto 1CTA/MC dispatch)",
    "sglang_v2": "SGLang v2 top-K (main 2026-07)",
    "flashinfer_topk": "FlashInfer top_k (0.6.11)",
}
OP_SHORT = {
    "gvr_cutedsl": "GVR-base", "gvr_multicta_cutedsl": "GVR-mCTA",
    "radix_cutedsl": "Radix", "radix_single_cuda": "Radix-1cta-CUDA",
    "radix_multi_cuda": "Radix-mCTA-CUDA", "op25_hls": "op25",
    "op27_hls": "op27", "op26_r0auto": "op26-R0",
    "sglang_v2": "SGLang-v2", "flashinfer_topk": "FlashInfer",
}
# op22 palette reused where the arm exists there (gvr_cutedsl / mcta /
# radix_cutedsl / sglang-red); new arms get distinct hues.
COL = {
    "gvr_cutedsl": "#b3e05a", "gvr_multicta_cutedsl": "#2ec4b6",
    "radix_cutedsl": "#4ea8de", "radix_single_cuda": "#b085f5",
    "radix_multi_cuda": "#f06595", "op25_hls": "#ffa94d",
    "op27_hls": "#ffd700", "op26_r0auto": "#7ee787",
    "sglang_v2": "#d62728", "flashinfer_topk": "#c0c6cf",
}
FP32_ONLY = {"sglang_v2", "flashinfer_topk"}

SCENARIOS = ["best", "worst"]
SCEN_LABEL = {
    "best": "BEST (GVR-base fastest pole)",
    "worst": "WORST (GVR-base slowest pole)",
}
SCEN_DASH = {"best": "solid", "worst": "dash"}
# scen_op30.json poles (also restated in §0 prose)
SCEN_POLE = {
    "best": {512: "beta_shallow hr0.30", 1024: "aggregate hr0.15",
             2048: "beta_shallow hr0.15"},
    "worst": {512: "beta_shallow hr0.90", 1024: "beta_deep hr0.85",
              2048: "aggregate hr0.85"},
}
KS = [512, 1024, 2048]
DTS = ["fp32", "bf16", "fp16"]
K_MODEL = {512: "V4-Flash", 1024: "V4-Pro", 2048: "V3.2"}
BS_ANCHOR_N = 131072
KPI_ARMS = ["op27_hls", "op26_r0auto", "sglang_v2", "radix_cutedsl"]
SUBS = [("seqlen", "seqlen_sweep"), ("bs", "bs_scaling"),
        ("bs_hugeN", "bs_hugeN")]
HUGE_NS = {524288, 1048576}


# ---------------- data ----------------

def load_all(root):
    """({scen: {sweep: {(K,dt,N,BS): {op: rec}}}}, err_counts, fails)."""
    data, errs, fails = {}, {}, []
    for scen in SCENARIOS:
        sweeps = {}
        for sweep, sub in SUBS:
            p = root / scen / sub / "results.jsonl"
            if not p.exists():
                continue
            cells = {}
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("exact") == "FAIL":
                    fails.append((scen, sweep, r["op"], r["K"],
                                  r["dtype"], r["N"]))
                if "error" in r:
                    key = f"{scen}/{sweep}"
                    errs[key] = errs.get(key, 0) + 1
                    continue
                if r.get("us_cold") is None and r.get("us_warm") is None:
                    continue
                cells.setdefault(
                    (r["K"], r["dtype"], r["N"], r["BS"]), {})[r["op"]] = r
            if cells:
                sweeps[sweep] = cells
        if sweeps:
            data[scen] = sweeps
    return data, errs, fails


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def pooled_cells(data, scen):
    """{(sweep,K,dt,N,BS): {op: rec}} over all three sweeps."""
    out = {}
    for sweep, _ in SUBS:
        out.update({(sweep,) + k: v
                    for k, v in data.get(scen, {}).get(sweep, {}).items()})
    return out


def speedups(cells, op, metric):
    """Per-cell speedup t(base)/t(op) (>1 => op faster than GVR-base)."""
    out = []
    for v in cells.values():
        a, b = v.get(MAIN), v.get(op)
        if a and b and a.get(metric) and b.get(metric):
            out.append(a[metric] / b[metric])
    return out


def wb_ratio(data, metric="us_cold"):
    """GVR-base worst/best time ratio on the common cells (>1 => worst
    slower). None until both scenarios have data."""
    pb, pw = pooled_cells(data, "best"), pooled_cells(data, "worst")
    rs = []
    for key, ops_b in pb.items():
        ops_w = pw.get(key)
        if not ops_w or MAIN not in ops_b or MAIN not in ops_w:
            continue
        a, b = ops_b[MAIN].get(metric), ops_w[MAIN].get(metric)
        if a and b:
            rs.append(b / a)
    return gm(rs), len(rs)


# ---------------- CSVs ----------------

CSV_EXTRA = [("mc_cluster_size", "gvr_multicta_cutedsl", "cluster_size"),
             ("op25_ms_path", "op25_hls", "ms_path"),
             ("op27_ms_path", "op27_hls", "ms_path"),
             ("r0_arm", "op26_r0auto", "r0_arm"),
             ("sglang_v2_cold_span_us", "sglang_v2", "us_cold_span")]


def write_csvs(data):
    outs = []
    for sweep, fn in (("seqlen", "op30_seqlen_data.csv"),
                      ("bs", "op30_bs_data.csv"),
                      ("bs_hugeN", "op30_bs_hugeN_data.csv")):
        path = HERE / fn
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            head = ["scenario", "K", "dtype", "N", "BS", "hit_rate", "layer"]
            for o in OPS:
                head += [f"{o}_cold_us", f"{o}_warm_us"]
            head += [f"speedup_vs_base_{o}" for o in RIVALS]
            head += [c for c, _, _ in CSV_EXTRA]
            w.writerow(head)
            nrow = 0
            for scen in SCENARIOS:
                cells = data.get(scen, {}).get(sweep, {})
                for (K, dt, N, BS), d in sorted(cells.items()):
                    ref = d.get(MAIN) or next(iter(d.values()))
                    hr = ref.get("hit_rate")
                    row = [scen, K, dt, N, BS,
                           round(hr, 4) if hr is not None else "",
                           ref.get("layer", "")]
                    for o in OPS:
                        r = d.get(o, {})
                        row += [round(r["us_cold"], 3)
                                if r.get("us_cold") else "",
                                round(r["us_warm"], 3)
                                if r.get("us_warm") else ""]
                    base = d.get(MAIN, {}).get("us_cold")
                    for o in RIVALS:
                        r = d.get(o, {})
                        row.append(round(base / r["us_cold"], 3)
                                   if (base and r.get("us_cold")) else "")
                    for _, o, field in CSV_EXTRA:
                        v = d.get(o, {}).get(field, "")
                        row.append(round(v, 3) if isinstance(v, float) else v)
                    w.writerow(row)
                    nrow += 1
        outs.append((fn, nrow))
    return outs


# ---------------- html helpers ----------------

def bi(en, zh, tag="div"):
    return (f'<{tag} class="i18n-en">{en}</{tag}>'
            f'<{tag} class="i18n-zh">{zh}</{tag}>')


def fmt_s(s):
    """Format a speedup (>1 good = op faster than GVR-base)."""
    if s is None:
        return "<td>—</td>"
    cls = "good" if s >= 1.05 else ("bad" if s <= 0.95 else "")
    return f"<td class='{cls}'>{s:.3f}×</td>"


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


# ---------------- TL;DR ----------------

def tldr(data):
    kpi_en, kpi_zh = [], []
    for scen in SCENARIOS:
        pooled = pooled_cells(data, scen)
        for arm in KPI_ARMS:
            g = gm(speedups(pooled, arm, "us_cold"))
            val = f"<b>{g:.3f}×</b>" if g else "<b>n/a</b>"
            kpi_en.append(f"<div class='kpi'>{scen.upper()} "
                          f"{OP_SHORT[arm]} {val}</div>")
            kpi_zh.append(f"<div class='kpi'>{scen.upper()} "
                          f"{OP_SHORT[arm]} {val}</div>")
    g_wb, n_wb = wb_ratio(data)
    wb = (f"<b>{g_wb:.3f}×</b> ({n_wb} cells)" if g_wb else "<b>n/a</b>")
    kpi_en.append(f"<div class='kpi'>GVR-base worst/best {wb}</div>")
    kpi_zh.append(f"<div class='kpi'>GVR-base worst/best {wb}</div>")
    en = ("<h3>TL;DR — verdict</h3>" + "".join(kpi_en)
          + "<p>(cold-L2 geomean speedup t<sub>GVR-base</sub>/t<sub>op</sub>, "
            "&gt;1 ⇒ op FASTER than the GVR (cuteDSL) baseline; pooled over "
            "seqlen + bs + bs_hugeN cells. Last chip = how much slower "
            "GVR-base itself runs on its WORST data vs its BEST data, same "
            "cells, cold.)</p>")
    zh = ("<h3>TL;DR — 结论</h3>" + "".join(kpi_zh)
          + "<p>（cold-L2 几何均值加速比 t<sub>GVR-base</sub>/t<sub>op</sub>，"
            "&gt;1 ⇒ 该臂比 GVR (cuteDSL) 基线更快；seqlen + bs + bs_hugeN "
            "全 cell 合并。末位芯片 = GVR-base 自身在 WORST 数据上比 BEST "
            "数据慢多少（同 cell，冷 L2）。）</p>")
    return en, zh


# ---------------- §0 calibration ----------------

def md_to_html(md):
    """Tiny stdlib markdown-table renderer for CALIBRATION.md."""
    out, intable = [], False
    for ln in md.splitlines():
        s = ln.strip()
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not intable:
                out.append('<div class="scrolltbl"><table>')
                tag, intable = "th", True
            else:
                tag = "td"
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>"
                                        for c in cells) + "</tr>")
            continue
        if intable:
            out.append("</table></div>")
            intable = False
        if s.startswith("## "):
            out.append(f"<h3>{s[3:]}</h3>")
        elif s.startswith("# "):
            continue
        elif s.startswith(("- ", "* ")):
            out.append(f"<p class='small'>• {s[2:]}</p>")
        elif s:
            out.append(f"<p class='small'>{s}</p>")
    if intable:
        out.append("</table></div>")
    return "".join(out)


def calibration_html():
    p = HERE / "CALIBRATION.md"
    tables = (md_to_html(p.read_text()) if p.exists()
              else "<p class='small'>CALIBRATION.md missing</p>")
    en = """
<p><b>How BEST/WORST were chosen.</b> A phase-1 calibration sweep measured the
<b>absolute</b> GVR-base (cuteDSL) cold-L2 kernel time over the full synth grid
cfg∈{aggregate, beta_shallow, beta_moderate, beta_deep} × hr∈{0.05..0.90}
(fp32, BS=1, N∈{16K, 64K, 256K}), scored each (cfg, hr) by its per-N-normalized
geomean, and took BEST = argmin, WORST = argmax per model
(<code>pick_scen_op30.py</code> → <code>scen_op30.json</code>). Poles:
v4flash BEST=beta_shallow hr0.30 / WORST=beta_shallow hr0.90;
v4pro BEST=aggregate hr0.15 / WORST=beta_deep hr0.85;
v32 BEST=beta_shallow hr0.15 / WORST=aggregate hr0.85 — WORST/BEST time ratios
2.17× / 1.75× / 1.54×. The radix control's spread stays ≤1.04× on the same
grid, confirming the poles are GVR-specific data sensitivity, not noise.</p>
<p><b>Contrast with op22.</b> op22's poles were <b>HLS-relative</b>
(BEST=hr0.55, WORST=beta_shallow hr0.05); for GVR-base those labels are
<b>reversed AND shifted</b>: GVR-base is FAST at low hr (boundary misses put
the mean(logits[preIdx]) threshold seed right at the K-th value → ev1
accepted) and SLOW at high hr (near-perfect hint pins the seed at the top-K
median → guaranteed undershoot → extra full-row re-scans) — its true worst
sits at hr 0.85–0.90, not at op22's 0.55 label. Hence this dedicated
calibration instead of reusing the op22 poles.</p>"""
    zh = """
<p><b>BEST/WORST 如何选定。</b>Phase-1 定标扫描测量 GVR-base（cuteDSL）的
<b>绝对</b> cold-L2 内核时间，覆盖 cfg∈{aggregate, beta_shallow,
beta_moderate, beta_deep} × hr∈{0.05..0.90}（fp32、BS=1、N∈{16K, 64K,
256K}），按 per-N 归一化几何均值打分，逐模型取 BEST = argmin、WORST = argmax
（<code>pick_scen_op30.py</code> → <code>scen_op30.json</code>）。极点：
v4flash BEST=beta_shallow hr0.30 / WORST=beta_shallow hr0.90；
v4pro BEST=aggregate hr0.15 / WORST=beta_deep hr0.85；
v32 BEST=beta_shallow hr0.15 / WORST=aggregate hr0.85 —— WORST/BEST 时间比
2.17× / 1.75× / 1.54×。radix 对照臂在同一网格上散布 ≤1.04×，证明极点是
GVR 特有的数据敏感性而非噪声。</p>
<p><b>与 op22 的对照。</b>op22 的极点是 <b>HLS-相对</b>的（BEST=hr0.55、
WORST=beta_shallow hr0.05）；对 GVR-base 而言这套标签<b>方向反转且位置偏移</b>：
GVR-base 在低 hr 时快（贴边界的 miss 使 mean(logits[preIdx]) 阈值种子恰落在
第 K 个值 → ev1 直接接受），高 hr 时慢（近乎完美的提示把种子钉在 top-K
中位数 → 必然 undershoot → 额外全行重扫）—— 其真实 worst 在 hr 0.85–0.90，
而不是 op22 标签的 0.55。因此单独做本次定标，而不复用 op22 极点。</p>"""
    return ('<div class="card">' + bi(en, zh)
            + bi("<h3>Calibration tables (CALIBRATION.md)</h3>",
                 "<h3>定标表（CALIBRATION.md）</h3>") + tables + "</div>")


# ---------------- full-data table ----------------

def fulldata_table(data, metric):
    head = ("<tr><th>scen</th><th>sweep</th><th>K</th><th>dtype</th>"
            "<th>N</th><th>BS</th><th>hr</th>"
            + "".join(f"<th style='color:{COL[o]}'>{OP_SHORT[o]} µs</th>"
                      for o in OPS) + "</tr>")
    rows = []
    for scen in SCENARIOS:
        for sweep, _ in SUBS:
            cells = data.get(scen, {}).get(sweep, {})
            for (K, dt, N, BS), d in sorted(cells.items()):
                ref = d.get(MAIN) or next(iter(d.values()))
                hr = ref.get("hit_rate")
                tds = []
                for o in OPS:
                    v = d.get(o, {}).get(metric)
                    tds.append(f"<td>{v:.1f}</td>" if v else "<td>–</td>")
                rows.append(
                    f"<tr><td>{scen}</td><td>{sweep}</td><td>{K}</td>"
                    f"<td>{dt}</td><td>{N}</td><td>{BS}</td>"
                    f"<td>{hr:.2f}</td>" + "".join(tds) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def geomean_table(data, metric):
    """Rows: (scenario,K,dtype); cols: geomean speedup vs base per arm."""
    head = ("<tr><th>scenario</th><th>K</th><th>dtype</th>"
            + "".join(f"<th>{OP_SHORT[o]}</th>" for o in RIVALS)
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
                tds, ncell = [], 0
                for o in RIVALS:
                    ss = speedups(cells, o, metric)
                    ncell = max(ncell, len(ss))
                    tds.append(fmt_s(gm(ss)))
                rows.append(f"<tr><td>{scen}</td><td>{K}</td><td>{dt}</td>"
                            + "".join(tds) + f"<td>{ncell}</td></tr>")
    if not rows:
        return "<p class='small'>no data</p>"
    return f"<table>{head}{''.join(rows)}</table>"


# ---------------- interactive panels (op22 pattern) ----------------

def chart_records(data):
    """Compact JS records: s scenario, w 'seq'|'bs', K, d dtype, N, B BS,
    o op, c cold-µs, h warm-µs. bs_hugeN folds into 'bs' (BS≤64 lines)."""
    recs = []
    for scen, sweeps in data.items():
        for sweep, cells in sweeps.items():
            w = "seq" if sweep == "seqlen" else "bs"
            for (K, dt, N, BS), ops in cells.items():
                for op, r in ops.items():
                    recs.append({"s": scen, "w": w, "K": K, "d": dt, "N": N,
                                 "B": BS, "o": op,
                                 "c": round(r.get("us_cold") or 0, 3),
                                 "h": round(r.get("us_warm") or 0, 3)})
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


def interactive_panels(data):
    scen_items = [(s, s.upper()) for s in SCENARIOS if s in data]
    op_items = [(o, OP_SHORT[o]) for o in OPS]
    ks_avail = sorted({k[0] for s in data.values()
                       for cells in s.values() for k in cells})
    dt_avail = sorted({k[1] for s in data.values()
                       for cells in s.values() for k in cells},
                      key=lambda d: DTS.index(d))
    k_items = [(str(k), f"K={k} ({K_MODEL[k]})") for k in KS
               if k in ks_avail] or [("2048", "K=2048 (V3.2)")]
    dt_items = [(d, d) for d in dt_avail] or [("fp32", "fp32")]
    k_def = "2048" if 2048 in ks_avail else k_items[0][0]
    dt_def = "fp32" if "fp32" in dt_avail else dt_items[0][0]
    bs_ns = sorted({N for s in data.values()
                    for sw in ("bs", "bs_hugeN") if sw in s
                    for (_, _, N, _) in s[sw]})
    n_items = [(str(n), f"{n // 1024}K" + (" (BS≤64)" if n in HUGE_NS else ""))
               for n in bs_ns]
    n_def = (str(BS_ANCHOR_N) if BS_ANCHOR_N in bs_ns
             else (n_items[0][0] if n_items else str(BS_ANCHOR_N)))
    scen_def = {s for s, _ in scen_items}

    ctrl_seq = ('<div class="ctl">'
                + bi("<b>scenarios</b>", "<b>场景</b>", "span") + " "
                + checks("sck1", scen_items, scen_def)
                + '<br>' + bi("<b>operators</b>", "<b>算子</b>", "span") + " "
                + checks("ock1", op_items, set(OPS))
                + '<br>' + radios("kk1", k_items, k_def) + " · "
                + radios("dd1", dt_items, dt_def) + "</div>")
    ctrl_bs = ('<div class="ctl">'
               + bi("<b>scenarios</b>", "<b>场景</b>", "span") + " "
               + checks("sck2", scen_items, scen_def)
               + '<br>' + bi("<b>operators</b>", "<b>算子</b>", "span") + " "
               + checks("ock2", op_items, set(OPS))
               + '<br>' + radios("kk2", k_items, k_def) + " · "
               + radios("dd2", dt_items, dt_def) + " · N: "
               + radios("nn2", n_items or [(n_def, "128K")], n_def)
               + "</div>")

    js_data = json.dumps(chart_records(data), separators=(",", ":"))
    js = """
<script>
const D=%s;
const COL=%s,DASH=%s,SHORT=%s,MAIN=%s;
const T={en:{lat:'Latency vs N',spd:'Speedup base/op (>1 = op faster than GVR-base)',
latb:'Latency vs BS',spdb:'Speedup base/op vs BS',us:'µs (log)',
n:'N (post-compress, log)',bs:'batch size (log)',r:'speedup ×'},
zh:{lat:'延迟 对 N',spd:'加速比 base/op（>1 = 该臂快于 GVR-base）',
latb:'延迟 对 BS',spdb:'加速比 base/op 对 BS',us:'µs（对数）',
n:'N（压缩后，对数）',bs:'batch size（对数）',r:'加速比 ×'}};
function lang(){return document.getElementById('lang-zh').checked?'zh':'en'}
function reg(){return document.getElementById('m-cold').checked?'c':'h'}
function vals(cls){return [...document.querySelectorAll('.'+cls+':checked')].map(x=>x.value)}
function rad(n){const e=document.querySelector('input[name='+n+']:checked');return e?e.value:null}
function LAY(t,xt,yt,ylog){return {title:{text:t,font:{size:15}},
paper_bgcolor:'#161b22',plot_bgcolor:'#0f1419',font:{color:'#e6e6e6',size:12},
margin:{t:42,r:10,b:48,l:56},showlegend:true,
legend:{orientation:'h',y:-0.22,font:{size:10.5}},
xaxis:{title:xt,type:'log',gridcolor:'#2a3340'},
yaxis:{title:yt,type:ylog?'log':'linear',gridcolor:'#2a3340'}}}
function pick(w,scens,K,d){const m={};
D.filter(r=>r.w==w&&scens.includes(r.s)&&r.K==K&&r.d==d).forEach(r=>{
(m[r.s]=m[r.s]||{});(m[r.s][r.o]=m[r.s][r.o]||{});m[r.s][r.o][w=='seq'?r.N:r.B+'|'+r.N]=r});
return m}
function seqDraw(){const L=T[lang()],rg=reg(),scens=vals('sck1'),ops=vals('ock1'),
K=rad('kk1'),d=rad('dd1'),m=pick('seq',scens,K,d),lat=[],spd=[];
for(const s of scens){if(!m[s])continue;
for(const o of ops){const c=m[s][o];if(!c)continue;
const xs=Object.keys(c).map(Number).sort((a,b)=>a-b).filter(N=>c[N][rg]);
if(!xs.length)continue;
lat.push({x:xs,y:xs.map(N=>c[N][rg]),name:s+'·'+SHORT[o],mode:'lines+markers',
marker:{size:5},line:{color:COL[o],dash:DASH[s]}});
if(o!=MAIN&&m[s][MAIN]){const tb=m[s][MAIN];
const xs2=xs.filter(N=>tb[N]&&tb[N][rg]);
spd.push({x:xs2,y:xs2.map(N=>tb[N][rg]/c[N][rg]),name:s+'·'+SHORT[o],
mode:'lines+markers',marker:{size:5},line:{color:COL[o],dash:DASH[s]}});}}}
spd.push({x:[4096,1048576],y:[1,1],mode:'lines',showlegend:false,
line:{color:'#888',width:1}});
Plotly.newPlot('p_slat',lat,LAY(L.lat+' (K='+K+', '+d+', BS=1, '+(rg=='c'?'cold':'warm')+'-L2)',L.n,L.us,true),{responsive:true});
Plotly.newPlot('p_sspd',spd,LAY(L.spd,L.n,L.r,false),{responsive:true});}
function bsDraw(){const L=T[lang()],rg=reg(),scens=vals('sck2'),ops=vals('ock2'),
K=rad('kk2'),d=rad('dd2'),N=rad('nn2'),m=pick('bs',scens,K,d),lat=[],spd=[];
for(const s of scens){if(!m[s])continue;
for(const o of ops){const c=m[s][o];if(!c)continue;
const bs=Object.keys(c).filter(k=>k.split('|')[1]==N).map(k=>Number(k.split('|')[0]))
.sort((a,b)=>a-b).filter(B=>c[B+'|'+N][rg]);
if(!bs.length)continue;
lat.push({x:bs,y:bs.map(B=>c[B+'|'+N][rg]),name:s+'·'+SHORT[o],mode:'lines+markers',
marker:{size:5},line:{color:COL[o],dash:DASH[s]}});
if(o!=MAIN&&m[s][MAIN]){const tb=m[s][MAIN];
const bs2=bs.filter(B=>tb[B+'|'+N]&&tb[B+'|'+N][rg]);
spd.push({x:bs2,y:bs2.map(B=>tb[B+'|'+N][rg]/c[B+'|'+N][rg]),
name:s+'·'+SHORT[o],mode:'lines+markers',marker:{size:5},
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
</script>""" % (js_data, json.dumps(COL), json.dumps(SCEN_DASH),
                json.dumps(OP_SHORT), json.dumps(MAIN))
    return ctrl_seq, ctrl_bs, js


# ---------------- §4 methodology ----------------

def arms_table():
    rows = [
        ("gvr_cutedsl", "harness <code>gvr_cutedsl</code> (CuTe DSL "
         "single-CTA #14602)", "<b>BASELINE</b> — all speedups are "
         "t(gvr_cutedsl)/t(op)"),
        ("gvr_multicta_cutedsl", "harness <code>gvr_multicta_cutedsl</code> "
         "(CuTe DSL DSMEM cluster, PR#15198)",
         "host auto cluster_size (recorded per cell)"),
        ("radix_cutedsl", "harness <code>radix_cutedsl</code>",
         "hint-blind exact radix"),
        ("radix_single_cuda", "harness <code>radix_single_cuda</code> "
         "(CUDA C++)", "hint-blind"),
        ("radix_multi_cuda", "harness <code>radix_multi_cuda</code> "
         "(CUDA C++)", "hint-blind"),
        ("op25_hls", "<code>gvr_ms_auto</code> @HEAD, "
         "<code>OP27_K2048_TAIL=0</code>",
         "op25 ship (w3a ladder + slot_scale2 + fp32 C8); coexists with "
         "op27_hls in ONE process — env re-read per call, qfracs in both "
         "compile-cache keys"),
        ("op27_hls", "<code>gvr_ms_auto</code> @HEAD, "
         "<code>OP27_K2048_TAIL=1</code>",
         "op27 ship (+ K2048 tail ladder 0.75/0.45/0.048); ms_path recorded"),
        ("op26_r0auto", "harness <code>op26_r0auto</code>",
         "op26 R0 dispatch (1cta/mc auto + small-N gate); r0_arm recorded"),
        ("sglang_v2", "harness <code>sglang_v2</code> (sglang@main 2026-07)",
         "fp32-only; plan/setup untimed; 2-kernel PDL path — kernel-sum can "
         "under-count overlap, honest wall-clock in the "
         "<code>*_span</code> columns"),
        ("flashinfer_topk", "harness <code>flashinfer_topk</code> "
         "(flashinfer 0.6.11)", "fp32-only"),
    ]
    body = "".join(
        f"<tr><td style='color:{COL[o]}'>{OP_LABEL[o]}</td>"
        f"<td>{src}</td><td>{note}</td></tr>" for o, src, note in rows)
    return ("<table><tr><th>arm</th><th>source</th><th>notes</th></tr>"
            + body + "</table>")


def methodology_html():
    en = f"""
<h3>Arms (10)</h3>{arms_table()}
<h3>Timing protocol</h3>
<p>nsys pure-kernel GPU time via per-cell NVTX range projection
(<code>nvtx_kern_sum</code>, the 512 MB L2-evict kernel filtered out;
<code>nvtx_gpu_proj_sum</code> adds the <code>*_span</code> wall-clock columns
for sglang_v2's PDL overlap). Per cell: 10 warmup, 50 warm-L2 reps
(<code>w|</code>), 20 cold-L2 reps (<code>c|</code>, 512 MB evict before each
launch — canonical metric), eager+sync inside the range, cudaProfilerApi
window; protocol byte-identical to <code>sweep_op22rr.py</code>
(<code>harness/sweep_nsys.measure_cell</code>).</p>
<h3>Data provenance</h3>
<p>Bundles from the <code>indexer-topk-temporal-synth</code> unified skill;
per-cell <code>seed = 42 + crc32("{{K}}|{{N}}") % 1e6</code> (op22 policy),
shared across dtypes and scenarios' N-grid; bundles generated once and
byte-identical across all 10 arms at every test point. Scenario cfg/hr per
model from <code>scen_op30.json</code> (§0).</p>
<h3>Node provenance</h3>
<p>Single node <b>umbriel-b200-047</b> (8× B200, sm_100), claim-queue
sharding over the 8 GPUs (atomic mkdir per batch) — no cross-node anchor
transfer; absolute µs are valid within this report.</p>
<h3>Exactness</h3>
<p>At BS=1, once per (arm, K, dtype, N): the arm's output index set is
gathered from the fp32 row and its sorted value-multiset compared against
<code>torch.topk</code> (GVR output order is atomicAdd-nondeterministic).
The build asserts zero FAIL rows.</p>"""
    zh = f"""
<h3>臂（10 个）</h3>{arms_table()}
<h3>计时协议</h3>
<p>nsys 纯 kernel GPU 时间，按 cell 的 NVTX 区间投影
（<code>nvtx_kern_sum</code>，512 MB L2 清空 kernel 已过滤；
<code>nvtx_gpu_proj_sum</code> 额外给出 <code>*_span</code> 墙钟列，覆盖
sglang_v2 的 PDL 重叠）。每 cell：10 次 warmup、50 次热 L2
（<code>w|</code>）、20 次冷 L2（<code>c|</code>，每次启动前清空 512 MB ——
规范指标），区间内 eager+sync，cudaProfilerApi 窗口；协议与
<code>sweep_op22rr.py</code>（<code>harness/sweep_nsys.measure_cell</code>）
逐字节一致。</p>
<h3>数据出处</h3>
<p>Bundle 由统一 skill <code>indexer-topk-temporal-synth</code> 生成；
每 cell <code>seed = 42 + crc32("{{K}}|{{N}}") % 1e6</code>（op22 策略），
跨 dtype 与场景 N 网格共享；每个测试点的输入在全部 10 个臂间字节一致
（生成一次、磁盘加载）。各模型场景 cfg/hr 见 <code>scen_op30.json</code>
（§0）。</p>
<h3>节点出处</h3>
<p>单节点 <b>umbriel-b200-047</b>（8× B200，sm_100），8 GPU 认领队列分片
（每批 atomic mkdir）—— 无跨节点锚点迁移；绝对 µs 仅在本报告内有效。</p>
<h3>精确性</h3>
<p>BS=1 时，每 (arm, K, dtype, N) 一次：从 fp32 行 gather 该臂输出索引集，
排序值多重集与 <code>torch.topk</code> 对比（GVR 输出顺序 atomicAdd
非确定）。构建时断言零 FAIL 行。</p>"""
    return '<div class="card">' + bi(en, zh) + "</div>"


# ---------------- main ----------------

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE.parent / "results_b200_op30"
    data, errs, fails = load_all(root)
    assert not fails, f"exactness FAIL rows: {fails}"
    have = {s: {sw: len(c) for sw, c in v.items()} for s, v in data.items()}
    print(f"loaded from {root}: {have}  errors-skipped: {errs or 'none'}")

    csv_outs = write_csvs(data)
    print("csv:", csv_outs)

    ctrl_seq, ctrl_bs, chart_js = interactive_panels(data)

    B = []
    B.append(f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>op30 — 10 arms on GVR-base BEST/WORST synthetic extremes</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head><body>
<input type="radio" name="lang" id="lang-zh" checked>
<input type="radio" name="lang" id="lang-en">
<input type="radio" name="metric" id="m-cold" checked>
<input type="radio" name="metric" id="m-warm">
<div class="wrap">
<div class="langbar"><label for="lang-zh">中文</label><label for="lang-en">English</label></div>
<h1>{bi("op30 — 10 arms on GVR-base BEST/WORST synthetic-data extremes",
        "op30 — GVR-base 最优/最差合成数据极点上的 10 臂重测", "span")}</h1>""")
    B.append('<p class="meta">2026-07-13 · umbriel-b200-047 (8× B200, '
             'sm_100) · branch <code>omni/op21-gvr-prod</code> · bucket '
             '<code>indexer_topk_op_bench/op30_gvrbase_extremes/</code></p>')

    # ---- intro ----
    B.append('<div class="card">' + bi(
        """<p>op22-style 10-arm top-K kernel benchmark on two NEW synthetic
scenarios defined relative to the <b>GVR (cuteDSL) baseline itself</b>:
<span class="reg">BEST</span> = the (cfg, hr) pole where GVR-base runs
fastest, <span class="reg">WORST</span> = where it runs slowest — chosen by
an absolute-time calibration sweep (§0), NOT the op22 HLS-relative labels.
Timing = <b>nsys pure-kernel GPU time</b>, <span class="reg">cold-L2</span>
canonical (20 reps, 512 MB evict) + <span class="reg">warm-L2</span>
(50 reps). All speedups = t(gvr_cutedsl)/t(op), &gt;1 ⇒ op faster than the
baseline. Single node, all arms read byte-identical bundles per cell.</p>""",
        """<p>op22 风格的 10 臂 top-K 内核基准，跑在两个<b>以 GVR (cuteDSL)
基线自身</b>定义的新合成场景上：<span class="reg">BEST</span> = GVR-base
最快的 (cfg, hr) 极点，<span class="reg">WORST</span> = 其最慢极点 —— 由
绝对时间定标扫描选出（§0），不是 op22 的 HLS-相对标签。计时 = <b>nsys
纯 kernel GPU 时间</b>，<span class="reg">冷 L2</span> 为规范口径（20 次，
每次 512 MB 清空）+ <span class="reg">热 L2</span>（50 次）。所有加速比 =
t(gvr_cutedsl)/t(op)，&gt;1 ⇒ 该臂快于基线。单节点，同一 cell 全部臂读取
字节一致的 bundle。</p>""") + "</div>")

    # ---- TL;DR ----
    tl_en, tl_zh = tldr(data)
    B.append(f'<div class="card" style="border-color:#76b900">'
             f'{bi(tl_en, tl_zh)}</div>')

    # ---- test environment / data quality ----
    ncells = {s: sum(len(v) for v in data.get(s, {}).values())
              for s in SCENARIOS}
    kpis = ("<div class='kpi'>node <b>umbriel-b200-047</b> (8× B200)</div>"
            "<div class='kpi'>timing <b>nsys pure-kernel, cold+warm L2</b>"
            "</div>"
            "<div class='kpi'>cells " + " · ".join(
                f"<b>{s}</b> {n}" for s, n in ncells.items()) + "</div>")
    err_note_en = err_note_zh = ""
    if errs:
        seg = ", ".join(f"{k}: {n}" for k, n in sorted(errs.items()))
        err_note_en = (f"<p class='small'>Data quality: {sum(errs.values())} "
                       f"row(s) with a recorded error were skipped "
                       f"({seg}).</p>")
        err_note_zh = (f"<p class='small'>数据质量：跳过 "
                       f"{sum(errs.values())} 条带 error 的记录（{seg}）。"
                       f"</p>")
    B.append('<div class="card">'
             + bi("<h3>Test environment</h3>" + kpis + err_note_en,
                  "<h3>测试环境</h3>" + kpis + err_note_zh) + "</div>")

    # ---- operator legend + L2 regime bar ----
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
             + bi("<span class='small'>(tables switch via CSS, Plotly "
                  "charts redraw)</span>",
                  "<span class='small'>（表格经 CSS 切换，Plotly 图表联动"
                  "重绘）</span>", "span") + "</div>")

    # ---- 0. calibration ----
    B.append("<h2>" + bi("0. Data calibration — deriving the GVR-base poles",
                         "0. 数据定标 — GVR-base 极点的推导", "span")
             + "</h2>")
    B.append(calibration_html())

    # ---- 1. seqlen ----
    B.append("<h2>" + bi("1. Seq-len sweep (BS=1)",
                         "1. 序列长度扫描（BS=1）", "span") + "</h2>")
    B.append(bi(
        "<p>Per (K, dtype): latency vs N (left, log-y) and speedup vs the "
        "GVR-base baseline vs N (right, &gt;1 ⇒ op faster), BEST/WORST "
        "selectable. sglang_v2 / flashinfer_topk are fp32-only and drop out "
        "of the bf16/fp16 views.</p>",
        "<p>每 (K, dtype)：延迟对 N（左，log-y）与对 GVR-base 基线的加速比"
        "对 N（右，&gt;1 ⇒ 该臂更快），BEST/WORST 可勾选。sglang_v2 / "
        "flashinfer_topk 仅 fp32，在 bf16/fp16 视图中自动消失。</p>"))
    B.append('<div class="card">' + ctrl_seq
             + '<div class="row"><div id="p_slat" class="plt"></div>'
               '<div id="p_sspd" class="plt"></div></div></div>')

    # ---- 2. BS scaling ----
    B.append("<h2>" + bi(
        "2. BS-scaling (BS 1→2048, N 4K→256K; stretch N 512K/1M at BS≤64)",
        "2. BS 扩展性（BS 1→2048，N 4K→256K；补充 N 512K/1M @ BS≤64）",
        "span") + "</h2>")
    B.append(bi(
        f"<p>Per (K, dtype): latency vs BS at the selected N (anchor "
        f"N={BS_ANCHOR_N // 1024}K) and speedup vs GVR-base vs BS. The "
        f"512K/1M N choices come from the bs_hugeN stretch grid and only "
        f"cover BS 2–64 (marked in the selector).</p>",
        f"<p>每 (K, dtype)：所选 N（锚点 N={BS_ANCHOR_N // 1024}K）处延迟对 "
        f"BS，及对 GVR-base 的加速比对 BS。N=512K/1M 两档来自 bs_hugeN "
        f"补充网格，仅覆盖 BS 2–64（选择器中已标注）。</p>"))
    B.append('<div class="card">' + ctrl_bs
             + '<div class="row"><div id="p_blat" class="plt"></div>'
               '<div id="p_bspd" class="plt"></div></div></div>')

    # ---- per-(scenario,K,dtype) geomean table ----
    B.append('<div class="card">'
             + bi("<h3>Geomean speedup vs GVR-base per (scenario, K, dtype) "
                  "— pooled seqlen+bs+hugeN cells (&gt;1 ⇒ op faster)</h3>",
                  "<h3>按 (scenario, K, dtype) 的几何均值加速比（对 "
                  "GVR-base；seqlen+bs+hugeN 合并；&gt;1 ⇒ 该臂更快）</h3>")
             + f'<div class="cold">{geomean_table(data, "us_cold")}</div>'
             + f'<div class="warm">{geomean_table(data, "us_warm")}</div>'
             + "</div>")

    # ---- 3. full data ----
    B.append("<h2>" + bi("3. Full data", "3. 全量数据表", "span") + "</h2>")
    B.append('<div class="card">'
             + bi("<p>Download: <a href='op30_seqlen_data.csv'>"
                  "op30_seqlen_data.csv</a> · <a href='op30_bs_data.csv'>"
                  "op30_bs_data.csv</a> · <a href='op30_bs_hugeN_data.csv'>"
                  "op30_bs_hugeN_data.csv</a> (per-arm cold/warm µs + cold "
                  "speedups vs base + dispatch extras + sglang span). Table "
                  "below: per-cell µs in the active L2 regime.</p>",
                  "<p>下载：<a href='op30_seqlen_data.csv'>"
                  "op30_seqlen_data.csv</a> · <a href='op30_bs_data.csv'>"
                  "op30_bs_data.csv</a> · <a href='op30_bs_hugeN_data.csv'>"
                  "op30_bs_hugeN_data.csv</a>（每臂冷/热 µs + 对基线冷加速比 "
                  "+ dispatch 额外列 + sglang span）。下表：当前 L2 状态下"
                  "的每-cell µs。</p>")
             + f'<div class="scrolltbl cold">'
               f'{fulldata_table(data, "us_cold")}</div>'
             + f'<div class="scrolltbl warm">'
               f'{fulldata_table(data, "us_warm")}</div>'
             + "</div>")

    # ---- 4. methodology ----
    B.append("<h2>" + bi("4. Methodology", "4. 方法论", "span") + "</h2>")
    B.append(methodology_html())

    B.append(chart_js)
    B.append("</div></body></html>")
    out = HERE / "REPORT.html"
    html = "\n".join(B)
    out.write_text(html)
    n_script = html.count("<script")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, <script> tags: "
          f"{n_script})")
    assert n_script == 2, n_script  # plotly CDN + chart driver


if __name__ == "__main__":
    main()
