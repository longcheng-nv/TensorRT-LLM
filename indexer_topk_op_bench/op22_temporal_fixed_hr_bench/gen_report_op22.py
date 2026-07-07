#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op22 W6 — build REPORT.html from the parsed nsys sweeps.

Bilingual (zh default / en) CSS-only report — ZERO <script> tags: language
and cold/warm-L2 toggles are radio + `:checked ~` CSS (op19 REPORT.html
pattern); all charts are server-rendered inline matplotlib SVG.

Tolerates partial data (missing scenarios/sweeps are skipped) so it can be
smoke-built mid-campaign and re-run when all batches land.

Usage: python3 gen_report_op22.py [--out-root ../results_b200_op22]
"""
import argparse
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

OPS = ["gvr_ms_auto", "gvr_cutedsl", "gvr_multicta_cutedsl",
       "radix_cutedsl", "sglang_streaming"]
MAIN = "gvr_ms_auto"
RIVALS = [o for o in OPS if o != MAIN]
OP_LABEL = {
    "gvr_ms_auto": "GVR op#21 ms_auto",
    "gvr_cutedsl": "GVR single-CTA (cuteDSL #14602)",
    "gvr_multicta_cutedsl": "GVR multi-CTA (PR#15198)",
    "radix_cutedsl": "Radix (cuteDSL)",
    "sglang_streaming": "SGLang StreamingTopK",
}
COL = {"gvr_ms_auto": "#e6a817", "gvr_cutedsl": "#76b900",
       "gvr_multicta_cutedsl": "#2ec4b6", "radix_cutedsl": "#4ea8de",
       "sglang_streaming": "#d62728"}
SCENARIOS = ["best", "worst", "real"]
SCEN_LABEL = {"best": "BEST (beta_deep, hr=0.90)",
              "worst": "WORST (beta_shallow, hr=0.05)",
              "real": "REAL (aggregate, sampled hr)"}
KS = [512, 1024, 2048]
DTS = ["fp32", "bf16", "fp16"]
K_MODEL = {512: "V4-Flash", 1024: "V4-Pro", 2048: "V3.2"}


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


def ratio_cells(cells, rival, metric):
    """Per-cell rival/main time ratio (>1 => op21 faster)."""
    out = {}
    for key, ops in cells.items():
        a, b = ops.get(MAIN), ops.get(rival)
        if a and b and a.get(metric) and b.get(metric):
            out[key] = b[metric] / a[metric]
    return out


# ---------------- figures ----------------

def svg_of(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.find("<svg"):]


def fig_seqlen(data, K, dt, metric):
    """One row of 3 subplots (scenario) — us vs N, 5 ops, log-log."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4), sharey=True)
    for ax, scen in zip(axes, SCENARIOS):
        cells = data.get(scen, {}).get("seqlen", {})
        for op in OPS:
            pts = sorted((N, ops[op][metric]) for (k, d, N, BS), ops
                         in cells.items()
                         if k == K and d == dt and BS == 1 and op in ops
                         and ops[op].get(metric))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker="o", ms=3, lw=1.4, color=COL[op],
                        label=OP_LABEL[op])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(SCEN_LABEL[scen], fontsize=9)
        ax.set_xlabel("N (post-compress)", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(f"{metric.replace('us_', '')}-L2 kernel µs", fontsize=8)
    axes[-1].legend(fontsize=6.5, loc="upper left")
    fig.suptitle(f"K={K} ({K_MODEL[K]})  {dt}  —  seq-len sweep BS=1",
                 fontsize=10, y=1.04)
    return svg_of(fig)


def fig_bs(data, K, dt, metric):
    """2 rows (op21 ratio vs radix, vs gvr_cutedsl) x 3 scenarios; lines=N."""
    rivals = ["radix_cutedsl", "gvr_cutedsl"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.0), sharex=True)
    ns = sorted({N for scen in SCENARIOS
                 for (k, d, N, BS) in data.get(scen, {}).get("bs", {})
                 if k == K and d == dt})
    cmap = plt.get_cmap("viridis")
    for ri, rival in enumerate(rivals):
        for ci, scen in enumerate(SCENARIOS):
            ax = axes[ri][ci]
            cells = data.get(scen, {}).get("bs", {})
            rat = ratio_cells({k: v for k, v in cells.items()
                               if k[0] == K and k[1] == dt}, rival, metric)
            for ni, N in enumerate(ns):
                pts = sorted((BS, r) for (k, d, n, BS), r in rat.items()
                             if n == N)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker="o", ms=2.5, lw=1.1,
                            color=cmap(ni / max(len(ns) - 1, 1)),
                            label=f"N={N//1024}K")
            ax.axhline(1.0, color="#999", lw=0.8, ls="--")
            ax.set_xscale("log", base=2)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7)
            if ri == 0:
                ax.set_title(SCEN_LABEL[scen], fontsize=9)
            if ri == len(rivals) - 1:
                ax.set_xlabel("BS", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{OP_LABEL[rival].split(' (')[0]} / op21",
                              fontsize=7.5)
    axes[0][-1].legend(fontsize=6, ncol=2, loc="upper right")
    fig.suptitle(f"K={K} ({K_MODEL[K]})  {dt}  —  BS-scaling: time ratio "
                 f"rival/op21 (>1 ⇒ op21 faster), {metric.replace('us_','')}-L2",
                 fontsize=10, y=0.99)
    return svg_of(fig)


# ---------------- tables ----------------

def fmt_r(r):
    if r is None:
        return "<td class='num'>—</td>"
    cls = "good" if r >= 1.05 else ("bad" if r <= 0.95 else "")
    return f"<td class='num {cls}'>{r:.3f}×</td>"


def geomean_table(data, metric):
    """Rows: (scenario, K, dtype); cols: op21 geomean ratio vs each rival
    (seqlen+bs cells pooled)."""
    head = ("<tr><th>scenario</th><th>K</th><th>dtype</th>"
            + "".join(f"<th class='num'>vs {OP_LABEL[r].split(' (')[0]}</th>"
                      for r in RIVALS) + "<th class='num'>cells</th></tr>")
    rows = []
    for scen in SCENARIOS:
        sd = data.get(scen, {})
        pooled = {}
        for sweep in ("seqlen", "bs"):
            pooled.update({(sweep,) + k: v
                           for k, v in sd.get(sweep, {}).items()})
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
                            + "".join(tds)
                            + f"<td class='num'>{ncell}</td></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def hr_sensitivity_table(data, metric):
    """Per op: geomean over common (K,dt,N,BS) cells of worst_us/best_us —
    how much slower each op gets when hit-rate drops 0.90 -> 0.05."""
    head = ("<tr><th>op</th>"
            + "".join(f"<th class='num'>K={K} {dt}</th>"
                      for K in KS for dt in DTS)
            + "<th class='num'>overall</th></tr>")
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
        rows.append(f"<tr><td>{OP_LABEL[op]}</td>{''.join(tds)}"
                    + fmt_r(gm(alls)) + "</tr>")
    return f"<table>{head}{''.join(rows)}</table>"


# ---------------- html ----------------

def bi(en, zh, tag="div"):
    return (f'<{tag} class="en">{en}</{tag}>'
            f'<{tag} class="zh">{zh}</{tag}>')


CSS = """
:root{--ink:#1a202c;--mut:#4a5568;--line:#e2e8f0;--acc:#2b6cb0;--good:#276749;--bad:#9b2c2c;--bg:#f7fafc;}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:var(--bg);line-height:1.62}
input[name=lang],input[name=metric]{position:absolute;left:-9999px}
.wrap{max-width:1180px;margin:0 auto;padding:28px 34px 80px}
.topbar{position:sticky;top:0;z-index:9;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);margin-bottom:18px}
.topbar label{display:inline-block;padding:5px 16px;border:1px solid var(--acc);color:var(--acc);cursor:pointer;font-weight:600;border-radius:4px;margin-right:8px;user-select:none}
#lang-zh:checked ~ .wrap label[for=lang-zh],#lang-en:checked ~ .wrap label[for=lang-en]{background:var(--acc);color:#fff}
#lang-zh:checked ~ .wrap .en{display:none}
#lang-en:checked ~ .wrap .zh{display:none}
#m-cold:checked ~ .wrap label[for=m-cold],#m-warm:checked ~ .wrap label[for=m-warm]{background:var(--acc);color:#fff}
#m-cold:checked ~ .wrap .warm{display:none}
#m-warm:checked ~ .wrap .cold{display:none}
h1{font-size:25px;margin:8px 0 2px}
h2{font-size:20px;margin:38px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--acc)}
h3{font-size:16px;margin:24px 0 8px;color:var(--acc)}
.meta{color:var(--mut);font-size:13px;margin-bottom:6px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;background:#fff}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:#edf2f7;font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.good{color:var(--good);font-weight:600}
.bad{color:var(--bad);font-weight:600}
.card{background:#fff;border:1px solid var(--line);border-radius:6px;padding:14px 18px;margin:14px 0}
.tldr{border-left:4px solid var(--acc)}
.warn{border-left:4px solid #c05621;background:#fffaf0}
.fig{background:#fff;border:1px solid var(--line);border-radius:6px;padding:10px;margin:16px 0;overflow-x:auto}
.fig svg{max-width:100%;height:auto}
.figcap{font-size:12.5px;color:var(--mut);margin-top:4px}
code,.mono{font-family:ui-monospace,Consolas,monospace;font-size:0.92em;background:#edf2f7;padding:1px 4px;border-radius:3px}
pre{background:#edf2f7;padding:10px 12px;border-radius:6px;overflow-x:auto;font-size:12.5px}
.small{font-size:12.5px;color:var(--mut)}
ol li,ul li{margin:4px 0}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(HERE.parent / "results_b200_op22"))
    ap.add_argument("--out", default=str(HERE / "REPORT.html"))
    args = ap.parse_args()
    root = Path(args.out_root)

    data = {s: load_scenario(root, s) for s in SCENARIOS}
    have = {s: sorted(data[s]) for s in SCENARIOS}
    print("loaded:", {s: v for s, v in have.items()})

    # ---- figures (cold + warm sets) ----
    figs = {"us_cold": {"seq": {}, "bs": {}}, "us_warm": {"seq": {}, "bs": {}}}
    for metric in figs:
        for K in KS:
            for dt in DTS:
                if any("seqlen" in data[s] for s in SCENARIOS):
                    figs[metric]["seq"][(K, dt)] = fig_seqlen(data, K, dt, metric)
                if any("bs" in data[s] for s in SCENARIOS):
                    figs[metric]["bs"][(K, dt)] = fig_bs(data, K, dt, metric)

    # ---- assemble ----
    B = []
    B.append(f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>op22 — GVR op21 vs rivals on temporal-synth fixed-hit-rate data</title>
<style>{CSS}</style></head><body>
<input type="radio" name="lang" id="lang-zh" checked>
<input type="radio" name="lang" id="lang-en">
<input type="radio" name="metric" id="m-cold" checked>
<input type="radio" name="metric" id="m-warm">
<div class="wrap">
<div class="topbar">
  <label for="lang-zh">中文</label><label for="lang-en">English</label>
  <span style="margin-left:22px"></span>
  <label for="m-cold">cold-L2 (canonical)</label><label for="m-warm">warm-L2</label>
  <span class="small" style="margin-left:12px">CSS-only toggles · zero JavaScript · inline SVG</span>
</div>
<h1>op22 — GVR op#21 vs 4 rivals · temporal-synth fixed-hit-rate 数据
<span class="en" style="display:none"></span></h1>""")
    B.append(bi(
        "<h1 style='font-size:16px;color:var(--mut);margin:0'>GVR op#21 vs 4 rivals on temporally-coherent synthetic data with FIXED hit rate (best / worst / realistic scenarios)</h1>",
        "<h1 style='font-size:16px;color:var(--mut);margin:0'>GVR op#21 对 4 个对手,在固定 hit-rate 的时序相关合成数据上(best / worst / realistic 三场景)</h1>"))
    B.append('<p class="meta">2026-07-07 · B200 (sm_100), GPU0 · nodes: umbriel-b200-040 '
             '(real 18/18 + best 16/18) / umbriel-b200-049 (best bs-K2048-bf16/fp16 + worst '
             '+ bs_hugeN) — absolute µs do not transfer across nodes; per-cell rival ratios '
             '(the canonical metric) are node-internal · '
             'branch <code>omni/op21-gvr-prod</code> · bucket <code>indexer_topk_op_bench/'
             'op22_temporal_fixed_hr_bench/</code> · nsys pure-kernel (NVTX→GPU projection), '
             'cold-L2 (512 MB evict) canonical + warm-L2 · data: <code>indexer-topk-temporal-synth</code> '
             '(5 gates PASS ×3 models) · exactness pre-gate 456/456</p>')

    # TL;DR placeholder — auto numbers
    tl_en, tl_zh = tldr(data)
    B.append(f'<div class="card tldr">{bi(tl_en, tl_zh)}</div>')

    # ---- 1. goal & objects ----
    B.append("<h2>" + bi("1. Goal & objects", "1. 目标与对象", "span") + "</h2>")
    B.append(bi(
        """<p>Measure the production-dispatch GVR kernel <b>op#21</b> (<code>gvr_ms_auto</code>, iter12 @f51f50f4da)
        against 4 rivals on data whose <b>temporal-hint quality (hit rate) is controlled</b>:
        GVR seeds its threshold search from the previous step's top-K hint, so its cost depends
        on the hint — radix/streaming rivals are hint-blind. (§7 shows the dependence is on
        threshold-init quality and is NON-monotone in hit rate.) Scenarios: BEST
        (deep-layer marginal, hr=0.90), WORST (shallow-layer marginal, hr=0.05), REAL
        (aggregate layer mixture, hr sampled from the real per-step distribution — the anchor
        comparable to <code>report/report.html</code>).</p>""",
        """<p>把生产 dispatch 的 GVR 内核 <b>op#21</b>(<code>gvr_ms_auto</code>, iter12 @f51f50f4da)
        与 4 个对手放在 <b>时序提示质量(hit rate)受控</b> 的数据上对比:GVR 以上一步 top-K 提示
        为阈值搜索的种子,代价依赖提示质量,而 radix/streaming 对手对提示不敏感。(§7 证明该依赖
        实为『阈值初始化质量』,且对 hit rate 非单调。)三场景:BEST
        (深层 marginal,hr=0.90)、WORST(浅层 marginal,hr=0.05)、REAL(aggregate 层混合、
        hr 按真实 per-step 分布采样 —— 与 <code>report/report.html</code> 可比的锚点)。</p>"""))
    B.append("<table><tr><th>op</th><th>impl</th></tr>" + "".join(
        f"<tr><td style='color:{COL[o]};font-weight:600'>{OP_LABEL[o]}</td>"
        f"<td><code>{o}</code>"
        + (" — fp32-only, K≤1024" if o == "sglang_streaming" else "")
        + "</td></tr>" for o in OPS) + "</table>")

    # ---- 2. methodology ----
    B.append("<h2>" + bi("2. Data methodology (reproducible)",
                         "2. 数据方法(可复现)", "span") + "</h2>")
    B.append(methodology_html())

    # ---- 3. seqlen ----
    B.append("<h2>" + bi("3. §1 Seq-len sweep (BS=1, N 4K→1M)",
                         "3. §1 序列长度扫描(BS=1,N 4K→1M)", "span") + "</h2>")
    B.append(bi(
        "<p>Absolute pure-kernel µs vs N, per (K, dtype), three scenarios side-by-side. "
        "In REAL, each N cell draws its own layer + hr (by design), so N-trends are jagged "
        "— judge per-cell ratios, not curve smoothness.</p>",
        "<p>纯核 µs 对 N,按 (K, dtype),三场景并排。REAL 场景每个 N cell 独立抽层与 hr"
        "(设计使然),N 趋势有锯齿 —— 看每 cell 的比值,不看曲线平滑度。</p>"))
    for metric, cls in (("us_cold", "cold"), ("us_warm", "warm")):
        B.append(f'<div class="{cls}">')
        for K in KS:
            for dt in DTS:
                svg = figs[metric]["seq"].get((K, dt))
                if svg:
                    B.append(f'<div class="fig">{svg}</div>')
        B.append("</div>")

    # ---- 4. BS scaling ----
    B.append("<h2>" + bi("4. §2 BS-scaling (BS 1→2048, N 4K→256K; stretch N 512K/1M at BS≤64)",
                         "4. §2 批量扩展(BS 1→2048,N 4K→256K;补充档 N 512K/1M 到 BS≤64)", "span") + "</h2>")
    B.append(bi(
        "<p>Time ratio rival/op21 vs BS (>1 ⇒ op21 faster); one line per N (the 512K/1M "
        "lines come from the bs_hugeN stretch grid, BS 2–64). Rows: vs Radix "
        "(hint-blind rival), vs GVR single-CTA (op21's own code start).</p>",
        "<p>rival/op21 时间比对 BS(>1 ⇒ op21 更快);每条线一个 N(512K/1M 两条线来自 "
        "bs_hugeN 补充档,BS 2–64)。两行:对 Radix(提示盲对手)、对 GVR 单 CTA"
        "(op21 的代码起点)。</p>"))
    for metric, cls in (("us_cold", "cold"), ("us_warm", "warm")):
        B.append(f'<div class="{cls}">')
        for K in KS:
            for dt in DTS:
                svg = figs[metric]["bs"].get((K, dt))
                if svg:
                    B.append(f'<div class="fig">{svg}</div>')
        B.append("</div>")

    # ---- 5. summary tables ----
    B.append("<h2>" + bi("5. Geomean ratio tables", "5. 几何均值比值表", "span") + "</h2>")
    B.append(bi("<p>Pooled seqlen+bs cells; time ratio rival/op21 (>1 ⇒ op21 faster).</p>",
                "<p>seqlen+bs cell 合并;时间比 rival/op21(>1 ⇒ op21 更快)。</p>"))
    B.append(f'<div class="cold">{geomean_table(data, "us_cold")}</div>')
    B.append(f'<div class="warm">{geomean_table(data, "us_warm")}</div>')

    B.append("<h3>" + bi("Hit-rate sensitivity (worst / best time, same cells)",
                         "hit-rate 敏感度(相同 cell 上 worst/best 用时比)", "span") + "</h3>")
    B.append(bi(
        "<p>How much slower each op runs when hr drops 0.90→0.05 (marginal also shifts "
        "deep→shallow). ≈1.0 ⇒ hint-insensitive; the GVR family's values quantify its "
        "worst-case exposure.</p>",
        "<p>hr 从 0.90→0.05(marginal 同时 deep→shallow)时各 op 变慢多少。≈1.0 ⇒ 对提示"
        "不敏感;GVR 家族的数值量化了其最坏暴露。</p>"))
    B.append(f'<div class="cold">{hr_sensitivity_table(data, "us_cold")}</div>')
    B.append(f'<div class="warm">{hr_sensitivity_table(data, "us_warm")}</div>')

    # ---- 6. findings ----
    B.append("<h2>" + bi("6. Findings — extreme cells (op21 vs Radix)",
                         "6. Findings — 极端 cell(op21 对 Radix)", "span") + "</h2>")
    B.append(bi(
        "<p>Per scenario, the 5 worst and 5 best cells by radix/op21 time ratio "
        "(cold-L2), with the cell's hit-rate, source layer and op21 dispatch path — "
        "the raw material for regime analysis. §7 identifies the mechanism behind "
        "the loss cells: extra full-row re-scans triggered by poisoned "
        "threshold-init, a pure function of the preIdx hint (NOT value "
        "tie-density, and NOT monotone in hr).</p>",
        "<p>每场景按 radix/op21 时间比(cold-L2)列最差 5 个与最好 5 个 cell,附该 cell 的 "
        "hit-rate、来源层与 op21 dispatch 路径 —— regime 分析的原始素材。失利 cell 背后的机制"
        "见 §7:阈值初始化被污染导致的额外全行重扫,纯粹由 preIdx 提示决定(与 value tie 密度"
        "无关,对 hr 也非单调)。</p>"))
    B.append(f'<div class="cold">{findings_html(data, "us_cold")}</div>')
    B.append(f'<div class="warm">{findings_html(data, "us_warm")}</div>')

    # ---- 7. mechanism ----
    B.append("<h2>" + bi("7. Mechanism — why hr=0.90 slows the GVR family down",
                         "7. 机制 — 为什么 hr=0.90 反而拖慢整个 GVR 家族", "span") + "</h2>")
    B.append(mech_html(data))

    B.append("</div></body></html>")
    out = Path(args.out)
    out.write_text("\n".join(B))
    n_script = "\n".join(B).count("<script")
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB, <script> tags: {n_script})")
    assert n_script == 0


def scen_dt_gm(data, scen, rival, dt, metric="us_cold"):
    sd = data.get(scen, {})
    pooled = {}
    for sweep in ("seqlen", "bs"):
        pooled.update({(sweep,) + k: v for k, v in sd.get(sweep, {}).items()})
    rs = [v[rival][metric] / v[MAIN][metric]
          for k, v in pooled.items() if k[2] == dt
          and MAIN in v and rival in v
          and v[MAIN].get(metric) and v[rival].get(metric)]
    return gm(rs), len(rs)


def tldr(data):
    """Auto TL;DR: per scenario, op21 vs gvr_cutedsl and vs radix per dtype."""
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
                seg_zh.append(f"对 {tag}:" + " / ".join(gs))
        if seg_en:
            lines_en.append(f"<b>{scen.upper()}</b> — " + "; ".join(seg_en))
            lines_zh.append(f"<b>{scen.upper()}</b> — " + ";".join(seg_zh))
    if not lines_en:
        return "<b>TL;DR</b> — data pending", "<b>TL;DR</b> — 数据未就绪"
    en = ("<b>TL;DR</b> (cold-L2 geomean, time ratio rival/op21, >1 ⇒ op21 faster)"
          "<br>" + "<br>".join(lines_en)
          + "<br><span class='small'>Sanity anchor: REAL vs gvr_cutedsl "
            "reproduces the op21-campaign B200 verdict (1.249/1.091/1.055 "
            "fp32/bf16/fp16) within −4.1%/+1.3%/+2.2%.</span>")
    zh = ("<b>TL;DR</b>(cold-L2 几何均值,时间比 rival/op21,>1 ⇒ op21 更快)"
          "<br>" + "<br>".join(lines_zh)
          + "<br><span class='small'>Sanity 锚点:REAL 对 gvr_cutedsl 复现 op21 "
            "战役 B200 收口值(fp32/bf16/fp16 = 1.249/1.091/1.055),偏差 "
            "−4.1%/+1.3%/+2.2%。</span>")
    return en, zh


def findings_html(data, metric="us_cold"):
    """Auto findings: extreme win/loss cells for op21 vs radix per scenario."""
    blocks = []
    for scen in SCENARIOS:
        sd = data.get(scen, {})
        pooled = {}
        for sweep in ("seqlen", "bs"):
            pooled.update({(sweep,) + k: v
                           for k, v in sd.get(sweep, {}).items()})
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
                    f"<td class='num'>{N}</td><td class='num'>{BS}</td>"
                    f"<td class='num'>{v[MAIN][metric]:.1f}</td>"
                    f"<td class='num'>{v['radix_cutedsl'][metric]:.1f}</td>"
                    f"<td class='num {cls}'>{r:.3f}×</td>"
                    f"<td class='num'>{hr:.2f}</td><td>L{lay} {path}</td></tr>")
        blocks.append(
            f"<h3>{SCEN_LABEL[scen]}</h3><table><tr><th></th><th>sweep</th>"
            "<th>K</th><th>dtype</th><th class='num'>N</th><th class='num'>BS</th>"
            "<th class='num'>op21 µs</th><th class='num'>radix µs</th>"
            "<th class='num'>radix/op21</th><th class='num'>hr</th>"
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
        wc_zh = ("<p><b>预言验证(构建时计算)</b> — 回放表明 WORST 零 refine(54/54 cell "
                 "ev1)⇒ refine 触发器缺席;单 CTA/多 CTA GVR 应与 REAL 持平。op21 的 msc "
                 "路径带第二触发器:worst 的『贴边界平坦』数据产生宽候选带(回放 cand→kC,"
                 "如 K2048 N=1M 时 4318)→ slot 溢出 fallback,因此 ms_auto 在 hugeN 即使 "
                 "ev1 也可能仍慢于 REAL。匹配 cell 上 t<sub>worst</sub>/t<sub>real</sub> "
                 "几何均值(&lt;1 ⇒ worst 更快):" + seg +
                 "。<i>注意:worst 在 b200-049、real 在 b200-040 —— 跨节点绝对 µs 对比,"
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
<p><b>结论:BEST 场景的变慢是真实、数据依赖的,但 tie 密度假设被证伪。</b>
代价来自阈值搜索期间的<b>额外全行重扫</b>次数,而该次数是 <b>preIdx 提示的纯函数</b>
—— 与 logits 值分布无关。证据链(详见 <code>MECH_FINDINGS.md</code>,工件
<code>mech_check_iters.py/.jsonl</code>、<code>mech_crossover.py</code>):</p>
<ol>
<li><b>非收敛从未发生。</b>用忠实复刻内核控制流的 host 回放
(<code>harness/count_gvr_iters.py</code>)在真实 bundle 上:162/162 cell P2 全收敛。
{mech_summary_table()}</li>
<li><b>GVR 单 CTA 的用时是 eval 次数的线性函数,场景变量消失</b>:K2048 fp32 N=1M
ev1→93.3 µs、ev3→136.7 µs(每次额外全行扫描 ≈22 µs);best/real N=524K 同为 ev4 →
86.2/86.3 µs(完全一致)。</li>
<li><b>交叉实验(决定性)</b>:在头条 cell 上把 best 与 real 的 logits、preIdx 两两互换
(CUDA-event 中位数,50 次 cold,b200-049 GPU1,仅筛查用)。用时<b>只跟随 preIdx 列</b>:
{xover}
K512 方向反转(real-preIdx 反而比 best-preIdx 慢)⇒ 对 hit rate 非单调。</li>
<li><b>为什么 hr→1 会毒化初始化</b>:GVR 用 mean(logits[preIdx]) 作阈值种子。近乎完美的
提示使它 ≈ top-K 的<b>中位数</b> → 初始计数 ≈ K/2 &lt; K → 必然 undershoot → 1–3 次
refine 重扫。而边界聚集的 miss(REAL,以及 miss-depth 模型贴着选择边界的 WORST)使种子
≈ 第 K 个值 → 首次计数即落入 [K, kC] 接受带。</li>
<li><b>op21 的悬崖更陡</b>:ladder 未命中或 slot 溢出时,msc 内核走 leader-CTA 全行
fallback(<code>gvr_msc_op.py:1096</code>,N=1M fp32 每遍 ~95 µs)—— 头条 cell
236 µs(ev3)对 28.6 µs(ev1)。K512 单独展示溢出触发:交叉组合回放全为 ev2,但
real-preIdx 的 cand≈4500(超 slot 容量)对 654 → fallback → 186 对 124 µs。</li>
</ol>
{wc_zh}
<p class="small">方法论注记:合成的固定 hr≥0.9 preIdx 是一种应力构造 —— 真实捕获中提示
均值从不落在 top-K 中位数上,因为真实 miss 聚集在选择边界。</p>"""
    return f'<div class="card">{bi(en, zh)}</div>'


def methodology_html():
    cli = """SKILL=.claude/skills/indexer-topk-temporal-synth
# per cell (model, dtype, N): seed = 42 + crc32("{K}|{N}") % 1e6   (see below)
python3 $SKILL/src/synth_temporal_data.py --model {v4flash|v4pro|v32} --N &lt;N&gt; \\
    --cfg beta_deep    --target_hr 0.90 --bs 1 --dtype &lt;dt&gt; --seed &lt;seed&gt; --outdir bundles/best/...   # BEST
python3 $SKILL/src/synth_temporal_data.py ... --cfg beta_shallow --target_hr 0.05 ...                  # WORST
python3 $SKILL/src/synth_temporal_data.py ... --cfg aggregate                    ...                  # REAL (hr sampled)
# every bundle's exact CLI incl. resolved seed: bundles/&lt;scen&gt;/&lt;model&gt;_&lt;dt&gt;_N&lt;N&gt;/*/meta.json "gen_cmd"
"""
    nl = ("使用 indexer-topk-temporal-synth skill 为 {V4-Flash|V4-Pro|V3.2} 生成 N=&lt;N&gt;、"
          "dtype=&lt;dtype&gt;、seed=&lt;cell_seed&gt; 的单行 decode logits + 时序相关 preIdx;"
          "GVR 最优场景用 deep-layer 分布 (--cfg beta_deep) 且固定 hit rate 0.90,"
          "GVR 最差场景用 shallow-layer 分布 (--cfg beta_shallow) 且固定 hit rate 0.05,"
          "realistic 锚点用 --cfg aggregate 并让 hit rate 按真实 per-step 分布采样。")
    gates = """<table><tr><th>gate</th><th>v32</th><th>v4flash</th><th>v4pro</th><th>limit</th></tr>
<tr><td>G1 per-layer KS max</td><td class=num>0.005</td><td class=num>0.003</td><td class=num>0.002</td><td class=num>≤0.05</td></tr>
<tr><td>G2 aggregate KS</td><td class=num>0.021</td><td class=num>0.018</td><td class=num>0.021</td><td class=num>≤0.05</td></tr>
<tr><td>G3 boundary mass @16K/64K/256K</td><td class=num>1.01/1.10/1.14</td><td class=num>1.11/1.12/1.11</td><td class=num>1.03/1.03/1.01</td><td class=num>0.80–1.25</td></tr>
<tr><td>G4 retention-curve max err</td><td class=num>0.030</td><td class=num>0.015</td><td class=num>0.021</td><td class=num>≤0.05</td></tr>
<tr><td>G5 realised-vs-target hr err</td><td class=num>0.000</td><td class=num>0.000</td><td class=num>0.000</td><td class=num>≤0.03</td></tr></table>"""
    en = f"""<p>Generator: <code>.claude/skills/indexer-topk-temporal-synth</code> (empirical
inverse-CDF + GPD tail marginal, rank-conditional temporal model; supersedes the legacy
single-Beta skills — motivating falsification: <code>synth_vs_real_validation/</code>).
Validation gates, all PASS (2026-07-06):</p>{gates}
<p><b>Seed policy</b>: <code>synthesize()</code> draws the row's layer as the FIRST rng call,
so a constant seed would collapse the layer mixture across the grid. Per-cell
<code>seed(K,N) = 42 + crc32("{{K}}|{{N}}") % 1e6</code>, shared across dtypes and scenarios;
fully deterministic from base 42. Realised hr verified ±0.03 per bundle (G5).</p>
<p><b>Canonical CLI</b> (verbatim; recorded per bundle in <code>meta.json.gen_cmd</code>):</p>
<pre>{cli}</pre>
<p><b>Natural-language prompt equivalent</b>:</p><pre>{nl}</pre>
<p><b>Protocol</b>: 1-row bundle replicated to BS (report.html convention; all rows share L2
lines at high BS — documented caveat), <code>seq_lens = N·cr</code> (== skill convention since
NEXT_N=1), nsys pure-kernel NVTX→GPU projection, eager+sync inside range, cold-L2 = 512 MB
evict before each timed call (canonical), warm-L2 = hot replay; 20 cold + 50 warm reps/cell;
exactness pre-gate 456/456 (sorted value-multiset — GVR output order is
atomicAdd-nondeterministic). N∈{{512K,1M}} rows are iid draws from the same empirical CDF+GPD
tail (marginals calibrated on 64K captures) — fine for kernel benchmarking, noted as caveat.</p>"""
    zh = f"""<p>生成器:<code>.claude/skills/indexer-topk-temporal-synth</code>(经验逆 CDF + GPD
尾部 marginal、按秩条件的时序模型;取代旧单-Beta skills —— 证伪研究见
<code>synth_vs_real_validation/</code>)。验证门全部 PASS(2026-07-06):</p>{gates}
<p><b>Seed 策略</b>:<code>synthesize()</code> 的第一次 rng 调用就是抽层,常量 seed 会让全网格
塌缩到同一层。每 cell <code>seed(K,N) = 42 + crc32("{{K}}|{{N}}") % 1e6</code>,跨 dtype 与场景共享;
从基 42 完全确定。每 bundle 实测 hr 校验 ±0.03(G5)。</p>
<p><b>规范 CLI</b>(逐字;每个 bundle 的 <code>meta.json.gen_cmd</code> 记录了带解析 seed 的等价命令):</p>
<pre>{cli}</pre>
<p><b>自然语言 prompt 等价物</b>:</p><pre>{nl}</pre>
<p><b>协议</b>:单行 bundle 复制到 BS(report.html 惯例;高 BS 下各行共享 L2 —— 已记录的 caveat),
<code>seq_lens = N·cr</code>(NEXT_N=1 时与 skill 约定相同),nsys 纯核 NVTX→GPU 投影,range 内
eager+sync,cold-L2 = 每次计时前 512 MB evict(canonical),warm-L2 = 热重放;每 cell 20 cold +
50 warm;exactness 预门 456/456(排序值多重集判据 —— GVR 输出顺序 atomicAdd 非确定)。
N∈{{512K,1M}} 行是同一经验 CDF+GPD 尾部的 iid 抽样(marginal 在 64K captures 上标定)——
对内核基准足够,已注明为 caveat。</p>"""
    return bi(en, zh)


if __name__ == "__main__":
    main()
