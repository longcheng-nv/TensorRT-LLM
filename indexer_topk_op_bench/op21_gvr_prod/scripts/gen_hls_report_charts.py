# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the interactive-data block (sections 6.6/6.7) of
HLS_VALIDATION_REPORT.html from the authoritative nsys artifacts.

Data sources (all committed / NFS-resident):
  results/nsys/iter13_ab_hls/       iter13 log-falsi A/B   (alpha=0.2)
  results/nsys/iter13_ab_hls_a01/   iter13 alpha=0.1 probe
  results/nsys/iter13_ab_hls_dist/  iter14 distributed A/B
  ../count_ge_multi_bench/results_m3.csv   tau(M) same-silicon sweep
  P0 spot medians: hardcoded from the measured runs (ITERATIONS iter13/14).

Output: replaces the block between <!-- HLS-CHARTS-BEGIN --> and
<!-- HLS-CHARTS-END --> in HLS_VALIDATION_REPORT.html (both language
variants inside), and ensures the plotly CDN tag exists in <head>.
Chart interaction follows the op22 REPORT.html conventions (plotly +
checkbox/radio controls). Bilingual labels via .en/.zh spans.

Section 7 (iter15 P0 switch verdict) is generated the same way between
<!-- P0-SWITCH-BEGIN --> and <!-- P0-SWITCH-END --> from
  results/nsys/p0batch/{ab3,hb3}_<scen>_<dtype>.{nsys-rep,jsonl}
(the 3-arm orig/legacy/shipped batch, scripts/ab_p0batch.py); the 17-cell
P0 no-regress grid @HEAD is a frozen snapshot of
`nsys_verdict.py msa fp32` on results/nsys/msa_* (2026-07-08).
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP21 = HERE.parents[0]
BENCH = OP21.parents[0]
sys.path.insert(0, str(BENCH / "report"))
from parse_nsys_full import parse_rep  # noqa: E402

RUNS = {
    "i13": OP21 / "results/nsys/iter13_ab_hls",
    "a01": OP21 / "results/nsys/iter13_ab_hls_a01",
    "i14": OP21 / "results/nsys/iter13_ab_hls_dist",
}
SCENS = ["best", "worst", "real"]

P0_SPOT = {  # cell label -> {run: [old_us, new_us]}
    "K512 fp32 262K":  {"i13": [18.14, 17.38], "i14": [17.47, 18.21]},
    "K1024 fp32 65K":  {"i13": [15.07, 14.29], "i14": [14.30, 14.82]},
    "K1024 fp32 262K": {"i13": [19.87, 19.01], "i14": [18.98, 19.78]},
    "K2048 fp32 262K": {"i13": [18.29, 19.33], "i14": [19.26, 19.10]},
    "K1024 bf16 262K": {"i13": [14.50, 14.59], "i14": [14.56, 14.56]},
}


def load_ab():
    """-> {run: {scen: [ {K,N,BS,ms_path,old,new,exact} ]}}"""
    out = {}
    for run, d in RUNS.items():
        out[run] = {}
        for scen in SCENS:
            rep = d / f"ab_{scen}_fp32.nsys-rep"
            jl = d / f"ab_{scen}_fp32.jsonl"
            if not rep.exists() or not jl.exists():
                continue
            kern = parse_rep(rep)
            cells = []
            for line in jl.read_text().splitlines():
                r = json.loads(line)
                if "error" in r:
                    continue
                base = (f"{r['scenario']}|{r['K']}|{r['dtype']}|{r['N']}"
                        f"|{r['BS']}")
                uo = kern.get(f"c|old|{base}")
                un = kern.get(f"c|new|{base}")
                if uo is None or un is None:
                    continue
                cells.append({"K": r["K"], "N": r["N"], "BS": r["BS"],
                              "path": r.get("ms_path"),
                              "old": round(uo, 2), "new": round(un, 2),
                              "exact": f"{r.get('exact_old','?')}/"
                                       f"{r.get('exact_new','?')}"})
            out[run][scen] = cells
    return out


def load_tau():
    """-> {dtype: {N: {M: us}}} from results_m3.csv"""
    out = {}
    p = BENCH / "count_ge_multi_bench" / "results_m3.csv"
    for r in csv.DictReader(open(p)):
        out.setdefault(r["dtype"], {}).setdefault(int(r["N"]), {})[
            int(r["M"])] = float(r["us_med"])
    return out


P0BATCH = OP21 / "results/nsys/p0batch"
P7_DTS = ["fp32", "bf16", "fp16"]
P7_ARMS = ["orig", "legacy", "shipped"]

# frozen snapshot of `nsys_verdict.py msa fp32` @HLS HEAD (2026-07-08):
# (K, N, BS, ms_us, rival_us, best_rival, gvrbest_us)
MSA_P0 = [
    (1024, 65536, 1, 14.02, 19.96, "radix_cutedsl_multi", 16.95),
    (1024, 65536, 4, 14.24, 20.10, "radix_cutedsl", 17.06),
    (1024, 65536, 8, 14.46, 20.54, "radix_cutedsl_multi", 17.45),
    (1024, 65536, 16, 14.94, 21.16, "radix_cutedsl_multi", 17.72),
    (1024, 131072, 1, 15.84, 19.91, "radix_cutedsl", 18.19),
    (1024, 131072, 4, 15.87, 20.11, "radix_cutedsl_multi", 18.52),
    (1024, 131072, 8, 16.29, 20.71, "radix_cutedsl", 19.06),
    (1024, 131072, 16, 16.90, 24.68, "radix_cutedsl", 19.84),
    (1024, 262144, 1, 18.75, 20.06, "radix_cutedsl", 20.34),
    (1024, 262144, 4, 18.82, 20.43, "radix_cutedsl", 20.87),
    (1024, 262144, 8, 19.33, 24.16, "radix_cutedsl", 21.43),
    (1024, 262144, 16, 19.97, 31.36, "radix_cutedsl", 22.68),
    (512, 131072, 1, 14.27, 19.12, "radix_cutedsl", 16.20),
    (512, 262144, 1, 17.25, 19.13, "radix_cutedsl", 19.95),
    (2048, 131072, 1, 18.50, 20.09, "radix_cutedsl", 19.11),
    (2048, 262144, 1, 18.88, 19.81, "radix_cutedsl_multi", 26.15),
    (2048, 262144, 16, 23.39, 31.97, "radix_cutedsl", 28.27),
]


def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def load_p0batch():
    """-> {"tail": {dt: {scen: cells}}, "hb": {dt: {scen: cells}}};
    cell = {K,N,BS,path,o,l,s,exact}"""
    out = {"tail": {}, "hb": {}}
    specs = [("tail", "ab3", dt) for dt in P7_DTS] + [("hb", "hb3", "fp32")]
    for kind, prefix, dt in specs:
        for scen in SCENS:
            rep = P0BATCH / f"{prefix}_{scen}_{dt}.nsys-rep"
            jl = P0BATCH / f"{prefix}_{scen}_{dt}.jsonl"
            if not rep.exists() or not jl.exists():
                continue
            kern = parse_rep(rep)
            cells = []
            for line in jl.read_text().splitlines():
                r = json.loads(line)
                if "error" in r:
                    continue
                base = (f"{r['scenario']}|{r['K']}|{r['dtype']}|{r['N']}"
                        f"|{r['BS']}")
                t = {a: kern.get(f"c|{a}|{base}") for a in P7_ARMS}
                if any(t[a] is None for a in P7_ARMS):
                    continue
                cells.append({
                    "K": r["K"], "N": r["N"], "BS": r["BS"],
                    "path": r.get("ms_path"),
                    "o": round(t["orig"], 2), "l": round(t["legacy"], 2),
                    "s": round(t["shipped"], 2),
                    "exact": "/".join((r.get(f"exact_{a}", "?") or "?")[:2]
                                      for a in P7_ARMS)})
            out[kind].setdefault(dt, {})[scen] = cells
    return out


def bi(en, zh):
    # inline spans; the charts-block CSS forces .zh display:inline in this
    # scope (the report default is block, which would break control rows)
    return f'<span class="en">{en}</span><span class="zh">{zh}</span>'


def detail_table(cells, run_label):
    rows = []
    for c in cells:
        ratio = c["old"] / c["new"] if c["new"] else float("nan")
        cls = ' class="hl-good"' if ratio > 1.02 else (
            ' class="hl-bad"' if ratio < 0.98 else "")
        rows.append(
            f"<tr><td>K{c['K']}</td><td class=num>{c['N']}</td>"
            f"<td class=num>{c['BS']}</td><td>{c['path']}</td>"
            f"<td class=num>{c['old']:.2f}</td><td class=num>{c['new']:.2f}"
            f"</td><td class=num{cls}>{ratio:.3f}</td>"
            f"<td>{c['exact']}</td></tr>")
    return ("<table><tr><th>K</th><th class=num>N</th><th class=num>BS</th>"
            "<th>path</th><th class=num>old µs</th><th class=num>new µs</th>"
            f"<th class=num>old/new</th><th>exact</th></tr>"
            + "".join(rows) + "</table>")


def p7_gm_matrix(p7, a, b):
    cols = [("fp32 tail", "tail", "fp32"), ("bf16 tail", "tail", "bf16"),
            ("fp16 tail", "tail", "fp16"), ("fp32 high-BS", "hb", "fp32")]
    rows = []
    for scen in SCENS:
        tds = []
        for _lab, kind, dt in cols:
            cells = p7[kind].get(dt, {}).get(scen, [])
            rs = [c[a] / c[b] for c in cells if c[b]]
            if not rs:
                tds.append("<td class=num>—</td>")
                continue
            g = gm(rs)
            w = sum(x > 1 for x in rs)
            cls = " hl-good" if g > 1.02 else (" hl-bad" if g < 0.98 else "")
            tds.append(f'<td class="num{cls}">{g:.3f} '
                       f'<span class="small">({w}/{len(rs)})</span></td>')
        rows.append(f"<tr><td>{scen}</td>{''.join(tds)}</tr>")
    hdr = ("<tr><th>scenario</th>"
           + "".join(f"<th class=num>{lab}</th>" for lab, _, _ in cols)
           + "</tr>")
    return "<table>" + hdr + "".join(rows) + "</table>"


def p7_detail_table(cells):
    rows = []
    for c in cells:
        ros, rls, rol = c["o"] / c["s"], c["l"] / c["s"], c["o"] / c["l"]
        cls = ' class="num hl-good"' if ros > 1.02 else (
            ' class="num hl-bad"' if ros < 0.98 else " class=num")
        rows.append(
            f"<tr><td>K{c['K']}</td><td class=num>{c['N']}</td>"
            f"<td class=num>{c['BS']}</td><td>{c['path']}</td>"
            f"<td class=num>{c['o']:.2f}</td><td class=num>{c['l']:.2f}</td>"
            f"<td class=num>{c['s']:.2f}</td><td{cls}>{ros:.3f}</td>"
            f"<td class=num>{rls:.3f}</td><td class=num>{rol:.3f}</td>"
            f"<td>{c['exact']}</td></tr>")
    return ("<table><tr><th>K</th><th class=num>N</th><th class=num>BS</th>"
            "<th>path</th><th class=num>orig µs</th><th class=num>legacy µs"
            "</th><th class=num>shipped µs</th><th class=num>o/s</th>"
            "<th class=num>l/s</th><th class=num>o/l</th><th>exact</th></tr>"
            + "".join(rows) + "</table>")


def build_p7(p7):
    data_js = json.dumps(p7, separators=(",", ":"))

    head = f"""
<h2><span class="no">7</span><span class="en">P0 switch verdict — three \
generations × three scenarios × three dtypes (added 2026-07-08)</span>\
<span class="zh">P0 切换判决 —— 三代 × 三场景 × 三 dtype(2026-07-08 增补)\
</span></h2>

<div class="en">
<p>Sections 1–6 validated the HLS levers <em>within</em> op21 (OLD/NEW knob A/Bs). This section settles the
remaining ship question — <strong>HLS-GVR vs the original production GVR</strong> — by a direct
<strong>three-arm</strong> paired A/B in one process (<code>scripts/ab_p0batch.py</code>):
<code>orig</code>&nbsp;= <code>gvr_cutedsl</code> (original single-CTA production GVR),
<code>legacy</code>&nbsp;= <code>gvr_ms_auto</code> with <code>OP21_FB_LOGFALSI=0&nbsp;+&nbsp;OP21_FB_DIST=0</code>
(pre-HLS op21), <code>shipped</code>&nbsp;= the HLS ship config (falsi ON + dist N-rule). op22 bundles, cold-L2,
arms interleaved per rep (throttle-immune), 30 reps/arm, sorted-set exactness per cell per arm.
Coverage: <em>tail</em> set (15 cells: BS=1 seqlen tail + 262K&nbsp;BS16, ×3&nbsp;K) × 3 scenarios ×
{{fp32, bf16, fp16}} on GPU1; <em>high-BS</em> set (25 cells, BS&nbsp;64–1024 incl. 1M&nbsp;BS64 — all
<code>ms_1cta</code>) × 3 scenarios, fp32, on GPU0. <strong>210 records × 3 arms: exact 630/630, 0
errors.</strong> Raw: <code>results/nsys/p0batch/</code>; drivers <code>drive_p0_gpu{{0,1}}.sh</code>; text
tables <code>parse_p0batch.py</code>; full narrative: <code>ITERATIONS.md</code> iter15.</p>
</div>
<div class="zh">
<p>§1–§6 验证的是 HLS 杠杆在 op21 <em>内部</em>的效果(OLD/NEW 开关 A/B)。本节回答剩下的 ship 问题 ——
<strong>HLS-GVR 对原始生产 GVR</strong> —— 用同进程<strong>三臂</strong>配对 A/B 直测
(<code>scripts/ab_p0batch.py</code>):<code>orig</code>&nbsp;= <code>gvr_cutedsl</code>(原始单 CTA 生产
GVR),<code>legacy</code>&nbsp;= <code>gvr_ms_auto</code> 且
<code>OP21_FB_LOGFALSI=0&nbsp;+&nbsp;OP21_FB_DIST=0</code>(HLS 前的 op21),<code>shipped</code>&nbsp;= HLS
ship 配置(falsi ON + dist N 规则)。op22 bundle,冷 L2,三臂逐 rep 交错(对节流免疫),每臂 30 rep,
逐格逐臂排序集合精确性校验。覆盖:<em>tail</em> 集(15 格:BS=1 seqlen 尾部 + 262K&nbsp;BS16,×3&nbsp;K)
× 3 场景 × {{fp32, bf16, fp16}},GPU1;<em>高 BS</em> 集(25 格,BS&nbsp;64–1024 含 1M&nbsp;BS64 —— 全部
<code>ms_1cta</code> 路径)× 3 场景,fp32,GPU0。<strong>210 条记录 × 3 臂:exact 630/630,0 错误。</strong>
原始数据:<code>results/nsys/p0batch/</code>;驱动 <code>drive_p0_gpu{{0,1}}.sh</code>;文本表
<code>parse_p0batch.py</code>;完整叙述:<code>ITERATIONS.md</code> iter15。</p>
</div>

<h3>7.1 {bi("Master matrices (gm of paired per-cell ratios; win counts in parentheses)",
            "总判决矩阵(逐格配对比值的几何均值;括号内为胜格数)")}</h3>
<p class="small">{bi(
  "gm orig/shipped — &gt;1 = HLS-GVR faster than the original production GVR:",
  "gm orig/shipped —— &gt;1 = HLS-GVR 快于原始生产 GVR:")}</p>
{p7_gm_matrix(p7, "o", "s")}
<p class="small">{bi(
  "gm legacy/shipped — the HLS lever effect inside op21 (&gt;1 = HLS faster than pre-HLS op21):",
  "gm legacy/shipped —— HLS 杠杆在 op21 内部的效果(&gt;1 = HLS 快于 HLS 前 op21):")}</p>
{p7_gm_matrix(p7, "l", "s")}
<p class="small">{bi(
  "gm orig/legacy — context: WITHOUT HLS, op21 was a real-only bet (&gt;1 = pre-HLS op21 faster than original):",
  "gm orig/legacy —— 背景:没有 HLS 时 op21 是一注只押 real 的赌(&gt;1 = HLS 前 op21 快于原始):")}</p>
{p7_gm_matrix(p7, "o", "l")}

<div class="en">
<div class="goodbox">
<strong>Verdict lines (direct-measure grade).</strong>
<ol>
<li><strong>Real-axis ranking FINAL, all dtypes: shipped(HLS) &gt; legacy &gt; orig.</strong> 16-bit
<em>amplifies</em> the HLS margin: o/s tail gm fp32 1.667 → fp16 1.801 → bf16 <strong>2.040</strong>
(bf16 sweeps 15/15).</li>
<li><strong>hugeN msc is HLS territory in every scenario × dtype</strong>: shipped takes ALL 1M msc-path
cells (o/s 1.35–2.18 worst, 1.68–2.76 best, 1.79–5.39 real) — the §6.3 distributed fallback closes the
worst-case collapse exactly as designed (K2048 1M worst 95→54 µs vs orig).</li>
<li><strong>The one systematic orig pocket = worst × <code>ms_1cta</code></strong> (mid-N BS1 tail + the
entire high-BS set): high-BS worst is 0/25 for shipped (gm o/s 0.788). The dist fallback does not exist on
the single-CTA ms path — the recorded follow-up lever is extending dist to ms (or a dispatch-to-orig
guard) for BS≥64 hugeN.</li>
<li><strong>K2048 falsi tax, direct-measured</strong>: fp32 real msc pocket l/s 0.944–0.965 (~5%, the low
end of the earlier ~8% read); in 16-bit the tax <em>vanishes</em> (l/s 0.94–1.07 mixed-sign, gm ≈ 1.0).
DSv4 geometry (K512/K1024) stays zero-tax (real l/s ≥ 0.98).</li>
<li><strong>Best-axis 16-bit flips to shipped-win</strong> (o/s 1.10–1.12 tail; fp32 was 0.99 neutral),
driven by the 1M msc cells (2.57–2.76×).</li>
</ol>
<strong>Switch decision: the ship rule stands — HLS op21 as default.</strong> The original GVR remains
preferable only in adversarial worst × single-CTA territory, which op22 already showed is not
production-representative. No dispatch change ships now.
</div>
</div>
<div class="zh">
<div class="goodbox">
<strong>判决(直测级)。</strong>
<ol>
<li><strong>real 轴排序最终确立,全 dtype 成立:shipped(HLS) &gt; legacy &gt; orig。</strong>16-bit
<em>放大</em> HLS 优势:o/s tail gm fp32 1.667 → fp16 1.801 → bf16 <strong>2.040</strong>(bf16 15/15
通吃)。</li>
<li><strong>hugeN msc 在所有场景 × 所有 dtype 都是 HLS 领地</strong>:shipped 拿下全部 1M msc 路径格
(o/s worst 1.35–2.18,best 1.68–2.76,real 1.79–5.39)—— §6.3 的分布式 fallback 按设计闭合了最坏
塌陷(K2048 1M worst 95→54 µs vs orig)。</li>
<li><strong>orig 唯一系统性口袋 = worst × <code>ms_1cta</code></strong>(中 N BS1 尾部 + 整个高 BS 集):
高 BS worst 段 shipped 0/25(gm o/s 0.788)。单 CTA ms 路径没有 dist fallback —— 已记录的后续杠杆 =
把 dist 扩展到 ms 路径(或对 BS≥64 hugeN 做 dispatch-to-orig 防御)。</li>
<li><strong>K2048 falsi 税(直测)</strong>:fp32 real msc 口袋 l/s 0.944–0.965(约 5%,处于此前 ~8%
读数的低端);16-bit 下税<em>消失</em>(l/s 0.94–1.07 混合符号,gm ≈ 1.0)。DSv4 几何(K512/K1024)
保持零税(real l/s ≥ 0.98)。</li>
<li><strong>best 轴 16-bit 翻为 shipped 胜</strong>(tail o/s 1.10–1.12;fp32 原为 0.99 中性),由 1M
msc 格(2.57–2.76×)驱动。</li>
</ol>
<strong>切换判决:ship 规则不变 —— HLS op21 为默认。</strong>原始 GVR 仅在对抗性 worst × 单 CTA 域占优,
而 op22 已证明该场景不具生产代表性。当前不改 dispatch。
</div>
</div>
"""

    # ------- 7.2 P0 no-regress grid @HEAD (frozen msa snapshot) -------
    msa_rows = []
    r_ratios, g_ratios = [], []
    for (K, N, BS, ms, rival, best_r, gbest) in MSA_P0:
        rr, gg = rival / ms, gbest / ms
        r_ratios.append(rr)
        g_ratios.append(gg)
        msa_rows.append(
            f"<tr><td>K{K}</td><td class=num>{N}</td><td class=num>{BS}</td>"
            f"<td class=num>{ms:.2f}</td><td class=num>{rival:.2f}</td>"
            f"<td class=\"num{' hl-good' if rr > 1 else ' hl-bad'}\">"
            f"{rr:.3f}</td><td>{best_r}</td>"
            f"<td class=num>{gbest:.2f}</td><td class=num>{gg:.3f}</td></tr>")
    msa_note_en = (f"gm rival/ms = <strong>{gm(r_ratios):.3f}</strong>, win "
                   f"{sum(x > 1 for x in r_ratios)}/{len(r_ratios)}; "
                   f"gm gvrbest/ms = {gm(g_ratios):.3f}")
    grid = f"""
<h3>7.2 {bi("P0 no-regress grid @HLS HEAD (17 cells, op21 synth, fp32)",
            "P0 无回归网格 @HLS HEAD(17 格,op21 synth,fp32)")}</h3>
<div class="en">
<p>The canonical 17-cell P0 grid (<code>drive_nsys_iter2.sh</code> → <code>nsys_verdict.py msa fp32</code>)
re-run at the HLS HEAD: {msa_note_en} — vs the iter7 anchor (1.249, 17/17) there is <strong>no
regression</strong> from iter13+14 (the N-gate keeps every N≤262144 binary bit-identical by construction;
the delta is run-to-run).</p>
</div>
<div class="zh">
<p>规范 17 格 P0 网格(<code>drive_nsys_iter2.sh</code> → <code>nsys_verdict.py msa fp32</code>)在 HLS
HEAD 重跑:{msa_note_en} —— 对照 iter7 锚点(1.249,17/17),iter13+14 <strong>无回归</strong>
(N 门控使所有 N≤262144 二进制按构造位相同;差值属跑批间漂移)。</p>
</div>
<details><summary>{bi("P0 grid per-cell (frozen 2026-07-08 snapshot)",
                      "P0 网格逐格(2026-07-08 冻结快照)")}</summary>
<table><tr><th>K</th><th class=num>N</th><th class=num>BS</th>
<th class=num>ms µs</th><th class=num>rival µs</th><th class=num>rival/ms</th>
<th>best rival</th><th class=num>gvr-best µs</th><th class=num>gvrbest/ms</th></tr>
{"".join(msa_rows)}</table></details>
<div class="en"><div class="warnbox">
<strong>Tooling gotcha</strong>: <code>nsys_verdict.py</code> defaults to <code>PREFIX="ms"</code>, which
silently reads the STALE iter1 <code>ms_*</code> reps still present in <code>results/nsys/</code> and
reproduces iter1's 0.830 — always pass <code>msa</code> explicitly.
</div></div>
<div class="zh"><div class="warnbox">
<strong>工具坑</strong>:<code>nsys_verdict.py</code> 默认 <code>PREFIX="ms"</code>,会静默读取仍留在
<code>results/nsys/</code> 的 iter1 陈旧 <code>ms_*</code> reps 并复现 iter1 的 0.830 —— 必须显式传
<code>msa</code>。
</div></div>
"""

    # ------- 7.3 interactive charts -------
    ctl = (
        '<div class="ctl">'
        + bi("<b>scenarios</b>", "<b>场景</b>") + " "
        + "".join(f'<label class="ck"><input type="checkbox" class="p7sck" '
                  f'value="{s}" checked>{s.upper()}</label> ' for s in SCENS)
        + " · " + bi("<b>pair</b>", "<b>比值对</b>") + " "
        '<label class="ck"><input type="radio" name="p7pr" value="os" checked>'
        'orig/shipped</label> '
        '<label class="ck"><input type="radio" name="p7pr" value="ls">'
        'legacy/shipped</label> '
        '<label class="ck"><input type="radio" name="p7pr" value="ol">'
        'orig/legacy</label><br>'
        + bi("<b>dtype</b>", "<b>数据类型</b>") + " "
        + "".join(f'<label class="ck"><input type="radio" name="p7dt" '
                  f'value="{d}"{" checked" if d == "fp32" else ""}>{d}'
                  f'</label> ' for d in P7_DTS)
        + " · "
        + "".join(f'<label class="ck"><input type="radio" name="p7kk" '
                  f'value="{k}"{" checked" if k == 1024 else ""}>K={k}'
                  f'</label> ' for k in (512, 1024, 2048))
        + "</div>")
    charts = f"""
<h3>7.3 {bi("Interactive charts (nsys cold-L2, paired ratios)",
            "交互图表(nsys 冷 L2,配对比值)")}</h3>
<p class="small">{bi(
  "Check scenarios / pick ratio pair + dtype + K. Left: tail set, paired "
  "ratio vs N (BS=1 cells; >1 = the second arm of the pair wins). Right: "
  "high-BS set (fp32 only, all ms_1cta), one bar per cell. Ratios are "
  "same-process paired — cross-run safe.",
  "勾选场景/选择比值对、dtype 与 K。左:tail 集,配对比值对 N(BS=1 格;"
  ">1 = 比值对中第二臂胜)。右:高 BS 集(仅 fp32,全部 ms_1cta),每格一根"
  "柱。比值为同进程配对 —— 跨轮安全。")}</p>
<div class="card">{ctl}
<div class="row"><div id="p7_ratio" class="plt"></div>
<div id="p7_hb" class="plt"></div></div></div>
"""

    # ------- 7.4 per-cell detail tables -------
    det = [f"<h3>7.4 {bi('Full per-cell data (210 cells × 3 arms)', '逐格全量数据(210 格 × 3 臂)')}</h3>"]
    for dt in P7_DTS:
        for scen in SCENS:
            cells = p7["tail"].get(dt, {}).get(scen)
            if not cells:
                continue
            det.append(
                f"<details><summary>{bi('tail 3-arm', 'tail 三臂')} — {dt} "
                f"{scen} ({len(cells)} cells)</summary>"
                + p7_detail_table(cells) + "</details>")
    for scen in SCENS:
        cells = p7["hb"].get("fp32", {}).get(scen)
        if not cells:
            continue
        det.append(
            f"<details><summary>{bi('high-BS 3-arm', '高 BS 三臂')} — fp32 "
            f"{scen} ({len(cells)} cells)</summary>"
            + p7_detail_table(cells) + "</details>")

    wiring = """
<script>
(() => {
const P7 = %DATA%;
const NS7 = [16384, 65536, 262144, 1048576];
const SC7 = {best: "#0f6bb3", worst: "#a12a1e", real: "#1a7a4a"};
const PAIRS7 = {os: ["o", "s", "orig/shipped"],
                ls: ["l", "s", "legacy/shipped"],
                ol: ["o", "l", "orig/legacy"]};
const fmtN7 = n => n >= 1048576 ? (n / 1048576) + "M"
                                : (n / 1024) + "K";
function p7Sel(cls) {
  return Array.from(document.querySelectorAll("input." + cls + ":checked"))
    .map(e => e.value);
}
function p7Val(name) {
  return document.querySelector('input[name="' + name + '"]:checked').value;
}
function p7Draw() {
  const scens = p7Sel("p7sck");
  const dt = p7Val("p7dt");
  const K = +p7Val("p7kk");
  const [a, b, plab] = PAIRS7[p7Val("p7pr")];
  const tr = [];
  for (const s of scens) {
    const cells = ((P7.tail[dt] || {})[s] || []).filter(
      c => c.K === K && c.BS === 1);
    cells.sort((x, y) => x.N - y.N);
    tr.push({x: cells.map(c => c.N), y: cells.map(c => c[a] / c[b]),
             name: s, mode: "lines+markers", line: {color: SC7[s]}});
  }
  tr.push({x: [NS7[0], NS7[NS7.length - 1]], y: [1, 1], mode: "lines",
           line: {color: "#888", dash: "dot"}, showlegend: false});
  Plotly.react("p7_ratio", tr, {
    title: {text: "tail  " + dt + "  K=" + K + "  " + plab
                  + "  (>1 = 2nd arm wins)", font: {size: 13}},
    height: 360, margin: {l: 55, r: 10, t: 36, b: 42},
    xaxis: {type: "log", title: "N", tickvals: NS7,
            ticktext: ["16K", "64K", "256K", "1M"]},
    yaxis: {title: plab},
    legend: {orientation: "h", font: {size: 10}}, showlegend: true});
  const bars = [];
  for (const s of scens) {
    const cells = ((P7.hb.fp32 || {})[s] || []).filter(c => c.K === K);
    cells.sort((x, y) => x.N - y.N || x.BS - y.BS);
    bars.push({x: cells.map(c => fmtN7(c.N) + " BS" + c.BS),
               y: cells.map(c => c[a] / c[b]),
               name: s, type: "bar", marker: {color: SC7[s]}});
  }
  Plotly.react("p7_hb", bars, {
    title: {text: "high-BS fp32 (ms_1cta)  K=" + K + "  " + plab,
            font: {size: 13}},
    height: 360, margin: {l: 55, r: 10, t: 36, b: 80},
    yaxis: {title: plab}, barmode: "group",
    shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1,
              line: {color: "#888", dash: "dot", width: 1}}],
    legend: {orientation: "h", font: {size: 10}}});
}
for (const e of document.querySelectorAll(
     "input.p7sck, input[name=p7pr], input[name=p7dt], input[name=p7kk]"))
  e.addEventListener("change", p7Draw);
p7Draw();
})();
</script>
""".replace("%DATA%", data_js)

    return ("<!-- P0-SWITCH-BEGIN -->" + head + grid + charts + "".join(det)
            + wiring + "\n<!-- P0-SWITCH-END -->")


def main():
    ab = load_ab()
    tau = load_tau()
    data_js = json.dumps({"ab": ab, "tau": tau, "p0": P0_SPOT},
                         separators=(",", ":"))

    # ---------------- section 6.6 (charts) ----------------
    ctl1 = (
        '<div class="ctl">'
        + bi("<b>scenarios</b>", "<b>场景</b>") + " "
        + "".join(f'<label class="ck"><input type="checkbox" class="hsck" '
                  f'value="{s}"{" checked" if s != "worst" else ""}>'
                  f'{s.upper()}</label> ' for s in SCENS)
        + " · " + bi("<b>run</b>", "<b>轮次</b>") + " "
        '<label class="ck"><input type="radio" name="hrun" value="i13">'
        'iter13 log-falsi</label> '
        '<label class="ck"><input type="radio" name="hrun" value="a01">'
        'iter13 α=0.1</label> '
        '<label class="ck"><input type="radio" name="hrun" value="i14" '
        'checked>iter14 distributed</label><br>'
        + "".join(f'<label class="ck"><input type="radio" name="hkk" '
                  f'value="{k}"{" checked" if k == 2048 else ""}>K={k}'
                  f'</label> ' for k in (512, 1024, 2048))
        + '</div>')
    charts = f"""
<h3>6.6 {bi("Interactive charts (nsys cold-L2, BS=1 seqlen cells)",
            "交互图表(nsys 冷 L2,BS=1 seqlen 格)")}</h3>
<p class="small">{bi(
  "Check scenarios / pick run + K. Left: absolute µs vs N (log-log; old "
  "dashed, new solid — pairs are same-process interleaved, so within-run "
  "comparisons are throttle-immune; absolute µs do NOT transfer across "
  "runs). Right: the paired old/new ratio vs N (cross-run safe; >1 = new "
  "wins). BS=16 cells live in the 6.7 tables.",
  "勾选场景/选择轮次与 K。左:绝对 µs 对 N(双对数;old 虚线、new 实线 —— "
  "两臂同进程交错,轮内比较对节流免疫;绝对 µs 不跨轮可比)。右:配对 "
  "old/new 比值对 N(跨轮安全;>1 = new 胜)。BS=16 格见 6.7 明细表。")}</p>
<div class="card">{ctl1}
<div class="row"><div id="h_abs" class="plt"></div>
<div id="h_ratio" class="plt"></div></div></div>

<p class="small">{bi(
  "Width-tax tau(M) measured same-silicon (Step 0) — tau(3) sits on the "
  "interpolated 1.2 at large N. P0 spot: the fast-path cost of each "
  "iteration's code mass on op21 synth data (light/no fallback).",
  "宽度税 tau(M) 同硅实测(Step 0)—— 大 N 处 tau(3) 落在插值 1.2 上。"
  "P0 spot:两轮代码质量对 op21 synth 数据(轻/无 fallback)fast path "
  "的代价。")}</p>
<div class="card"><div class="ctl">
<label class="ck"><input type="checkbox" class="htck" value="2" checked>M=2</label>
<label class="ck"><input type="checkbox" class="htck" value="3" checked>M=3</label>
<label class="ck"><input type="checkbox" class="htck" value="4" checked>M=4</label>
 · <label class="ck"><input type="checkbox" class="htdt" value="fp32" checked>fp32</label>
<label class="ck"><input type="checkbox" class="htdt" value="fp16" checked>fp16</label>
</div>
<div class="row"><div id="h_tau" class="plt"></div>
<div id="h_p0" class="plt"></div></div></div>
"""

    # ---------------- section 6.7 (detail tables) ----------------
    det = [f"<h3>6.7 {bi('Full per-cell data', '逐格全量数据')}</h3>"]
    names = {"i13": ("iter13 log-falsi A/B (α=0.2)", "iter13 log-falsi A/B(α=0.2)"),
             "a01": ("iter13 α=0.1 probe", "iter13 α=0.1 探针"),
             "i14": ("iter14 distributed-fallback A/B", "iter14 分布式 fallback A/B")}
    for run in ("i13", "a01", "i14"):
        for scen in SCENS:
            cells = ab.get(run, {}).get(scen)
            if not cells:
                continue
            en, zh = names[run]
            det.append(
                f"<details><summary>{bi(en, zh)} — {scen} "
                f"({len(cells)} cells)</summary>"
                + detail_table(cells, run) + "</details>")
    tau_rows = []
    for dt in ("fp32", "fp16"):
        for N in sorted(tau.get(dt, {})):
            ms = tau[dt][N]
            if 1 not in ms:
                continue
            tau_rows.append(
                f"<tr><td>{dt}</td><td class=num>{N}</td>"
                + "".join(f"<td class=num>{ms.get(m, float('nan')):.3f}</td>"
                          for m in (1, 2, 3, 4))
                + "".join(f"<td class=num>{ms[m]/ms[1]:.3f}</td>"
                          if m in ms else "<td class=num>—</td>"
                          for m in (2, 3, 4))
                + "</tr>")
    det.append(
        f"<details><summary>{bi('tau(M) same-silicon sweep (Step 0)', 'tau(M) 同硅扫描(Step 0)')}"
        "</summary><table><tr><th>dtype</th><th class=num>N</th>"
        "<th class=num>M1 µs</th><th class=num>M2 µs</th>"
        "<th class=num>M3 µs</th><th class=num>M4 µs</th>"
        "<th class=num>τ(2)</th><th class=num>τ(3)</th><th class=num>τ(4)</th></tr>"
        + "".join(tau_rows) + "</table></details>")
    p0_rows = []
    for cell, d in P0_SPOT.items():
        r13 = d["i13"][0] / d["i13"][1]
        r14 = d["i14"][0] / d["i14"][1]
        p0_rows.append(
            f"<tr><td>{cell}</td>"
            f"<td class=num>{d['i13'][0]:.2f}</td><td class=num>{d['i13'][1]:.2f}</td>"
            f"<td class=num>{r13:.3f}</td>"
            f"<td class=num>{d['i14'][0]:.2f}</td><td class=num>{d['i14'][1]:.2f}</td>"
            f"<td class=num>{r14:.3f}</td></tr>")
    det.append(
        f"<details><summary>{bi('P0 no-regress spot (op21 synth)', 'P0 无回归 spot(op21 synth)')}"
        "</summary><table><tr><th>cell</th>"
        "<th class=num>i13 old</th><th class=num>i13 new</th><th class=num>old/new</th>"
        "<th class=num>i14 old</th><th class=num>i14 new</th><th class=num>old/new</th></tr>"
        + "".join(p0_rows) + "</table></details>")

    wiring = """
<script>
const HD = %DATA%;
const NS = [16384, 65536, 262144, 1048576];
const SCOL = {best: "#0f6bb3", worst: "#a12a1e", real: "#1a7a4a"};
const LAYOUT = (title, ylab, ylog) => ({
  title: {text: title, font: {size: 13}}, height: 360,
  margin: {l: 55, r: 10, t: 36, b: 42},
  xaxis: {type: "log", title: "N", tickvals: NS,
          ticktext: ["16K", "64K", "256K", "1M"]},
  yaxis: {type: ylog ? "log" : "linear", title: ylab},
  legend: {orientation: "h", font: {size: 10}}, showlegend: true});
function hSel(cls) {
  return Array.from(document.querySelectorAll("input." + cls + ":checked"))
    .map(e => e.value);
}
function hDraw() {
  const scens = hSel("hsck");
  const run = document.querySelector('input[name="hrun"]:checked').value;
  const K = +document.querySelector('input[name="hkk"]:checked').value;
  const tAbs = [], tRat = [];
  for (const s of scens) {
    const cells = ((HD.ab[run] || {})[s] || []).filter(
      c => c.K === K && c.BS === 1);
    cells.sort((a, b) => a.N - b.N);
    const xs = cells.map(c => c.N);
    tAbs.push({x: xs, y: cells.map(c => c.old), name: s + " old",
               mode: "lines+markers", line: {color: SCOL[s], dash: "dash"}});
    tAbs.push({x: xs, y: cells.map(c => c.new), name: s + " new",
               mode: "lines+markers", line: {color: SCOL[s]}});
    tRat.push({x: xs, y: cells.map(c => c.old / c.new), name: s,
               mode: "lines+markers", line: {color: SCOL[s]}});
  }
  tRat.push({x: [NS[0], NS[NS.length - 1]], y: [1, 1], name: "parity",
             mode: "lines", line: {color: "#888", dash: "dot"},
             showlegend: false});
  Plotly.react("h_abs",
    tAbs, LAYOUT(run + "  K=" + K + "  cold-L2 µs (old dashed / new solid)",
                 "µs", true));
  Plotly.react("h_ratio",
    tRat, LAYOUT(run + "  K=" + K + "  paired old/new ratio (>1 = new wins)",
                 "old/new", false));
}
function hDrawTau() {
  const ms = hSel("htck").map(Number);
  const dts = hSel("htdt");
  const tr = [];
  const mcol = {2: "#0f6bb3", 3: "#1a7a4a", 4: "#a12a1e"};
  for (const dt of dts) {
    const byN = HD.tau[dt] || {};
    const xs = Object.keys(byN).map(Number).sort((a, b) => a - b)
      .filter(n => byN[n]["1"] !== undefined);
    for (const m of ms) {
      const y = xs.map(n => byN[n][m] !== undefined
                            ? byN[n][m] / byN[n]["1"] : null);
      tr.push({x: xs, y: y, name: dt + " τ(" + m + ")",
               mode: "lines+markers",
               line: {color: mcol[m], dash: dt === "fp16" ? "dash" : "solid"}});
    }
  }
  tr.push({x: [4096, 1048576], y: [1.2, 1.2], name: "HLS τ(3)=1.2",
           mode: "lines", line: {color: "#1a7a4a", dash: "dot", width: 1}});
  const lay = LAYOUT("width tax τ(M) = t(M)/t(M=1), cold-L2", "τ", false);
  lay.xaxis.tickvals = [4096, 16384, 65536, 262144];
  lay.xaxis.ticktext = ["4K", "16K", "64K", "256K"];
  Plotly.react("h_tau", tr, lay);
  const cells = Object.keys(HD.p0);
  const bars = [
    {x: cells, y: cells.map(c => 100 * (HD.p0[c].i13[0] / HD.p0[c].i13[1] - 1)),
     name: "iter13", type: "bar", marker: {color: "#0f6bb3"}},
    {x: cells, y: cells.map(c => 100 * (HD.p0[c].i14[0] / HD.p0[c].i14[1] - 1)),
     name: "iter14 (dist forced)", type: "bar", marker: {color: "#a12a1e"}}];
  Plotly.react("h_p0", bars, {
    title: {text: "P0 fast-path spot: paired gain % (neg = code-mass tax)",
            font: {size: 13}},
    height: 360, margin: {l: 55, r: 10, t: 36, b: 90},
    yaxis: {title: "old/new − 1  (%)"}, barmode: "group",
    legend: {orientation: "h", font: {size: 10}}});
}
for (const e of document.querySelectorAll(
     "input.hsck, input[name=hrun], input[name=hkk]"))
  e.addEventListener("change", hDraw);
for (const e of document.querySelectorAll("input.htck, input.htdt"))
  e.addEventListener("change", hDrawTau);
hDraw(); hDrawTau();
</script>
""".replace("%DATA%", data_js)

    css = """
<style>
  .card { background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; padding: .7rem .9rem; margin: 1rem 0; }
  .ctl { font-size: .84rem; margin-bottom: .4rem; }
  .ck { margin-right: .55rem; white-space: nowrap; cursor: pointer; }
  .row { display: flex; flex-wrap: wrap; gap: .6rem; }
  .plt { flex: 1 1 440px; min-width: 380px; }
  details { margin: .5rem 0; }
  details summary { cursor: pointer; font-size: .9rem; color: var(--accent); }
  details table { font-size: .8rem; }
  /* charts block: bilingual spans must stay INLINE (report default for
     .zh is display:block, which breaks control rows / summaries) */
  #lang-zh:checked ~ .wrap .ctl .zh,
  #lang-zh:checked ~ .wrap summary .zh,
  #lang-zh:checked ~ .wrap h3 .zh,
  #lang-zh:checked ~ .wrap p.small .zh { display: inline; }
</style>"""

    frag = ("<!-- HLS-CHARTS-BEGIN -->" + css + charts + "".join(det)
            + wiring + "\n<!-- HLS-CHARTS-END -->")

    p7 = load_p0batch()
    frag7 = build_p7(p7)

    p = OP21 / "HLS_VALIDATION_REPORT.html"
    t = p.read_text()
    if "HLS-CHARTS-BEGIN" in t:
        pre, rest = t.split("<!-- HLS-CHARTS-BEGIN -->", 1)
        _, post = rest.split("<!-- HLS-CHARTS-END -->", 1)
        t = pre + frag + post
    else:
        anchor = "<footer>"
        t = t.replace(anchor, frag + "\n" + anchor, 1)
    if "P0-SWITCH-BEGIN" in t:
        pre, rest = t.split("<!-- P0-SWITCH-BEGIN -->", 1)
        _, post = rest.split("<!-- P0-SWITCH-END -->", 1)
        t = pre + frag7 + post
    else:
        t = t.replace("<footer>", frag7 + "\n<footer>", 1)
    if "cdn.plot.ly" not in t:
        t = t.replace("</title>",
                      '</title>\n<script src="https://cdn.plot.ly/'
                      'plotly-2.35.2.min.js"></script>', 1)
    p.write_text(t)
    n_cells = sum(len(c) for r in ab.values() for c in r.values())
    n7 = sum(len(c) for kind in p7.values()
             for dts in kind.values() for c in dts.values())
    print(f"inserted charts block: {n_cells} A/B cells, "
          f"{len(tau_rows)} tau rows, {len(P0_SPOT)} p0 cells; "
          f"section 7: {n7} three-arm cells, {len(MSA_P0)} P0-grid cells")


if __name__ == "__main__":
    main()
