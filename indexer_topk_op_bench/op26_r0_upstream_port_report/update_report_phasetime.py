#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Idempotent injector: §9e "BS=1 real-data phase-time breakdown at the PR head"
into REPORT.html (marker PHASETIME:BEGIN/END, inserted after HEADFULL:END).

Source of truth:
  p4f1_harness/phase_breakdown_ptime/phase_full_full.csv   (865 cells)
  p4f1_harness/phase_breakdown_ptime/phase_analysis.json   (analyze_phases.py)

All charts are Plotly (already loaded by REPORT.html), self-contained script
block inside the marker region; bilingual via the report's lang-en/lang-zh
span convention. Prose findings live in FINDINGS_* below.
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "REPORT.html")
PBP = os.path.join(HERE, "p4f1_harness", "phase_breakdown_ptime")
BEGIN = "<!-- PHASETIME:BEGIN (update_report_phasetime.py) -->"
END = "<!-- PHASETIME:END -->"
ANCHOR_AFTER = "<!-- HEADFULL:END -->"

PHASES = ["p1_gather_stats", "smem_stage", "p1b_rungs", "p2_count_admission",
          "p3_collect", "p4_select", "epilogue"]
PLAB = ["P1 gather/stats", "smem-stage", "P1b rungs", "P2 count+adm",
        "P3 collect", "P4 select(+tail)", "epilogue"]
PCOL = ["#4c78a8", "#9ecae9", "#f2cf5b", "#f58518", "#54a24b", "#e45756",
        "#b3b3b3"]
ISL_ORDER = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]


def load_cells():
    rows = []
    for r in csv.DictReader(open(os.path.join(PBP, "phase_full_full.csv"))):
        d = dict(u=r["uuid"], m=r["model"], i=r["isl"], l=int(r["layer"]),
                 K=int(r["K"]), N=int(r["N"]), h=float(r["hit"]),
                 cs=int(r["cs"]), T=int(r["T"]),
                 us=round(float(r["us_prod_nsys"]), 3),
                 ov=round(float(r["overhead"]), 4))
        d["f"] = [round(float(r[f"frac_{p}"]), 5) for p in PHASES]
        d["uu"] = [round(float(r[f"us_{p}"]), 4) for p in PHASES]
        rows.append(d)
    meta = {f"{m['model']}_{m['isl']}_L{int(m['layer']):02d}": m
            for m in csv.DictReader(open(os.path.join(HERE, "real_3arm_layers_full.csv")))}
    for d in rows:
        mm = meta[d["u"]]
        d["pv"] = round(float(mm["pr_vs_base"]), 3) if mm["pr_vs_base"] else None
        d["rg"] = ("cs1-small" if d["cs"] == 1 and d["N"] <= 8448 else
                   "cs1-mid" if d["cs"] == 1 else
                   "cs4" if d["cs"] == 4 else f"cs8-T{d['T']}")
        pv = d["pv"]
        d["pc"] = ("n/a" if pv is None else "strong-win" if pv >= 1.15 else
                   "win" if pv >= 1.05 else "parity" if pv >= 0.95 else "loss")
        d["ht"] = "hi" if d["h"] >= 0.60 else ("mid" if d["h"] >= 0.35 else "lo")
    # within-(model,isl) speed quartiles
    from collections import defaultdict
    g = defaultdict(list)
    for d in rows:
        g[(d["m"], d["i"])].append(d)
    for grp in g.values():
        srt = sorted(grp, key=lambda d: d["us"])
        nq = max(1, len(srt) // 4)
        for d in srt[:nq]:
            d["sq"] = "fastest25"
        for d in srt[-nq:]:
            d["sq"] = "slowest25"
        for d in srt:
            d.setdefault("sq", "mid50")
    return rows


# ---------------------------------------------------------------- prose ----
# (filled from PHASE_FULL_ANALYSIS.md after the run; keep EN/ZH in sync)
FINDINGS_EN = """
<p><b>Findings (each claim is per-cell, no cross-cell averaging).</b></p>
<ol>
<li><b>P4 dominance is a per-cell fact, not an average artifact.</b> P4 select(+tail) is the largest
phase in <b>827/865</b> cells; its share is med 44% with a full range of 1.8–62%. In absolute terms the
per-rung medians are: cs1-small 4.5&thinsp;µs (44% of 10.1&thinsp;µs), cs1-mid 4.8&thinsp;µs (36% of
13.1&thinsp;µs), cs4 8.2&thinsp;µs (49% of 16.7&thinsp;µs), cs8 9.6&thinsp;µs (50% of 19.3&thinsp;µs).
This silicon-confirms distP4 (parallelize the leader-only final collect across cluster CTAs) as the #1
remaining lever on the current head — the 07-20 warp-redundant P4 search landed and P4 still holds
~half the kernel wherever clusters are in play.</li>
<li><b>The dispatch boundary reshapes the composition.</b> Within cs=1, P4's share falls monotonically
with N (flash: 48% at 4k → 31% at 128k) while P2+P3 grow to ~50% — the single-CTA scan is the cost
center just below the cluster threshold. Crossing to cs≥4 parallelizes P2/P3 (P3 drops from ~25% to
~13%) but the leader-only P4 — which also absorbs the wait for the slowest peer's collect — re-balloons
to 47–55%. The most balanced compositions on the whole grid are the cs=1 N≈32k cells (flash/pro 128k
ISL, v32 32k).</li>
<li><b>High hint quality is not free speed on the PR head.</b> Within each (model, ISL) group, the
slowest-quartile layers have a <i>higher</i> hit median than the fastest quartile (0.84 vs 0.76) and a
P2 share of 0.27 vs 0.20. In the cs1-small rung, Spearman(hit, P2 µs) = +0.62 and Spearman(hit, total
µs) = +0.40; at cs8, hit correlates with P1b rung-ladder µs at +0.58. The mechanism matches the known
real-data undershoot bias: a tight hint-seeded threshold makes the admission/refine walk (fb_fix) work
harder, and P4 does not shrink in return. (Dispatch still must not branch on hit — it is unknowable at
inference; this is a kernel-internal cost story.)</li>
<li><b>The degenerate fast path exists but only at tiny N.</b> 3 cells (N=1027, hit ≥ 0.86, incl. two
hit = 1.0 pro layers) collapse P2–P3 and show P4 at 1.8–2.8%, running at the 5.3–6.1&thinsp;µs grid
floor — the only cells where the threshold prior alone essentially solves the problem.</li>
<li><b>A 37-cell P2-dominant minority (P2 share 0.31–0.49) sits at both hint extremes</b> (hit 0.02
and 1.0), spread over 13 (model, ISL) groups: these are refine-walk-bound cells — deep undershoot walks
at low hit, tight-boundary walks at high hit — the population the secant/fb_fix levers address.</li>
<li><b>PR-loss cells are an admission/tie story, not a P4 story.</b> The 36 cells where the PR kernel
loses to base (§7b ratios, classification only) concentrate in v32 large-N high-hit cells (19/36 at
v32 64k–256k; class hit med 0.85, us med 17.9). Their phase mix is P2-heavier (0.22 vs 0.20 grid-wide)
with P4 at the grid-typical 0.49 — consistent with tie-dense K=2048 exact-tail work, not with a
structural P4 deficit.</li>
<li><b>Validation.</b> 865/865 exact on BOTH arms, 865/865 monotone timestamps, instrumentation
overhead med +0.5% / p95 +4.4%. 28 cells exceed the ±7% gate: 27 are ~10&thinsp;µs launch-floor cells
(pro 4k–16k) where the delta is negative (noise), and flash_128k_L42 reproducibly pays ~+10% stamp tax
while its phase fractions reproduce (P4 0.528 vs 0.531 across independent runs) — fractions kept,
absolute anchored to the pristine arm as everywhere else.</li>
</ol>"""
FINDINGS_ZH = """
<p><b>结论(每条都基于逐 cell 数据,全程不做跨 cell 平均)。</b></p>
<ol>
<li><b>P4 主导是逐格事实,不是平均值假象。</b>P4 select(+tail) 在 <b>827/865</b> 个 cell 中都是最大相;
份额中位 44%,全距 1.8–62%。绝对值上各 rung 中位:cs1-small 4.5&thinsp;µs(占 10.1&thinsp;µs 的 44%)、
cs1-mid 4.8&thinsp;µs(36%/13.1&thinsp;µs)、cs4 8.2&thinsp;µs(49%/16.7&thinsp;µs)、cs8 9.6&thinsp;µs
(50%/19.3&thinsp;µs)。这在当前 head 上以硅上数据再次确认:distP4(把 leader 独占的最终收集并行到
cluster 各 CTA)仍是第一杠杆——07-20 的 warp-redundant P4 搜索已落地,凡有 cluster 的地方 P4 仍占
kernel 约一半。</li>
<li><b>分派边界重塑相位构成。</b>cs=1 内部,P4 份额随 N 单调下降(flash:4k 的 48% → 128k 的 31%),
P2+P3 升至 ~50%——紧贴 cluster 门槛之下,单 CTA 扫描是成本中心。跨入 cs≥4 后 P2/P3 被并行化
(P3 从 ~25% 降到 ~13%),但 leader 独占的 P4——还吸收等最慢 peer collect 的时间——重新膨胀到
47–55%。全网格构成最均衡的是 cs=1、N≈32k 的格(flash/pro 128k ISL、v32 32k)。</li>
<li><b>在 PR head 上,高 hint 质量不等于更快。</b>各 (model, ISL) 组内,最慢四分位层的 hit 中位反而
高于最快四分位(0.84 对 0.76),P2 份额 0.27 对 0.20。cs1-small rung 内 Spearman(hit, P2 µs) = +0.62、
Spearman(hit, 总 µs) = +0.40;cs8 上 hit 与 P1b 阶梯 µs 相关 +0.58。机理与已知的真实数据 undershoot
偏置一致:hint 种子给出的阈值越紧,admission/refine(fb_fix)走得越多,而 P4 并不因此缩短。
(分派仍然不得依据 hit 分支——推理时不可知;这是 kernel 内部的成本叙事。)</li>
<li><b>退化快路径存在,但只在极小 N。</b>3 个 cell(N=1027,hit ≥ 0.86,含两个 hit = 1.0 的 pro 层)
把 P2–P3 塌缩掉,P4 仅 1.8–2.8%,跑在 5.3–6.1&thinsp;µs 的全网格地板——只有这些格,阈值先验本身就
基本解决了问题。</li>
<li><b>37 个 P2 主导的少数派(P2 份额 0.31–0.49)分布在 hint 两个极端</b>(hit 0.02 与 1.0),
横跨 13 个 (model, ISL) 组:这是 refine-walk 受限的群体——低 hit 是深 undershoot 行走,高 hit 是
紧边界行走——正是 secant/fb_fix 杠杆面向的人群。</li>
<li><b>PR 输格是 admission/tie 叙事,不是 P4 叙事。</b>PR 对 base 落败的 36 格(§7b 比值,仅用于分类)
集中在 v32 大 N 高 hit(19/36 位于 v32 64k–256k;类 hit 中位 0.85,us 中位 17.9)。其相位构成 P2 偏重
(0.22 对全网格 0.20),P4 处于网格常态 0.49——与 K=2048 tie 密集触发 exact-tail 的机制一致,
而非 P4 结构性劣势。</li>
<li><b>验证。</b>两臂 865/865 精确,865/865 时间戳单调,插桩开销中位 +0.5%、p95 +4.4%。28 格超出
±7% 门:27 格是 ~10&thinsp;µs launch-floor 格(pro 4k–16k)且偏差为负(噪声);flash_128k_L42 可复现地
付 ~+10% 插桩税,但其相位份额跨独立运行复现(P4 0.528 对 0.531)——保留其份额,绝对值一如其他格
锚定在纯净臂。</li>
</ol>"""


def bi(en, zh, tag="p"):
    return (f"<div class='lang-en'><{tag}>{en}</{tag}></div>"
            f"<div class='lang-zh'><{tag}>{zh}</{tag}></div>")


def build(A, cells):
    v = A["validation"]
    dom = A["dominant_phase_counts"]
    p4 = A["p4_frac_overall"]
    dom_s = ", ".join(f"{PLAB[PHASES.index(k)]} {n}" for k, n in
                      sorted(dom.items(), key=lambda kv: -kv[1]))
    kpis = f"""<div class="kpis">
<div class="kpi"><div class="v" style="color:#6ede8a">{v['n']}/865</div><div class="l lang-en">cells measured — exact {v['n']-v['inexact']}, monotone {v['n']-v['nonmono']}</div><div class="l lang-zh">测量格数——精确 {v['n']-v['inexact']},时间戳单调 {v['n']-v['nonmono']}</div></div>
<div class="kpi"><div class="v">{p4['med']*100:.0f}%</div><div class="l lang-en">P4 select median share (range {p4['min']*100:.0f}–{p4['max']*100:.0f}%)</div><div class="l lang-zh">P4 select 份额中位(范围 {p4['min']*100:.0f}–{p4['max']*100:.0f}%)</div></div>
<div class="kpi"><div class="v">{dom.get('p4_select',0)}/{v['n']}</div><div class="l lang-en">cells where P4 is the dominant phase</div><div class="l lang-zh">P4 为最大相的格数</div></div>
<div class="kpi"><div class="v">{v['ovh']['med']*100:+.1f}%</div><div class="l lang-en">instrumentation overhead median (nsys timed/prod; p95 within gate)</div><div class="l lang-zh">插桩开销中位(nsys timed/prod;p95 在门内)</div></div>
</div>"""

    # ---- per model x ISL table (frac med [p25,p75]) ----
    def mi_table():
        hdr = ("<tr><th>model/ISL</th><th>n</th><th>cs/T</th>"
               "<th><span class='lang-en'>PR µs med [min,max]</span>"
               "<span class='lang-zh'>PR µs 中位[最小,最大]</span></th>" +
               "".join(f"<th>{p}</th>" for p in PLAB) + "</tr>")
        trs = []
        for key in sorted(A["per_model_isl"],
                          key=lambda k: (k.split("|")[0],
                                         ISL_ORDER.index(k.split("|")[1]))):
            e = A["per_model_isl"][key]
            tds = [key.replace("|", "/"), str(e["n"]), f"{e['cs']}/{e['T']}",
                   f"{e['us']['med']:.1f} [{e['us']['min']:.1f},{e['us']['max']:.1f}]"]
            for ph in PHASES:
                f = e[f"f_{ph}"]
                tds.append(f"{f['med']:.2f} <span class='mut'>[{f['p25']:.2f},{f['p75']:.2f}]</span>")
            trs.append("<tr>" + "".join(f"<td>{t}</td>" for t in tds) + "</tr>")
        return "<table>" + hdr + "".join(trs) + "</table>"

    # ---- class table ----
    def cls_table(key, order, label_en, label_zh):
        hdr = (f"<tr><th><span class='lang-en'>{label_en}</span>"
               f"<span class='lang-zh'>{label_zh}</span></th><th>n</th>"
               "<th>PR µs med</th><th>hit med</th><th>pr_vs_base med</th>" +
               "".join(f"<th>{PLAB[PHASES.index(p)]}</th>" for p in
                       ["p1_gather_stats", "p1b_rungs", "p2_count_admission",
                        "p3_collect", "p4_select"]) + "</tr>")
        trs = []
        for cls in order:
            e = A[f"by_{key}"].get(cls)
            if not e:
                continue
            pv = e["pr_vs_base"]
            tds = [cls, str(e["n"]), f"{e['us']['med']:.1f}",
                   f"{e['hit']['med']:.2f}",
                   f"{pv['med']:.2f}" if pv.get("med") is not None else "—"]
            for ph in ["p1_gather_stats", "p1b_rungs", "p2_count_admission",
                       "p3_collect", "p4_select"]:
                f = e[f"f_{ph}"]
                tds.append(f"{f['med']:.2f} <span class='mut'>[{f['p25']:.2f},{f['p75']:.2f}]</span>")
            trs.append("<tr>" + "".join(f"<td>{t}</td>" for t in tds) + "</tr>")
        return "<table>" + hdr + "".join(trs) + "</table>"

    data_js = json.dumps([[d["u"], d["m"], d["i"], d["l"], d["N"], d["h"],
                           d["us"], d["f"], d["uu"], d["pv"], d["rg"],
                           d["pc"], d["ht"], d["sq"]] for d in cells],
                         separators=(",", ":"))

    ctrl_css = """<style>
.ptc{margin:6px 0 2px;font-size:13px}.ptc label{margin-right:10px;cursor:pointer}
.ptc input{vertical-align:middle}.ptfig{min-height:380px}
</style>"""

    def radios(name, opts, checked, lab=None):
        return " ".join(
            f"<label><input type='radio' name='{name}' value='{o}'"
            f"{' checked' if o == checked else ''}> {lab[k] if lab else o}</label>"
            for k, o in enumerate(opts))

    def checks(name, opts, on, lab=None):
        return " ".join(
            f"<label><input type='checkbox' class='{name}' value='{o}'"
            f"{' checked' if o in on else ''}> {lab[k] if lab else o}</label>"
            for k, o in enumerate(opts))

    figs = f"""
{ctrl_css}
<h3><span class="lang-en">Fig. P1 — per-layer stacked phase time (no averaging: every layer shown)</span><span class="lang-zh">图 P1——逐 layer 分相堆叠耗时(不做平均:每层单独显示)</span></h3>
<div class="ptc">model: {radios('ptm1', ['flash','pro','v32'], 'pro')} &nbsp; ISL: {radios('pti1', ISL_ORDER, '128k')} &nbsp; unit: {radios('ptu1', ['us','frac'], 'us')}</div>
<div id="ptfig1" class="ptfig"></div>
<h3><span class="lang-en">Fig. P2 — phase-share distributions across layers, by ISL (boxes = layer spread)</span><span class="lang-zh">图 P2——各 ISL 下 phase 份额跨 layer 分布(箱线 = 层间散布)</span></h3>
<div class="ptc">model: {radios('ptm2', ['flash','pro','v32'], 'pro')} &nbsp; phases: {checks('ptp2', PHASES, {'p1_gather_stats','p2_count_admission','p3_collect','p4_select'}, PLAB)}</div>
<div id="ptfig2" class="ptfig"></div>
<h3><span class="lang-en">Fig. P3 — phase share vs previous-step hint hit-rate (each point = one cell)</span><span class="lang-zh">图 P3——phase 份额对上一步 hint 命中率(每点 = 一个 cell)</span></h3>
<div class="ptc">phase: {radios('ptp3', PHASES, 'p2_count_admission', PLAB)} &nbsp; unit: {radios('ptu3', ['frac','us'], 'frac')} &nbsp; model: {checks('ptm3', ['flash','pro','v32'], {'flash','pro','v32'})}</div>
<div id="ptfig3" class="ptfig"></div>
<h3><span class="lang-en">Fig. P4 — phase composition by performance class</span><span class="lang-zh">图 P4——按性能表现分类的 phase 构成</span></h3>
<div class="ptc">class axis: {radios('ptc4', ['pclass','htier','squart','rung'], 'pclass')} &nbsp; phase: {radios('ptp4', PHASES, 'p4_select', PLAB)} &nbsp; unit: {radios('ptu4', ['frac','us'], 'frac')}</div>
<div id="ptfig4" class="ptfig"></div>
<script>
const PTD={data_js};
const PTPH={json.dumps(PHASES)}, PTLAB={json.dumps(PLAB)}, PTCOL={json.dumps(PCOL)};
const PTISL={json.dumps(ISL_ORDER)};
const PTCLS={{pclass:['strong-win','win','parity','loss'],htier:['hi','mid','lo'],
  squart:['fastest25','mid50','slowest25'],rung:['cs1-small','cs1-mid','cs4','cs8-T512']}};
const PTCI={{u:0,m:1,i:2,l:3,N:4,h:5,us:6,f:7,uu:8,pv:9,rg:10,pc:11,ht:12,sq:13}};
function ptv(n){{const e=document.querySelector(`input[name=${{n}}]:checked`);return e?e.value:null}}
function ptcks(c){{return [...document.querySelectorAll(`input.${{c}}:checked`)].map(e=>e.value)}}
const PTLY={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#c9d1d9',size:11}},margin:{{t:10,r:10,b:40,l:50}},
  legend:{{orientation:'h',y:1.12}}}};
function ptDraw1(){{
  const m=ptv('ptm1'),i=ptv('pti1'),u=ptv('ptu1');
  const cs=PTD.filter(d=>d[1]===m&&d[2]===i).sort((a,b)=>a[3]-b[3]);
  const x=cs.map(d=>'L'+d[3]);
  const tr=PTPH.map((p,k)=>({{type:'bar',name:PTLAB[k],x:x,
    y:cs.map(d=>u==='us'?d[8][k]:d[7][k]),marker:{{color:PTCOL[k]}},
    customdata:cs.map(d=>[d[0],(d[5]*100).toFixed(0)]),
    hovertemplate:'%{{customdata[0]}} hit=%{{customdata[1]}}% '+PTLAB[k]+': %{{y:.2f}}<extra></extra>'}}));
  Plotly.react('ptfig1',tr,{{...PTLY,barmode:'stack',
    yaxis:{{title:u==='us'?'µs (nsys cold-L2)':'fraction',gridcolor:'#30363d'}},
    xaxis:{{title:'layer',tickangle:-45}}}},{{displayModeBar:false}});
}}
function ptDraw2(){{
  const m=ptv('ptm2'),sel=ptcks('ptp2');
  const tr=[];
  PTPH.forEach((p,k)=>{{if(!sel.includes(p))return;
    const cs=PTD.filter(d=>d[1]===m);
    tr.push({{type:'box',name:PTLAB[k],x:cs.map(d=>d[2]),y:cs.map(d=>d[7][k]),
      marker:{{color:PTCOL[k]}},boxpoints:false}});}});
  Plotly.react('ptfig2',tr,{{...PTLY,boxmode:'group',
    yaxis:{{title:'phase fraction',gridcolor:'#30363d'}},
    xaxis:{{categoryorder:'array',categoryarray:PTISL}}}},{{displayModeBar:false}});
}}
function ptDraw3(){{
  const p=ptv('ptp3'),u=ptv('ptu3'),ms=ptcks('ptm3'),k=PTPH.indexOf(p);
  const mc={{flash:'#4c78a8',pro:'#e45756',v32:'#f2cf5b'}};
  const tr=ms.map(m=>{{const cs=PTD.filter(d=>d[1]===m);
    return {{type:'scatter',mode:'markers',name:m,x:cs.map(d=>d[5]),
      y:cs.map(d=>u==='us'?d[8][k]:d[7][k]),
      marker:{{color:mc[m],size:5,opacity:0.55}},
      customdata:cs.map(d=>d[0]),
      hovertemplate:'%{{customdata}} hit=%{{x:.2f}} %{{y:.3f}}<extra></extra>'}};}});
  Plotly.react('ptfig3',tr,{{...PTLY,
    xaxis:{{title:'hint hit-rate',gridcolor:'#30363d'}},
    yaxis:{{title:PTLAB[k]+(u==='us'?' µs':' fraction'),gridcolor:'#30363d'}}}},{{displayModeBar:false}});
}}
function ptDraw4(){{
  const ax=ptv('ptc4'),p=ptv('ptp4'),u=ptv('ptu4'),k=PTPH.indexOf(p);
  const ci={{pclass:PTCI.pc,htier:PTCI.ht,squart:PTCI.sq,rung:PTCI.rg}}[ax];
  const tr=PTCLS[ax].map((cls,j)=>{{
    const cs=PTD.filter(d=>d[ci]===cls);
    return {{type:'box',name:cls+' (n='+cs.length+')',
      y:cs.map(d=>u==='us'?d[8][k]:d[7][k]),
      boxpoints:'all',jitter:0.5,pointpos:0,marker:{{size:3,opacity:0.4}},
      line:{{color:PTCOL[(j*2+1)%7]}},
      text:cs.map(d=>d[0]),hovertemplate:'%{{text}}: %{{y:.3f}}<extra></extra>'}};}});
  Plotly.react('ptfig4',tr,{{...PTLY,showlegend:false,
    yaxis:{{title:PTLAB[k]+(u==='us'?' µs':' fraction'),gridcolor:'#30363d'}}}},{{displayModeBar:false}});
}}
function ptDrawAll(){{try{{ptDraw1()}}catch(e){{}} try{{ptDraw2()}}catch(e){{}}
  try{{ptDraw3()}}catch(e){{}} try{{ptDraw4()}}catch(e){{}}}}
document.querySelectorAll("input[name^=pt],input.ptp2,input.ptm3").forEach(
  e=>e.addEventListener('change',()=>setTimeout(ptDrawAll,0)));
if(window.Plotly) ptDrawAll(); else window.addEventListener('load',ptDrawAll);
</script>"""

    body = f"""
<h2 id="sec-phasetime">9e · BS=1 real-data phase-time breakdown at the PR head (865 cells) / BS=1 真实数据分相耗时全格分解</h2>
{bi(
"<b>What.</b> In-kernel per-phase timing of the production GVR top-K kernel at the kf-campaign "
"PR#16457 head (<code>kf_campaign/gvrpkg_head</code>, kernel md5 <code>94d0cc5c</code>) on the FULL "
"865-cell real decode grid of §7b — every (model × ISL × layer) case, BS=1 fp32, no averaging across "
"cells anywhere in this section. Method: a spliced twin of the head kernel "
"(<code>gvrpkgtimed_head</code>, <code>[ptime]</code> markers) stamps <code>clock64()</code> at 8 phase "
"boundaries (leader CTA, thread 0); fractions come from per-phase MEDIAN cycles over 20 cold-L2 launches; "
"absolute µs = fraction × the pristine prod arm's nsys kernel time measured back-to-back on the same GPU "
"(10 warmup + 20 cold-L2 launches per arm, 512 MB evict outside the NVTX range, 8-way GPU sharding on "
"umbriel-b200-037). Validation per cell: tie-robust exactness of BOTH arms vs torch.topk, timestamp "
"monotonicity on every launch, and an instrumentation-overhead gate (nsys timed/prod).",
"<b>做了什么。</b>在 kf 战役 PR#16457 head(<code>kf_campaign/gvrpkg_head</code>,kernel md5 "
"<code>94d0cc5c</code>)上,对 §7b 的完整 865 格真实 decode 网格做 kernel 内分相计时——逐 "
"(model × ISL × layer) case,BS=1 fp32,本节所有结果均不做跨 cell 平均。方法:head kernel 的孪生插桩副本 "
"(<code>gvrpkgtimed_head</code>,<code>[ptime]</code> 标记)由 leader CTA 线程 0 在 8 个相边界打 "
"<code>clock64()</code> 戳;份额 = 20 次冷 L2 发射的逐相中位周期;绝对 µs = 份额 × 同卡背靠背实测的"
"纯净 prod 臂 nsys kernel 时间(每臂 10 预热 + 20 冷发射,512MB evict 在 NVTX 区间外,8 卡分片于 "
"umbriel-b200-037)。逐格验证:两臂对 torch.topk 的 tie-robust 精确性、每次发射的时间戳单调性、"
"插桩开销门(nsys timed/prod)。")}
{kpis}
{bi(
"<b>Phase map.</b> P1 gather/stats = preIdx gather + min/max/mean seed; P1b = h-space rung-ladder build; "
"P2 = R0 M-ary count + admission (+ fb_fix refine / secant fallback); P3 = candidate collect "
"(leader's own slice; at cs&gt;1 the P4 bucket also absorbs the leader's wait for the slowest peer's "
"collect via cluster handoff); P4 = final select incl. DSMEM gather, rank-scatter and the "
"p4-exact-tail/p4tt path; epilogue = final cluster barrier.",
"<b>相位图。</b>P1 gather/stats = preIdx 收集 + min/max/mean 种子;P1b = h 空间 rung 阶梯构建;"
"P2 = R0 M 叉计数 + admission(+ fb_fix 精化/secant 回退);P3 = 候选收集(leader 自身分片;cs&gt;1 时 "
"P4 桶还吸收 leader 等最慢 peer collect 的 cluster handoff);P4 = 最终选择,含 DSMEM 收集、rank-scatter "
"与 p4-exact-tail/p4tt 路径;epilogue = 收尾 cluster barrier。")}
{figs}
<div class="lang-en">{FINDINGS_EN}</div>
<div class="lang-zh">{FINDINGS_ZH}</div>
<details><summary class="mut"><span class="lang-en">per (model, ISL) phase-share table — median [p25, p75] across layers</span><span class="lang-zh">逐 (model, ISL) 相份额表——跨 layer 中位 [p25, p75]</span></summary>
{mi_table()}
</details>
<details><summary class="mut"><span class="lang-en">performance-class tables (PR-vs-base class / hint tier / within-group speed quartile)</span><span class="lang-zh">性能分类表(PR 对 base 分类 / hint 档 / 组内速度四分位)</span></summary>
<h4><span class="lang-en">by PR-vs-base class (§7b frozen ratios, used for classification only)</span><span class="lang-zh">按 PR 对 base 表现分类(§7b 冻结比值,仅用于分类)</span></h4>
{cls_table('pclass', ['strong-win','win','parity','loss'], 'PR-vs-base class', 'PR对base类')}
<h4><span class="lang-en">by hint tier (hit ≥0.60 hi / 0.35–0.60 mid / &lt;0.35 lo)</span><span class="lang-zh">按 hint 档(hit ≥0.60 高 / 0.35–0.60 中 / &lt;0.35 低)</span></h4>
{cls_table('htier', ['hi','mid','lo'], 'hint tier', 'hint 档')}
<h4><span class="lang-en">by within-(model,ISL) PR-time quartile</span><span class="lang-zh">按组内 PR 耗时四分位</span></h4>
{cls_table('squart', ['fastest25','mid50','slowest25'], 'speed quartile', '速度四分位')}
</details>
<div class="lang-en"><p class="mut"><b>Provenance</b>: 2026-07-21, umbriel-b200-037 GPUs 0-7, nsys cold-L2,
<code>p4f1_harness/phase_breakdown_ptime/</code> (measure_phases_full.py / aggregate_phases.py /
analyze_phases.py; raw: phase_full_full.csv, phase_analysis.json, shard logs).</p></div>
<div class="lang-zh"><p class="mut"><b>数据来源</b>:2026-07-21,umbriel-b200-037 GPU0-7,nsys 冷 L2,
<code>p4f1_harness/phase_breakdown_ptime/</code>(measure_phases_full.py / aggregate_phases.py /
analyze_phases.py;原始数据:phase_full_full.csv、phase_analysis.json、分片日志)。</p></div>
"""
    return BEGIN + body + "\n" + END


def main():
    A = json.load(open(os.path.join(PBP, "phase_analysis.json")))
    cells = load_cells()
    block = build(A, cells)
    assert "%%FINDINGS" not in block, "fill FINDINGS_EN/ZH before injecting"
    html = open(REPORT, encoding="utf-8").read()
    if BEGIN in html:
        html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                      lambda _: block, html, flags=re.S)
        action = "replaced"
    else:
        if ANCHOR_AFTER not in html:
            raise SystemExit(f"anchor {ANCHOR_AFTER!r} not found")
        html = html.replace(ANCHOR_AFTER, ANCHOR_AFTER + "\n" + block + "\n", 1)
        action = "inserted"
    open(REPORT, "w", encoding="utf-8").write(html)
    print(f"§9e PHASETIME block {action} ({len(block)} chars, "
          f"{len(cells)} cells embedded)")


if __name__ == "__main__":
    main()
