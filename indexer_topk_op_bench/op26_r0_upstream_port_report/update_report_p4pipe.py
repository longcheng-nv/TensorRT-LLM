#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Idempotent injector: §9f "P4 pipeline decomposition + instruction accounting"
into REPORT.html (marker P4PIPE:BEGIN/END, inserted after PHASETIME:END).

Source of truth:
  p4f1_harness/p4_pipeline/p4pipe_full.csv       (865 cells, Exp A)
  p4f1_harness/p4_pipeline/p4pipe_analysis.json  (analyze_p4pipe.py)
  p4f1_harness/p4_pipeline/ncu_p4_summary.json   (Exp B, 13 cells)

Charts are Plotly (already loaded by REPORT.html); bilingual via the
report's lang-en/lang-zh convention.
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "REPORT.html")
PP = os.path.join(HERE, "p4f1_harness", "p4_pipeline")
BEGIN = "<!-- P4PIPE:BEGIN (update_report_p4pipe.py) -->"
END = "<!-- P4PIPE:END -->"
ANCHOR_AFTER = "<!-- PHASETIME:END -->"

SUB = ["p4_peer_wait", "p4_dsmem_gather", "p4_minmax", "p4_coarse_hist",
       "p4_coarse_search", "p4_fine", "p4_scatter", "p4_tail"]
SLAB = ["peer wait", "DSMEM gather", "min/max", "coarse hist",
        "coarse search", "fine recursion", "scatter", "tail repair"]
SCOL = ["#8f6bb8", "#b279a2", "#4c78a8", "#9ecae9", "#f2cf5b", "#e45756",
        "#f58518", "#54a24b"]
ISL_ORDER = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]


def load_cells():
    rows = []
    for r in csv.DictReader(open(os.path.join(PP, "p4pipe_full.csv"))):
        cs = int(r["cs"])
        N = int(r["N"])
        d = dict(u=r["uuid"], m=r["model"], i=r["isl"], l=int(r["layer"]),
                 N=N, h=float(r["hit"]), cs=cs, K=int(r["K"]),
                 us=round(float(r["us_prod_nsys"]), 3),
                 p4=round(float(r["us_p4_select"]), 3))
        d["su"] = [round(float(r[f"us_{s}"]), 4) for s in SUB]
        d["sp"] = [round(float(r[f"p4share_{s}"] or 0), 5) for s in SUB]
        d["sk"] = [round(float(r[f"frac_{s}"]), 5) for s in SUB]
        d["rg"] = ("cs1-small" if cs == 1 and N <= 8448 else
                   "cs1-mid" if cs == 1 else "cs4" if cs == 4 else "cs8")
        rows.append(d)
    return rows


def load_ncu():
    S = json.load(open(os.path.join(PP, "ncu_p4_summary.json")))
    out = {}
    for cell, rec in S.items():
        stg = {}
        for st in SUB:
            v = rec["stages"].get(st)
            if not v:
                continue
            stg[st] = dict(
                inst=round(v["inst"]),
                cycP4=round((v["expA_share_of_p4"] or 0), 4),
                ko=" ".join(f"{k}:{n:.0f}" for k, n in v["key_ops"].items()),
                top=" ".join(f"{k}×{n:.0f}" for k, n in
                             list(v["top5"].items())[:3]),
            )
        out[cell] = dict(cs=rec["cs"], N=rec["N"], K=rec["K"], hit=rec["hit"],
                         stalls=dict(list(rec["stalls_kernelwide"].items())[:4]),
                         stages=stg)
    return out


def bi(en, zh, tag="p"):
    return (f"<div class='lang-en'><{tag}>{en}</{tag}></div>"
            f"<div class='lang-zh'><{tag}>{zh}</{tag}></div>")


FINDINGS_EN = """
<p><b>Findings (per-cell; medians are stated as medians, never silent averages).</b></p>
<ol>
<li><b>P4's four-scan select pipeline is now priced per stage.</b> The rank-scatter P4 runs
min/max &rarr; coarse-hist(+search) &rarr; fine 256-bin recursion &rarr; classify/scatter over the
candidate buffer. At cs=1 the <b>fine recursion is the single largest stage in 491/865 cells</b>
(med 1.53&ndash;1.75&thinsp;µs, 31&ndash;34% of P4), followed by scatter (18&ndash;25%) and coarse
search (15&ndash;19%). Together fine+scatter hold a median 23% of the whole kernel at cs=1 — the
biggest single-CTA lever after the P2 walk.</li>
<li><b>The cluster tax, quantified: at cs=8 the leader spends 47% of P4 before selecting anything.</b>
peer-wait (handoff #2 arrive+wait for the slowest peer's collect) med 1.41&thinsp;µs + DSMEM gather
med 3.11&thinsp;µs (max 4.2) = 4.68&thinsp;µs = 47% of P4 = 24% of the whole kernel; at cs=4 it is
3.00&thinsp;µs / 35% / 17%. The gather is the dominant sub-stage in 267/865 cells — all clusters.
Spearman(N, peer-wait µs) = +0.82/+0.78 within cs4/cs8: the wait tracks the slowest peer's scan
length, as expected.</li>
<li><b>The core select cost is rung-invariant — cluster P4 ballooning is pure wait+gather.</b>
Summing min/max&rarr;scatter gives med 4.27&thinsp;µs (cs1), 4.96 (cs4), 4.76 (cs8): the leader's
select work is pinned by the candidate-buffer size (&le;kC), not by N. Everything §9e saw P4 gain
when crossing the dispatch boundary is the serial DSMEM machinery — this is the precise distP4
target (parallelize/overlap gather, or pre-place peer candidates), worth up to ~4.7&thinsp;µs/cell
at cs=8.</li>
<li><b>Fine recursion is latency-bound, not instruction-bound.</b> NCU: its instruction share of
the P4 region is well below its cycle share (e.g. flash_4k: 1842 inst vs 37% of P4 cycles), it
carries 3 CTA-wide barriers (BAR&times;96 per launch at T512) and its smem traffic is modest —
warp-stall sampling puts it and coarse-search at the top of the cs=1 stall locations. Shaving a
barrier or fusing the fine build+search would cut dead time, not work.</li>
<li><b>The tail-repair blow-up class is a K=512 gate hole.</b> p4tt (tiny-tie fast path) is gated
<code>top_k &ge; 1024</code>, so K=512 cells that trigger exact-tail run the 4-level radix rewrite:
flash_128k_L42 (hit 0.86) pays <b>5823 tail instructions (26&times; the non-firing 226)</b>,
41.6% of its P4, with BSSY/BAR-heavy stalls — the §9e "+10% stamp-tax outlier" is really an
exact-tail cell. Across the grid the tail is benign at the median (5&ndash;6% of P4) but 34 v32
(K=2048) cells and 2 K=512 cells exceed 15%; extending p4tt below K=1024 (or seeding its collect
with the fine-bin count) caps the K=512 escape.</li>
<li><b>coarse-hist prices the kNumBins=K zeroing.</b> Its cost steps with K: med 0.37 / 0.50 /
0.60&thinsp;µs for K=512/1024/2048 — the re-zero + scatter-add over K bins; a shared-zero or
smaller first-level histogram is worth ~0.2&thinsp;µs on v32 only.</li>
<li><b>Degenerate copy-out exists at the same 3 tiny-N cells as §9e</b> (flash_4k_L06,
pro_4k_L02/L16): cand==K collapses the pipeline; P4 falls to a 0.2&ndash;0.5&thinsp;µs copy.</li>
<li><b>Stall-reason profile agrees: clusters wait, single CTAs starve on icache.</b> Kernel-wide
NCU stall ranking flips from <i>no_instruction</i> 17.6 cyc/issue (flash_4k, the known short-
single-CTA icache effect) to <i>barrier</i> 23&ndash;24 cyc/issue at cs&ge;4 (flash_512k,
v32_128k) — matching the sub-P4 timing attribution independently.</li>
<li><b>Validation.</b> 865/865 exact both arms, 865/865 monotone 16-stamp chains, overhead med
+1.3% / p95 +4.6% (8 cells over the ±7% gate: 7 are flash 256k/1024k borderline +7.1&ndash;8.2%,
plus flash_128k_L42 at +16.6% — the same reproducible stamp-tax cell as §9e; fractions reproduce,
absolutes anchored to the pristine arm). Top-level P4 share drifts vs §9e by med +0.004
(max |0.061|) — the added stamps did not distort the composition. NCU cross-check: per-stage
instruction shares track the clock-stamp cycle shares in rank order for all 13 profiled cells.</li>
</ol>"""

FINDINGS_ZH = """
<p><b>结论(逐 cell;凡中位数均明说,不做静默平均)。</b></p>
<ol>
<li><b>P4 四遍扫描选择流水线已逐段定价。</b>rank-scatter P4 依次跑 min/max &rarr; 粗直方图(+搜索)
&rarr; 细 256-bin 递归 &rarr; 分类/scatter 写回。cs=1 下<b>细递归是 491/865 格的最大子相</b>
(中位 1.53&ndash;1.75&thinsp;µs,占 P4 的 31&ndash;34%),其后是 scatter(18&ndash;25%)与粗搜索
(15&ndash;19%)。fine+scatter 合计占整个 kernel 中位 23%(cs=1)——是 P2 行走之后最大的单 CTA 杠杆。</li>
<li><b>cluster 税被量化:cs=8 时 leader 在真正开始选择前已花掉 P4 的 47%。</b>peer-wait
(handoff #2 等最慢 peer collect)中位 1.41&thinsp;µs + DSMEM gather 中位 3.11&thinsp;µs(最大 4.2)
= 4.68&thinsp;µs = P4 的 47% = 整个 kernel 的 24%;cs=4 为 3.00&thinsp;µs / 35% / 17%。gather 是
267/865 格的最大子相——全部是 cluster 格。cs4/cs8 组内 Spearman(N, peer-wait µs) = +0.82/+0.78:
等待随最慢 peer 的扫描长度增长,符合预期。</li>
<li><b>核心选择成本跨 rung 不变——cluster 端 P4 膨胀纯粹是 wait+gather。</b>把 min/max&rarr;scatter
求和:中位 4.27&thinsp;µs(cs1)、4.96(cs4)、4.76(cs8)——leader 的选择工作量由候选缓冲上限
(&le;kC)钉死,与 N 无关。§9e 观察到的跨分派边界后 P4 重新膨胀,全部来自串行 DSMEM 机制——这就是
distP4 的精确靶点(并行化/重叠 gather,或 peer 候选预放置),cs=8 下每格最多可回收 ~4.7&thinsp;µs。</li>
<li><b>细递归是 latency-bound,不是指令量 bound。</b>NCU:其指令份额显著低于其周期份额
(如 flash_4k:1842 条指令对 37% 的 P4 周期),自带 3 个 CTA 级 barrier(T512 下每发射 BAR&times;96),
smem 流量不大——warp-stall 采样把它和粗搜索列为 cs=1 的最热 stall 位置。省一个 barrier 或融合
fine 的 build+search,砍的是死等,不是工作量。</li>
<li><b>tail 修复爆炸类是 K=512 的门洞。</b>p4tt(小 tie 快路径)门是 <code>top_k &ge; 1024</code>,
K=512 一旦触发 exact-tail 只能走 4 级 radix 重写:flash_128k_L42(hit 0.86)付出
<b>5823 条 tail 指令(未触发格 226 条的 26 倍)</b>、占其 P4 的 41.6%,stall 偏向 BSSY/BAR——§9e 的
"+10% 插桩税异常格"实为 exact-tail 格。全网格看 tail 中位无害(P4 的 5&ndash;6%),但 34 个 v32
(K=2048)格与 2 个 K=512 格超过 15%;把 p4tt 下探到 K&lt;1024(或用细 bin 计数种子其 collect)
可封死 K=512 逃逸。</li>
<li><b>粗直方图定价了 kNumBins=K 的清零。</b>成本随 K 阶梯:K=512/1024/2048 中位 0.37 / 0.50 /
0.60&thinsp;µs——K 个 bin 的重清零 + scatter-add;共享清零或更小的一级直方图只在 v32 上值 ~0.2&thinsp;µs。</li>
<li><b>退化 copy-out 与 §9e 完全同 3 个极小 N 格</b>(flash_4k_L06、pro_4k_L02/L16):cand==K 使
流水线塌缩;P4 降为 0.2&ndash;0.5&thinsp;µs 的拷贝。</li>
<li><b>stall 原因画像独立吻合:cluster 在等,单 CTA 在饿指令。</b>NCU kernel 级 stall 排名从
<i>no_instruction</i> 17.6 cyc/issue(flash_4k,已知的短单 CTA icache 效应)翻转为 cs&ge;4 的
<i>barrier</i> 23&ndash;24 cyc/issue(flash_512k、v32_128k)——与子相计时归因相互印证。</li>
<li><b>验证。</b>两臂 865/865 精确,865/865 十六戳链单调,插桩开销中位 +1.3%、p95 +4.6%
(8 格超 ±7% 门:7 格为 flash 256k/1024k 的 +7.1&ndash;8.2% 边缘,另有 flash_128k_L42 +16.6%——
与 §9e 同一个可复现插桩税格;份额复现,绝对值一如既往锚定纯净臂)。顶层 P4 份额对 §9e 漂移中位
+0.004(最大 |0.061|)——新增戳未扭曲构成。NCU 交叉验证:13 个剖析格的逐段指令份额与时钟戳周期
份额在秩序上一致。</li>
</ol>"""


def build(A, cells, ncu):
    dom = A["dominant_substage_counts"]
    ct8 = A["cluster_tax_cs8"]
    kpis = f"""<div class="kpis">
<div class="kpi"><div class="v" style="color:#6ede8a">{A['n']}/865</div><div class="l lang-en">cells; exact {A['exact']}, monotone {A['mono']}, P4-share drift vs §9e med {A['drift_med']:+.3f}</div><div class="l lang-zh">测量格数;精确 {A['exact']},单调 {A['mono']},P4 份额对 §9e 漂移中位 {A['drift_med']:+.3f}</div></div>
<div class="kpi"><div class="v">{dom.get('p4_fine',0)}/865</div><div class="l lang-en">cells where the fine 256-bin recursion is the largest P4 stage (DSMEM gather: {dom.get('p4_dsmem_gather',0)}, scatter: {dom.get('p4_scatter',0)})</div><div class="l lang-zh">细递归为最大 P4 子相的格数(DSMEM gather:{dom.get('p4_dsmem_gather',0)},scatter:{dom.get('p4_scatter',0)})</div></div>
<div class="kpi"><div class="v">{ct8['us_med']:.1f}µs</div><div class="l lang-en">cs=8 cluster tax (peer-wait + DSMEM gather) = {100*ct8['shP4_med']:.0f}% of P4 = {100*ct8['shK_med']:.0f}% of kernel</div><div class="l lang-zh">cs=8 cluster 税(peer-wait + DSMEM gather)= P4 的 {100*ct8['shP4_med']:.0f}% = kernel 的 {100*ct8['shK_med']:.0f}%</div></div>
<div class="kpi"><div class="v">{A['ovh_med']*100:+.1f}%</div><div class="l lang-en">instrumentation overhead median (16-stamp twin, nsys timed/prod)</div><div class="l lang-zh">插桩开销中位(16 戳孪生,nsys timed/prod)</div></div>
</div>"""

    def mi_table():
        hdr = ("<tr><th>model/ISL</th><th>n</th><th>cs</th>"
               "<th><span class='lang-en'>P4 µs med</span>"
               "<span class='lang-zh'>P4 µs 中位</span></th>" +
               "".join(f"<th>{s}</th>" for s in SLAB) + "</tr>")
        trs = []
        for key in sorted(A["per_model_isl"],
                          key=lambda k: (k.split("/")[0],
                                         ISL_ORDER.index(k.split("/")[1]))):
            e = A["per_model_isl"][key]
            tds = [key, str(e["n"]), str(e["cs"]), f"{e['p4us']:.1f}"]
            for s in SUB:
                tds.append(f"{e[s]:.2f}")
            trs.append("<tr>" + "".join(f"<td>{t}</td>" for t in tds) + "</tr>")
        return ("<table>" + hdr + "".join(trs) + "</table>"
                + bi("Cell values = median share of P4 across layers.",
                     "表内数值 = 跨 layer 的 P4 内份额中位。", "p"))

    def ncu_table():
        hdr = ("<tr><th>cell</th><th>cs</th><th>stage</th>"
               "<th><span class='lang-en'>inst</span><span class='lang-zh'>指令数</span></th>"
               "<th><span class='lang-en'>cyc %P4</span><span class='lang-zh'>周期%P4</span></th>"
               "<th><span class='lang-en'>key ops</span><span class='lang-zh'>关键指令</span></th>"
               "<th>top mnemonics</th></tr>")
        trs = []
        for cell, rec in ncu.items():
            first = True
            for st, v in rec["stages"].items():
                if v["inst"] < 40 and v["cycP4"] < 0.01:
                    continue
                c1 = (f"<td rowspan_placeholder>{cell}</td>"
                      f"<td>{rec['cs']}</td>") if first else "<td></td><td></td>"
                first = False
                trs.append(
                    f"<tr>{c1}<td>{st.replace('p4_','')}</td>"
                    f"<td>{v['inst']}</td><td>{100*v['cycP4']:.1f}</td>"
                    f"<td>{v['ko']}</td><td class='mut'>{v['top']}</td></tr>")
        return ("<table>" + hdr + "".join(trs) + "</table>").replace(
            " rowspan_placeholder", "")

    def stall_tbl():
        hdr = ("<tr><th>cell</th><th>cs</th>"
               "<th><span class='lang-en'>top warp-stall reasons (cycles per issue-active)</span>"
               "<span class='lang-zh'>最重 warp-stall 原因(cyc/issue-active)</span></th></tr>")
        trs = []
        for cell, rec in ncu.items():
            st = ", ".join(f"{k} {v:.1f}" for k, v in rec["stalls"].items())
            trs.append(f"<tr><td>{cell}</td><td>{rec['cs']}</td><td>{st}</td></tr>")
        return "<table>" + hdr + "".join(trs) + "</table>"

    data_js = json.dumps([[d["u"], d["m"], d["i"], d["l"], d["N"], d["h"],
                           d["cs"], d["us"], d["p4"], d["su"], d["sp"],
                           d["sk"], d["rg"]] for d in cells],
                         separators=(",", ":"))
    ncu_js = json.dumps(
        {c: {"cs": r["cs"],
             "st": {s.replace("p4_", ""): [r["stages"][s]["inst"],
                                           r["stages"][s]["cycP4"],
                                           r["stages"][s]["ko"]]
                    for s in r["stages"]}}
         for c, r in ncu.items()}, separators=(",", ":"))

    def radios(name, opts, checked, lab=None):
        return " ".join(
            f"<label><input type='radio' name='{name}' value='{o}'"
            f"{' checked' if o == checked else ''}> {lab[k] if lab else o}</label>"
            for k, o in enumerate(opts))

    figs = f"""
<h3><span class="lang-en">Fig. Q1 — per-layer stacked P4 sub-stage time (every layer shown)</span><span class="lang-zh">图 Q1——逐 layer 的 P4 子相堆叠耗时(每层单独显示)</span></h3>
<div class="ptc">model: {radios('pqm1', ['flash','pro','v32'], 'pro')} &nbsp; ISL: {radios('pqi1', ISL_ORDER, '512k')} &nbsp; unit: {radios('pqu1', ['us','shareP4','shareKernel'], 'us')}</div>
<div id="pqfig1" class="ptfig"></div>
<h3><span class="lang-en">Fig. Q2 — P4 sub-stage share distributions by ISL (boxes = layer spread)</span><span class="lang-zh">图 Q2——各 ISL 下 P4 子相份额分布(箱线 = 层间散布)</span></h3>
<div class="ptc">model: {radios('pqm2', ['flash','pro','v32'], 'pro')} &nbsp; unit: {radios('pqu2', ['shareP4','us'], 'shareP4')}</div>
<div id="pqfig2" class="ptfig"></div>
<h3><span class="lang-en">Fig. Q3 — rung medians: the P4 pipeline reshapes across the dispatch ladder</span><span class="lang-zh">图 Q3——rung 中位:P4 流水线随分派阶梯重塑</span></h3>
<div class="ptc">unit: {radios('pqu3', ['us','shareP4','shareKernel'], 'us')}</div>
<div id="pqfig3" class="ptfig"></div>
<h3><span class="lang-en">Fig. Q4 — NCU: instructions vs cycles per P4 stage (13 profiled cells)</span><span class="lang-zh">图 Q4——NCU:各 P4 子相的指令份额对周期份额(13 个剖析格)</span></h3>
<div class="ptc">cell: {radios('pqc4', list(ncu.keys()), 'flash_512k_L02')}</div>
<div id="pqfig4" class="ptfig"></div>
<script>
const PQD={data_js};
const PQN={ncu_js};
const PQS={json.dumps([s.replace('p4_','') for s in SUB])},PQL={json.dumps(SLAB)},PQC={json.dumps(SCOL)};
const PQISL={json.dumps(ISL_ORDER)};
const PQI={{u:0,m:1,i:2,l:3,N:4,h:5,cs:6,us:7,p4:8,su:9,sp:10,sk:11,rg:12}};
function pqv(n){{const e=document.querySelector(`input[name=${{n}}]:checked`);return e?e.value:null}}
const PQLY={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#c9d1d9',size:11}},margin:{{t:10,r:10,b:40,l:50}},
  legend:{{orientation:'h',y:1.14}}}};
function pqArr(d,u){{return u==='us'?d[PQI.su]:(u==='shareP4'?d[PQI.sp]:d[PQI.sk])}}
function pqDraw1(){{
  const m=pqv('pqm1'),i=pqv('pqi1'),u=pqv('pqu1');
  const cs=PQD.filter(d=>d[1]===m&&d[2]===i).sort((a,b)=>a[3]-b[3]);
  const x=cs.map(d=>'L'+d[3]);
  const tr=PQS.map((s,k)=>({{type:'bar',name:PQL[k],x:x,
    y:cs.map(d=>pqArr(d,u)[k]),marker:{{color:PQC[k]}},
    customdata:cs.map(d=>[d[0],(d[5]*100).toFixed(0)]),
    hovertemplate:'%{{customdata[0]}} hit=%{{customdata[1]}}% '+PQL[k]+': %{{y:.2f}}<extra></extra>'}}));
  Plotly.react('pqfig1',tr,{{...PQLY,barmode:'stack',
    yaxis:{{title:u==='us'?'µs (nsys cold-L2)':u,gridcolor:'#30363d'}},
    xaxis:{{title:'layer',tickangle:-45}}}},{{displayModeBar:false}});
}}
function pqDraw2(){{
  const m=pqv('pqm2'),u=pqv('pqu2');
  const tr=[];
  PQS.forEach((s,k)=>{{
    const cs=PQD.filter(d=>d[1]===m);
    tr.push({{type:'box',name:PQL[k],x:cs.map(d=>d[2]),
      y:cs.map(d=>u==='us'?d[PQI.su][k]:d[PQI.sp][k]),
      marker:{{color:PQC[k]}},boxpoints:false}});}});
  Plotly.react('pqfig2',tr,{{...PQLY,boxmode:'group',
    yaxis:{{title:u==='us'?'µs':'share of P4',gridcolor:'#30363d'}},
    xaxis:{{categoryorder:'array',categoryarray:PQISL}}}},{{displayModeBar:false}});
}}
function pqMed(v){{const s=[...v].sort((a,b)=>a-b);return s.length?s[Math.floor(s.length/2)]:0}}
function pqDraw3(){{
  const u=pqv('pqu3');
  const rungs=['cs1-small','cs1-mid','cs4','cs8'];
  const tr=PQS.map((s,k)=>({{type:'bar',name:PQL[k],x:rungs,
    y:rungs.map(rg=>pqMed(PQD.filter(d=>d[PQI.rg]===rg).map(d=>pqArr(d,u)[k]))),
    marker:{{color:PQC[k]}},
    hovertemplate:'%{{x}} '+PQL[k]+': %{{y:.2f}}<extra></extra>'}}));
  Plotly.react('pqfig3',tr,{{...PQLY,barmode:'stack',
    yaxis:{{title:u==='us'?'median µs':'median '+u,gridcolor:'#30363d'}}}},{{displayModeBar:false}});
}}
function pqDraw4(){{
  const c=pqv('pqc4'),R=PQN[c];if(!R)return;
  const ss=PQS.filter(s=>R.st[s]);
  const ti=ss.reduce((a,s)=>a+R.st[s][0],0);
  const tr=[
    {{type:'bar',name:'inst share of P4 region',x:ss,y:ss.map(s=>R.st[s][0]/ti),
      marker:{{color:'#4c78a8'}},customdata:ss.map(s=>R.st[s][2]),
      hovertemplate:'%{{x}} inst=%{{y:.1%}}<br>%{{customdata}}<extra></extra>'}},
    {{type:'bar',name:'cycle share of P4 (clock stamps)',x:ss,y:ss.map(s=>R.st[s][1]),
      marker:{{color:'#e45756'}},hovertemplate:'%{{x}} cyc=%{{y:.1%}}<extra></extra>'}}];
  Plotly.react('pqfig4',tr,{{...PQLY,barmode:'group',
    yaxis:{{title:'share',tickformat:'.0%',gridcolor:'#30363d'}}}},{{displayModeBar:false}});
}}
function pqDrawAll(){{try{{pqDraw1()}}catch(e){{}} try{{pqDraw2()}}catch(e){{}}
  try{{pqDraw3()}}catch(e){{}} try{{pqDraw4()}}catch(e){{}}}}
document.querySelectorAll("input[name^=pq]").forEach(
  e=>e.addEventListener('change',()=>setTimeout(pqDrawAll,0)));
if(window.Plotly) pqDrawAll(); else window.addEventListener('load',pqDrawAll);
</script>"""

    body = f"""
<h2 id="sec-p4pipe">9f · P4 pipeline decomposition + instruction accounting (865 cells + 13-cell NCU) / P4 流水线拆解与指令归账</h2>
{bi(
"<b>What.</b> §9e established that P4 select holds ~44% of the kernel in 827/865 real-data cells. "
"This section opens the P4 box: the same 865-cell grid (BS=1 fp32, kf PR#16457 head) re-measured "
"with a 16-slot twin (<code>gvrpkgp4t_head</code>, <code>[ptime]+[p4sub]</code>) that stamps "
"<code>clock64()</code> at the 8 internal barrier boundaries of <code>phase4_rank_scatter</code> — "
"peer-wait, DSMEM gather, min/max, coarse hist, coarse search, fine recursion, scatter, tail repair "
"— plus an NCU pass on 13 representative cells where the executed clock stamps serve as SASS "
"landmarks: every instruction between consecutive CS2R clock reads is bucketed to its stage, giving "
"per-stage instruction counts, opcode mixes and stall attribution. Methodology (arms, cold-L2, nsys "
"anchor, gates) is identical to §9e, with one added gate: the top-level P4 share must reproduce §9e.",
"<b>做了什么。</b>§9e 确立了 P4 select 在 827/865 个真实数据格中占 kernel 约 44%。本节打开 P4 黑盒:"
"同一 865 格网格(BS=1 fp32,kf PR#16457 head)用 16 槽孪生 kernel "
"(<code>gvrpkgp4t_head</code>,<code>[ptime]+[p4sub]</code>)重测,在 "
"<code>phase4_rank_scatter</code> 的 8 个内部 barrier 边界打 <code>clock64()</code> 戳——"
"peer-wait、DSMEM gather、min/max、粗直方图、粗搜索、细递归、scatter、tail 修复——并在 13 个代表格上"
"跑 NCU:以实际执行的 clock 戳为 SASS 地标,把相邻 CS2R 之间的每条指令归入其子相,得到逐子相的"
"指令数、指令类别构成与 stall 归因。方法学(双臂、冷 L2、nsys 锚、验证门)与 §9e 完全一致,"
"另加一门:顶层 P4 份额必须复现 §9e。")}
{kpis}
{bi(
"<b>Stage map (all inside the §9e P4 bucket, in execution order).</b> peer wait = cluster handoff #2 "
"arrive+wait (leader waits for the slowest peer's collect; cs=1: zero-width) &rarr; DSMEM gather = "
"leader pulls peer candidates over <code>mapa</code>/generic loads &rarr; min/max block reduce over "
"the candidate buffer &rarr; coarse kNumBins(=K) histogram zero+build (smem atomics) &rarr; 3-step "
"high&rarr;low bin search &rarr; fine 256-bin re-zero+build+search on the straddling bin &rarr; "
"classify+scatter writeback &rarr; tail = output pad + p4-exact-tail / p4tt tie repair.",
"<b>子相图(全部位于 §9e 的 P4 桶内,按执行顺序)。</b>peer wait = cluster handoff #2 "
"arrive+wait(leader 等最慢 peer 的 collect;cs=1 零宽)&rarr; DSMEM gather = leader 通过 "
"<code>mapa</code>/generic load 拉取 peer 候选 &rarr; 候选缓冲的 min/max 块规约 &rarr; 粗 "
"kNumBins(=K) 直方图清零+构建(smem 原子加)&rarr; 3 步高&rarr;低 bin 搜索 &rarr; 骑跨 bin 上的"
"细 256-bin 重清零+构建+搜索 &rarr; 分类+scatter 写回 &rarr; tail = 输出填充 + p4-exact-tail / "
"p4tt tie 修复。")}
{figs}
<div class="lang-en">{FINDINGS_EN}</div>
<div class="lang-zh">{FINDINGS_ZH}</div>
{bi(
"<b>Lever map (ordered by ceiling on this grid).</b> ① distP4 — parallelize/overlap the leader-only "
"DSMEM gather and absorb the peer wait: up to ~4.7&thinsp;µs/cell at cs=8, ~3.0 at cs=4 (24%/17% of "
"kernel). ② fine-recursion barrier diet / build+search fusion: 1.5&ndash;1.8&thinsp;µs everywhere, "
"latency not work. ③ p4tt below K=1024: caps the 26&times; tail escapes (2 K=512 cells today, but "
"the class is input-dependent). ④ coarse-hist zero sharing at K=2048: ~0.2&thinsp;µs, v32 only.",
"<b>杠杆图(按本网格上限排序)。</b>① distP4——并行化/重叠 leader 独占的 DSMEM gather 并吸收 "
"peer 等待:cs=8 每格最多 ~4.7&thinsp;µs,cs=4 ~3.0(kernel 的 24%/17%)。② 细递归 barrier 减负 / "
"build+search 融合:各处 1.5&ndash;1.8&thinsp;µs,砍延迟不砍工作量。③ p4tt 下探 K&lt;1024:封顶 "
"26&times; 的 tail 逃逸(当前 2 个 K=512 格,但该类随输入漂移)。④ K=2048 粗直方图共享清零:"
"~0.2&thinsp;µs,仅 v32。")}
<details><summary class="mut"><span class="lang-en">per (model, ISL) P4 sub-stage share table — median across layers</span><span class="lang-zh">逐 (model, ISL) P4 子相份额表——跨 layer 中位</span></summary>
{mi_table()}
</details>
<details><summary class="mut"><span class="lang-en">NCU per-stage instruction accounting (13 cells; stages ≥40 inst or ≥1% cyc)</span><span class="lang-zh">NCU 逐子相指令归账(13 格;仅列 ≥40 条指令或 ≥1% 周期的子相)</span></summary>
{ncu_table()}
<h4><span class="lang-en">kernel-wide warp-stall reasons</span><span class="lang-zh">kernel 级 warp-stall 原因</span></h4>
{stall_tbl()}
</details>
<div class="lang-en"><p class="mut"><b>Provenance</b>: 2026-07-22, umbriel-b200-093 GPUs 0-7 (Exp A) / GPU 0 (NCU),
nsys cold-L2 + ncu --set full (clock-stamp SASS landmark segmentation),
<code>p4f1_harness/p4_pipeline/</code> (splice_p4sub.py / measure_p4pipe_full.py / aggregate_p4pipe.py /
analyze_p4pipe.py / parse_ncu_p4.py / summarize_ncu_p4.py; raw: p4pipe_full.csv, p4pipe_analysis.json,
ncu_p4_summary.json, shard logs).</p></div>
<div class="lang-zh"><p class="mut"><b>数据来源</b>:2026-07-22,umbriel-b200-093 GPU0-7(实验 A)/ GPU0(NCU),
nsys 冷 L2 + ncu --set full(clock 戳 SASS 地标分段),
<code>p4f1_harness/p4_pipeline/</code>(splice_p4sub.py / measure_p4pipe_full.py / aggregate_p4pipe.py /
analyze_p4pipe.py / parse_ncu_p4.py / summarize_ncu_p4.py;原始数据:p4pipe_full.csv、p4pipe_analysis.json、
ncu_p4_summary.json、分片日志)。</p></div>
"""
    return BEGIN + body + "\n" + END


def main():
    A = json.load(open(os.path.join(PP, "p4pipe_analysis.json")))
    cells = load_cells()
    ncu = load_ncu()
    block = build(A, cells, ncu)
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
    print(f"§9f P4PIPE block {action} ({len(block)} chars, "
          f"{len(cells)} cells + {len(ncu)} ncu cells embedded)")


if __name__ == "__main__":
    main()
