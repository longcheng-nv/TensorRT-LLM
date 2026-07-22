# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate REPORT_BSEXT.html — bilingual (EN/中文, CSS-only toggle, no JS)
report of the compB BS>1 optimization campaign. Data = final_bs.csv."""
import csv
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "final_bs.csv")))
cells = {}
for r in rows:
    key = (r["kind"], f"{r['model']}_{r['isl']}", int(r["N"]), int(r["K"]))
    cells.setdefault(key, {}).setdefault(int(r["BS"]), {})[r["op"]] = r

BSL = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ARM = {"0": "v4", "1": "tp4", "2": "tp3", "3": "tp2"}
COLORS = {"flash_512k": "#2563eb", "pro_512k": "#d97706",
          "flash_256k": "#059669", "pro_1024k": "#7c3aed"}

series, tables = {}, {}
for (kind, tag, N, K), d in sorted(cells.items()):
    sp, tab = [], []
    for bs in BSL:
        g, a = d[bs]["gvr_pr"], d[bs]["auto"]
        s = float(g["us"]) / float(a["us"])
        sp.append(s)
        tab.append((bs, float(g["us"]), float(a["us"]),
                    ARM[a["pick"]], s, a["exact"]))
    series[tag] = (kind, N, K, sp)
    tables[tag] = tab

tgt_sp = [s for t, (k, n, kk, sp) in series.items() if k == "target" for s in sp]
all_sp = [s for (_, _, _, sp) in series.values() for s in sp]
gm_t, gm_a = st.geometric_mean(tgt_sp), st.geometric_mean(all_sp)
mn = min(all_sp)


def svg_chart():
    W, H, ML, MR, MT, MB = 860, 400, 60, 20, 20, 46
    pw, ph = W - ML - MR, H - MT - MB
    ymin, ymax = 0.9, 4.2
    xs = {bs: ML + pw * i / (len(BSL) - 1) for i, bs in enumerate(BSL)}
    def Y(v):
        return MT + ph * (1 - (math.log(v) - math.log(ymin))
                          / (math.log(ymax) - math.log(ymin)))
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'style="max-width:100%;height:auto;background:#fff">']
    for gv in (1.0, 1.2, 1.5, 2.0, 3.0, 4.0):
        y = Y(gv)
        dash = ' stroke-dasharray="5 4"' if gv in (1.0, 1.2, 2.0) else ""
        col = "#dc2626" if gv == 1.0 else "#b45309" if gv == 1.2 else \
              "#15803d" if gv == 2.0 else "#e5e7eb"
        w = "1.4" if gv in (1.0, 1.2, 2.0) else "1"
        out.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W-MR}" y2="{y:.1f}" '
                   f'stroke="{col}" stroke-width="{w}"{dash}/>')
        out.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" '
                   f'font-size="12" fill="#374151">{gv:g}×</text>')
    for bs in BSL:
        x = xs[bs]
        out.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H-MB}" '
                   f'stroke="#f3f4f6"/>')
        out.append(f'<text x="{x:.1f}" y="{H-MB+18}" text-anchor="middle" '
                   f'font-size="12" fill="#374151">{bs}</text>')
    out.append(f'<text x="{ML+pw/2}" y="{H-8}" text-anchor="middle" '
               f'font-size="13" fill="#111">BS (batch size)</text>')
    for tag, (kind, N, K, sp) in series.items():
        c = COLORS[tag]
        pts = " ".join(f"{xs[bs]:.1f},{Y(s):.1f}" for bs, s in zip(BSL, sp))
        dash = "" if kind == "target" else ' stroke-dasharray="7 4"'
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" '
                   f'stroke-width="2.4"{dash}/>')
        for bs, s in zip(BSL, sp):
            out.append(f'<circle cx="{xs[bs]:.1f}" cy="{Y(s):.1f}" r="3.4" '
                       f'fill="{c}"/>')
    lx = ML + 10
    for i, (tag, (kind, N, K, sp)) in enumerate(series.items()):
        c = COLORS[tag]
        y = MT + 14 + 18 * i
        out.append(f'<rect x="{lx}" y="{y-9}" width="18" height="4" '
                   f'fill="{c}"/>')
        star = "" if kind == "target" else " (gen)"
        out.append(f'<text x="{lx+24}" y="{y}" font-size="12" '
                   f'fill="#111">{tag} N={N} K={K}{star}</text>')
    out.append("</svg>")
    return "".join(out)


def bil(en, zh):
    return (f'<span class="l-en">{en}</span><span class="l-zh">{zh}</span>')


def cell_table(tag):
    kind, N, K, _ = series[tag]
    h = [f'<h3>{tag} — N={N}, K={K} '
         f'({bil("target envelope","目标包络") if kind=="target" else bil("generalization","泛化检验")})</h3>',
         '<table><tr><th>BS</th><th>gvr_pr (µs)</th><th>auto (µs)</th>'
         f'<th>{bil("arm","臂")}</th><th>{bil("speedup","加速比")}</th></tr>']
    for bs, g, a, arm, s, ex in tables[tag]:
        cls = ' class="miss"' if s < 1.2 else (' class="hi"' if s >= 2 else "")
        h.append(f'<tr{cls}><td>{bs}</td><td>{g:.2f}</td><td>{a:.2f}</td>'
                 f'<td>{arm}</td><td><b>{s:.3f}×</b></td></tr>')
    h.append("</table>")
    return "".join(h)


CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;
     margin:0;background:#f8fafc;color:#111827;line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding:24px 28px 80px}
h1{font-size:26px;margin:10px 0 2px}h2{font-size:20px;border-bottom:2px solid
#e5e7eb;padding-bottom:6px;margin-top:36px}h3{font-size:16px;margin:22px 0 8px}
table{border-collapse:collapse;margin:8px 0 18px;font-size:13.5px;width:100%}
th,td{border:1px solid #e5e7eb;padding:5px 10px;text-align:right}
th{background:#f1f5f9;text-align:center}td:nth-child(4){text-align:center}
tr.miss td{background:#fef2f2}tr.hi td{background:#f0fdf4}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}
.card{flex:1 1 150px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;
padding:14px 16px}.card .v{font-size:26px;font-weight:700}
.card .t{font-size:12.5px;color:#6b7280;margin-top:2px}
.pass{color:#15803d}.warn{color:#b45309}
.note{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;
border-radius:6px;font-size:14px}
.lesson{background:#eff6ff;border-left:4px solid #3b82f6;padding:10px 14px;
border-radius:6px;font-size:14px;margin:10px 0}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:13px}
/* CSS-only language toggle */
.l-zh{display:none}
#lang-zh:checked ~ .wrap .l-zh{display:inline}
#lang-zh:checked ~ .wrap .l-en{display:none}
#lang-zh:checked ~ .wrap div.l-zh,#lang-zh:checked ~ .wrap p.l-zh,
#lang-zh:checked ~ .wrap li.l-zh{display:block}
div.l-zh,p.l-zh,li.l-zh{display:none}
#lang-en:checked ~ .wrap div.l-en,#lang-en:checked ~ .wrap p.l-en{display:block}
.toggle{position:sticky;top:0;background:#111827;color:#fff;padding:9px 28px;
z-index:5;font-size:14px}
.toggle label{cursor:pointer;padding:4px 14px;border:1px solid #4b5563;
border-radius:6px;margin-right:8px}
#lang-en:checked ~ .toggle label[for=lang-en],
#lang-zh:checked ~ .toggle label[for=lang-zh]{background:#2563eb;
border-color:#2563eb}
input[name=lang]{display:none}
"""

perf_rows_summary = {
    2: (2.54, 1.67), 8: (2.18, 1.57), 16: (1.46, 1.22), 32: (1.19, 1.11),
    64: (1.70, 1.58), 256: (3.07, 2.11), 1024: (3.55, 2.26)}

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>compB BS&gt;1 Optimization Campaign — Final Report</title>
<style>{CSS}</style></head><body>
<input type="radio" name="lang" id="lang-en" checked>
<input type="radio" name="lang" id="lang-zh">
<div class="toggle"><label for="lang-en">English</label>
<label for="lang-zh">中文</label>
<span style="opacity:.65">compB BS&gt;1 campaign · umbriel-b200-039 · 2026-07-22</span></div>
<div class="wrap">
<h1>{bil("compB BS&gt;1 Optimization Campaign — Final Report",
         "compB BS&gt;1 优化战役 — 最终报告")}</h1>
<p style="color:#6b7280;font-size:14px">
{bil("Batched exact top-K for the DSv4 indexer (compB lineage) vs PR#16457 head (gvr_pr). "
     "nsys cold-L2 kernel-sum, real decode-capture rows, same-GPU paired, all-row tie-robust exactness. "
     "Code: branch <code>kf/compb-bs-ext</code> @ longcheng-nv/TensorRT-LLM (3 staged commits).",
     "DSv4 indexer 批量精确 top-K(compB 谱系)对 PR#16457 head(gvr_pr)。"
     "nsys 冷 L2 kernel-sum,真实 decode 捕获数据,同卡配对,全行 tie-鲁棒精确性校验。"
     "代码:<code>kf/compb-bs-ext</code> 分支 @ longcheng-nv/TensorRT-LLM(3 个阶段提交)。")}</p>

<div class="cards">
<div class="card"><div class="v pass">{gm_t:.3f}×</div>
<div class="t">{bil("geomean vs PR head, target envelope (goal ≥2.0 — PASS)",
                    "目标包络几何均值 vs PR head(目标 ≥2.0 — 达成)")}</div></div>
<div class="card"><div class="v">{gm_a:.3f}×</div>
<div class="t">{bil("geomean, all 4 cells pooled (incl. generalization)",
                    "全部 4 cell 池化几何均值(含泛化)")}</div></div>
<div class="card"><div class="v warn">{mn:.3f}×</div>
<div class="t">{bil("minimum case (goal ≥1.2 — missed at BS=32 only, 4/40 points)",
                    "最小 case(目标 ≥1.2 — 仅 BS=32 未达,4/40 点)")}</div></div>
<div class="card"><div class="v pass">0/40</div>
<div class="t">{bil("regressions vs PR head (goal: none — PASS)",
                    "对 PR head 退化数(目标零退化 — 达成)")}</div></div>
<div class="card"><div class="v pass">80/80</div>
<div class="t">{bil("exact results (all rows, tie-robust)","精确结果(全行,tie-鲁棒)")}</div></div>
</div>

<h2>{bil("1 · Speedup vs PR head over BS","1 · 对 PR head 的加速比随 BS 变化")}</h2>
{svg_chart()}
<p class="l-en">Dashed horizontal lines: 1.0× (no-regression floor), 1.2× (min-case goal),
2.0× (average goal). Solid series = target envelope (N=131075); dashed series =
generalization cells (N=65538 / N=262127).</p>
<p class="l-zh">水平虚线:1.0×(零退化底线)、1.2×(最小 case 目标)、2.0×(平均目标)。
实线 = 目标包络(N=131075);虚线 = 泛化 cell(N=65538 / N=262127)。</p>

<h2>{bil("2 · The kernel portfolio and dispatch",
         "2 · 内核组合与派发")}</h2>
<table>
<tr><th>{bil("zone","区间")}</th><th>{bil("arm","臂")}</th>
<th>{bil("design","设计")}</th><th>{bil("typical speedup","典型加速")}</th></tr>
<tr><td>BS ≤ rows_per_wave (~4-9)</td><td>ext_v4</td>
<td>{bil("register-resident row teams, one co-resident wave, launch_bounds(512,4) diet",
"寄存器驻留 row team,单一 co-resident 波,launch_bounds(512,4) 瘦身")}</td><td>1.57–2.54×</td></tr>
<tr><td>{bil("mid BS, slice fits smem","中段 BS,slice 装进 smem")}</td><td>tp4</td>
<td>{bil("fused exact-hist 2-pass: hist → barrier → direct emit; smem row cache makes pass 2 mostly smem-resident",
"融合精确直方图双遍:hist → 屏障 → 直写输出;smem 行缓存让第二遍基本走 smem")}</td><td>1.11–1.70×</td></tr>
<tr><td>{bil("high-mid BS","中高段 BS")}</td><td>tp3</td>
<td>{bil("fused sampled single-pass: 1/16 sample → budget b_safe → candidate collect → CTA0 finish; smem-staged candidates",
"融合采样单遍:1/16 采样 → 预算 b_safe → 候选收集 → CTA0 finish;候选 smem staging")}</td><td>1.14–3.88×</td></tr>
<tr><td>BS &gt; {bil("one wave","单波")} (~592)</td><td>tp2</td>
<td>{bil("3-kernel sampled single-pass (launch boundaries as sync)",
"三内核采样单遍(发射边界即同步)")}</td><td>2.26–3.55×</td></tr>
</table>

<h2>{bil("3 · Per-cell results","3 · 分 cell 结果")}</h2>
{cell_table("flash_512k")}
{cell_table("pro_512k")}
{cell_table("flash_256k")}
{cell_table("pro_1024k")}

<h2>{bil("4 · What shipped (staged commits on kf/compb-bs-ext)",
         "4 · 交付内容(kf/compb-bs-ext 分阶段提交)")}</h2>
<ul>
<li class="l-en"><b>dd9cd928ef</b> baseline: validated D1+D2 state (best-arm gm 1.597× over BS 8–1024).</li>
<li class="l-zh"><b>dd9cd928ef</b> 基线:已验证的 D1+D2 状态(BS 8–1024 best-arm gm 1.597×)。</li>
<li class="l-en"><b>f102594ba0</b> tp3 fused single-kernel arm + <b>smem-staged collect</b> — the same-address
per-row candidate-counter atomics (~5K serialized L2 RMWs) were BOTH the mid-BS valley and the flash@1024 anomaly;
one bulk reservation per CTA fixed both.</li>
<li class="l-zh"><b>f102594ba0</b> tp3 融合单核 + <b>smem staging collect</b> —— per-row 候选计数器的同地址原子
(~5K 次串行 L2 RMW)同时是中段 BS 谷与 flash@1024 反常的根因;每 CTA 一次批量占位同时修复两者。</li>
<li class="l-en"><b>4daeefed1f</b> tp4 exact-hist fused 2-pass arm (smem row cache) + measured (N,BS) dispatch
table + the restored <b>consumer-fence contract</b>.</li>
<li class="l-zh"><b>4daeefed1f</b> tp4 精确直方图融合双遍臂(smem 行缓存)+ 实测 (N,BS) 派发表 + 恢复的
<b>消费者 fence 契约</b>。</li>
</ul>

<h2>{bil("5 · Correctness contracts (hard rules learned)",
         "5 · 正确性契约(用 bug 换来的硬规则)")}</h2>
<div class="lesson">{bil(
"<b>Fence-less barrier data contract:</b> any data produced on one CTA and consumed by another across the "
"relaxed spinning barrier must be written with <code>atomicExch</code> AND read after a consumer "
"<code>fence.acq_rel.gpu</code> — compB's own tail refine does exactly this. Plain stores intermittently "
"miss in-flight entries (~2 wrong tie members per bad row, stochastic, found via randn adversarial rows).",
"<b>fence-less 屏障数据契约:</b>跨 relaxed 自旋屏障由一个 CTA 生产、另一个 CTA 消费的数据,必须用 "
"<code>atomicExch</code> 写并在消费侧先执行 <code>fence.acq_rel.gpu</code> —— compB 自己的 tail refine 正是这样做的。"
"plain store 会间歇性丢失 in-flight 条目(每坏行约错 2 个 tie 成员,随机;由 randn 对抗行捕获)。")}</div>
<div class="lesson">{bil(
"<b>Load-then-rezero needs a barrier:</b> a kernel that plain-loads a shared global value and then zeroes the "
"same buffer must put <code>__syncthreads()</code> between them (hit twice this campaign; stochastic single-row "
"corruption at high BS).",
"<b>load 后清零必须夹栅栏:</b>同一内核先 plain-load 共享全局值、再清零同一缓冲,两者之间必须有 "
"<code>__syncthreads()</code>(本战役踩中两次;高 BS 下随机单行损坏)。")}</div>
<div class="lesson">{bil(
"<b>All exactness gates:</b> adversarial rows (const / 3-level quantized / randn, forcing every fallback path), "
"real-capture rows with ALL-row tie-robust value-set checks, and 40× stress repeats at racy batch sizes.",
"<b>精确性门:</b>对抗行(常数 / 3-level 量化 / randn,强制覆盖所有兜底路径)+ 真实捕获行全行 tie-鲁棒值集校验 + "
"易竞态 BS 的 40× 压力重复。")}</div>

<h2>{bil("6 · Falsified on the way (measured cold, reverted)",
         "6 · 途中证伪并回退的方案(全部冷判决)")}</h2>
<ul>
<li class="l-en">Parallel finish emit (3rd barrier + global tie round trip cost more than the saved single-CTA scan).</li>
<li class="l-zh">并行 finish emit(第三道屏障 + 全局 tie 往返比省下的单 CTA 扫描更贵)。</li>
<li class="l-en">Dropping the gt staging arena for a bigger row cache (per-hit gt cursor atomics regressed BS=32 by 7–15%).</li>
<li class="l-zh">砍 gt staging arena 换更大行缓存(gt 逐命中游标原子使 BS=32 回退 7–15%)。</li>
<li class="l-en">Warp-aggregated candidate atomics — the per-element ballot tax, learned three times now.</li>
<li class="l-zh">候选原子 warp 聚合 —— 逐元素 ballot 税,本战役已三次验证。</li>
<li class="l-en">(Earlier, same day) B' persistent work-queue: chases the ≤2% launch-gap while paying a ≥5% in-kernel sync floor.</li>
<li class="l-zh">(同日更早)B' persistent 工作队列:追 ≤2% 发射间隙却付 ≥5% 核内同步地板。</li>
</ul>

<h2>{bil("7 · The BS=32 residual","7 · BS=32 残余分析")}</h2>
<p class="note l-en"><b>Why BS=32 resists:</b> gvr_pr's latency-flat plateau ends exactly there
(gvr@64 ≈ 2× gvr@32) — at BS=32 the PR arm is at its best operating point, while every exact
full-read arm still pays ≥1 full pass + 2 team barriers + a tail. Best arms land 1–9% short of the
1.2× bar (14.3–24.7µs vs needed 14.1–22.8µs). Identified next lever (out of scope):
a hint-assisted arm consuming prev-step <code>preIdx</code> — the same legitimate inference-time input
gvr itself uses — to skip the histogram pass in the high-hit regime.</p>
<p class="note l-zh"><b>为什么 BS=32 难啃:</b>gvr_pr 的延迟平台恰好在此结束(gvr@64 ≈ 2× gvr@32)——
BS=32 是 PR 臂的最佳工作点,而所有精确全读臂仍要付 ≥1 遍全读 + 2 道 team 屏障 + 尾部。最优臂距 1.2× 线差
1–9%(14.3–24.7µs vs 需要 14.1–22.8µs)。已识别的下一杠杆(超出本战役范围):
消费上一步 <code>preIdx</code> 的 hint 辅助臂 —— 与 gvr 使用同一合法推理时输入 —— 在高命中区跳过直方图遍。</p>

<h2>{bil("8 · Methodology","8 · 方法学")}</h2>
<ul>
<li class="l-en">Timing: nsys cold-L2 kernel-sum (20 cold reps, 512MB L2 evict outside the NVTX range), one GPU per cell, arms paired back-to-back. Warm event timing was used ONLY for iteration, never for verdicts (at mid BS the whole batch fits B200's 126MB L2 and warm numbers mislead).</li>
<li class="l-zh">计时:nsys 冷 L2 kernel-sum(20 次冷重复,NVTX 区间外 512MB L2 驱逐),每 cell 独占一卡、臂背靠背配对。warm 事件计时只用于迭代、从不作判决(中段 BS 整批装进 B200 的 126MB L2,warm 数字有系统性误导)。</li>
<li class="l-en">Data: real DSv4 decode captures (V4-Flash K=512, V4-Pro K=1024), row replicated across the batch (§7.8 protocol).</li>
<li class="l-zh">数据:真实 DSv4 decode 捕获(V4-Flash K=512、V4-Pro K=1024),行复制成批(§7.8 协议)。</li>
<li class="l-en">Full provenance: kf_bs_scaling/ext/ (RESULTS.md, final_bs.csv, per-stage CSVs); ledger: kf_campaign/R3_LEDGER.md.</li>
<li class="l-zh">完整溯源:kf_bs_scaling/ext/(RESULTS.md、final_bs.csv、各阶段 CSV);台账:kf_campaign/R3_LEDGER.md。</li>
</ul>
</div></body></html>
"""
(HERE / "REPORT_BSEXT.html").write_text(html)
print("REPORT_BSEXT.html written,", len(html), "bytes")
