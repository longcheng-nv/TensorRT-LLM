#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotently inject the §8.1 SGLang-v2 conditional-exactness note + the
§8.2 FlashInfer validation note into REPORT.html §8 (between the rival table
<details> and the next <h2>9 heading, whatever section 9 currently is).
Re-running replaces the marker-delimited block in place; gen_report.py calls
this after every full regen so the block survives regeneration. See NOTE.md
for the underlying analysis; scripts live alongside this file."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "REPORT.html"
MARK_A = "<!-- SGLV2-TIEBIN-NOTE -->"
MARK_B = "<!-- /SGLV2-TIEBIN-NOTE -->"
ANCHOR = "<h2>9 · "

NOTE = MARK_A + """
<h3>8.1 · Correctness caveat: SGLang v2 is only <i>conditionally</i> exact / 正确性注记:SGLang v2 仅条件精确</h3>
<div class="lang-en"><p>
A Slack report (Yue Weng, 2026-07-14) showed SGLang v2 returning badly wrong top-K on uniform-[0,1) logits at
kv_len 128K. Root cause (<code>sgl_kernel/deepseek_v4/topk_impl.cuh</code>): the kernel histograms the top
<code>kHistBits</code> bits of the <b>fp16-cast</b> score (cluster path 10 bits, register/streaming 12 bits);
elements falling in the <i>threshold bin</i> go to a tie buffer capped at <code>kMaxNumTie&nbsp;=&nbsp;2048</code>,
and overflow candidates are <b>silently dropped</b> (arrival order decides survivors, so the error is also
non-deterministic). The kernel is exact iff the threshold bin holds ≤ 2048 elements; for K=2048 the cap has
<b>zero structural headroom</b> (<code>kMaxNumTie == kMaxTopK</code>). We reproduced the uniform failure end-to-end
(N=128K, K=2048: 2017/2048 slots wrong) — the §8 <code>torch.topk</code> gate catches it, so the all-exact table
above reflects <b>data coverage, not gate laxness</b>.</p>
<p>Extending the scan beyond the §8 benchmark slice (bench layer × last decode step) to <b>ALL layers × ALL ISLs ×
ALL decode steps</b> of the same real captures:</p>
<table><tr><th>real data</th><th>coverage</th><th>max threshold-bin count</th><th>margin vs 2048</th><th>end-to-end kernel</th></tr>
<tr><td>V4 Flash (K=512)</td><td>21 layers × 9 ISL (4K–1M), last step</td><td>247 (@1M)</td><td><b>≥ 8.3×</b></td><td>exact — SAFE</td></tr>
<tr><td>V4 Pro (K=1024)</td><td>30 layers × 9 ISL (4K–1M), last step</td><td>395 (@1M)</td><td><b>≥ 5.2×</b></td><td>exact — SAFE</td></tr>
<tr><td>V3.2 (K=2048)</td><td>58 layers × 7 ISL, last step</td><td>1466 (256k, L52)</td><td>1.40×</td><td>exact (thin margin)</td></tr>
<tr><td>V3.2 128k</td><td>58 layers × 15 steps (870 cells)</td><td><b>2278 (L52 step 4) — 1 cell OVER cap</b></td><td>0.90×</td><td rowspan="2"><b>FAIL</b>: L52 steps 3/6/12 @256k exact=False,
25–168/2048 slots wrong, max value err 0.0088, non-deterministic</td></tr>
<tr><td>V3.2 256k</td><td>58 layers × 15 steps (870 cells)</td><td><b>2214 (L52 step 6) — 3 cells OVER cap</b></td><td>0.93×</td></tr>
<tr><td class="mut">synth worst-case (K=2048), extrapolated</td><td class="mut">N up to 2M</td><td class="mut">1467 (@512K)</td><td class="mut">1.40×</td><td class="mut">exact — single-row synth lacks the layer/step extreme-value statistic</td></tr></table>
<p><b>Verdict:</b> the §8 exact=True results are genuine but <b>slice-conditional</b>. For the V4 Flash/Pro
deployment envelope SGLang v2 is safe by ≥5× margin up to ISL=1M (and, by calibrated-synth extrapolation, far
beyond). For <b>V3.2 (K=2048, cr=1) real production captures already cross the cap at ISL 128K/256K</b> on the
flat-distribution layer 52 (~0.1–0.3% of layer-step cells); every decode token runs 58 indexer layers, so one bad
layer-step feeds a wrong top-K set into sparse attention. SGLang v2 should not be labeled "exact" for the K=2048
use case; the GVR and Radix arms are unconditionally exact. Analysis + repro scripts:
<code>sglv2_correctness/</code> (NOTE.md, tiebin_extended.py, sglv2_real_overflow.py).</p></div>
<div class="lang-zh"><p>
Slack 报告(Yue Weng, 2026-07-14)显示 SGLang v2 在 kv_len 128K、uniform-[0,1) logits 上输出严重错误的 top-K。
根因(<code>sgl_kernel/deepseek_v4/topk_impl.cuh</code>):内核对分数的 <b>fp16 转换值</b>取高
<code>kHistBits</code> 位做直方图(cluster 路径 10 位,register/streaming 12 位);落在<i>阈值 bin</i> 内的元素进入容量
<code>kMaxNumTie&nbsp;=&nbsp;2048</code> 的 tie buffer,溢出候选被<b>静默丢弃</b>(到达序决定保留谁,故错误还是非确定的)。
内核精确当且仅当阈值 bin ≤ 2048 个元素;K=2048 时上限<b>结构性零余量</b>(<code>kMaxNumTie == kMaxTopK</code>)。
我们端到端复现了 uniform 失败(N=128K、K=2048:2017/2048 槽位错)——§8 的 <code>torch.topk</code> 校验门能抓到它,
故上表全 exact 反映的是<b>数据覆盖问题,而非校验门太松</b>。</p>
<p>把扫描从 §8 基准切片(代表层 × 最后 decode step)扩展到同一批真实采集的<b>全部层 × 全部 ISL × 全部 decode step</b>:</p>
<table><tr><th>真实数据</th><th>覆盖</th><th>阈值 bin 最大元素数</th><th>对 2048 余量</th><th>端到端内核</th></tr>
<tr><td>V4 Flash (K=512)</td><td>21 层 × 9 ISL (4K–1M),最后 step</td><td>247 (@1M)</td><td><b>≥ 8.3×</b></td><td>精确 — 安全</td></tr>
<tr><td>V4 Pro (K=1024)</td><td>30 层 × 9 ISL (4K–1M),最后 step</td><td>395 (@1M)</td><td><b>≥ 5.2×</b></td><td>精确 — 安全</td></tr>
<tr><td>V3.2 (K=2048)</td><td>58 层 × 7 ISL,最后 step</td><td>1466 (256k, L52)</td><td>1.40×</td><td>精确(余量薄)</td></tr>
<tr><td>V3.2 128k</td><td>58 层 × 15 step(870 cell)</td><td><b>2278(L52 step4)— 1 cell 越限</b></td><td>0.90×</td><td rowspan="2"><b>失败</b>:256k L52 step 3/6/12 exact=False,
25–168/2048 槽位错,最大值误差 0.0088,非确定</td></tr>
<tr><td>V3.2 256k</td><td>58 层 × 15 step(870 cell)</td><td><b>2214(L52 step6)— 3 cell 越限</b></td><td>0.93×</td></tr>
<tr><td class="mut">synth 最坏场景 (K=2048) 外推</td><td class="mut">N 至 2M</td><td class="mut">1467 (@512K)</td><td class="mut">1.40×</td><td class="mut">精确 — 单行 synth 缺层/步极值统计</td></tr></table>
<p><b>结论:</b>§8 的 exact=True 真实但<b>依赖切片条件</b>。V4 Flash/Pro 部署包络内 SGLang v2 至 ISL=1M 有 ≥5× 余量
(标定 synth 外推更远也安全)。而 <b>V3.2(K=2048, cr=1)的真实生产采集在 ISL 128K/256K 已越限</b>——平坦分布的
layer 52(约 0.1–0.3% 的层×步 cell);每个 decode token 要过 58 个 indexer 层,单个坏层步就会把错误 top-K 送入稀疏
注意力。K=2048 场景不应给 SGLang v2 标"exact";GVR 与 Radix 臂无条件精确。分析与复现脚本:
<code>sglv2_correctness/</code>(NOTE.md、tiebin_extended.py、sglv2_real_overflow.py)。</p></div>

<h3>8.2 · FlashInfer top_k put through the same battery — exact on all 2245 checks / FlashInfer 同套测试全通过</h3>
<div class="lang-en"><p>
The same correctness battery that falsified SGLang v2's unconditional-exactness was applied to
<b>FlashInfer <code>top_k</code> 0.6.11</b> (the §8 <code>flashinfer_topk</code> arm; B200 default =
<code>fast_topk_clusters_exact</code>, an 8-bit-radix multi-phase refinement over the <b>full fp32 bits</b>).
Structurally it has no SGLang-style tie cap: the per-phase global overflow buffer is sized
<code>max_model_len&nbsp;/&nbsp;num_clusters</code> per cluster — i.e. the whole row fits — so boundary ties are
never silently dropped. Empirically (<code>fi_topk_correctness.py</code>, B200, both the default clusters path and
<code>deterministic=True</code> radix path): <b>2245 / 2245 checks exact</b> vs <code>torch.topk</code>.</p>
<table><tr><th>battery</th><th>coverage</th><th>result</th></tr>
<tr><td>adversarial synthetic</td><td>uniform-[0,1) N=128K/1M × K∈{512,1024,2048} (the SGLang-killer case) · all-equal row (total tie) ·
<b>fp16-collision block</b> (8192 fp32-distinct values inside one fp16 ulp straddling the K boundary — a 16-bit histogram structurally cannot separate them) · fp32 + bf16 · default + deterministic</td><td><b>exact</b> (SGLang v2 fails the uniform case 2017/2048 slots)</td></tr>
<tr><td>SGLang-failing real rows</td><td>V3.2 256k L52 steps 3/6/12 + 128k L52 step 4 (the over-cap rows) + 3 below-cap controls, both modes</td><td><b>exact</b></td></tr>
<tr><td>broad real sweep</td><td>V4 Flash (21 layers × 9 ISL, K=512) + V4 Pro (30 layers × 9 ISL, K=1024) + V3.2 bench layers × 7 ISL (K=2048), last decode step</td><td><b>exact</b> (all rows)</td></tr>
<tr><td>V3.2 full temporal grid</td><td>ALL 58 layers × ALL 15 decode steps × {128k, 256k} = 1740 cells — the grid where SGLang v2's overflow lives</td><td><b>exact 1740/1740</b></td></tr></table>
<p><b>Verdict:</b> FlashInfer <code>top_k</code> is <b>unconditionally exact</b> on everything we can throw at it —
including the exact rows and the exact adversarial distribution that break SGLang v2. Among the external arms, the
correctness ranking is FlashInfer = Radix = GVR (unconditional) &gt; SGLang v2 (conditional, K=2048 real data already
over the cap). Script: <code>sglv2_correctness/fi_topk_correctness.py</code>.</p></div>
<div class="lang-zh"><p>
把证伪 SGLang v2 无条件精确性的同一套正确性测试施加于 <b>FlashInfer <code>top_k</code> 0.6.11</b>
(§8 的 <code>flashinfer_topk</code> 臂;B200 默认走 <code>fast_topk_clusters_exact</code>,对<b>完整 fp32 位</b>做
8-bit radix 多阶段细化)。结构上它没有 SGLang 式 tie 上限:每阶段的全局 overflow buffer 按
<code>max_model_len&nbsp;/&nbsp;num_clusters</code> × 每 cluster 配置——即整行都装得下,边界 tie 永远不会被静默丢弃。
实测(<code>fi_topk_correctness.py</code>,B200,default clusters 路径 + <code>deterministic=True</code> radix 路径
两种模式):对 <code>torch.topk</code> <b>2245 / 2245 项检查全部精确</b>。</p>
<table><tr><th>测试组</th><th>覆盖</th><th>结果</th></tr>
<tr><td>对抗合成</td><td>uniform-[0,1) N=128K/1M × K∈{512,1024,2048}(击垮 SGLang 的用例)· 全等值行(全体 tie)·
<b>fp16 碰撞块</b>(8192 个 fp32 互异值挤在一个 fp16 ulp 内并跨越 K 边界——16-bit 直方图结构上无法分辨)· fp32 + bf16 · 两种模式</td><td><b>精确</b>(SGLang v2 在 uniform 用例错 2017/2048 槽)</td></tr>
<tr><td>SGLang 失败的真实行</td><td>V3.2 256k L52 step 3/6/12 + 128k L52 step 4(越限行)+ 3 个未越限对照行,两种模式</td><td><b>精确</b></td></tr>
<tr><td>广域真实扫描</td><td>V4 Flash(21 层 × 9 ISL,K=512)+ V4 Pro(30 层 × 9 ISL,K=1024)+ V3.2 bench 层 × 7 ISL(K=2048),最后 decode step</td><td><b>精确</b>(全部行)</td></tr>
<tr><td>V3.2 全时序网格</td><td>全部 58 层 × 全部 15 decode step × {128k, 256k} = 1740 cell——SGLang v2 溢出所在的网格</td><td><b>精确 1740/1740</b></td></tr></table>
<p><b>结论:</b>FlashInfer <code>top_k</code> 在我们能构造的全部数据上<b>无条件精确</b>——包括击垮 SGLang v2 的那些
真实行与对抗分布。外部臂的正确性排序:FlashInfer = Radix = GVR(无条件)&gt; SGLang v2(条件精确,K=2048 真实数据已越限)。
脚本:<code>sglv2_correctness/fi_topk_correctness.py</code>。</p></div>
""" + MARK_B + "\n\n"

html = REPORT.read_text()
if MARK_A in html:
    pre, rest = html.split(MARK_A, 1)
    _, post = rest.split(MARK_B, 1)
    html = pre + NOTE.rstrip("\n") + post
    action = "replaced"
else:
    assert ANCHOR in html, "anchor <h2>9 not found"
    html = html.replace(ANCHOR, NOTE + ANCHOR, 1)
    action = "inserted"
REPORT.write_text(html)
print(f"§8.1 sglang-v2 correctness note {action} in {REPORT}")
