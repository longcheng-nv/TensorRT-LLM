#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotently inject the SGLang-v2 conditional-exactness correctness note
into REPORT.html §8 (between the rival table <details> and the <h2>9 heading).
Re-running replaces the marker-delimited block in place. See NOTE.md for the
underlying analysis; scripts live alongside this file."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "REPORT.html"
MARK_A = "<!-- SGLV2-TIEBIN-NOTE -->"
MARK_B = "<!-- /SGLV2-TIEBIN-NOTE -->"
ANCHOR = "<h2>9 · Test data"

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
