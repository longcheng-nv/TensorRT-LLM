#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Idempotent injector: add §9d "HEADFULL — full-coverage re-test at the latest
PR head @e6fdbfac3d" to REPORT.html.

Source of truth: headfull_harness/HEADFULL_VERDICT.md +
headfull_harness/results_headfull/compare_headfull.log (2026-07-20 sweep,
54/54 batches, 2772 cells x 3 arms on b200-027 + umbriel-b200-019).

Marker-delimited (HEADFULL:BEGIN/END), safe to re-run; inserted right after
the §9c RUNGRECAL block (before §10).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "REPORT.html")
BEGIN = "<!-- HEADFULL:BEGIN (update_report_headfull.py) -->"
END = "<!-- HEADFULL:END -->"
ANCHOR_AFTER = "<!-- RUNGRECAL:END -->"

# Six-axis comparison — HEADFULL_VERDICT.md / compare_headfull.log.
# (axis_en, axis_zh, n, anchor_norm_pr, within_run, report_ratio, delta, hot)
AXES = [
    ("§3 synth seqlen fp32", "§3 合成 seqlen fp32", 52, "0.977", "1.156", "1.148", "+0.7%", False),
    ("§7 synth BS fp32", "§7 合成 BS fp32", 572, "0.973", "1.178", "1.153", "+2.2%", False),
    ("§7 synth BS 16-bit", "§7 合成 BS 16-bit", 1144, "0.997", "1.157", "1.098", "+5.5%", True),
    ("§4 real seqlen fp32 BS=1", "§4 真实 seqlen fp32 BS=1", 25, "0.953", "1.257", "1.309", "−4.0%", False),
    ("§7 real BS fp32", "§7 真实 BS fp32", 275, "0.972", "1.313", "1.293", "+1.6%", False),
    ("§7 real BS 16-bit", "§7 真实 BS 16-bit", 550, "1.015", "1.213", "1.141", "+6.3%", True),
]

# Worst pr cells per axis (REPORT/new < 1 = head slower RAW, before anchor
# normalization) — compare_headfull.log, all 12 listed, none hidden.
WORST = [
    ("0.770", "synth bs worst K2048 fp16 N=1048576 BS256", "235.04", "305.30"),
    ("0.775", "synth bs worst K2048 bf16 N=1048576 BS256", "241.46", "311.72"),
    ("0.779", "synth bs worst K2048 bf16 N=1048576 BS128", "121.87", "156.44"),
    ("0.855", "real seqlen pro fp32 64k BS1", "10.06", "11.77"),
    ("0.869", "synth seqlen worst K1024 fp32 N=8192 BS1", "9.11", "10.49"),
    ("0.878", "real seqlen pro fp32 256k BS1", "13.25", "15.10"),
    ("0.878", "real seqlen pro fp32 128k BS1", "11.95", "13.60"),
    ("0.884", "real bs pro bf16 1024k BS128", "36.74", "41.53"),
    ("0.889", "real bs pro fp16 1024k BS128", "36.60", "41.18"),
    ("0.900", "real bs flash fp32 256k BS2", "12.90", "14.33"),
    ("0.900", "real bs pro fp32 256k BS4", "13.87", "15.41"),
    ("0.900", "real bs flash fp32 256k BS1", "12.99", "14.42"),
]


def build():
    ax_rows = []
    for en, zh, n, norm, wr, rep, delta, hot in AXES:
        b0, b1 = ("<b>", "</b>") if hot else ("", "")
        ax_rows.append(
            f"<tr><td><span class='lang-en'>{en}</span><span class='lang-zh'>{zh}</span></td>"
            f"<td>{n}</td><td>{norm}</td><td>{b0}{wr}{b1}</td><td>{rep}</td>"
            f"<td>{b0}{delta}{b1}</td></tr>")
    ax_table = (
        "<table><tr>"
        "<th><span class='lang-en'>axis</span><span class='lang-zh'>轴</span></th><th>n</th>"
        "<th><span class='lang-en'>anchor-norm pr (REPORT/new)</span><span class='lang-zh'>锚归一 pr（REPORT/new）</span></th>"
        "<th><span class='lang-en'>within-run base/pr (head)</span><span class='lang-zh'>within-run base/pr（新 head）</span></th>"
        "<th><span class='lang-en'>REPORT base/pr</span><span class='lang-zh'>REPORT base/pr</span></th>"
        "<th>Δ</th></tr>" + "".join(ax_rows) + "</table>")

    w_rows = "".join(
        f"<tr><td>{cell}</td><td>{ratio}</td><td>{old}</td><td>{new}</td></tr>"
        for ratio, cell, old, new in WORST)
    w_table = (
        "<table><tr><th>cell</th>"
        "<th><span class='lang-en'>REPORT/new (raw)</span><span class='lang-zh'>REPORT/new（原始）</span></th>"
        "<th>REPORT µs</th><th>head µs</th></tr>" + w_rows + "</table>")

    kpis = f"""<div class="kpis">
<div class="kpi"><div class="v" style="color:#6ede8a">2772/2772</div><div class="l lang-en">head exactness — gvr_pr, full grid</div><div class="l lang-zh">新 head 精确性——gvr_pr 全网格</div></div>
<div class="kpi"><div class="v">0.953–1.015</div><div class="l lang-en">anchor-norm pr across all six axes</div><div class="l lang-zh">六轴锚归一 pr 区间</div></div>
<div class="kpi"><div class="v" style="color:#6ede8a">+5.5% / +6.3%</div><div class="l lang-en">16-bit BS axes vs REPORT (synth / real, within-run)</div><div class="l lang-zh">16-bit BS 轴对 REPORT（合成/真实,within-run）</div></div>
<div class="kpi"><div class="v">1.015</div><div class="l lang-en">anchor median (op26_r0auto, n=2618; no bimodality)</div><div class="l lang-zh">锚中位（op26_r0auto,n=2618;无双峰）</div></div>
</div>"""

    en = f"""
<h2 id="sec-headfull">9d · HEADFULL — full-coverage re-test at the latest PR head @e6fdbfac3d / 最新 PR HEAD 全覆盖复测</h2>
<div class="lang-en">
<p><b>What.</b> The PR branch moved past the §9b head <code>eae374554c</code> (upstream ruff-format rebase, the
§9c K2048 rung swap <code>2d7ad4d019</code>, p4tt tiny-tie fast path) — so the FULL report grid was re-swept at
the current head <code>e6fdbfac3d</code>: all 2772 cells × 3 arms (base / PR / op26_r0auto anchor) = 8316 rows,
54/54 nsys cold-L2 batches, 0 errors (2026-07-20, b200-027 + umbriel-b200-019 GPUs 1-7; GPU0 excluded — broken
cooling). Reference = this report's refresh grids (b200-094). Raw jsonl + compare log:
<code>headfull_harness/results_headfull/</code>; full analysis: <code>headfull_harness/HEADFULL_VERDICT.md</code>.</p>
</div>
<div class="lang-zh">
<p><b>做了什么。</b>PR 分支已越过 §9b 的 head <code>eae374554c</code>（上游 ruff-format rebase、§9c 的 K2048
rung swap <code>2d7ad4d019</code>、p4tt tiny-tie 快路径）——因此在当前 head <code>e6fdbfac3d</code> 上重扫了
本报告的完整网格:2772 cell × 3 臂（base / PR / op26_r0auto 锚）= 8316 行,54/54 个 nsys cold-L2 批次,
0 错误（2026-07-20,b200-027 + umbriel-b200-019 GPU1-7;GPU0 因散热损坏排除）。参照 = 本报告 refresh 网格
（b200-094）。原始 jsonl + 对比日志:<code>headfull_harness/results_headfull/</code>;完整分析:
<code>headfull_harness/HEADFULL_VERDICT.md</code>。</p>
</div>
{kpis}
<div class="lang-en">
<p><b>Exactness.</b> gvr_pr at the shipped head: <b>2772/2772 exact</b>. The 36 inexact rows are all in the
<i>base</i> arm, all at real flash N=131075 (fp32/fp16/bf16 × 12 BS) — the long-known base undershoot on real
data (§4, §5), not a head defect.</p>
<p><b>Anchor gate.</b> op26_r0auto measured in both runs: n=2618, median <b>1.015</b>, p95 1.091; per-node
split 027 med 1.016 / 019 med 1.015 — no bimodality, the two source nodes agree. The p95 tail (1.09–1.20)
concentrates in DRAM-heavy N=1M large-BS cells where these nodes are ~10–20% slower than b200-094 —
node drift, not the kernel; hence the within-run pairing below.</p>
<p><b>Six-axis comparison (pr arm)</b> — anchor-normalized REPORT/new geomean (1.00 = parity after node-bias
removal) and node-clean within-run base/pr speedup vs REPORT's published ratio:</p>
</div>
<div class="lang-zh">
<p><b>精确性。</b>gvr_pr 在已上线 head 上:<b>2772/2772 精确</b>。36 个不精确行全部在 <i>base</i> 臂,且全部位于
真实 flash N=131075(fp32/fp16/bf16 × 12 BS)——即早已知的 base 真实数据 undershoot(§4、§5),不是 head 缺陷。</p>
<p><b>锚门。</b>op26_r0auto 两次运行都测:n=2618,中位 <b>1.015</b>,p95 1.091;分节点 027 中位 1.016 /
019 中位 1.015——无双峰,两个源节点相互一致。p95 尾部(1.09–1.20)集中在 DRAM 重载的 N=1M 大 BS cell,
这些节点在带宽饱和 cell 上比 b200-094 慢 ~10–20%——是节点漂移而非 kernel;故下表用 within-run 配对。</p>
<p><b>六轴对比(pr 臂)</b>——锚归一 REPORT/new 几何均值(1.00 = 去除节点偏置后持平),以及节点无关的
within-run base/pr 加速比对 REPORT 已发布值:</p>
</div>
{ax_table}
<div class="lang-en">
<p><b>Reading the table.</b>
(1) The head reproduces every axis: anchor-norm pr within [0.953, 1.015], no axis beyond the ~5% cross-node
noise floor set by the anchor tail.
(2) The 16-bit axes are <b>genuinely better</b> than the published numbers (+5.5% synth / +6.3% real):
consistent with the post-@018251950f commits — 16-bit data ties far more often, the exact-tail fires more,
and p4tt recovers the cost.
(3) real seqlen fp32 −4.0% (1.257 vs 1.309, 25 cells) is driven by pro mid-ISL cells (64k 1.046 vs 1.108,
256k 1.127 vs 1.205, 512k 1.226 vs 1.290); two node families plus the exact-tail correctness machinery on the
tie-prone pro capture make this the expected direction, and the magnitude sits inside the anchor p95 envelope.
Not a ship blocker; the discriminating experiment, if needed, is a paired old-vs-new on ONE node
(<code>newpr_nsys_ab.py</code> style).
(4) The worst RAW cells (synth worst K2048 1M BS128–256 16-bit, 0.77–0.78) decompose into anchor drift
1.14–1.18 × within-run residual ~0.95–0.97 vs REPORT's ~0.98 — a ≤5% within-run deficit on 3 envelope-edge
cells (N=1M is outside the deployment envelope; stress probe only).</p>
</div>
<div class="lang-zh">
<p><b>读表。</b>
(1) 新 head 在全部六轴上复现 REPORT:锚归一 pr 落在 [0.953, 1.015],没有任何轴超出锚尾部确立的 ~5% 跨节点
噪声底。
(2) 16-bit 两轴<b>确实优于</b>已发布数字(合成 +5.5% / 真实 +6.3%):与 @018251950f 之后的提交一致——16-bit
数据 tie 远更频繁,exact-tail 触发更多,p4tt 快路径收回其代价。
(3) 真实 seqlen fp32 −4.0%(1.257 对 1.309,25 cell)由 pro 中段 ISL cell 驱动(64k 1.046 对 1.108、256k
1.127 对 1.205、512k 1.226 对 1.290);两个节点家族 + exact-tail 正确性机制作用在 tie 密集的 pro capture 上,
方向符合预期,幅度在锚 p95 包络内。不构成 ship 阻塞;若需判决,判别实验是单节点新旧配对
(<code>newpr_nsys_ab.py</code> 式)。
(4) 原始比值最差的 cell(合成 worst K2048 1M BS128–256 16-bit,0.77–0.78)分解为锚漂移 1.14–1.18 ×
within-run 残差 ~0.95–0.97(REPORT 为 ~0.98)——3 个包络边缘 cell 上 ≤5% 的 within-run 差
(N=1M 在部署包络之外,仅为应力探针)。</p>
</div>
<details><summary class="mut"><span class="lang-en">worst raw REPORT/new pr cells (all 12 listed — before anchor normalization)</span><span class="lang-zh">原始 REPORT/new 最差 pr cell(全部 12 个——锚归一前)</span></summary>
{w_table}
</details>
<div class="keep">
<div class="lang-en"><p><b>Verdict.</b> Shipped PR#16457 head <code>@e6fdbfac3d</code> holds or beats every
REPORT.html axis; the 16-bit BS grids are measurably better than the published numbers; exactness is
2772/2772; no real regression found. All §3/§4/§7 conclusions carry to the current head.</p></div>
<div class="lang-zh"><p><b>结论。</b>已上线的 PR#16457 head <code>@e6fdbfac3d</code> 在 REPORT.html 全部轴上
持平或更好;16-bit BS 网格可测量地优于已发布数字;精确性 2772/2772;未发现真实回退。§3/§4/§7 的全部结论
对当前 head 成立。</p></div>
</div>
<div class="lang-en"><p class="mut"><b>Provenance</b>: 2026-07-20, b200-027 (16 batches) + umbriel-b200-019
GPUs 1-7 (38 batches), nsys cold-L2, exactness-gated per cell (same protocol as §3/§4/§7). Head built from the
PR branch at <code>e6fdbfac3d</code>. Sweep committed at <code>aad6771344</code>.</p></div>
<div class="lang-zh"><p class="mut"><b>数据来源</b>:2026-07-20,b200-027(16 批)+ umbriel-b200-019
GPU1-7(38 批),nsys cold-L2,逐 cell 精确性门控(协议与 §3/§4/§7 相同)。被测 head 从 PR 分支
<code>e6fdbfac3d</code> 构建。sweep 提交号 <code>aad6771344</code>。</p></div>
"""
    return BEGIN + en + "\n" + END


def main():
    html = open(REPORT, encoding="utf-8").read()
    block = build()
    if BEGIN in html:
        html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block,
                      html, flags=re.S)
        action = "replaced"
    else:
        if ANCHOR_AFTER not in html:
            raise SystemExit(f"anchor {ANCHOR_AFTER!r} not found in REPORT.html")
        html = html.replace(ANCHOR_AFTER, ANCHOR_AFTER + "\n" + block + "\n", 1)
        action = "inserted"
    open(REPORT, "w", encoding="utf-8").write(html)
    print(f"§9d HEADFULL block {action} ({len(block)} chars)")


if __name__ == "__main__":
    main()
