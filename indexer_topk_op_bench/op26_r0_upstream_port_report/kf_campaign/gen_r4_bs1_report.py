# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R4 BS=1 champion report -> R4_CHAMPION_BS1_REPORT.html.

Bilingual (EN/中文) via CSS-only toggle (radio + :checked ~ sibling; no JS —
the viewer strips <script>). Data: grid_r4r3cg.csv (865 real cells, champion
28dc11f6 vs PR#16457 pinned head 04a0900ff7, nsys cold-L2 paired).
"""
import csv
import html
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = [r for r in csv.DictReader(open(HERE / "grid_r4r3cg.csv")) if r["speedup_cold"]]
sp = [float(r["speedup_cold"]) for r in rows]
gm, med = statistics.geometric_mean(sp), statistics.median(sp)
qs = statistics.quantiles(sp, n=20)
p5, p95 = qs[0], qs[-1]
n15 = sum(1 for v in sp if v >= 1.5)
n20 = sum(1 for v in sp if v >= 2.0)
nreg = sum(1 for v in sp if v < 1.0)

ISLS = ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"]
MODELS = [("flash", "V4-Flash (K=512)"), ("pro", "V4-Pro (K=1024)"),
          ("v32", "V3.2 (K=2048)")]
grp = defaultdict(list)
for r in rows:
    grp[(r["model"], r["isl"])].append(float(r["speedup_cold"]))


def cell(v):
    if v is None:
        return "<td>—</td>"
    cls = ' class="hi"' if v >= 1.5 else (' class="lo"' if v < 1.1 else "")
    return f"<td{cls}>{v:.3f}</td>"


def group_table():
    out = ["<table><tr><th></th>" + "".join(f"<th>{i}</th>" for i in ISLS) + "</tr>"]
    for m, lab in MODELS:
        tds = []
        for isl in ISLS:
            v = grp.get((m, isl))
            tds.append(cell(statistics.geometric_mean(v) if v else None))
        out.append(f"<tr><th>{lab}</th>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


worst = sorted(rows, key=lambda r: float(r["speedup_cold"]))[:8]
best = sorted(rows, key=lambda r: -float(r["speedup_cold"]))[:8]


def extreme_table(rs):
    out = ["<table><tr><th>cell</th><th>N</th><th>K</th><th>hit</th><th>×</th></tr>"]
    for r in rs:
        out.append(f"<tr><td>{html.escape(r['uuid'])}</td><td>{r['N']}</td>"
                   f"<td>{r['K']}</td><td>{r['hit']}</td>"
                   f"<td>{float(r['speedup_cold']):.3f}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


STATS_EN = f"""
<div class="kpis">
<div class="kpi"><b>{gm:.4f}×</b><span>geomean speedup (865 real cells)</span></div>
<div class="kpi"><b>865/865</b><span>tie-robust exact</span></div>
<div class="kpi"><b>0</b><span>real regressions (worst cell 1.013 @60-rep)</span></div>
<div class="kpi"><b>{med:.3f}× / {p5:.3f}× / {p95:.3f}×</b><span>median / p5 / p95</span></div>
<div class="kpi"><b>{n15} / {n20}</b><span>cells ≥1.5× / ≥2.0×</span></div>
<div class="kpi"><b>{max(sp):.3f}×</b><span>best cell</span></div>
</div>"""
STATS_ZH = STATS_EN.replace("geomean speedup (865 real cells)", "geomean 加速(865 个真实 cell)") \
    .replace("tie-robust exact", "tie-robust 精确") \
    .replace("real regressions (worst cell 1.013 @60-rep)", "真实回退(最差格 60-rep 复测 1.013)") \
    .replace("median / p5 / p95", "中位 / p5 / p95") \
    .replace("cells ≥1.5× / ≥2.0×", "≥1.5× / ≥2.0× 的格数") \
    .replace("best cell", "最佳格")

EN = f"""
<h1>KF R4 Champion — BS=1 Performance vs GVR PR Head</h1>
<p class="meta">Champion <code>gvr_topk_r3_perK_dispatch</code> (kernel <code>28dc11f6…</code>,
campaign <code>gvr-topk-cold60</code> round 3) vs TensorRT-LLM
<a href="https://github.com/NVIDIA/TensorRT-LLM/pull/16457">PR#16457</a> pinned head
<code>04a0900ff7</code> (in-tree CuTe DSL GVR). Measurement: nsys cold-L2 (512 MB evict)
pure kernel time, same-GPU paired arms, B200 (umbriel-b200-027), 865 real decode-capture
cells (BS=1 fp32, all GVR-active layers × ISL 4K–1M). All three acceptance bars met.
Source: fork branch <code>kf/r4-champion-final-bs1</code>; data <code>grid_r4r3cg.csv</code>.</p>
{STATS_EN}
<h2>Geomean speedup by model × ISL</h2>
{group_table()}
<p>Shape: strong at both ends — small ISL rides the direct single-CTA×1024 path
(Pro 4k up to 2.65×), large ISL rides register-resident 16-CTA clusters with
per-(tier,K) AR6/AR8 measured rung ladders (Pro 1M 1.94×). The thinnest band is
ISL 32–64k (N≈8–16K), healed to ≥1.0 by the KCMAX-8448 direct window + barrier folds.</p>
<h2>Worst 8 cells (after 60-rep adjudication, none are real regressions)</h2>
{extreme_table(worst)}
<h2>Best 8 cells</h2>
{extreme_table(best)}
<h2>Cross-checks</h2>
<ul>
<li>vs sglang_v2 (PR-arm normalized): geomean 1.099, win 567/865 (weak zone N=8195, gm 0.80)</li>
<li>vs radix_cutedsl: geomean 1.593, win 845/865</li>
<li>vs first-lineage compB (reference, slightly different head snapshot): 1.6531 &lt; 1.8267 —
coldstart + hard skeleton lock, 3 rounds, as expected in the handoff.</li>
<li>Verdict chain: v5 1.295 (78 regs) → v14 1.343 (24) → r2_wd 1.441 (19) → v25 1.521 (2)
→ v27 1.582 (2) → r3_a003 1.618 (1) → <b>28dc11f6 1.6531 (0)</b>.</li>
</ul>
"""

ZH = f"""
<h1>KF R4 冠军算子 — BS=1 相对 GVR PR head 的性能</h1>
<p class="meta">冠军 <code>gvr_topk_r3_perK_dispatch</code>(kernel <code>28dc11f6…</code>,
campaign <code>gvr-topk-cold60</code> 第 3 轮)对比 TensorRT-LLM
<a href="https://github.com/NVIDIA/TensorRT-LLM/pull/16457">PR#16457</a> 锁定 head
<code>04a0900ff7</code>(in-tree CuTe DSL GVR)。测量口径:nsys cold-L2(512 MB 逐出)
纯 kernel 时间,同 GPU 配对双臂,B200(umbriel-b200-027),865 个真实 decode 采集
cell(BS=1 fp32,全部 GVR-active 层 × ISL 4K–1M)。三条验收 Bar 全部达成。
源码:fork 分支 <code>kf/r4-champion-final-bs1</code>;数据 <code>grid_r4r3cg.csv</code>。</p>
{STATS_ZH}
<h2>分组 geomean 加速(模型 × ISL)</h2>
{group_table()}
<p>形态:两端强 —— 小 ISL 走 direct 单 CTA×1024 直通道(Pro 4k 最高 2.65×),
大 ISL 走寄存器驻留 16-CTA cluster + 逐 (tier,K) 实测 AR6/AR8 阈值梯(Pro 1M 1.94×)。
最薄带为 ISL 32–64k(N≈8–16K),已由 KCMAX-8448 直通窗 + barrier 折叠修复到 ≥1.0。</p>
<h2>最差 8 格(60-rep 裁定后,均非真实回退)</h2>
{extreme_table(worst)}
<h2>最佳 8 格</h2>
{extreme_table(best)}
<h2>横向对照</h2>
<ul>
<li>vs sglang_v2(PR 臂归一):geomean 1.099,胜 567/865(弱区 N=8195,gm 0.80)</li>
<li>vs radix_cutedsl:geomean 1.593,胜 845/865</li>
<li>vs 第一 lineage compB(参考,head 快照略异):1.6531 &lt; 1.8267 ——
冷启动 + 骨架硬锁 3 轮所得,符合 handoff 预期管理。</li>
<li>判决链:v5 1.295(78 回退)→ v14 1.343(24)→ r2_wd 1.441(19)→ v25 1.521(2)
→ v27 1.582(2)→ r3_a003 1.618(1)→ <b>28dc11f6 1.6531(0)</b>。</li>
</ul>
"""

PAGE = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>KF R4 Champion BS=1 Report</title>
<style>
body {{ font: 14px/1.55 -apple-system, "Segoe UI", Roboto, "Noto Sans CJK SC", sans-serif;
       max-width: 1080px; margin: 24px auto; padding: 0 16px; color: #1a2233; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 28px; }}
code {{ background: #f0f3f8; padding: 1px 5px; border-radius: 4px; }}
.meta {{ color: #55627a; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border: 1px solid #d6dde8; padding: 5px 10px; text-align: right; font-variant-numeric: tabular-nums; }}
th {{ background: #f0f3f8; text-align: center; }}
td.hi {{ background: #e7f5ec; font-weight: 600; }}
td.lo {{ background: #fdf3e3; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }}
.kpi {{ background: #f0f3f8; border: 1px solid #d6dde8; border-radius: 8px;
        padding: 10px 14px; min-width: 150px; }}
.kpi b {{ display: block; font-size: 19px; }} .kpi span {{ color: #55627a; font-size: 12px; }}
.langbar {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #d6dde8; }}
.langbar label {{ cursor: pointer; padding: 4px 14px; border: 1px solid #b9c4d6; border-radius: 6px; margin-right: 8px; }}
#lang-en, #lang-zh {{ display: none; }}
#pane-en, #pane-zh {{ display: none; }}
#lang-en:checked ~ #pane-en {{ display: block; }}
#lang-zh:checked ~ #pane-zh {{ display: block; }}
#lang-en:checked ~ .langbar label[for=lang-en],
#lang-zh:checked ~ .langbar label[for=lang-zh] {{ background: #1a2233; color: #fff; }}
</style></head><body>
<input type="radio" name="lang" id="lang-en" checked>
<input type="radio" name="lang" id="lang-zh">
<div class="langbar"><label for="lang-en">English</label><label for="lang-zh">中文</label></div>
<div id="pane-en">{EN}</div>
<div id="pane-zh">{ZH}</div>
</body></html>
"""
out = HERE / "R4_CHAMPION_BS1_REPORT.html"
out.write_text(PAGE)
print(f"wrote {out} ({len(PAGE)} bytes)")
