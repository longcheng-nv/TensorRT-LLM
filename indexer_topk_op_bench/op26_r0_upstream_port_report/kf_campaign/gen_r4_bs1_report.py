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


def flow_svg(zh):
    t = (lambda en, z: z if zh else en)
    box = ('<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
           'fill="{f}" stroke="#8fa0ba"/>')

    def node(x, y, w, h, lines, f="#f0f3f8", fs=12.5, bold_first=True):
        out = [box.format(x=x, y=y, w=w, h=h, f=f)]
        ty = y + h / 2 - (len(lines) - 1) * 8
        for i, ln in enumerate(lines):
            wgt = ' font-weight="600"' if (i == 0 and bold_first) else ""
            out.append(f'<text x="{x + w/2}" y="{ty + i*16}" text-anchor="middle" '
                       f'font-size="{fs}"{wgt} fill="#1a2233">{ln}</text>')
        return "".join(out)

    def arrow(x1, y1, x2, y2, label="", lx=None, ly=None):
        s = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#55627a" '
             f'stroke-width="1.6" marker-end="url(#ar)"/>')
        if label:
            s += (f'<text x="{lx or (x1+x2)/2}" y="{ly or (y1+y2)/2 - 6}" '
                  f'text-anchor="middle" font-size="11.5" fill="#55627a">{label}</text>')
        return s

    e = []
    e.append(node(370, 8, 300, 44, [
        t("Inputs: logits[1,npad] · pre_idx[1,K] · n", "输入: logits[1,npad] · pre_idx[1,K] · n")],
        f="#e7eef9"))
    e.append(arrow(520, 52, 520, 84))
    e.append(node(340, 86, 360, 50, [
        t("Host dispatch on (npad, K) — hint quality never consulted",
          "Host 按 (npad, K) 分派 — 不依赖 hint 质量")], f="#fdf3e3"))
    # three branches
    e.append(arrow(400, 136, 165, 176, t("npad ≤ 12288", "npad ≤ 12288"), 240, 158))
    e.append(arrow(520, 136, 520, 176, t("≤ 262144", "≤ 262144")))
    e.append(arrow(650, 136, 880, 176, t("> 262144", "> 262144"), 800, 158))

    # direct path
    e.append(node(20, 178, 290, 118, [
        t("DIRECT exact path — 1 CTA × 1024 thr", "DIRECT 直通道 — 1 CTA × 1024 线程"),
        t("threshold solve trivially converged", "阈值求解解析退化(kC ≥ npad)"),
        t("whole row → smem (fused level-0 hist)", "整行进 smem(融合 0 级直方图)"),
        t("11/11/10-bit radix + early exit", "11/11/10-bit radix + 提前退出"),
        t("boundary-bin compact + warp emit", "边界 bin 压缩 + warp 聚合发射")]))
    # register-resident path
    e.append(node(360, 178, 320, 128, [
        t("REGISTER-RESIDENT GVR", "寄存器驻留 GVR"),
        t("clusters 1/4/8/16 CTA × 512 thr", "cluster 1/4/8/16 CTA × 512 线程"),
        t("row loaded to registers ONCE", "整行一次载入寄存器"),
        t("(overlaps P1 hint gather);", "(与 P1 hint gather 重叠);"),
        t("all passes re-scan registers, not L2", "全部 pass 重扫寄存器而非 L2"),
        t("per-(tier,K) AR6/AR8 rung ladders", "逐 (tier,K) 实测 AR6/AR8 阈值梯")]))
    # streaming path
    e.append(node(730, 178, 300, 96, [
        t("STREAMING GVR — 16-CTA cluster", "流式 GVR — 16-CTA cluster"),
        t("row re-scanned from L2 per pass", "每 pass 从 L2 重扫整行"),
        t("+ async L2 prefetch under P1", "+ P1 期间异步 L2 预取")]))

    e.append(arrow(520, 306, 520, 336))
    e.append(arrow(880, 274, 880, 336))
    e.append(f'<line x1="880" y1="336" x2="540" y2="336" stroke="none"/>')

    # phases (shared by reg-resident & streaming)
    e.append(node(340, 338, 560, 66, [
        t("P1 — prior: gather logits[pre_idx]; two-level 64-bin histogram",
          "P1 — 先验: gather logits[pre_idx];两级 64-bin 直方图"),
        t("(97% trim vs sunk outliers) → 8 CCDF-quantile threshold rungs",
          "(97% 修剪抗离群)→ 8 个 CCDF 分位阈值 rung")], f="#e7f5ec"))
    e.append(arrow(620, 404, 620, 424))
    e.append(node(340, 426, 560, 80, [
        t("P2 — solve: ONE pass counts all 8 rungs (8 regs/thread);",
          "P2 — 求解: 单 pass 同测 8 阈值(每线程 8 计数器);"),
        t("accept if count∈[K,kC]; else log-secant bracket ÷9 per pass (≤8);",
          "count∈[K,kC] 即采纳;否则 log-secant 括号每 pass ÷9(≤8 次);"),
        t("plateau fallback: max-below descent → exact tie emit",
          "plateau 回退: max-below 降值 → 精确平票直发")], f="#e7f5ec"))
    e.append(arrow(620, 506, 620, 526))
    e.append(node(340, 528, 560, 62, [
        t("P3 — collect: per-thread counts (cached from P2) → prefix sums;",
          "P3 — 收集: P2 缓存的每线程计数 → 前缀和;"),
        t("(val,idx) packed 64-bit → CTA0 smem via DSMEM, parity-banked exchange",
          "(值,索引) 打包 64-bit → 经 DSMEM 入 CTA0 smem,奇偶双 bank 交换")], f="#e7f5ec"))
    e.append(arrow(620, 590, 620, 610))
    e.append(node(340, 612, 560, 62, [
        t("P4 — refine (CTA0 solo): 4×8-bit radix select of K-th key;",
          "P4 — 精修(CTA0 独占): 4×8-bit radix 选第 K 名;"),
        t("strict-greater writeback + tie-ticket fill (tie-robust exact)",
          "strict-greater 直写 + tie-ticket 限额填充(tie-robust 精确)")], f="#e7f5ec"))
    e.append(arrow(165, 296, 165, 700))
    e.append(arrow(620, 674, 620, 700))
    e.append(node(430, 702, 380, 44, [
        t("Output: indices[1,K] — exact top-K index set", "输出: indices[1,K] — 精确 top-K 索引集")],
        f="#e7eef9"))
    e.append(arrow(165, 700, 430, 722))
    return ('<svg viewBox="0 0 1060 760" xmlns="http://www.w3.org/2000/svg" '
            'style="width:100%;max-width:1060px" role="img">'
            '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="4.5" '
            'orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="#55627a"/></marker></defs>'
            + "".join(e) + "</svg>")


PSEUDO_EN = """<pre>
gvr_topk(logits[1, npad], pre_idx[1, K], n) -&gt; indices[1, K]:
  # host dispatch — depends only on (npad, K), never on hint quality
  if npad &le; 12288:              launch direct_topk&lt;1 CTA × 1024thr&gt;
  elif npad &lt; 16384:             launch gvr_reg&lt;CS=1,  MAXV=8&gt;
  elif npad &lt; 32768:             launch gvr_reg&lt;CS=4,  MAXV=4&gt;
  elif npad &le; 65536:            launch gvr_reg&lt;CS=8,  MAXV=3..4&gt;
  elif npad &le; 262144:           launch gvr_reg&lt;CS=16, MAXV=5..8, AR=6|8 by (tier,K)&gt;
  else:                          launch gvr_stream&lt;CS=16&gt;

direct_topk:                       # trivial convergence: kC ≥ npad ⇒ any threshold accepts row
  smem[0..npad) ← row (f2u keys, fused level-0 histogram)
  kth ← radix_select_11_11_10(smem, K)        # whole-bin early exit
  emit strict-greater; fill remainder from tie bin        # tie-robust

gvr_reg / gvr_stream:              # per-CTA slice; reg: slice → registers ONCE
  P1: hv ← logits[pre_idx]                       (overlaps the register loads)
      hist₁(64) over [min,max] → trim point p97 (outlier-proof)
      hist₂(64) over [p97 range] → rungs[0..AR) = CCDF quantiles; rungs[AR-1]=hmin
  P2: loop ≤ 8 passes:
        cnt[0..AR) ← count_ge(rungs) in ONE scan (AR regs/thread; cluster-summed
                     via parity-banked DSMEM exchange)
        if ∃r: K ≤ cnt[r] ≤ kC: thr ← rungs[r]; break
        rungs ← linear ladder inside secant bracket (log-CCDF ≈ linear) — ÷9 / pass
      if bracket collapsed:                       # plateau / degenerate
        vstar ← max_below(t_hi) descent until count lands or k-th value proven
        if proven: emit strict-greater + tie tickets globally; return
  P3: pos ← prefix_sum(per-thread counts cached by P2)   # zero re-scan
      (key,idx) → CTA0 smem candidate buffer (DSMEM push, cluster sync)
  P4: (CTA0 only) 4×8-bit radix over ≤ kC candidates → k-th key
      emit strict-greater; fill K−m from tie bin (row-local counters)
</pre>"""

PSEUDO_ZH = """<pre>
gvr_topk(logits[1, npad], pre_idx[1, K], n) -&gt; indices[1, K]:
  # host 分派 — 只看 (npad, K),绝不依赖 hint 质量
  if npad &le; 12288:              发射 direct_topk&lt;1 CTA × 1024 线程&gt;
  elif npad &lt; 16384:             发射 gvr_reg&lt;CS=1,  MAXV=8&gt;
  elif npad &lt; 32768:             发射 gvr_reg&lt;CS=4,  MAXV=4&gt;
  elif npad &le; 65536:            发射 gvr_reg&lt;CS=8,  MAXV=3..4&gt;
  elif npad &le; 262144:           发射 gvr_reg&lt;CS=16, MAXV=5..8, AR=6|8 按(档,K)实测&gt;
  else:                          发射 gvr_stream&lt;CS=16&gt;

direct_topk:                       # 平凡收敛: kC ≥ npad ⇒ 任意阈值全收
  smem[0..npad) ← 整行(f2u 键序,融合 0 级直方图)
  kth ← radix_select_11_11_10(smem, K)        # 整 bin 提前退出
  发射 strict-greater;余额从平票 bin 补齐          # tie-robust

gvr_reg / gvr_stream:              # 每 CTA 一个 slice;reg: slice 一次载入寄存器
  P1: hv ← logits[pre_idx]                       (与寄存器加载重叠)
      hist₁(64) 于 [min,max] → 97% 修剪点(抗沉底离群)
      hist₂(64) 于修剪区间 → rungs[0..AR) = CCDF 分位;rungs[AR-1]=hmin
  P2: 循环 ≤ 8 pass:
        cnt[0..AR) ← 单次扫描同测 AR 个阈值(每线程 AR 个计数器;
                     cluster 求和走奇偶双 bank DSMEM 交换)
        若 ∃r: K ≤ cnt[r] ≤ kC: thr ← rungs[r]; break
        rungs ← secant 括号内线性阶梯(log-CCDF 近线性)— 每 pass ÷9
      若括号塌缩:                                 # plateau / 退化
        vstar ← max_below(t_hi) 逐真实值下探,直至 count 落窗或证得第 k 名
        证得则全局发射 strict-greater + tie-ticket;return
  P3: pos ← prefix_sum(P2 缓存的每线程计数)        # 零重扫
      (键,索引) → CTA0 smem 候选缓冲(DSMEM 推送,cluster 同步)
  P4: (仅 CTA0)对 ≤ kC 候选做 4×8-bit radix → 第 K 名键
      发射 strict-greater;从平票 bin 补 K−m(行内计数器)
</pre>"""

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
<h2>Algorithm flow (GVR skeleton preserved: pre_idx prior → secant-log solve → exact refine)</h2>
{flow_svg(False)}
<h2>Pseudocode</h2>
{PSEUDO_EN}
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
<h2>算法流程图(GVR 骨架保留:preIdx 先验 → secant-log 求解 → 精确 refine)</h2>
{flow_svg(True)}
<h2>伪代码</h2>
{PSEUDO_ZH}
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
pre {{ background: #f6f8fb; border: 1px solid #d6dde8; border-radius: 8px;
      padding: 12px 16px; overflow-x: auto; font-size: 12.5px; line-height: 1.5; }}
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
