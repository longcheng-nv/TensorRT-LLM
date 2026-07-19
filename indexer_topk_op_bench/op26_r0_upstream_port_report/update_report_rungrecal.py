#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Idempotent injector: add §9c "Rung qfracs recalibration study" to REPORT.html.

Syncs the conclusions of the decode-capture study
  E2E_exp/indexer_decode_capture/  (report §5c-CCDF / §5c-CCDF-b)
into this op26 PR report: the R0 ladder's shipped q=(0.85, 0.35) validated /
recalibrated on random-token AND real SWE-bench captures via full admission
replay (analysis/stats_rung_table_eval.json + stats_rung_table_real.json).

Marker-delimited (RUNGRECAL:BEGIN/END), safe to re-run; inserted before §10.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "REPORT.html")
ANA = ("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/E2E_exp/"
       "indexer_decode_capture/analysis")
BEGIN = "<!-- RUNGRECAL:BEGIN (update_report_rungrecal.py) -->"
END = "<!-- RUNGRECAL:END -->"


def f3(x): return f"{x:.3f}"


def pct(x): return f"{100*x:.0f}%"


def build():
    ev = json.load(open(os.path.join(ANA, "stats_rung_table_eval.json")))["results"]
    rr = json.load(open(os.path.join(ANA, "stats_rung_table_real.json")))["results"]
    MN = {"flash": "Flash (K=512)", "pro": "Pro (K=1024)", "v32": "V3.2* (K=2048)"}
    CEIL_PL = {"flash": "~3%", "pro": "~11%", "v32": "~17%"}
    CEIL_G = {"flash": "~3%", "pro": "~13%", "v32": "~29%"}

    # table A: random-token envelope (ISL>=32K)
    ta = ["<table><tr><th>random-token sweep, envelope ISL≥32K — E[extra P2 passes]</th>"
          "<th>G ship (0.85,0.35)</th><th>Gbest (retuned global)</th>"
          "<th>PL (per-layer table)</th><th>PLN</th><th>PL cut vs G</th>"
          "<th>kernel ceiling</th></tr>"]
    for m in ("flash", "pro", "v32"):
        e = ev[m]["env"]
        cut = 1 - e["PL"]["extra"] / e["G"]["extra"]
        ta.append(f"<tr><td><b>{MN[m]}</b></td><td>{f3(e['G']['extra'])}</td>"
                  f"<td>{f3(e['Gbest']['extra'])}</td><td><b>{f3(e['PL']['extra'])}</b></td>"
                  f"<td>{f3(e['PLN']['extra'])}</td><td>−{pct(cut)}</td><td>{CEIL_PL[m]}</td></tr>")
    ta.append("</table>")

    # table B: real SWE captures
    tb = ["<table><tr><th>real SWE captures (odd-step eval) — E[extra] (ok%)</th>"
          "<th>G ship</th><th>Gbest_r (rand-calib global)</th>"
          "<th>PL_rand (rand-calib table)</th><th>PL_real (real-calib table)</th>"
          "<th>ORACLE</th><th>same-pair layers</th></tr>"]
    CAPN = {"flash": "Flash 64K+100K", "pro": "Pro 32K+64K", "v32": "V3.2* 64K"}
    for m in ("flash", "pro", "v32"):
        r = rr[m]; taag = r["table_agreement"]
        row = [f"<tr><td><b>{CAPN[m]}</b></td>"]
        for k in ("G", "Gbest_r", "PL_rand", "PL_real", "ORACLE"):
            d = r[k]
            b = ("<b>", "</b>") if k == "PL_real" else ("", "")
            row.append(f"<td>{b[0]}{f3(d['extra'])} ({pct(d['ok'])}){b[1]}</td>")
        row.append(f"<td>{taag['same_pair']}/{taag['n_layers']}</td></tr>")
        tb.append("".join(row))
    tb.append("</table>")

    gb = {m: ev[m]["gbest_pair"] for m in ("flash", "pro", "v32")}

    en = f"""
<h2 id="sec-rungrecal">9c · Rung qfracs recalibration study — retune the global pair first; per-layer table second / rung 参数重标研究</h2>
<div class="lang-en">
<p><b>What.</b> The R0 ladder's shipped rung quantiles q=(0.85, 0.35) were validated by a full admission
REPLAY on captured data (hint = logits[preIdx], 256-bin kernel-path rungs, full-row counts on a 14-q grid;
cost proxy E[extra P2 full-row passes] = P(bracket)·1 + P(fallback)·2; calibrate on even steps, evaluate on
odd). Data: the 9-ISL random-token decode-capture sweep (865 layer-cells) and the 2026-05-20 REAL SWE-bench
captures (Flash 64K/100K, Pro 32K/64K native preIdx; V3.2 64K reconstructed). Full study:
<code>E2E_exp/indexer_decode_capture/</code> report §5c-CCDF / §5c-CCDF-b, scripts
<code>src/rung_table_eval.py</code> + <code>src/rung_table_real.py</code>.</p>
{''.join(ta)}
{''.join(tb)}
<p><b>Findings.</b></p>
<ol>
<li><b>The shipped pair is a bug-grade mismatch on real V3.2</b>: (0.85, 0.35) lands in the bracket state on
<b>85.7% of real V3.2-64K steps</b> (one extra falsi pass nearly every step, E[extra]=0.857; Pro SWE-32K
41.8%). This is the same fat-admission family as §9's flash-L22 finding, but on the K2048 arm it is chronic.
Switching to (0.8, 0.5) fixes it (0.023).</li>
<li><b>Retune the GLOBAL pair first.</b> Train-optimal global pairs are Flash {tuple(gb['flash'])} /
Pro {tuple(gb['pro'])} / V3.2 {tuple(gb['v32'])} — all higher-q than shipped, consistent with the measured
hint-rank of the true threshold (median r*/K 0.5–0.8). On real data a retuned global pair already reaches
E[extra] 0.119 / 0.023 / 0.023 (vs shipped 0.189 / 0.327 / 0.857) — kernel-latency ceilings vs ship
≈ Flash {CEIL_G['flash']} / Pro {CEIL_G['pro']} / V3.2 {CEIL_G['v32']} (memory-bound, kernel ≈ 2+E
full-row-read equivalents at large N).</li>
<li><b>A static per-layer table is the second-order lever.</b> Layer id is known at inference; qneeds become
a tiny per-layer constant lookup (no extra rungs / SMEM / registers — a different failure class from the
silicon-falsified uh4 M=4). On the random sweep it beats the global retune clearly in the envelope
(PL cut −41% / −74% / −86% vs ship); on real data its net over the retuned global shrinks to ~0.3–2.3%
kernel ceiling (real cross-layer r* spread is tighter) — maybe worth it on Flash only.</li>
<li><b>Calibrate on real content.</b> Specific pair values do NOT transfer (random vs real tables agree on
4/21 / 10/30 / 0/7 layers; the random per-layer table is WORSE than its own global pair on real V3.2,
0.297 vs 0.023). Random data only gets the direction right (go higher q).</li>
</ol>
<p><b>Caveats.</b> ① The replayed "G" is the pre-vseed ladder — the shipped HEAD's r0_vseed (pmean virtual
rung) already recovers part of these misses, so true headroom sits between the G and Gbest rows.
② Admission ≠ latency (the uh4 lesson): falsi/fallback passes may be partially hidden by PDL/L2 — final
sign-off needs a silicon nsys cold-L2 A/B on this PR branch. ③ Real captures are single-ISL points
(32–100K); long-ISL layer-structure gains unverified. ④ fallback=2 passes is an assumption.</p>
<p><b>Silicon A/B verdict (2026-07-19, b200-027, shipped head eae374554c via launch(r0_qfracs=…), nsys
cold-L2, fp32 BS=1 seq-len scan, 3 arms paired per cell, 77 cells all exact).</b> Arms: <code>ship</code>
defaults; <code>qr2</code> = real-calibrated pairs (K512 (0.9,0.5) / K1024 (0.95,0.6) / K2048 (0.6,0.35));
<code>qr1</code> = column-count-preserving variant (K512 (0.9,) / K1024 (0.95,) / K2048 = qr2, serving as a
duplicate noise control — its two arms agreed within 0.1%).</p>
<table><tr><th>geomean ship/new (&gt;1 = new faster)</th><th>qr2</th><th>qr1</th></tr>
<tr><td>real Flash / Pro / V3.2</td><td>0.965 / 0.987 / <b>1.023</b></td><td>0.999 / 0.985 / <b>1.022</b></td></tr>
<tr><td>synth best K512 / K1024 / K2048</td><td>0.940 / 0.993 / <b>1.096</b></td><td>0.980 / 0.969 / <b>1.093</b></td></tr>
<tr><td>synth worst K512 / K1024 / K2048</td><td>0.975 / 0.963 / 0.994</td><td>1.010 / 1.007 / 0.995</td></tr>
<tr><td>ALL (77 cells)</td><td>0.990</td><td>1.005</td></tr></table>
<p><b>Outcome — the uh4 lesson repeats: admission ≠ latency.</b> (1) <b>V4 (K512/K1024): DO NOT change</b> —
qr2's extra explicit count column costs a real 3–7% (real Flash gm −3.5%) and even the column-preserving qr1
is a wash (the falsi/fallback passes the admission model charges for are largely hidden by PDL / warm-L2 in
wall-clock; the naive memory-bound ceiling over-translated by ~10×). (2) <b>K2048: (0.85,0.35)→(0.6,0.35) is
a real but modest win</b> — real V3.2 gm <b>+2.2%</b> (8k +11.6%, no loser), synth-best gm <b>+9.6%</b>
(65–262K up to +18%, 1M +14%), synth-worst gm −0.6% (deepest single cell −2.4% @262k). Candidate one-line
per-K default change; before PR: re-run the −2.4% worst cell (noise check) + 16-bit + BS-axis spot checks.
Harness/CSV: <code>qfracs_ab/</code>.</p>
<p><b>Pre-PR validation (2026-07-19, all three gates PASSED — change committed to the PR branch as
2d7ad4d019).</b> ① <i>Worst-cell noise re-check</i>: the −2.4% @262k did NOT reproduce (4 independent runs:
0.976/1.020/1.014/0.935 — scattered both ways; whole adverse batch 4-run geomean <b>1.007 = wash</b>;
single-cell excursions in the adverse scenario span 0.64–1.51 across runs, i.e. noise-floor-dominated — per
the stated priority, no further pair tuning attempted for the adverse axis). ② <i>16-bit</i>: wins carry —
synth-best bf16 <b>+9.4%</b> / fp16 <b>+9.9%</b> (1M to +17.5%), real V3.2 bf16 <b>+2.8%</b> / fp16
<b>+2.8%</b>, synth-worst bf16 +1.0% / fp16 +0.4% (wash-positive), all exact. ③ <i>BS axis</i>: real V3.2
128K/…/4K–256K × BS 1–1024 geomean <b>+2.2%</b> with the 8K rung at <b>+10–13% at every BS</b> and no loser
cell; synth-best full 8N×11BS grid <b>+10.9%</b> (BS-invariant), synth-worst grid <b>1.000</b> dead wash;
352 grid cells, 0 errors, 0 inexact. Final shipped semantics: <code>K2048 (vseed): (0.85,0.35)→(0.6,0.35)</code>;
K512/K1024 and the pre-vseed fallback unchanged.</p>
<p class="note"><b>Remaining follow-up</b>: per-layer qneeds table demoted further by this result (its extra
admission gain over a retuned global pair translated to ~nothing on V4); the K2048 pair swap is the only
piece that survived silicon — now validated on all three gates and committed (2d7ad4d019, push = maintainer's
call).</p>
</div>
<div class="lang-zh">
<p><b>做了什么。</b>用捕获数据对 R0 阶梯出厂分位 q=(0.85, 0.35) 做全量 admission <b>复演</b>(hint =
logits[preIdx],kernel 路径 256-bin rung,14 档 q 网格全行计数;代价代理 E[额外 P2 全行遍数] =
P(bracket)·1 + P(fallback)·2;偶数步标定、奇数步评估)。数据 = 9 档 ISL 随机 token 解码捕获(865 层单元)
+ 2026-05-20 <b>真实 SWE-bench 捕获</b>(Flash 64K/100K、Pro 32K/64K 原生 preIdx;V3.2 64K 重建)。完整研究见
<code>E2E_exp/indexer_decode_capture/</code> 报告 §5c-CCDF / §5c-CCDF-b,脚本
<code>src/rung_table_eval.py</code> + <code>src/rung_table_real.py</code>。</p>
<p><b>结论。</b></p>
<ol>
<li><b>出厂对在真实 V3.2 上是 bug 级失配</b>:(0.85, 0.35) 在真实 V3.2-64K 上 <b>85.7% 的步落入 bracket</b>
(几乎每步多一遍 falsi 补射,E[extra]=0.857;Pro SWE-32K 41.8%)。与 §9 的 flash-L22 fat-admission 同族,
但在 K2048 臂上是常态。换 (0.8, 0.5) 即修复(0.023)。</li>
<li><b>第一优先:重调全局对。</b>训练最优全局对 Flash {tuple(gb['flash'])} / Pro {tuple(gb['pro'])} /
V3.2 {tuple(gb['v32'])} —— 均比出厂偏高 q,与实测真实阈值的 hint-rank(r*/K 中位 0.5–0.8)一致。真实数据上
重调后的全局对即达 E[extra] 0.119 / 0.023 / 0.023(出厂 0.189 / 0.327 / 0.857)—— 对出厂的 kernel 延迟上限
≈ Flash {CEIL_G['flash']} / Pro {CEIL_G['pro']} / V3.2 {CEIL_G['v32']}(大 N 内存受限,kernel ≈ 2+E 次全行等效读)。</li>
<li><b>静态 per-layer 表是二阶杠杆。</b>layer id 推理时已知;qneeds 改为按层查小常量表即可(不加 rung、
不加 SMEM/寄存器 —— 与硅上证伪的 uh4 M=4 不同类)。随机 sweep 包络内 per-layer 明显优于调全局
(对出厂减遍 −41% / −74% / −86%);但真实数据上其相对重调全局的净增量缩到 kernel 上限 ~0.3–2.3%
(真实数据层间 r* 分布更收敛)—— 或许只在 Flash 上值得。</li>
<li><b>必须用真实语料标定。</b>具体表值不可跨数据迁移(随机表与真实表同对的层仅 4/21 / 10/30 / 0/7;
随机 per-layer 表在真实 V3.2 上反而劣于其自身全局对,0.297 vs 0.023)。随机数据只把「方向偏高 q」标对了。</li>
</ol>
<p><b>Caveat。</b>① 复演的「G」是 vseed 之前的阶梯 —— 已上线 HEAD 的 r0_vseed(pmean 虚拟阶梯)已回收部分
miss,真实余量介于 G 行与 Gbest 行之间。② admission ≠ latency(uh4 教训):falsi/fallback 遍可能被 PDL/L2
部分隐藏 —— 最终判定需在本 PR 分支上做 nsys 冷 L2 A/B。③ 真实捕获仅 32–100K 单点 ISL,长 ISL 层结构增益未验证。
④ fallback=2 遍为假设值。</p>
<p><b>硅上 A/B 判定(2026-07-19,b200-027,shipped head eae374554c 经 launch(r0_qfracs=…),nsys 冷 L2,
fp32 BS=1 序列长扫描,逐 cell 三臂配对,77 cell 全精确)。</b>臂:<code>ship</code> 默认;<code>qr2</code> =
真实标定对(K512 (0.9,0.5) / K1024 (0.95,0.6) / K2048 (0.6,0.35));<code>qr1</code> = 保列数变体
(K512 (0.9,) / K1024 (0.95,) / K2048 同 qr2,充当重复噪声对照——两臂互差 ≤0.1%)。</p>
<table><tr><th>几何均值 ship/new(&gt;1 = 新参更快)</th><th>qr2</th><th>qr1</th></tr>
<tr><td>real Flash / Pro / V3.2</td><td>0.965 / 0.987 / <b>1.023</b></td><td>0.999 / 0.985 / <b>1.022</b></td></tr>
<tr><td>synth best K512 / K1024 / K2048</td><td>0.940 / 0.993 / <b>1.096</b></td><td>0.980 / 0.969 / <b>1.093</b></td></tr>
<tr><td>synth worst K512 / K1024 / K2048</td><td>0.975 / 0.963 / 0.994</td><td>1.010 / 1.007 / 0.995</td></tr>
<tr><td>全部 77 cell</td><td>0.990</td><td>1.005</td></tr></table>
<p><b>结论 —— uh4 教训重演:admission ≠ latency。</b>(1) <b>V4(K512/K1024):不换</b> —— qr2 多一列显式
计数实付 3–7%(real Flash gm −3.5%),保列数的 qr1 也只是 wash(admission 模型记账的 falsi/fallback 遍在
wall-clock 里大部分被 PDL/热 L2 掩盖;朴素内存受限上限高估约 10×)。(2) <b>K2048:(0.85,0.35)→(0.6,0.35)
是真实但温和的赢</b> —— real V3.2 gm <b>+2.2%</b>(8k +11.6%,无输点),synth-best gm <b>+9.6%</b>
(65–262K 最高 +18%,1M +14%),synth-worst gm −0.6%(最深单点 −2.4% @262k)。候选一行 per-K 默认值改动;
进 PR 前需:复测该 −2.4% 单点(噪声排查)+ 16-bit + BS 轴抽查。Harness/CSV:<code>qfracs_ab/</code>。</p>
<p><b>进 PR 前验证(2026-07-19,三道门全部通过 —— 改动已提交 PR 分支 2d7ad4d019)。</b>①
<i>worst 单点噪声复测</i>:−2.4% @262k <b>不复现</b>(4 次独立 run:0.976/1.020/1.014/0.935,双向散布;
整个逆风批 4-run 几何均值 <b>1.007 = wash</b>;逆风场景单 cell 跨 run 波动 0.64–1.51,噪声地板主导 ——
按既定优先级,不再为逆风轴调替代对)。② <i>16-bit</i>:收益保持 —— synth-best bf16 <b>+9.4%</b> /
fp16 <b>+9.9%</b>(1M 至 +17.5%),real V3.2 bf16 <b>+2.8%</b> / fp16 <b>+2.8%</b>,synth-worst bf16 +1.0% /
fp16 +0.4%(wash 偏正),全部精确。③ <i>BS 轴</i>:real V3.2 全 ISL × BS 1–1024 几何均值 <b>+2.2%</b>,
8K 档在<b>每个 BS 上 +10–13%</b> 且无输点;synth-best 全 8N×11BS 网格 <b>+10.9%</b>(BS 不变),
synth-worst 网格 <b>1.000</b> 纯 wash;352 个网格 cell,0 错误,0 不精确。最终上线语义:
<code>K2048(vseed):(0.85,0.35)→(0.6,0.35)</code>;K512/K1024 与 pre-vseed 回退不变。</p>
<p class="note"><b>剩余后续</b>:per-layer qneeds 表被本结果进一步降级(其相对重调全局对的额外 admission
增益在 V4 上折算 ≈ 零);唯一过硅的是 K2048 换对 —— 现已通过全部三道验证并提交(2d7ad4d019,推送与否由
维护者决定)。</p>
</div>
"""
    return BEGIN + en + END


def main():
    html = open(REPORT, encoding="utf-8").read()
    block = build()
    if BEGIN in html:
        html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block,
                      html, flags=re.S)
        print("[rungrecal] replaced existing §9c block")
    else:
        anchor = "<h2>10 ·"
        i = html.find(anchor)
        if i < 0:
            sys.exit("anchor <h2>10 · not found")
        html = html[:i] + block + "\n\n" + html[i:]
        print("[rungrecal] inserted §9c before §10")
    open(REPORT, "w", encoding="utf-8").write(html)
    print(f"[rungrecal] wrote {REPORT} ({os.path.getsize(REPORT)//1024} KB)")


if __name__ == "__main__":
    main()
