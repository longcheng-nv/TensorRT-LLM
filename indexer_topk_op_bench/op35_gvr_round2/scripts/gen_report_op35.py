# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op35 final bilingual REPORT.html generator (CSS-only zh/en toggle, no JS).

Self-contained last-writer: re-derives every table from the raw results CSVs
(results/replay_b1.csv, nsys_oracle_decomp.csv, nsys_ab_verdict.csv) at each
run. Numbers in prose blocks are filled from the same CSVs where possible.
"""
import csv
import html
import math
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OP35 = _HERE.parent
R = _OP35 / "results"


def rd(name):
    p = R / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def gm(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    h = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"
    b = ""
    for r in rows:
        b += "<tr>" + "".join(
            f"<td>{html.escape(str(fmt.get(c, lambda v: v)(r.get(c, ''))))}</td>" for c in cols
        ) + "</tr>"
    return f"<table>{h}{b}</table>"


def main():
    verdict = rd("nsys_ab_verdict.csv")
    decomp = rd("nsys_oracle_decomp.csv")
    replay = rd("replay_b1.csv")

    vd = {r["cell"]: r for r in verdict}
    axes = {}
    for name, pred in (("synth-best", lambda c: c.startswith("synth_best")),
                       ("synth-worst", lambda c: c.startswith("synth_worst")),
                       ("real (3-model)", lambda c: c.startswith("real")),
                       ("ALL 77", lambda c: True)):
        sub = [float(r["ratio_med"]) for r in verdict if pred(r["cell"])]
        axes[name] = (gm(sub), min(sub, default=float("nan")), max(sub, default=float("nan")), len(sub))
    k2048 = [float(r["ratio_med"]) for r in verdict
             if "K2048" in r["cell"] or r["cell"].startswith("real_v32")]
    lose = [r for r in verdict if float(r["ratio_med"]) < 0.97]

    ax_rows = [dict(axis=k, geomean=f"{v[0]:.4f}", min=f"{v[1]:.3f}",
                    max=f"{v[2]:.3f}", n=v[3]) for k, v in axes.items()]
    ax_tbl = table(ax_rows, ["axis", "geomean", "min", "max", "n"])

    best_rows = sorted(verdict, key=lambda r: -float(r["ratio_med"]))[:10]
    worst_rows = sorted(verdict, key=lambda r: float(r["ratio_med"]))[:6]
    cell_cols = ["cell", "base_us", "var_us", "ratio_med", "ratio_min", "ratio_max"]

    dec_rows = sorted(decomp, key=lambda r: r["cell"])
    dec_cols = ["cell", "base", "fl_pct", "P3_pct", "P4_pct", "mid_pct"]

    zh = en = ""  # filled below via one combined body with lang blocks

    def bil(zh_txt, en_txt, tag="p"):
        return (f'<{tag} class="zh">{zh_txt}</{tag}>'
                f'<{tag} class="en">{en_txt}</{tag}>')

    body = f"""
<h1>{bil("op35 — GVR 第二轮优化战役报告 (post-PR#16457)",
         "op35 — GVR round-2 optimization campaign report (post-PR#16457)", "span")}</h1>
<p class="meta">2026-07-16 · umbriel-b200-081 (8×B200) · baseline = PR#16457 worktree HEAD
eae374554c · metric = op26 REPORT §6 cells (synth 52 + real 25, BS=1 fp32) ·
harness: op35_gvr_round2/scripts/ · verdicts: nsys cold-L2, single-GPU paired, 3 in-process rounds</p>

<h2>{bil("0 · 执行摘要", "0 · Executive summary", "span")}</h2>
{bil("目标 +40% (geomean t(PR)/t(op35) ≥ 1.40)。本轮判定:提案的扫描侧杠杆 (H1/H2 已在 PR 内、"
     "B1 上限 ~1.07、HLS 尾梯被 vseed 取代) 无法达标;nsys 四臂 oracle 将战场改判为 P4blk "
     "(握手#2+P4+写回, 中位 ~37%)。已收割 iter3 束 (skip_h1 + p4_fuse_mmz + kNumBins@K2048→512),"
     "nsys ×3 判决见 §3。+40% 判定为在 GVR 骨架内不可达 (见 §5 定界)。",
     "Target +40% (geomean ≥1.40). This round's finding: the proposal's scan-side levers "
     "cannot reach it (H1/H2 already in the PR; B1 ceiling ~1.07; the HLS tail ladder is "
     "superseded by vseed). The nsys 4-arm oracle re-attributes the battleground to P4blk "
     "(handoff2+P4+writeback, median ~37%). Harvested: the iter3 bundle "
     "(skip_h1 + p4_fuse_mmz + kNumBins@K2048→512) — nsys ×3 verdict in §3. "
     "+40% is assessed INFEASIBLE within the GVR skeleton (§5).")}

<h2>{bil("1 · 提案对照与差距分析", "1 · Proposal vs code-reality gap analysis", "span")}</h2>
{bil("PROPOSAL_GVR_NEXT_OPT.html 的 H1 (log-falsi fallback) 与 H2 (分布式 fallback) 在 PR HEAD "
     "已存在 (fb_fix=True α=0.2; cluster count-merge 全路径);H3 尾梯被 vseed 虚拟档取代 "
     "(实测证伪, 全轴 -3~-5%);H4 native 16-bit 不影响 fp32 指标集;B1 字面整窗跳块 replay 上限≈0。"
     "新增本战役发现的杠杆:P4blk 屏障链、kNumBins 瘦身、warp-窗粒度 B1 (上限 56-79% of P3, "
     "但 P3 仅 0-26%)。",
     "H1 (log-falsi fallback) and H2 (distributed fallback) from the proposal ALREADY exist at "
     "PR HEAD (fb_fix=True α=0.2; cluster count-merge on all paths); H3's tail ladder is "
     "superseded by the vseed virtual rung (falsified here: −3..−5% on every axis); H4 (16-bit) "
     "does not move the fp32 metric set; the literal whole-window B1 has ~zero replay ceiling. "
     "New levers found by this campaign: the P4blk barrier chain, kNumBins diet, and "
     "(warp,window)-granular B1 (56-79% of P3 — but P3 is only 0-26% of the kernel).")}

<h2>{bil("2 · nsys 四臂相位归因 (77 cells)", "2 · nsys 4-arm phase attribution (77 cells)", "span")}</h2>
{bil("臂: base / P3-removed / P4blk-removed / floor (P1+启动+写回)。中位份额: P4blk ~37% "
     "(23-58%), mid (P1b+P2+falsi+握手#1) 17-48%, floor 13-40%, P3 0-26%。"
     "UB(P4blk 归零)=1.578; UB(P3+P4blk 归零)=1.771。NCU: cs8 barrier 停顿 61.4%; "
     "cs1 小 N icache 31% + barrier 26% (纯延迟链)。",
     "Arms: base / P3-removed / P4blk-removed / floor (P1+launch+writeback). Median shares: "
     "P4blk ~37% (23-58%), mid (P1b+P2+falsi+handoff1) 17-48%, floor 13-40%, P3 0-26%. "
     "UB(zero P4blk)=1.578; UB(zero P3+P4blk)=1.771. NCU: cs8 barrier stalls 61.4%; "
     "cs1 small-N icache 31% + barrier 26% (pure latency chain).")}
<details><summary>{bil("逐 cell 分解表", "Per-cell decomposition", "span")}</summary>
{table(dec_rows, dec_cols)}</details>

<h2>{bil("3 · iter3 收割束 — nsys ×3 判决", "3 · The iter3 harvest bundle — nsys ×3 verdict", "span")}</h2>
{bil("束 = skip_h1 (删 Phase2 末冗余 cluster 握手) + p4_fuse_mmz (P4 minmax 遍与 hist 清零融合, "
     "省 2 屏障 + 1 遍) + kNumBins@K2048: 2048→512 (仅 K2048; 全局套用会伤 K1024 大 N)。"
     "正确性: 77/77 tie-aware value-multiset exact (L1) + nsys 判决轮全绿。",
     "Bundle = skip_h1 (drop the redundant end-of-P2 cluster handshake) + p4_fuse_mmz (fuse "
     "P4's min/max pass with the hist zero: −2 barriers, −1 pass) + kNumBins@K2048: 2048→512 "
     "(K2048 only; applying it globally regresses K1024 large-N). Correctness: 77/77 tie-aware "
     "value-multiset exact (L1) + all-green verdict rounds.")}
{ax_tbl}
{bil(f"K2048 域 geomean = {gm(k2048):.4f} (n={len(k2048)}); 回退 >3% 的 cell 数 = {len(lose)}。",
     f"K2048-domain geomean = {gm(k2048):.4f} (n={len(k2048)}); cells regressing >3% = {len(lose)}.")}
<h3>{bil("最优 / 最差 cell", "Best / worst cells", "span")}</h3>
{table(best_rows, cell_cols)}
{table(worst_rows, cell_cols)}

<h2>{bil("4 · 证伪台账 (本战役)", "4 · Falsification ledger (this campaign)", "span")}</h2>
<ul>
<li>{bil("H3 尾梯 qfracs (0.75,0.45,0.048) @K2048: 全轴 0.95-0.97 — vseed 已覆盖尾部内点, 额外列=纯 P2 税。",
        "H3 tail-ladder qfracs @K2048: 0.95-0.97 on every axis — vseed already covers the tail interior; extra rungs are pure P2 tax.")}</li>
<li>{bil("p4_fused_hist (P3 写时建直方图): exact 但 ~-15% — P3 热扫描环污染税 (op21 iter14 同类)。",
        "p4_fused_hist (hist during P3 writes): exact but ~−15% — hot-scan-loop pollution tax (op21 iter14 class).")}</li>
<li>{bil("P4 scatter 同地址原子假设: WASH — SM100 smem 原子已流水化。",
        "P4 scatter same-address-atomic hypothesis: WASH — SM100 smem atomics are pipelined.")}</li>
<li>{bil("launch-config 精调 (cs2/4/8, nt): 逐 cell best-of-5 天花板 1.025 — pick_config 已近最优。",
        "Launch-config refinement: per-cell best-of-5 ceiling 1.025 — pick_config already near-optimal.")}</li>
<li>{bil("kNumBins=512 全局: K1024 大 N 低命中 cell 0.84 回退 → 必须 per-K (K2048 only)。kNumBins=256 = exact-tail scratch 越界 UB, 不可用。",
        "Global kNumBins=512: 0.84 regression at K1024 large-N low-hit → must be per-K (K2048 only). kNumBins=256 = exact-tail scratch OOB (UB), unusable.")}</li>
<li>{bil("提案字面 B1 (整窗 8K 粒度): 真实数据 replay 跳过率 ≈0 — 粒度必须是 (warp,window) 256 元素。",
        "The proposal's literal B1 (whole 8K windows): ~zero skip rate on real data — the right quantum is (warp,window)=256 elems.")}</li>
</ul>

<h2>{bil("5 · +40% 可达性定界", "5 · Feasibility bound for the +40% ask", "span")}</h2>
{bil("双锁: ① 信息下界 — 精确 top-K 必须每步读全行一遍 (对手可把新最大值藏进任意块), "
     "P2 计数遍不可稀疏化 (跨步侧带 B3 无法免读地验证); ② 松约束对照 — 四臂 oracle 实测把 "
     "P3+P4blk 全部归零 (物理不可能的松弛) 也只有 geomean 1.771 < 1.40 所需的持续富余 "
     "(且其中 P2/P1b/floor 不可去)。现实可达栈 (iter3 已收 + distP4 + B1-warp) 估计 ~1.10-1.25。"
     "结论: +40% 在 GVR 骨架内不可达; 更高收益需要换算法 (并行战役 op35_apex_topk 正在验证采样过滤路线)。",
     "Double lock: ① information floor — an exact top-K must read the whole row once per step "
     "(an adversary can hide a new max in any block), so P2's count pass cannot be sparsified "
     "(the cross-step B3 sideband cannot verify without reading); ② relaxed-constraint control — "
     "the measured 4-arm oracle shows even zeroing ALL of P3+P4blk (physically impossible) yields "
     "only geomean 1.771, and the remaining floor/P1b/P2 cannot be removed. The realistic "
     "reachable stack (iter3 harvested + distP4 + warp-B1) is ~1.10-1.25. Conclusion: +40% is "
     "INFEASIBLE within the GVR skeleton; larger gains require a different algorithm "
     "(the concurrent op35_apex_topk campaign is validating a sampling-filter approach).")}

<h2>{bil("6 · 流程台账 (skills/harness 调用顺序)", "6 · Process ledger (skills/harnesses, call order)", "span")}</h2>
<ol>
<li>{bil("记忆/交接读取: PROPOSAL_GVR_NEXT_OPT.html, RESUME_SESSION_HANDOFF.md, OPT_CAMPAIGN_ANALYSIS.md, 证伪史 memory",
        "Recall: PROPOSAL_GVR_NEXT_OPT.html, RESUME_SESSION_HANDOFF.md, OPT_CAMPAIGN_ANALYSIS.md, falsification-history memory")}</li>
<li>{bil("/omni-kernel skill 加载 (战役协议: 探针梯 / 三轨 exactness / nsys 仲裁 / 台账纪律)",
        "/omni-kernel skill loaded (campaign protocol: probe ladder / 3-track exactness / nsys arbiter / ledger discipline)")}</li>
<li>{bil("脚手架: PLAN/ANALYSIS/AUTONOMY/ITERATIONS/FALSIFIED + gvrpkg_head 基线快照 (PR HEAD) + variant 包 (全部旗标默认关=字节等同)",
        "Scaffold: PLAN/ANALYSIS/AUTONOMY/ITERATIONS/FALSIFIED + gvrpkg_head baseline snapshot (PR HEAD) + variant package (flags default-off = byte-identical)")}</li>
<li>{bil("rung1 replay_b1.py (77 cells) → B1 粒度选型; rung2 oracle 旗标 (p3/floor/p4skip) event 筛 → nsys_oracle.py 四臂 L2 归因; ncu_cell.py L3 stall 归因",
        "rung1 replay_b1.py (77 cells) → B1 granularity; rung2 oracle flags event screens → nsys_oracle.py 4-arm L2 attribution; ncu_cell.py L3 stall attribution")}</li>
<li>{bil("ab_op35.py L1 事件配对筛 (8卡 shard-by-cell): h3tail/cfg/kb512/kb256/iter3; 离群点单卡复测 + 逐旗标消融",
        "ab_op35.py L1 event-paired screens (8-GPU shard-by-cell): h3tail/cfg/kb512/kb256/iter3; idle-GPU outlier re-runs + per-flag ablation")}</li>
<li>{bil("nsys_ab.py + drive_nsys_ab.sh L2 判决 (单卡配对, 3 进程内轮次) → parse_nsys_ab.py",
        "nsys_ab.py + drive_nsys_ab.sh L2 verdict (single-GPU paired, 3 in-process rounds) → parse_nsys_ab.py")}</li>
<li>{bil("每 iter git commit + 台账回写; RESUME_PROMPT.md 断线保险",
        "Per-iter git commit + ledger write-back; RESUME_PROMPT.md session-loss insurance")}</li>
</ol>

<h2>{bil("7 · 成本", "7 · Cost", "span")}</h2>
<div id="cost-section">COST_PLACEHOLDER</div>

<h2>{bil("8 · 后续", "8 · Next steps", "span")}</h2>
{bil("① iter3 束 → 独立 follow-up PR (不并入 #16457): variant 三处小改 + GvrParams K2048 kNumBins=512; "
     "② distP4 (cluster 握手#2 值搬运废除 + leader P4 并行化) = 剩余最大单杠杆 (~cs>1 cells P4blk 29-58%); "
     "③ B1 (warp,window) 侧带 = 512K-1M 段增量; ④ 采样过滤新算法见 op35_apex_topk。",
     "① iter3 bundle → separate follow-up PR (NOT into #16457): three small variant diffs + "
     "GvrParams K2048 kNumBins=512; ② distP4 (kill handoff2 value shipping + parallelize leader "
     "P4) = the largest remaining single lever (P4blk 29-58% at cs>1); ③ (warp,window) B1 "
     "sideband = incremental at 512K-1M; ④ sampling-filter new algorithm → op35_apex_topk.")}
"""

    css = """
body{font-family:system-ui,-apple-system,sans-serif;margin:24px auto;max-width:1100px;
     color:#1a1a2e;line-height:1.55;padding:0 16px}
h1{font-size:1.4em;border-bottom:3px solid #3b6ea5;padding-bottom:8px}
h2{font-size:1.15em;color:#24435f;margin-top:28px;border-left:4px solid #3b6ea5;padding-left:10px}
table{border-collapse:collapse;margin:12px 0;font-size:0.85em}
th,td{border:1px solid #c5d3e0;padding:4px 8px;text-align:right}
th{background:#eaf1f7}td:first-child,th:first-child{text-align:left}
.meta{color:#667;font-size:0.85em}
#lang-en:checked ~ .content .zh{display:none}
#lang-zh:checked ~ .content .en{display:none}
.lang-toggle{position:sticky;top:0;background:#fff;padding:8px 0;z-index:5}
.lang-toggle label{border:1px solid #3b6ea5;padding:4px 14px;cursor:pointer;border-radius:4px;margin-right:6px}
#lang-zh:checked ~ .lang-toggle label[for=lang-zh],
#lang-en:checked ~ .lang-toggle label[for=lang-en]{background:#3b6ea5;color:#fff}
details{margin:10px 0}summary{cursor:pointer;color:#24435f}
"""
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>op35 GVR round-2 campaign report</title><style>{css}</style></head><body>
<input type="radio" name="lang" id="lang-zh" checked hidden>
<input type="radio" name="lang" id="lang-en" hidden>
<div class="lang-toggle"><label for="lang-zh">中文</label><label for="lang-en">English</label></div>
<div class="content">{body}</div></body></html>"""
    out = _OP35 / "REPORT.html"
    cost_md = (_OP35 / "COST.md").read_text() if (_OP35 / "COST.md").exists() else ""
    page = page.replace("COST_PLACEHOLDER", f"<pre>{html.escape(cost_md)}</pre>")
    out.write_text(page)
    print("wrote", out, len(page), "bytes")


if __name__ == "__main__":
    main()
