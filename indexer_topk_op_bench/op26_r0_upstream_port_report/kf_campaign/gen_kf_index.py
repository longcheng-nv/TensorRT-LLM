# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KF campaigns index -> KF_CAMPAIGNS_INDEX.html.

Master table + per-campaign cards (goal/constraints/code/data locations,
internal vs local-nsys verified speedups) + interactive (CSS-only) SVG
relationship graph: nodes link to card anchors, hover highlights.
Campaign facts curated from R4_RUN_STATE.md / R3_LEDGER.md / COST_LEDGER.md.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
KFP = "indexer_topk_op_bench/op26_r0_upstream_port_report/kf_campaign"

C = [  # (anchor, id, name, status, rounds, internal, local, start, end, cost, owner_session, goal, constraint, code, data, note)
 dict(a="c1", id="tfb91bvwm972kfyf1bc1trj5e0", name="gvr-topk-bs1-real(第一期)", st="Cancelled", rd=2,
      internal="1.3385", local="champion c74f_sbx **1.6828**(865 格, 0 回退)", t0="07-21 10:15", t1="07-21 ~18:11",
      cost="≈$690", own="本会话系(第一 lineage)",
      goal="BS=1 865 真实格 beat 旧 PR head(e6fdbfac3d),nsys 冷判", cons="GVR 骨架(执行中放宽 — 见备注)",
      code=f"{KFP}/harvest/r2_c74fb3c0 + c74f_sbx(graft) · **fork:kf/gvr-topk-c74fsbx** @193156c8", data=f"{KFP}/grid_c74fsbx.csv 等",
      note="round-harvest-verdict 流程首建;两次 grid 因双 driver 作废(纪律起源)。**⚠ champion c74f_sbx 源码 0 次引用 pre_idx — preIdx 先验被放弃**(实测证伪 hint 收益后 Bar-first 取向)"),
 dict(a="kf1", id="k7rhy79h015gn3fpqv5cs154w8", name="gvr-topk-r3(baseline 验证失败)", st="Failed", rd=0,
      internal="—", local="—", t0="07-21 13:36", t1="+40s", cost="~0", own="本会话系",
      goal="R3 起跑(baseline-solution 现测)", cons="—", code="—", data="—",
      note="平台缺口 D1 首次发现: baseline evaluator 不 stage 资产"),
 dict(a="r3", id="e5q1zgrfhs0z57dj6850kc444r", name="gvr-topk-r3", st="Cancelled", rd=4,
      internal="1.1111", local="operator 复合 compB **1.8267**(865 格, 0 回退)", t0="07-22 03:15", t1="07-22 ~15:49",
      cost="$761", own="本会话系(第一 lineage R3)",
      goal="从 champion c74f_sbx 再推高(baselines=平台 champion 时延)", cons="**骨架放宽(R3_LEDGER D4, 用户 2026-07-22 裁定)**: (a) preIdx 先验由实测证据废止(12 项证伪 + hint 挂载 WASH 1.0001);(b) 保留为直方图前缀阈值精化;(c) 精确 refine 完整保留;禁装饰性 hint 路径,报告须连证据链明示",
      code=f"{KFP}/harvest/(aef33fac/becd/30e7 系)+ compB · **fork:kf/gvr-topk-compB** @9dbd6ee2(= c74fsbx 直系升级: +topk_mid&lt;3|4&gt;@1024 中段 rung 治 8195≤n≤16387 弱带 +19%、+v30::topk_coop K2048 专用 3-pass 梯 16896&lt;n≤140k、+aefm::topk_fast 非协作快尾;对 c74fsbx +8.1%;kernel.cu diff +1230 行)", data=f"{KFP}/grid_r3grid*.csv",
      note="平台 4 轮未超 harvested aef33fac;compB 为 operator 收口拼装。**⚠ compB 属 直方图+radix 系(无 pre_idx),其 1.8267× 为放宽骨架口径;R4 因此立项硬锁骨架,真 GVR 达 1.6531×(差 ~10% = 先验约束实测代价)**"),
 dict(a="r4f", id="6qfpzj957x01z2epbyr51srp68", name="R4 baseline-eval(失败)", st="Failed", rd=0,
      internal="—", local="—", t0="07-22 02:40", t1="+40s", cost="~0", own="本会话",
      goal="R4 prepare 的 cute_dsl baseline 现测", cons="—", code="—", data="—",
      note="D1 复现(0/28 safetensors missing)→ 改走 baselines.jsonl"),
 dict(a="r4", id="pra6srbd7h4pqecqbgxgm15rgg", name="gvr-topk-cold60(R4 冷启动)", st="Cancelled(预算)", rd=3,
      internal="1.3701", local="champion 28dc11f6 **1.6531**(865 格, 865 exact, 0 真实回退)→ 三 Bar 全达成",
      t0="07-22 02:48", t1="07-23 ~14:2x", cost="$1110.62", own="本会话(R4)",
      goal="第二 lineage 冷启动: 以 PR#16457 pinned head 04a0900ff7 为唯一起点, 865 格 geomean ≥1.60 + 0 回退 + exact",
      cons="骨架硬锁(preIdx 先验+secant-log+refine);禁注入第一 lineage 解法;死路/陷阱/REPORT 事实可注入",
      code=f"{KFP}/harvest/r3_28dc11f6 · fork:kf/r4-champion-final-bs1 @e1049bca",
      data=f"{KFP}/grid_r4r3cg.csv(判决)· grid_r4pr2.csv(分母)· R4_CLOSEOUT.md · R4_CHAMPION_BS1_REPORT.html",
      note="判决链 v5→v14→r2wd→v25→v27→r3a003→28dc(1.295→1.653);prompt=32KB 限内源码 digest"),
 dict(a="r5a", id="rngnxv95cx5qfdmte69vz0b0n8", name="gvr-topk-bs2x(R5 v1)", st="Cancelled(止损)", rd=2,
      internal="—(全 0%)", local="—", t0="07-22 22:04", t1="07-23 ~05:0x", cost="$9.38", own="本会话(R5)",
      goal="BS=2-1024 §7b 750 格 avg ≥2.0× vs head 原生批 + 全格≥1.0 + BS=1 守 R4 水位",
      cons="GVR 骨架;custom_inputs 按 cell 物化 [BS,npad]",
      code="—(无有效产出)", data=f"{KFP}/r5_bs/(工装)",
      note="平台 submit 评测对 custom_inputs 全判 0%(bugs 26583-26602)→ 止损重开"),
 dict(a="r5b", id="vk9m3tetqh165a5y01j7nnrns0", name="gvr-topk-bs2x-v2(R5 v2)", st="Cancelled(DQ→fork)", rd=3,
      internal="0.9799", local="v23/v39 全格 0.9348/0.9426 + **8 例盲区 inexact(DQ)**",
      t0="07-23 06:59", t1="07-23 ~13:1x", cost="$719.02", own="本会话(R5)",
      goal="同 R5 v1(改物化 safetensors 30 workloads/210MB)", cons="同上;平台盲区 = 512k×bs2-16 / bs64",
      code=f"{KFP}/harvest/r5_7bea12c6 · r5_5a1431a5", data=f"{KFP}/r5_bs/grid_r5g_v23.csv · grid_r5g_v39.csv",
      note="lineage 遗传 row0 竞态(cs>1 cluster × b>1)→ cancel + fork 注入 steering"),
 dict(a="r5c", id="befh5fh2595es8ztpcg0nmq6q8", name="gvr-topk-bs2x-v3(R5 fork)", st="Cancelled(预算, r5 完成)", rd=5,
      internal="1.0756", local="champion 156ab438: **全 exact**(750+865)但 750 格 gm **0.9862**/391 回退; BS=1 1.2233(=R4champ×0.743)→ 目标未达成",
      t0="07-23 19:23(rounds 3-5)", t1="07-24 ~07:4x", cost="$596.91", own="本会话(R5)",
      goal="继承 v2 44 candidates + steering(bug 图谱/自测清单/bs128 洞/吞吐方向), rounds 3-5",
      cons="同 R5;append-prompt 不可改 workloads",
      code=f"{KFP}/harvest/r5final_156ab438 · fork:kf/r5-champion-bs-combined @3e04d248",
      data=f"{KFP}/r5_bs/grid_r5g_final.csv · grid_r5bs1guard.csv · R5_CLOSEOUT.md",
      note="exactness 战役成功(8→0);吞吐域(bs16-256 中大N)为结构墙;部署建议 = (b,npad,K) 三路分派"),
 dict(a="fr", id="6em6mf55g11g767p5wcepgy07w", name="gvr-topk-pr16457-fresh", st="Completed", rd=3,
      internal="0.84", local="未复核(本会话外)", t0="07-23 21:48", t1="07-24 ~10:1x", cost="未核", own="**另一会话**",
      goal="artifact indexer-topk-decode-fresh-full(推测: 全格 fresh 分母重跑)", cons="未知(另一会话)",
      code="未知(另一会话)", data="未知(另一会话)", note="内部 0.84 未过平价;如需盘点可接管"),
 dict(a="d1", id="9dprgt29j515d75q1b6gyyqw1g", name="dsl-fp4-paged-mqa-logits", st="Completed", rd=3,
      internal="1.04", local="未复核(本会话外)", t0="07-23 15:58", t1="07-23 ~22:2x", cost="未核", own="**另一会话**",
      goal="cuteDSL FP4 paged-MQA logits 算子", cons="未知", code="未知", data="未知", note=""),
 dict(a="d2", id="66mcp39jf15g38gnn54y46nc7w", name="dsl-fp8-paged-mqa-logits", st="Completed", rd=3,
      internal="0.83", local="未复核(本会话外)", t0="07-23 15:53", t1="07-23 ~22:1x", cost="未核", own="**另一会话**",
      goal="cuteDSL FP8 paged-MQA logits 算子", cons="未知", code="未知", data="未知", note=""),
 dict(a="d0", id="0mv4xsw7… / 3adykw9p… / 79q6bd0j… / mbcva85f… / ex0e6fp0… / ewhqkn25… / xc71bent…",
      name="dsl-fp4/fp8 早期 7 次(round-0 ×2 + Failed ×5)", st="Completed×2 / Failed×5", rd=0,
      internal="1.00 / —", local="—", t0="07-23 08:22-08:32", t1="即刻~短", cost="未核", own="**另一会话**",
      goal="同上(起跑调试)", cons="未知", code="未知", data="未知", note="连续失败后 round-0 冒烟成功,再起 3 轮正式跑"),
]


PROMPTS = {  # anchor -> [(label, path)]
    "c1": [("campaign prompt(v1, 开跑原文)", "prompt.md"),
           ("prompt v2(round-2 前更新)", "prompt_v2.md")],
    "r3": [("campaign prompt(含 champion 源码内联)", "gvr-topk-r3/prompt.md")],
    "r4": [("campaign prompt(§B v3-coldstart + 19.9KB 源码 digest)", "gvr-topk-cold60/prompt.md")],
    "r5a": [("campaign prompt(v1-bs)", "gvr-topk-bs2x/prompt.md")],
    "r5b": [("campaign prompt(v2, 物化 safetensors 版)", "gvr-topk-bs2x-v2/prompt.md")],
    "r5c": [("继承 v2 prompt(不变)", "gvr-topk-bs2x-v2/prompt.md"),
            ("fork --append-prompt(steering 追加段)", "r5_bs/fork_steering.md")],
}


def prompt_block(anchor):
    import html as _h
    ent = PROMPTS.get(anchor)
    if not ent:
        return ('<tr><th>发出 prompt</th><td class="dim">无(prepare 期评测无 prompt)或'
                '归另一会话本地,平台 API 不暴露 prompt 原文</td></tr>')
    parts = []
    for label, rel in ent:
        p = HERE / rel
        if not p.exists():
            parts.append(f"<p class='dim'>{label}: 文件缺失 {rel}</p>")
            continue
        txt = _h.escape(p.read_text())
        parts.append(f"<details><summary>{label} — <code>{rel}</code> "
                     f"({p.stat().st_size} B, 点击展开)</summary>"
                     f"<pre class='pmt'>{txt}</pre></details>")
    return f'<tr><th>发出 prompt</th><td>{"".join(parts)}</td></tr>'


def md(s):
    import re
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"fork:(kf/[\w./-]+)",
                  r'<a href="https://github.com/longcheng-nv/TensorRT-LLM/tree/\1" target="_blank">fork:\1</a>', s)


rows_html = []
cards = []
for c in C:
    rows_html.append(
        f'<tr id="row-{c["a"]}"><td><a href="#{c["a"]}">{c["name"]}</a></td>'
        f'<td class="mono">{c["id"][:14]}…</td><td>{c["st"]}</td><td>{c["rd"]}</td>'
        f'<td>{c["internal"]}</td><td>{md(c["local"])}</td>'
        f'<td>{c["t0"]}</td><td>{c["t1"]}</td><td>{c["cost"]}</td><td>{md(c["own"])}</td></tr>')
    cards.append(f"""<div class="card" id="{c['a']}">
<h3>{c['name']} <span class="chip">{c['st']}</span></h3>
<table class="kv">
<tr><th>ID</th><td class="mono">{c['id']}</td></tr>
<tr><th>轮次 / 内部 speedup</th><td>{c['rd']} / {c['internal']}</td></tr>
<tr><th>本地 nsys 复核</th><td>{md(c['local'])}</td></tr>
<tr><th>起止(UTC)</th><td>{c['t0']} → {c['t1']}</td></tr>
<tr><th>花费</th><td>{c['cost']}</td></tr>
<tr><th>发起目标</th><td>{md(c['goal'])}</td></tr>
<tr><th>约束</th><td>{c['cons']}</td></tr>
<tr><th>代码位置</th><td class="mono">{md(c['code'])}</td></tr>
<tr><th>性能数据位置</th><td class="mono">{c['data']}</td></tr>
<tr><th>备注</th><td>{md(c['note'])}</td></tr>
{prompt_block(c['a'])}
</table></div>""")


def node(x, y, w, h, anchor, lines, cls="n"):
    tspan = "".join(
        f'<tspan x="{x + w/2}" dy="{"1.2em" if i else 0}">{ln}</tspan>'
        for i, ln in enumerate(lines))
    ty = y + h / 2 - (len(lines) - 1) * 8 + 4
    return (f'<a href="#{anchor}" class="{cls}">'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9"/>'
            f'<text x="{x + w/2}" y="{ty}" text-anchor="middle">{tspan}</text></a>')


GH = "https://github.com/longcheng-nv/TensorRT-LLM/tree/"


def onode(x, y, w, h, lines, gh=None):
    """Output artifact node; gh=<branch> adds a clickable GitHub link line."""
    tspan = "".join(
        f'<tspan x="{x + w/2}" dy="{"1.2em" if i else 0}">{ln}</tspan>'
        for i, ln in enumerate(lines))
    nl = len(lines) + (1 if gh else 0)
    ty = y + h / 2 - (nl - 1) * 8 + 4
    s = (f'<g class="o"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9"/>'
         f'<text x="{x + w/2}" y="{ty}" text-anchor="middle">{tspan}</text>')
    if gh:
        s += (f'<a href="{GH}{gh}" target="_blank" class="gh">'
              f'<text x="{x + w/2}" y="{ty + (nl - 1) * 16}" text-anchor="middle">'
              f'⧉ github: {gh}</text></a>')
    return s + "</g>"


def edge(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="5,4"' if dash else ""
    return f'<path d="M{x1},{y1} C{(x1+x2)//2},{y1} {(x1+x2)//2},{y2} {x2},{y2}" class="e"{d}/>'


def hedge(x1, y, x2, dash=False, lab=None):
    """Straight horizontal arrow at height y."""
    d = ' class="e d"' if dash else ' class="e"'
    s = f'<line x1="{x1}" y1="{y}" x2="{x2 - 6}" y2="{y}"{d} marker-end="url(#arr)"/>'
    if lab:
        s += f'<text x="{(x1 + x2) // 2}" y="{y - 7}" class="elab" text-anchor="middle">{lab}</text>'
    return s


def vedge(x, y1, y2, dash=False, lab=None):
    """Straight vertical arrow at column x."""
    d = ' class="e d"' if dash else ' class="e"'
    dy = -6 if y2 < y1 else 6
    s = f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - dy}"{d} marker-end="url(#arr)"/>'
    if lab:
        s += (f'<text x="{x + 8}" y="{(y1 + y2) // 2 + 4}" class="elab">{lab}</text>')
    return s


def ledge(x1, y1, xm, y2, x2, dash=False, lab=None):
    """L-shaped arrow: horizontal to xm, vertical to y2, horizontal to x2."""
    d = ' class="e d"' if dash else ' class="e"'
    s = (f'<path d="M{x1},{y1} L{xm},{y1} L{xm},{y2} L{x2 - 6},{y2}"{d} '
         f'marker-end="url(#arr)"/>')
    if lab:
        s += f'<text x="{xm + 8}" y="{(y1 + y2) // 2 + 4}" class="elab">{lab}</text>'
    return s


G = []
LANE_BG = [(16, 240, "#f8fafd", "第一 lineage(BS=1 → 收口后 BS-scaling 直系延伸)· 07-21 → 07-22"),
           (448, 128, "#f8fafd", "第二 lineage R4(BS=1 冷启动, 骨架硬锁)· 07-22 → 07-23"),
           (608, 190, "#f8fafd", "R5(BS=2-1024)· 07-22 → 07-24"),
           (830, 178, "#faf8fd", "另一会话(含 r3_v11 批量化并行线)")]
for y, h, col, lab in LANE_BG:
    G.append(f'<rect x="8" y="{y}" width="1104" height="{h}" rx="10" fill="{col}" stroke="#e3e9f2"/>')
    G.append(f'<text x="20" y="{y + 20}" class="lane">{lab}</text>')

# ---- Lane 1: campaign-1 -> c74f_sbx -> R3 -> compB (one clean left-to-right chain)
Y1 = 66
G.append(node(28, Y1, 168, 62, "c1", ["campaign-1 bs1-real", "2轮 · 内部 1.34", "$690"]))
G.append(onode(252, Y1, 210, 62, ["c74f_sbx 1.6828×(865格)", "⚠ 无 pre_idx(放宽骨架)"], gh="kf/gvr-topk-c74fsbx"))
G.append(node(518, Y1, 168, 62, "r3", ["gvr-topk-r3", "4轮 · 内部 1.11", "$761"]))
G.append(onode(742, Y1, 246, 62, ["compB 1.8267×(865格 · 0回退)", "= c74fsbx + topk_mid/coop/fast"], gh="kf/gvr-topk-compB"))
G.append(node(518, Y1 - 44, 168, 34, "kf1", ["baseline 验证 Failed(D1)"], cls="n bad"))
G.append(hedge(196, Y1 + 31, 252, lab="产出+graft"))
G.append(hedge(462, Y1 + 31, 518, lab="作 baseline"))
G.append(hedge(686, Y1 + 31, 742, lab="复合+graft"))
G.append(vedge(602, Y1 - 10, Y1, dash=True))
# post-close BS-scaling extension of compB (two parallel follow-ups)
G.append(onode(742, Y1 + 84, 246, 62, ["§7.8 BS-scaling 实测(收口即测)", "BS=1 全胜; crossover@BS=2;", "bs1024 批式臂快 45-125× ⇒ BS==1 门"]))
G.append(vedge(865, Y1 + 62, Y1 + 84))
G.append('<g class="n other"><rect x="446" y="{y}" width="250" height="62" rx="9"/>'
         '<text x="571" y="{t1}" text-anchor="middle">kf/compb-bs-ext(另一会话)</text>'
         '<text x="571" y="{t2}" text-anchor="middle">tp4 mid-BS 融合直方图臂 + 统一 dispatcher</text></g>'.format(y=Y1+84, t1=Y1+104, t2=Y1+120))
G.append(f'<a href="{GH}kf/compb-bs-ext" target="_blank" class="gh"><text x="571" y="{Y1+136}" text-anchor="middle">⧉ github: kf/compb-bs-ext</text></a>')
G.append(f'<line x1="742" y1="{Y1+115}" x2="702" y2="{Y1+115}" class="e" marker-end="url(#arr)"/>')
G.append(f'<text x="722" y="{Y1+108}" class="elab" text-anchor="middle">扩 BS>1</text>')

# ---- Lane 2: R4
Y2 = 498
G.append(node(28, Y2, 168, 62, "r4f", ["baseline-eval", "Failed(D1 复现)"], cls="n bad"))
G.append(node(252, Y2, 210, 62, "r4", ["gvr-topk-cold60(R4)", "3轮 · 内部 1.3701 · $1110"]))
G.append(onode(518, Y2, 220, 62, ["champion 28dc11f6", "865格 1.6531× · 0 回退 · 真 GVR"], gh="kf/r4-champion-final-bs1"))
G.append(onode(794, Y2, 246, 62, ["R4_CLOSEOUT.md", "R4_CHAMPION_BS1_REPORT.html", "grid_r4r3cg.csv / grid_r4pr2.csv"]))
G.append(hedge(196, Y2 + 31, 252, dash=True, lab="改走 baselines.jsonl"))
G.append(hedge(462, Y2 + 31, 518, lab="round-3 收割"))
G.append(hedge(738, Y2 + 31, 794))
# knowledge flow lane1 -> lane2 (single clean vertical, right side, no crossings)
G.append(vedge(1060, Y1 + 31, Y2 - 6, dash=True))
G.append(f'<path d="M988,{Y1 + 31} L1060,{Y1 + 31}" class="e d"/>')
G.append(f'<path d="M1060,{Y2 - 6} L1052,{Y2 - 6}" class="e d" marker-end="url(#arr)"/>')
G.append(f'<text x="1066" y="{(Y1 + Y2) // 2}" class="elab" writing-mode="tb">仅死路清单/陷阱/REPORT 事实(禁解法)</text>')

# ---- Lane 3: R5 (chain + fan-out done with L-edges at distinct heights)
Y3 = 658
G.append(node(28, Y3, 168, 56, "r5a", ["bs2x v1 · $9", "custom_inputs 全 0%"], cls="n bad"))
G.append(node(252, Y3, 210, 56, "r5b", ["bs2x-v2 · 3轮 · $719", "内部 0.98 · lineage DQ"]))
G.append(node(518, Y3, 220, 56, "r5c", ["bs2x-v3 fork(rounds3-5)", "内部 1.0756 · $597"]))
G.append(onode(794, Y3 - 6, 246, 46, ["champion 156ab438(全 exact)", "750格 0.9862 → 目标未达成"]))
G.append(onode(794, Y3 + 48, 246, 36, ["fork 留档 →"], gh="kf/r5-champion-bs-combined"))
G.append(onode(794, Y3 + 92, 246, 36, ["R5_CLOSEOUT.md · grid_r5g_final.csv"]))
G.append(hedge(196, Y3 + 28, 252, lab="重构 workloads"))
G.append(hedge(462, Y3 + 28, 518, lab="cancel+fork+steering"))
G.append(hedge(738, Y3 + 20, 794))
G.append(ledge(738, Y3 + 40, 766, Y3 + 66, 794))
G.append(ledge(738, Y3 + 40, 760, Y3 + 110, 794))
# R4 champion -> R5 prompt digest (single vertical at champion column)
G.append(vedge(560, Y2 + 62, Y3 - 6, dash=True, lab="champion digest → prompt 起点"))

# ---- Lane 4: other session
Y4 = 878
G.append(node(28, Y4, 210, 52, "d0", ["dsl-fp4/fp8 起跑 ×7", "Failed×5 · round0×2"], cls="n other"))
G.append(node(294, Y4, 190, 52, "d2", ["dsl-fp8 · 3轮 · 0.83"], cls="n other"))
G.append(node(540, Y4, 190, 52, "d1", ["dsl-fp4 · 3轮 · 1.04"], cls="n other"))
G.append(node(794, Y4, 246, 52, "fr", ["gvr-topk-pr16457-fresh", "3轮 · 0.84 · 未本地复核"], cls="n other"))
G.append(hedge(238, Y4 + 26, 294))
G.append(hedge(484, Y4 + 26, 540))
# parallel line: op38 batched r3_v11 (other session)
G.append('<g class="n other"><rect x="28" y="{y}" width="456" height="46" rx="9"/>'
         '<text x="256" y="{t1}" text-anchor="middle">op38_r3v11_bs: grid.y 批量化 R4 中期 champion r3_v11(与 R5 平行的第三条 BS 线)</text>'
         '<text x="256" y="{t2}" text-anchor="middle">发现: 寄存器驻留在高 BS 占用锁死 1 CTA/SM(该结论已注入 R5 prompt)</text></g>'.format(y=Y4+66, t1=Y4+84, t2=Y4+100))
G.append(f'<a href="{GH}kf/gvr-topk-r3v11-bs" target="_blank" class="gh"><text x="560" y="{Y4+95}">⧉ github: kf/gvr-topk-r3v11-bs(+ kf/gvr-topk-r3v11)</text></a>')

# ---- skeleton-compliance annotation between lane1 and lane2
G.append('<g class="warn"><rect x="8" y="180" width="1104" height="158" rx="10"/>'
         '<text x="24" y="296" font-weight="700">⚠ 骨架合规注记(第一 lineage)</text>'
         '<text x="24" y="320">· 终版算子 c74f_sbx / compA / compB 源码 0 次引用 pre_idx — preIdx 先验被整体放弃,实际走向 直方图前缀阈值梯 + radix 精确尾(GVR 三要素仅存 b 变体 + c)。</text>'
         '<text x="24" y="344">· 依据 R3_LEDGER D4(用户 2026-07-22 裁定): Bar-first 放宽骨架 — (a) 先验由实测证据废止(12 项证伪 + hint 挂载 WASH 1.0001),(b) 保留为直方图前缀阈值精化,(c) 精确 refine 完整保留。</text>'
         '<text x="24" y="368">· 两分支代际关系: compB = c74fsbx 直系升级(+topk_mid 中段 rung 治 N=16387 弱带 +19% · K2048 专用 v30::topk_coop 梯 · aefm::topk_fast 非协作快尾),对 c74fsbx +8.1%。</text>'
         '<text x="24" y="392">· ⇒ 1.8267× 与 1.6828× 均为"放宽骨架"口径,不可与硬锁骨架的 R4 1.6531× 直接混比;R4 因此立项(真 GVR: P1 消费 pre_idx),差值 ~10% = 先验约束实测代价。</text>'
         '<text x="24" y="416">· 两分支分母亦不同: c74fsbx vs 旧 head e6fdbfac3d;compB vs R3 时点 head(已含 #16424 优化)。</text></g>')

SVG = ('<svg viewBox="0 0 1120 1032" xmlns="http://www.w3.org/2000/svg" style="width:100%">'
       '<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" '
       'orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="#7f93b3"/></marker></defs>'
       '<style>'
       '.n rect{fill:#eef3fb;stroke:#7f93b3;stroke-width:1.3}'
       '.n text{font:12px sans-serif;fill:#1a2233}'
       '.n:hover rect{fill:#dbe7fb;stroke:#1a2233;stroke-width:2}'
       '.n.bad rect{fill:#fdeee3;stroke:#c98a4b}'
       '.n.other rect{fill:#f2eefb;stroke:#8d7fb3}'
       '.o rect{fill:#e7f5ec;stroke:#5c9a74;stroke-width:1.2}'
       '.o text{font:11.5px sans-serif;fill:#173d27}'
       '.gh text{font:11px ui-monospace,monospace;fill:#23508f;text-decoration:underline}'
       '.gh:hover text{fill:#0b2f66}'
       '.e{fill:none;stroke:#7f93b3;stroke-width:1.6}'
       '.e.d{stroke-dasharray:5,4;stroke:#a9b6ca}'
       '.elab{font:10.5px sans-serif;fill:#55627a}'
       '.lane{font:600 13px sans-serif;fill:#55627a}'
       '.warn rect{fill:#fff8ec;stroke:#c98a4b;stroke-width:1.2}'
       '.warn text{font:12px sans-serif;fill:#6b4a1b}'
       '</style>' + "".join(G) + "</svg>")

PAGE = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>KF campaigns 台账索引</title>
<style>
body {{ font: 13.5px/1.55 -apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC",sans-serif;
       max-width: 1160px; margin: 24px auto; padding: 0 16px; color: #1a2233; }}
h1 {{ font-size: 21px; }} h2 {{ font-size: 16.5px; margin-top: 30px; }}
.mono {{ font-family: ui-monospace, monospace; font-size: 12px; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border: 1px solid #d6dde8; padding: 4px 9px; text-align: left; vertical-align: top; }}
th {{ background: #f0f3f8; }}
.card {{ border: 1px solid #d6dde8; border-radius: 10px; padding: 6px 16px 12px; margin: 14px 0; }}
.card:target {{ border-color: #1a2233; box-shadow: 0 0 0 2px #dbe7fb; }}
.card h3 {{ font-size: 15px; }}
.kv th {{ width: 150px; background: #f7f9fc; font-weight: 600; }}
.chip {{ font-size: 11.5px; background: #f0f3f8; border: 1px solid #b9c4d6; border-radius: 10px; padding: 1px 9px; margin-left: 8px; }}
.legend span {{ margin-right: 18px; }}
.dim {{ color: #8a94a8; }}
details {{ margin: 4px 0; }}
summary {{ cursor: pointer; color: #23508f; }}
pre.pmt {{ background: #f6f8fb; border: 1px solid #d6dde8; border-radius: 8px;
          padding: 10px 14px; overflow-x: auto; font-size: 11.5px; line-height: 1.45;
          max-height: 480px; overflow-y: auto; white-space: pre-wrap; }}
.sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 4px; vertical-align: -1px; }}
</style></head><body>
<h1>KernelFactory campaigns 台账索引(loncheng@nvidia.com,2026-07-21 → 07-24)</h1>
<p>共 18 个 campaign(本会话系 8 + 另一会话 10)。判决口径:内部 speedup = KF 平台
(含 ~15µs eval 地板,只作轮内排序);<b>本地 nsys 复核 = B200 cold-L2 配对纯 kernel
时间(ship 判据)</b>。KF 总花费(本会话系,终账):<b>$3196.93</b>
(第一期 ≈$690 + R3 $761 + R4 $1110.62 + R5 $1325.31)。</p>

<h2>① 任务关系图(可点击节点跳转对应卡片;hover 高亮;虚线 = 知识/材料流)</h2>
<p class="legend"><span><span class="sw" style="background:#eef3fb;border:1px solid #7f93b3"></span>campaign</span>
<span><span class="sw" style="background:#fdeee3;border:1px solid #c98a4b"></span>失败/止损</span>
<span><span class="sw" style="background:#e7f5ec;border:1px solid #5c9a74"></span>产出(champion/分支/报告)</span>
<span><span class="sw" style="background:#f2eefb;border:1px solid #8d7fb3"></span>另一会话</span></p>
{SVG}

<h2>② 总表</h2>
<table><tr><th>名称</th><th>ID</th><th>状态</th><th>轮</th><th>内部</th><th>本地 nsys 复核</th>
<th>开始</th><th>结束</th><th>花费</th><th>归属</th></tr>
{"".join(rows_html)}
</table>

<h2>③ 逐 campaign 卡片</h2>
{"".join(cards)}

<h2>④ 公共工装 / 纪律速查</h2>
<ul>
<li>工装: {KFP}/(export_cells*.py · nsys_ab.py/nsys_bs.py · drive_grid_*.sh · aggregate_*.py · monitor_campaign.sh)</li>
<li>分母: grid_r4pr2.csv(BS=1 865)· r5_bs/grid_r5pr.csv(BS 批式 750)— 均 pinned head 04a0900ff7 本地现测</li>
<li>平台已知缺口: D1 baseline evaluator 不 stage 资产;custom_inputs 在 submit 评测端全判 0%;custom 与非 custom 输入不可混</li>
<li>纪律: 平台分数仅轮内排序;ship 判决 = 本地全格 + 逐行 exact + 锚检查;launch 前后 pgrep+GPU 双查;tag 不复用</li>
</ul>
</body></html>"""
out = HERE / "KF_CAMPAIGNS_INDEX.html"
out.write_text(PAGE)
print(f"wrote {out} ({len(PAGE)} bytes)")
