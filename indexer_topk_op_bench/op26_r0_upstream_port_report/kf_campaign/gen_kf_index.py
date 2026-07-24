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
      goal="BS=1 865 真实格 beat 旧 PR head(e6fdbfac3d),nsys 冷判", cons="GVR 骨架;无 coldstart 限制",
      code=f"{KFP}/harvest/r2_c74fb3c0 + c74f_sbx(graft)", data=f"{KFP}/grid_c74fsbx.csv 等",
      note="round-harvest-verdict 流程首建;两次 grid 因双 driver 作废(纪律起源)"),
 dict(a="kf1", id="k7rhy79h015gn3fpqv5cs154w8", name="gvr-topk-r3(baseline 验证失败)", st="Failed", rd=0,
      internal="—", local="—", t0="07-21 13:36", t1="+40s", cost="~0", own="本会话系",
      goal="R3 起跑(baseline-solution 现测)", cons="—", code="—", data="—",
      note="平台缺口 D1 首次发现: baseline evaluator 不 stage 资产"),
 dict(a="r3", id="e5q1zgrfhs0z57dj6850kc444r", name="gvr-topk-r3", st="Cancelled", rd=4,
      internal="1.1111", local="operator 复合 compB **1.8267**(865 格, 0 回退)", t0="07-22 03:15", t1="07-22 ~15:49",
      cost="$761", own="本会话系(第一 lineage R3)",
      goal="从 champion c74f_sbx 再推高(baselines=平台 champion 时延)", cons="骨架放宽(campaign-1 先例)",
      code=f"{KFP}/harvest/(aef33fac/becd/30e7 系)+ compB", data=f"{KFP}/grid_r3grid*.csv",
      note="平台 4 轮未超 harvested aef33fac;compB 为 operator 收口拼装"),
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
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)


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
<tr><th>代码位置</th><td class="mono">{c['code']}</td></tr>
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


def onode(x, y, w, h, lines):  # output artifact node (no link target card)
    tspan = "".join(
        f'<tspan x="{x + w/2}" dy="{"1.2em" if i else 0}">{ln}</tspan>'
        for i, ln in enumerate(lines))
    ty = y + h / 2 - (len(lines) - 1) * 8 + 4
    return (f'<g class="o"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9"/>'
            f'<text x="{x + w/2}" y="{ty}" text-anchor="middle">{tspan}</text></g>')


def edge(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="5,4"' if dash else ""
    return f'<path d="M{x1},{y1} C{(x1+x2)//2},{y1} {(x1+x2)//2},{y2} {x2},{y2}" class="e"{d}/>'


G = []
# Lane labels
for y, lab in [(30, "第一 lineage(BS=1)"), (208, "第二 lineage R4(BS=1 冷启动)"),
               (386, "R5(BS=2-1024)"), (596, "另一会话")]:
    G.append(f'<text x="12" y="{y}" class="lane">{lab}</text>')
# Lineage 1
G.append(node(30, 44, 190, 56, "c1", ["campaign-1", "bs1-real · 1.34 内部"]))
G.append(node(280, 44, 140, 44, "kf1", ["r3 baseline", "Failed(D1)"], cls="n bad"))
G.append(node(470, 44, 180, 56, "r3", ["gvr-topk-r3", "4轮 · 1.11 内部"]))
G.append(onode(720, 44, 220, 56, ["产出: compB 1.8267×", "(865 格 · 0 回退)"]))
G.append(edge(220, 72, 280, 66, dash=True))
G.append(edge(220, 72, 470, 72))
G.append(edge(650, 72, 720, 72))
# Lineage 2 (R4)
G.append(node(30, 222, 190, 56, "r4f", ["R4 baseline-eval", "Failed(D1 复现)"], cls="n bad"))
G.append(node(280, 222, 200, 56, "r4", ["gvr-topk-cold60", "3轮 · 1.3701 内部"]))
G.append(onode(560, 210, 230, 46, ["champion 28dc11f6", "865格 1.6531× · 0 回退"]))
G.append(onode(560, 268, 230, 40, ["fork: kf/r4-champion-final-bs1"]))
G.append(onode(820, 210, 220, 46, ["R4_CLOSEOUT.md ·", "R4_CHAMPION_BS1_REPORT.html"]))
G.append(edge(230, 250, 280, 250, dash=True))
G.append(edge(480, 250, 560, 233))
G.append(edge(480, 250, 560, 288))
G.append(edge(790, 233, 820, 233))
G.append(edge(720, 100, 350, 210, dash=True))  # 死路/陷阱注入(限清单)
# R5 chain
G.append(node(30, 400, 190, 56, "r5a", ["bs2x v1", "custom_inputs 0%"], cls="n bad"))
G.append(node(280, 400, 190, 56, "r5b", ["bs2x v2", "3轮 · 0.98 · DQ"]))
G.append(node(530, 400, 200, 56, "r5c", ["bs2x-v3 fork", "rounds3-5 · 1.0756"]))
G.append(onode(800, 388, 240, 46, ["champion 156ab438", "750格 0.9862 · 全 exact"]))
G.append(onode(800, 446, 240, 40, ["fork: kf/r5-champion-bs-combined"]))
G.append(onode(800, 498, 240, 40, ["R5_CLOSEOUT.md · grid_r5g_final.csv"]))
G.append(edge(220, 428, 280, 428))
G.append(edge(470, 428, 530, 428))
G.append(edge(730, 428, 800, 411))
G.append(edge(730, 428, 800, 466))
G.append(edge(730, 428, 800, 518))
G.append(edge(660, 256, 380, 398, dash=True))  # R4 champion digest -> R5 prompt
G.append(edge(430, 456, 530, 445, dash=True))  # steering fork
# other session
G.append(node(30, 612, 250, 50, "d0", ["dsl-fp4/fp8 ×7(调试)", "Failed×5 · r0×2"], cls="n other"))
G.append(node(340, 612, 200, 50, "d2", ["dsl-fp8 · 3轮 · 0.83"], cls="n other"))
G.append(node(340, 674, 200, 50, "d1", ["dsl-fp4 · 3轮 · 1.04"], cls="n other"))
G.append(node(620, 612, 240, 50, "fr", ["gvr-topk-pr16457-fresh", "3轮 · 0.84"], cls="n other"))
G.append(edge(280, 637, 340, 637))
G.append(edge(280, 637, 340, 699))

SVG = ('<svg viewBox="0 0 1080 740" xmlns="http://www.w3.org/2000/svg" style="width:100%">'
       '<style>'
       '.n rect{fill:#eef3fb;stroke:#7f93b3;stroke-width:1.3}'
       '.n text{font:12.5px sans-serif;fill:#1a2233}'
       '.n:hover rect{fill:#dbe7fb;stroke:#1a2233;stroke-width:2}'
       '.n.bad rect{fill:#fdeee3;stroke:#c98a4b}'
       '.n.other rect{fill:#f2eefb;stroke:#8d7fb3}'
       '.o rect{fill:#e7f5ec;stroke:#5c9a74;stroke-width:1.2}'
       '.o text{font:12px sans-serif;fill:#173d27}'
       '.e{fill:none;stroke:#8fa0ba;stroke-width:1.5}'
       '.lane{font:600 13px sans-serif;fill:#55627a}'
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
