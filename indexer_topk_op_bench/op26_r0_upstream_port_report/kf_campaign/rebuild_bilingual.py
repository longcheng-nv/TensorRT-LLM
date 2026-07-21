# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot repair + restructure of KF_PROCESS_LOG.html for bilingual parity.

Damage being fixed: gen_final_section.py's pre-fix idempotency bug ate the
<div class="lang-cn"> opening tag and CN sections 1-3 across repeated runs.

New structure (content parity by construction):
  shared header + §0 diagrams (bilingual captions)     [always visible]
  <div class="lang-en"> §1-3 EN prose </div>           [toggle]
  <div class="lang-cn"> §1-3 CN prose (restored) </div>[toggle]
  shared §4 Timeline / 时间线        (CN gloss per row)
  shared §5 Round-by-round / 逐轮结果 (CN gloss per row)
  shared §6 Final verdict / 终审      (gen_final_section.py, bilingual)

Inputs: current KF_PROCESS_LOG.html + pristine.html (git show 8601b19d2e).
Run gen_final_section.py AFTER this script.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/claude-150713/-home-scratch-loncheng-gpu-workspace-perf-workloads"
               "-DSV4-TensorRT-LLM/d7b4b629-e643-40dd-8351-1f01ea9f950c/scratchpad")

CN4 = [
    "kf 1.0.0 安装完成;SSO 认证通过;技能包就位。",
    "导出 28 个 cell(资产 9.8 MB);撰写 definition/workload/baselines/prompt。",
    "prepare 通过(修复 shape 维度为 axis 名字符串,常量轴 b=1)。",
    "战役启动:tfb91bvwm972kfyf1bc1trj5e0,phase Pending。",
    "第 1 轮运行,6 个 agent 启动;挂载后台 monitor(2 分钟轮询 → status_log.jsonl)。",
    "本机验证臂就绪(umbriel-b200-027):gvrpkg_head/ = PR 分支 HEAD e6fdbfac3d kernel 快照。",
]

CN5 = [
    "首轮爬坡:四个候选 0.29→0.95×,均低于基线,未收割。",
    "混合 radix-select:小 N 单 CTA smem + 大 N 多 CTA(11-bit 全局直方图、epoch 网格屏障),完全忽略 pre_idx。"
    "本地 28 格探针 geomean 冷 1.372×/warm 1.297×,28/28 精确;campaign 内部口径 1.04× 严重低估 nsys 核时间比(其计时含每次调用开销),导向偏保守但方向安全。",
    "全网格终审:geomean 1.316×(≥1.20 达标),865/865 精确,但 137 格回归 — 全部位于 N∈[16k,65k] 单CTA→多CTA 切换带。"
    "导向方向:单 CTA smem 路径上推(B200 smem 227KB/CTA 可容 ~48k)或更轻的中 N 协作方案。",
    "混合谱系第 1 轮演化;20 格探针显示回归带整体上移(v32_16k 0.75→0.97,v32_32k 0.90→1.22-1.31),大 N 端小幅回吐;硬核残留 = N≈16k 低 hit(K=512/1024)。",
    "多 CTA 侧重写为 3 遍 11/11/10-bit 协作 radix + float4 载入;单 CTA 切换点升至 n≤16384;直方图复零挪到上一调用 collect 尾。"
    "28 格探针冷 1.436×/warm 1.405×,28/28 精确,探针集内回归清零;疑似 sb 分支 8448<n≤16384 覆盖洞交由全网格裁决。"
    "运维注:/tmp/gvrlayers overlay 被 tmp 清理抹掉致 make_fragment 报错,已从 NFS userbase 重建。",
    "第 1 轮收束 → 第 2 轮启动(新增 7 agent,共 13)。胜者 a005(opus-4-8)ba1020ce 1.0768;仅 a000(fable-5,41a94aaa 1.041)另过 1.0。成本 ~$458 / 13.3 agent-hours。"
    "多 agent 独立共识死路与我们 6 月战役史互证:pre_idx warm-hint 加速 radix 无效;B200 私有直方图输给裸 smem 原子;n≤16k 多 CTA 无效;大 N CTA 过多变差。"
    "评测地板 ~15µs 压缩了小 N 的表观收益(campaign 口径 ≈ 本地 nsys 的 1/1.3-1.4)。",
    "全网格终审:geomean 1.3662×,865/865 精确,回归 137→85,且全部塌缩为 dispatch 边界伪影:候选切换点 SMALL_N=16384 恰比网格 16k npad(16387)低 3 个元素,"
    "这些格子跌进仅 ~8 CTA 的协作路径。本地 sb17 探针(SMALL_N→16896 + sb<22>@768 线程)启动验证修复方向。",
    "工程师探针双重确认切换点上推方向:sb17(768 线程×KPT22)flash_64k 0.74-0.79→0.85-0.92;sb17b(1024 线程×KPT17)→0.89-0.99,pro_64k→0.88-1.06,"
    "v32_16k 净赢 1.32-1.47;全部精确。残余 flash/pro_64k 1-11% 差距与 N=32771 档留作第 2 轮战场/导向素材。",
    "第 2 轮新谱系(a002/gpt-5.6-sol 第 1 轮仅 0.29,借共享洞察重生):bottom3 补集特化(n−k=3 小行)+ sb 阶梯补齐 8448 洞 + 协作核逐遍 early-exit"
    "(边界 bin 计数=剩余配额即跳过后续 pass 与 grid.sync)。28 格探针冷 1.699×/warm 1.647×,28/28 精确,最低格 1.18 — 战役首个零回归探针。大 N 跃升(v32_256k 1.43→1.96)。",
    "(后被作废)全网格 geomean “1.7758”、2 回归 — 经跨 run 锚检发现 PR 臂整档虚慢 25-50%,测量污染,作废重测。",
    "(后被作废)0197c2a1 全网格 “1.7101”、3 回归;本地冠军仍为 0260cee7;campaign 内部排名(0197>0260)被全网格翻转 — 子集口径 ≠ 部署包络。同样因污染作废。",
    "测量污染事件:跑格期间在 GPU6/7 发探针(双 driver,重演 op22 事故)→ r2a/r2b 两个终审作废。"
    "新常设协议:本地 GPU 工作全串行;每次网格做逐档 pr_cold 锚检(对同 PR 臂历史 run,须 ±3% 内);先安静 GPU 探针再跑。",
    "0260cee7 干净复测(锚检再抓 2 个外部占卡污染档 N=131087/163775,于安静 GPU 重测回填,回填锚 1.022/1.005):geomean 1.6421×,865/865 精确,4 回归"
    "(全在 N=16387,最深 flash_64k_L28 0.877)。本地排名:0260cee7 > sbx 1.6053/8回归 > ba1020ce 1.3662/85回归。",
    "c74fb3c0 干净终审:geomean 1.6713×,865/865 精确,5 回归(最深 0.846)— 按 geomean 登顶。对手横评(compare_rivals.py,PR 臂逐格归一,校准中位 1.010):"
    "vs sglang_v2 1.111(胜 569/865,首个在完整真实包络击败 sglang v2 的树内家族内核;sglang 残余堡垒 = 中 N 8k-32k);vs radix_cutedsl 1.611(胜 864/865)。"
    "精确性优势:冠军无条件精确,sglang v2 受 kMaxNumTie=2048 条件精确限制。",
    "达成 ship 门:c74f_sbx = 第 2 轮冠军 c74fb3c0 + 工程师 sb<17>@1024 线程档(8448<n≤16896,其 topk_small 本就以 blockDim.x 参数化,嫁接仅 3 行)。"
    "geomean 1.6828×,865/865 精确,零冷回归(2 个 0.993/0.998 边缘格经 60 rep 裁决为噪声:1.068/1.042);锚检干净;边界档愈合 flash_64k_L28 0.846→1.253。"
    "warm 轴次要备注:少数 64k 格 0.93-0.94(BS=1 decode 以冷-L2 为准绳)。campaign 继续跑,后续收割须击败该组合。",
]


def gloss_rows(seg, glosses):
    rows = list(re.finditer(r'<tr>(.*?)</tr>', seg, re.S))
    data = [m for m in rows if '<td' in m.group(1)]
    assert len(data) == len(glosses), (len(data), len(glosses))
    out, last = [], 0
    for m, g in zip(data, glosses):
        body = m.group(1)
        k = body.rfind('</td>')
        body = body[:k] + f'<br><span class="cng">{g}</span>' + body[k:]
        out.append(seg[last:m.start()] + '<tr>' + body + '</tr>')
        last = m.end()
    out.append(seg[last:])
    return ''.join(out)


def main():
    h = (HERE / "KF_PROCESS_LOG.html").read_text()
    p = SCRATCH.joinpath("pristine.html").read_text()

    # pieces from current file
    head_end = h.find('<div class="lang-en">')
    header = h[:head_end]
    en_body = h[head_end:h.find('<h2>4 · Timeline</h2>')]          # includes opening div
    sec4 = h[h.find('<h2>4 · Timeline</h2>'):h.find('<h2>5 · Round-by-round results</h2>')]
    sec5 = h[h.find('<h2>5 · Round-by-round results</h2>'):h.find('<h2>6 · Final verdict</h2>')]

    # restored CN prose 1-3
    cn = p[p.find('<div class="lang-cn">'):]
    cn = cn[:cn.find('<h2>4 · 时间线')] + '</div>\n'

    # close EN div after §3 (strip any trailing partial content after last EN §3 block)
    en_body = en_body.rstrip() + '\n</div>\n'

    # bilingual heading + CN gloss for shared sections
    sec4 = sec4.replace('<h2>4 · Timeline</h2>', '<h2>4 · Timeline / 时间线</h2>')
    sec4 = gloss_rows(sec4, CN4)
    sec5 = sec5.replace('<h2>5 · Round-by-round results</h2>',
                        '<h2>5 · Round-by-round results / 逐轮结果</h2>')
    sec5 = gloss_rows(sec5, CN5)

    # h1 bilingual line + gloss style
    header = header.replace(
        'zero regression</small></h1>',
        'zero regression<br>目标:§4 真实数据(865 格)上对 PR#16457 GVR 平均 +20%,零回归</small></h1>')
    header = header.replace('</style>',
        '.cng { color:#555; font-size:0.93em; display:inline-block; margin-top:3px; }\n</style>')

    doc = (header + en_body + '\n' + cn
           + '\n' + sec4 + sec5
           + '<h2>6 · Final verdict / 终审</h2>\n'
           + '\n</div>\n</body>\n</html>\n')
    (HERE / "KF_PROCESS_LOG.html").write_text(doc)
    print("rebuilt:", len(doc), "bytes; run gen_final_section.py next")


if __name__ == "__main__":
    main()
