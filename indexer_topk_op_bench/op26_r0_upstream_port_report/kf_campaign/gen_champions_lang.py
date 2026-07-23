#!/usr/bin/env python3
"""Idempotently upgrade KF_PROCESS_LOG.html:
  1. Working 3-state language toggle (双语 default / EN / 中文), CSS-only.
     - EN mode hides .zh /.lang-cn /.cng; 中文 mode hides .en /.lang-en.
     - Wraps the duplicated EN & 中文 copies of sections 1-3 so they really switch.
  2. New section 8: per-round KF champion operators + brief algorithm flows
     (C1 / R3 / R4 ladders), fully .en/.zh tagged.
Marker-based; re-run any time.
"""
import re, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, 'KF_PROCESS_LOG.html')
h = open(REPORT).read()

def B(en, zh):
    return f'<span class="en">{en}</span><span class="zh">{zh}</span>'

# ---------------- 1a. CSS ----------------
NEWCSS = '''/* KF-LANGCSS:START */
.langbar{position:fixed;top:12px;right:18px;z-index:10;display:flex;gap:4px;background:#fff;border:1.5px solid var(--acc);border-radius:20px;padding:3px 6px;box-shadow:0 1px 4px rgba(0,0,0,.12)}
.langbar label{cursor:pointer;padding:2px 12px;border-radius:14px;font-size:.88em;color:#333;user-select:none}
#lang-bi2:checked~.langbar label[for=lang-bi2],#lang-en2:checked~.langbar label[for=lang-en2],#lang-zh2:checked~.langbar label[for=lang-zh2]{background:var(--acc);color:#fff}
#lang-en2:checked~.content .zh,#lang-en2:checked~.content .lang-cn,#lang-en2:checked~.content .cng{display:none!important}
#lang-zh2:checked~.content .en,#lang-zh2:checked~.content .lang-en{display:none!important}
/* KF-LANGCSS:END */'''
OLDCSS = ('''/* bilingual CSS-only toggle: EN default */
.lang-cn { display:none; }
#lang-toggle:checked ~ .content .lang-cn { display:block; }
#lang-toggle:checked ~ .content .lang-en { display:none; }''')
if '/* KF-LANGCSS:START */' in h:
    h = re.sub(r'/\* KF-LANGCSS:START \*/.*?/\* KF-LANGCSS:END \*/', lambda m: NEWCSS, h, flags=re.S)
else:
    assert OLDCSS in h, 'old toggle CSS not found'
    h = h.replace(OLDCSS, NEWCSS)

# ---------------- 1b. UI ----------------
NEWUI = ('<!-- KF-LANGUI:START -->'
         '<input type="radio" name="langsel" id="lang-bi2" checked style="display:none">'
         '<input type="radio" name="langsel" id="lang-en2" style="display:none">'
         '<input type="radio" name="langsel" id="lang-zh2" style="display:none">'
         '<div class="langbar"><label for="lang-bi2">双语</label>'
         '<label for="lang-en2">EN</label><label for="lang-zh2">中文</label></div>'
         '<!-- KF-LANGUI:END -->')
OLDUI = ('<input type="checkbox" id="lang-toggle" style="display:none">\n'
         '<label class="toggle-label" for="lang-toggle">EN / 中文</label>')
if '<!-- KF-LANGUI:START -->' in h:
    h = re.sub(r'<!-- KF-LANGUI:START -->.*?<!-- KF-LANGUI:END -->', lambda m: NEWUI, h, flags=re.S)
else:
    assert OLDUI in h, 'old toggle UI not found'
    h = h.replace(OLDUI, NEWUI)

# ---------------- 1c. sections 1-3 already carry original lang-en / lang-cn
# wrappers in the source HTML — just verify; also undo the misaligned wrap a
# previous version of this script inserted (double divs broke balance).
h = h.replace('<!-- KF-ENBLOCK:START --><div class="lang-en">', '')
h = h.replace('</div><!-- KF-ENBLOCK:END --><!-- KF-CNBLOCK:START --><div class="lang-cn">', '')
h = h.replace('</div><!-- KF-CNBLOCK:END -->', '')
assert h.count('<div class="lang-en">') == 1 and h.count('<div class="lang-cn">') == 1

# h1 subtitle: tag EN/中文 lines
SUB_OLD = ('target: +20% avg vs PR#16457 GVR on §4 real data (865 cells), zero regression<br>'
           '目标:§4 真实数据(865 格)上对 PR#16457 GVR 平均 +20%,零回归')
SUB_NEW = ('<span class="en">target: +20% avg vs PR#16457 GVR on §4 real data (865 cells), zero regression</span>'
           '<span class="zh">目标:§4 真实数据(865 格)上对 PR#16457 GVR 平均 +20%,零回归</span>')
if SUB_OLD in h:
    h = h.replace(SUB_OLD, SUB_NEW)

# ---------------- 2. section 8: round champions ----------------
def card(tag, verdict, algo_en, algo_zh):
    return (f'<tr><td style="white-space:nowrap">{tag}</td>'
            f'<td style="white-space:nowrap">{verdict}</td><td>{B(algo_en, algo_zh)}</td></tr>')

C1 = ''.join([
 card('<b>R1 winner</b><br><code>ba1020ce</code> “hybrid_v28”<br>(n3 opus-4.8, a005)',
      'gm <b>1.3662×</b><br>85 regs · 865/865 exact',
      'Hybrid radix-select, <i>pre_idx ignored</i>. n≤16384: single-CTA smem radix — fused first 10-bit round, '
      'warp-parity histogram replicas. Large n: cooperative multi-CTA 3-pass 11/11/10-bit radix, float4 loads, '
      'epoch grid barrier; histogram re-zero folded into the previous call&rsquo;s collect tail (one grid.sync saved). '
      'All 85 regressions = the 16k dispatch-boundary artifact (SMALL_N cutoff 3 elements below the 16k npad).',
      '混合 radix-select,完全忽略 pre_idx。小 N(≤16384)单 CTA smem radix:首轮 10-bit 融合、warp 奇偶直方图副本;'
      '大 N 协作多 CTA 三遍 11/11/10-bit radix + float4 载入 + epoch 网格屏障;直方图复零折叠进上次调用的 collect 尾'
      '(省一次 grid.sync)。85 个回归全部是 16k 派发边界伪影(切换点比 16k npad 低 3 个元素)。'),
 card('<b>R2 winner</b><br><code>c74fb3c0</code> (a003)',
      'gm <b>1.6713×</b><br>5 regs · 865/865 exact',
      'Same lineage; <b>deletes the 16384 single-CTA rung</b> — cooperative path from 8448 up (~148 CTAs at 16k) '
      'with early-exit, flipping the round-1 “more CTAs is worse” consensus. Runner-up 0260cee7 (1.6421, 4 regs) '
      'kept the rung; the full grid, not the campaign-internal subset metric, decided the ranking.',
      '同血统;<b>删除 16384 单 CTA 档</b>——协作路径从 8448 起步(16k 处 ~148 CTA)+ 提前退出,'
      '翻转第 1 轮“CTA 越多越差”共识。亚军 0260cee7(1.6421,4 回归)保留该档;排名由全网格而非 campaign 内部子集口径裁定。'),
 card('<b>SHIP</b><br><code>c74f_sbx</code><br>= c74fb3c0 + graft',
      'gm <b>1.6828×</b><br><b>0 reg</b> · 865/865 exact<br>~$751 total',
      'Engineer dispatch graft: <code>topk_small&lt;17&gt;</code> @1024 threads single-CTA rung for 8448&lt;n≤16896 '
      '(3-line graft — topk_small was already blockDim.x-parameterized) heals the boundary band. '
      'First in-tree-family kernel to beat sglang v2 on the full real envelope (1.119×).',
      '工程师派发嫁接:8448&lt;n≤16896 走 <code>topk_small&lt;17&gt;</code>@1024 线程单 CTA 档(3 行嫁接,'
      'topk_small 本就参数化了 blockDim.x),治愈边界带。首个在完整真实包络击败 sglang v2 的树内家族内核(1.119×)。'),
])
R3T = ''.join([
 card('<code>09d13c81</code> (r2)', 'gm 1.7553× · 0 reg',
      '<b>Regular launch + fence-less sense-token spin grid barrier</b> (host generation counter ⇒ tokens never '
      'collide across launches, no per-launch reset), grid sized to co-residency. Every formally-ordered variant '
      'measured 8–11% slower — the win IS the omitted ordering (safety argument in R3_LEDGER).',
      '<b>常规 launch + 免序 sense 令牌自旋网格屏障</b>(host 代数计数器,跨 launch 永不撞号,无需逐 launch 复位),'
      '网格按共驻留定容。一切形式化内存序变体慢 8–11%——胜利正来自省掉排序(安全论证见 R3_LEDGER)。'),
 card('<code>30e79029</code> (r3)', 'gm 1.7714× · 0 reg',
      '+ <b>contiguous-slice scan</b>: per-block contiguous float4 slices replace grid-stride interleave — better cold-data locality.',
      '+ <b>连续切片扫描</b>:每块扫连续 float4 切片,替代 grid-stride 交错,冷数据局部性更好。'),
 card('<code>becdc5c7</code> (r3)', 'gm 1.7848× · 0 reg',
      '<b>Adaptive post-pass-0 finish + register-cached row</b>: one 11-bit MSB histogram pass, then 3-way dispatch '
      'on boundary-bucket size T — whole-bucket direct write (1 barrier) / T≤4096 smem-staged compaction + '
      'non-spinning rendezvous + last-arriver single-block 21-bit refine / classic 11/11/10 ladder. Keys live in '
      'registers (1×float4 + tail scalar per thread) — zero global re-reads across passes.',
      '<b>自适应收尾 + 寄存器缓存整行</b>:一遍 11-bit MSB 直方图后按边界桶大小 T 三路分派——整桶直写(1 屏障)/ '
      'T≤4096 smem 压缩 + 非自旋会合 + 末位到达块独占 21-bit 精化 / 经典 11/11/10 梯子;keys 全程驻寄存器,后续 pass 零全局重读。'),
 card('<code>compA</code> (engineer composite)', 'gm 1.7873× · 1 reg (0.999)',
      'becd ⊕ 30e7 dispatch: K=2048 ∧ 16896&lt;n≤140000 → the 30e7 ladder (v32 mid-n prefers it).',
      '工程师复合分派 #1:becd ⊕ 30e7——K=2048 且 16896&lt;n≤140000 走 30e7 梯子(v32 中档偏好)。'),
 card('<b>SHIP</b> <code>compB</code>', 'gm <b>1.8267×</b><br><b>0 reg</b> (min 1.140) · 865/865<br>$764.66',
      '<code>aef33fac</code> (= becd + <b>topk_mid single-CTA two-level-histogram tail-selection rungs</b> for '
      '4·k≤n, 8195≤n≤16387 — heals the weakest band +19%; the regressing n≈4099 mid&lt;1&gt; rung gated out) '
      '⊕ 30e7 K2048 dispatch. +8.1% over the campaign-1 champion.',
      '<code>aef33fac</code>(= becd + <b>topk_mid 单 CTA 两级直方图尾部选择档</b>,4·k≤n 且 8195≤n≤16387,'
      '治愈最弱档 +19%;实测回退的 n≈4099 mid&lt;1&gt; 档已裁除)⊕ 30e7 K2048 分派。对第一期冠军净增 +8.1%。'),
])
R4T = ''.join([
 card('ladder v5 → v27<br>(rounds 1–2)', 'gm 1.295→1.582<br>regs 78→2',
      'Cold-start lineage under skeleton hard-lock (denominator = pinned head 04a0900ff7): '
      'v5 1.295 (78 regs) → v14 1.343 (24) → r2_wd 1.441 (19) → v25 1.521 (2) → v27 1.582 (2) → r3_a003 1.618 (1). '
      'Several campaign-1 conclusions independently re-discovered (1024-thread small-N rung, 8–16 CTA large-N, prior-free dead end).',
      '骨架硬锁下的冷启动血统(分母 = pin 死的 head 04a0900ff7):判决链 v5 1.295(78 回归)→ v14 1.343(24)→ '
      'r2_wd 1.441(19)→ v25 1.521(2)→ v27 1.582(2)→ r3_a003 1.618(1)。多项第一期结论被独立重发现'
      '(1024 线程小 N 档、8–16 CTA 大 N、prior-free 死路)。'),
 card('<b>SHIP champion</b><br><code>28dc11f6</code><br>“r3 perK-dispatch”', 'gm <b>1.6531×</b><br>0 real reg · 865/865<br>$1110.62',
      'Pipeline: P1 hint-CCDF two-level histogram → 8-quantile threshold ladder; P2 single-pass 8-threshold '
      'counting + log-secant bracket (~9×/pass shrink) + plateau exact fallback; P3 DSMEM collect (odd/even dual '
      'bank); P4 CTA0 4×8-bit radix + tie-ticket. Dispatch on (npad, K) only: ≤12288 direct 1×1024; ≤262144 '
      '<b>register-resident GVR</b> 1/4/8/16-CTA (whole row in registers, multi-pass zero re-scan, measured '
      'per-(tier,K) AR6/AR8 ladder); &gt;262144 streaming 16-CTA. Compliance note: the direct path consumes no '
      'pre_idx (analytic degenerate limit at kC≥npad) — adjudicated compliant.',
      '流水线:P1 hint-CCDF 两级直方图 → 8 分位阈值梯;P2 单 pass 8 阈值计数 + log-secant 括号(每 pass ~9× 收缩)'
      '+ plateau 精确回退;P3 DSMEM 收集(奇偶双 bank);P4 CTA0 4×8-bit radix + tie-ticket。仅按 (npad, K) 派发:'
      '≤12288 direct 1×1024;≤262144 <b>寄存器驻留 GVR</b> 1/4/8/16-CTA(整行进寄存器,多 pass 零重扫,'
      'per-(tier,K) 实测 AR6/AR8 梯);&gt;262144 流式 16-CTA。合规附注:direct 路径不消费 pre_idx'
      '(kC≥npad 的解析退化极限),操作员裁定合规。'),
])
CHAMPS = f'''<!-- KF-CHAMPS:START -->
<h2 id="sec-champs">8 · Round-champion operators &amp; algorithms / 各轮冠军算子与算法一览</h2>
<style>#champs td,#champs th{{vertical-align:top}}#champs h3{{margin-top:1.4em}}</style>
<div id="champs">
<p>{B('One card per harvested round champion across all campaigns, with the full-865 verdict (nsys cold-L2, paired vs the campaign&rsquo;s PR-head denominator) and a compressed algorithm flow. Details: §5 (campaign-1 round log), §7.1–7.2 (R3 ladder), §7.9 (BS-ext arms), R4_CLOSEOUT.md.',
      '各战役逐轮收割冠军一卡一览:全 865 格终判(nsys 冷-L2,对各期 PR-head 分母同卡配对)+ 压缩算法流程。细节见 §5(第一期逐轮日志)、§7.1–7.2(R3 阶梯)、§7.9(BS-ext 各臂)、R4_CLOSEOUT.md。')}</p>

<h3>{B('Campaign-1 <code>tfb91bvwm…</code> (bs1-real, denominator PR head @e6fdbfac3d)', '第一期 <code>tfb91bvwm…</code>(bs1-real,分母 PR head @e6fdbfac3d)')}</h3>
<table><tr><th style="width:17%">{B('round / candidate','轮次 / 候选')}</th><th style="width:16%">{B('full-865 verdict','全 865 格终判')}</th><th>{B('algorithm flow','算法流程')}</th></tr>{C1}</table>

<h3>{B('R3 <code>e5q1zgrf…</code> (beyond-champion, denominator current head b14ec40e1b) — increment ladder', 'R3 <code>e5q1zgrf…</code>(超越冠军,分母最新 head b14ec40e1b)——增量阶梯')}</h3>
<table><tr><th style="width:17%">{B('candidate','候选')}</th><th style="width:16%">{B('full-865 verdict','全 865 格终判')}</th><th>{B('algorithm increment','算法增量')}</th></tr>{R3T}</table>

<h3>{B('R4 <code>pra6srbd…</code> (lineage-2 coldstart, skeleton hard-lock, denominator pinned 04a0900ff7)', 'R4 <code>pra6srbd…</code>(第二血统冷启动,骨架硬锁,分母 pin 死 04a0900ff7)')}</h3>
<table><tr><th style="width:17%">{B('round / candidate','轮次 / 候选')}</th><th style="width:16%">{B('full-865 verdict','全 865 格终判')}</th><th>{B('algorithm flow','算法流程')}</th></tr>{R4T}</table>

<p>{B('<b>BS-ext</b> (local engineering campaign): the batched five-arm operator <code>run_batch_auto</code> — grid.y batching / ext_v4 row-teams / tp4 exact-hist fused 2-pass / tp3 fused sampled single-pass / tp2 3-kernel pipeline + measured (N,BS) dispatch — TARGET gm 2.083× over BS 2–1024; full algorithm flows in <a href="#sec-7bsext">§7.9</a>. <b>R5</b> (lineage-2 BS campaign, <code>vk9m3tet…</code>): in flight — champions will be appended here as rounds close.',
      '<b>BS-ext</b>(本地工程战役):批式五臂算子 <code>run_batch_auto</code>——grid.y 批化 / ext_v4 row-teams / tp4 精确直方图双遍 / tp3 融合采样单遍 / tp2 三核流水 + 实测 (N,BS) 派发表——全段 TARGET gm 2.083×;完整算法流程见 <a href="#sec-7bsext">§7.9</a>。<b>R5</b>(第二血统 BS 战役 <code>vk9m3tet…</code>):进行中——各轮冠军收口后回填至此。')}</p>
</div>
<!-- KF-CHAMPS:END -->'''

if '<!-- KF-CHAMPS:START -->' in h:
    h = re.sub(r'<!-- KF-CHAMPS:START -->.*?<!-- KF-CHAMPS:END -->', lambda m: CHAMPS, h, flags=re.S)
else:
    i = h.find('<!-- KF-R3:END -->')
    assert i >= 0
    h = h[:i] + CHAMPS + '\n' + h[i:]

open(REPORT, 'w').write(h)
print('OK bytes', len(h))
