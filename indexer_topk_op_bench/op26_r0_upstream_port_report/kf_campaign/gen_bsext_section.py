#!/usr/bin/env python3
"""Idempotently inject into KF_PROCESS_LOG.html:
  1. <!-- KF-MASTERTL --> master optimization timeline (top, before section 0)
  2. <!-- KF-BSEXT -->  section 7.9: compB BS-ext campaign final operator + perf
CSS-only (no <script>). Data: kf_bs_scaling/ext/final_bs.csv (last-writer safe:
only replaces its own marker blocks). Re-run any time.
"""
import csv, math, collections, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, 'KF_PROCESS_LOG.html')
FINAL = os.path.join(HERE, '..', 'kf_bs_scaling', 'ext', 'final_bs.csv')

BSL = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# ---------------- data ----------------
rows = list(csv.DictReader(open(FINAL)))
d = collections.defaultdict(dict)
for r in rows:
    d[(r['model'], r['isl'], int(r['N']), int(r['K']), int(r['BS']))][r['op']] = float(r['us'])
cells = sorted(set(k[:4] for k in d), key=lambda c: (c[0], c[2]))
sp = {}           # (cell, BS) -> speedup
gm_bs = {}        # BS -> pooled geomean
for bs in BSL:
    vals = []
    for c in cells:
        k = c + (bs,)
        s = d[k]['gvr_pr'] / d[k]['auto']
        sp[(c, bs)] = s
        vals.append(s)
    gm_bs[bs] = math.exp(sum(map(math.log, vals)) / len(vals))
allv = [sp[(c, b)] for c in cells for b in BSL]
pooled = math.exp(sum(map(math.log, allv)) / len(allv))

RIVAL = os.path.join(HERE, '..', 'rival_bs_layers.csv')
CELL_L = {('flash', '256k', 65538, 512): '22', ('flash', '512k', 131075, 512): '22',
          ('pro', '512k', 131075, 1024): '30', ('pro', '1024k', 262127, 1024): '30'}
rrows = list(csv.DictReader(open(RIVAL)))
rd = collections.defaultdict(dict)
for r in rrows:
    rd[(r['model'], r['isl'], r['L'], int(r['BS']))][r['op']] = float(r['us'])
rsp = {}   # (arm, cell, BS) -> speedup vs gvr_pr (within-REPORT b200-027 ratio)
rgm = {}   # (arm, BS) -> pooled geomean
for arm in ['sglang_v2', 'flashinfer_topk']:
    for bs in BSL:
        vals = []
        for c, L in CELL_L.items():
            k = (c[0], c[1], L, bs)
            s = rd[k]['gvr_pr'] / rd[k][arm]
            rsp[(arm, c, bs)] = s
            vals.append(s)
        rgm[(arm, bs)] = math.exp(sum(map(math.log, vals)) / len(vals))

CELL_LBL = {('flash', '256k', 65538, 512): 'flash 256k · N=65538 · K512',
            ('flash', '512k', 131075, 512): 'flash 512k · N=131075 · K512',
            ('pro', '512k', 131075, 1024): 'pro 512k · N=131075 · K1024',
            ('pro', '1024k', 262127, 1024): 'pro 1024k · N=262127 · K1024'}
CELL_COLOR = {('flash', '256k', 65538, 512): '#7fb3e8',
              ('flash', '512k', 131075, 512): '#2a78d6',
              ('pro', '512k', 131075, 1024): '#f0a8c0',
              ('pro', '1024k', 262127, 1024): '#e87ba4'}

# ---------------- interactive SVG curve (CSS-checkbox toggles, no JS) ------
W, H, ML, MR, MT, MB = 760, 340, 52, 14, 18, 40
PW, PH = W - ML - MR, H - MT - MB
YMAX = 4.2
def X(bs): return ML + PW * (math.log2(bs) - 1) / 9.0
def Y(v):  return MT + PH * (1 - min(v, YMAX) / YMAX)

ARMS = [  # (key, css-class, color, bold-line-label, pooled getter, per-cell getter)
    ('AUTO', '#0b0b0b', 'ours run_batch_auto (local b200-039)',
     lambda b: gm_bs[b], lambda c, b: sp[(c, b)]),
    ('SGL', '#008300', 'sglang v2 (REPORT b200-027)',
     lambda b: rgm[('sglang_v2', b)], lambda c, b: rsp[('sglang_v2', c, b)]),
    ('FI', '#eda100', 'flashinfer top-K (REPORT b200-027)',
     lambda b: rgm[('flashinfer_topk', b)], lambda c, b: rsp[('flashinfer_topk', c, b)]),
]
svg = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;background:#fcfcfb;border:1px solid #e4e3df;border-radius:6px">']
for g in [0.5, 1, 2, 3, 4]:
    lbl = f'{g:.1f}×'
    svg.append(f'<line x1="{ML}" y1="{Y(g):.1f}" x2="{W-MR}" y2="{Y(g):.1f}" stroke="#e4e3df"/>'
               f'<text x="{ML-6}" y="{Y(g)+4:.1f}" text-anchor="end" font-size="11" fill="#52514e">{lbl}</text>')
for lv, lbl, col in [(2.0, 'target avg 2.0×', '#070'), (1.2, 'floor 1.2×', '#b26a00'), (1.0, 'parity', '#8b8a85')]:
    svg.append(f'<line x1="{ML}" y1="{Y(lv):.1f}" x2="{W-MR}" y2="{Y(lv):.1f}" stroke="{col}" stroke-dasharray="5 4" stroke-width="1.2"/>'
               f'<text x="{W-MR-4}" y="{Y(lv)-4:.1f}" text-anchor="end" font-size="10.5" fill="{col}">{lbl}</text>')
for bs in BSL:
    svg.append(f'<text x="{X(bs):.1f}" y="{H-MB+16}" text-anchor="middle" font-size="11" fill="#52514e">{bs}</text>')
svg.append(f'<text x="{ML+PW/2:.0f}" y="{H-6}" text-anchor="middle" font-size="11.5" fill="#0b0b0b">BS (log2)</text>')
# thin per-cell lines (gated by both the arm chip and the cells chip)
for key, col, _lbl, _gmf, cellf in ARMS:
    for c in cells:
        pts = ' '.join(f'{X(b):.1f},{Y(cellf(c, b)):.1f}' for b in BSL)
        svg.append(f'<g class="s{key} cl"><polyline points="{pts}" fill="none" stroke="{col}" '
                   f'stroke-width="1.3" opacity="0.42"><title>{CELL_LBL[c]}</title></polyline>')
        for b in BSL:
            v = cellf(c, b)
            svg.append(f'<circle cx="{X(b):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{col}" opacity="0.42">'
                       f'<title>{CELL_LBL[c]} · BS={b}: {v:.3f}×</title></circle>')
        svg.append('</g>')
# bold pooled lines
for key, col, lbl, gmf, _cellf in ARMS:
    pts = ' '.join(f'{X(b):.1f},{Y(gmf(b)):.1f}' for b in BSL)
    svg.append(f'<g class="s{key}"><polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.6"/>')
    for b in BSL:
        v = gmf(b)
        dy = -8 if key == 'AUTO' else (14 if key == 'SGL' else -8)
        svg.append(f'<circle cx="{X(b):.1f}" cy="{Y(v):.1f}" r="3.4" fill="{col}">'
                   f'<title>{lbl} · BS={b}: pooled gm {v:.3f}×</title></circle>')
        if key == 'AUTO':
            svg.append(f'<text x="{X(b):.1f}" y="{Y(v)+dy:.1f}" text-anchor="middle" font-size="10" fill="{col}">{v:.2f}</text>')
    svg.append('</g>')
svg.append('</svg>')
chips_css = ''.join(
    f'#fx-{k}:not(:checked) ~ * .s{k}{{display:none}}'
    f'#fx-{k}:checked ~ .chips label[for=fx-{k}]{{border-color:{col};color:#0b0b0b;box-shadow:inset 0 0 0 1px {col}}}'
    for k, col, _l, _g, _c in ARMS) + (
    '#fx-CL:not(:checked) ~ * .cl{display:none}'
    '#fx-CL:checked ~ .chips label[for=fx-CL]{border-color:#555;color:#0b0b0b;box-shadow:inset 0 0 0 1px #555}')
chips = ''.join(f'<input type="checkbox" id="fx-{k}" checked>'
                for k, _c, _l, _g, _cf in ARMS) + '<input type="checkbox" id="fx-CL">'
chip_labels = ('<div class="chips">series ▸ ' + ''.join(
    f'<label for="fx-{k}"><i style="background:{col}"></i>{lbl}</label>'
    for k, col, lbl, _g, _c in ARMS) +
    '<label for="fx-CL"><i style="background:#999"></i>per-cell thin lines / 分 cell 细线</label></div>')
SVG = (f'<div class="fbsx"><style>.fbsx{{position:relative}}.fbsx .chips{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;'
       f'font-size:0.85em;color:#52514e;align-items:center}}.fbsx .chips label{{border:1.5px solid #c9c8c2;border-radius:16px;'
       f'padding:3px 12px;cursor:pointer;user-select:none}}.fbsx .chips label i{{display:inline-block;width:10px;height:10px;'
       f'border-radius:5px;margin-right:6px}}.fbsx>input{{position:absolute;opacity:0;pointer-events:none}}'
       f'{chips_css}</style>{chips}{chip_labels}' + ''.join(svg) + '</div>')

# ---------------- table ----------------
def cellfmt(v):
    if v < 1.0:  bg = '#f8d7da'
    elif v < 1.2: bg = '#fdecd3'
    elif v >= 2.0: bg = '#d9efd9'
    else: bg = 'transparent'
    return f'<td style="background:{bg}">{v:.3f}</td>'
tbl = ['<table style="border-collapse:collapse;font-size:0.88em"><tr><th style="text-align:left">cell \\ BS</th>' +
       ''.join(f'<th>{b}</th>' for b in BSL) + '</tr>']
for c in cells:
    tbl.append(f'<tr><td style="text-align:left">{CELL_LBL[c]}</td>' + ''.join(cellfmt(sp[(c, b)]) for b in BSL) + '</tr>')
tbl.append('<tr><td style="text-align:left"><b>pooled geomean</b></td>' +
           ''.join(f'<td><b>{gm_bs[b]:.3f}</b></td>' for b in BSL) + '</tr></table>')
TBL = ''.join(tbl)

# ---------------- section 7.9 ----------------
BSEXT = f'''<!-- KF-BSEXT:START -->
<h3 id="sec-7bsext">7.9 compB BS-ext campaign — batched operator / BS&gt;1 批式扩展战役(消除 §7.8 悬崖)</h3>
<style>#sec79 td,#sec79 th{{border:1px solid #ddd;padding:3px 9px;text-align:right}}#sec79 th{{background:#f4f4f2}}
#sec79 .gate{{display:inline-block;border-radius:8px;padding:8px 14px;margin:4px 6px 4px 0;font-size:0.92em;border:1.5px solid}}
#sec79 .ok{{background:#e7f6e7;border-color:#070}} #sec79 .warn{{background:#fdf3e0;border-color:#b26a00}}
#sec79 .arm{{border-left:3px solid #2a78d6;padding:2px 10px;margin:6px 0}}</style>
<div id="sec79">
<p><b>Request / 任务请求.</b> §7.8 quantified the cliff: shipped compB is a BS=1 latency specialist (crossover at BS=2, batched arms 45–125× faster at BS=1024).
This local engineering campaign (2026-07-22, umbriel-b200-039, GPU-quiet paired nsys cold-L2) set out to <b>extend the compB lineage to BS&gt;1</b>. Bars:
<b>avg ≥ 2.0× vs gvr_pr (PR#16457 arm) over BS 2–1024</b> · <b>every point ≥ 1.2×</b> · <b>zero regression</b> · ALL-row exactness.
Envelope = 4 real capture cells (flash 256k/512k, pro 512k/1024k) × BS {{2…1024}}, same real row replicated.
/ §7.8 量化了悬崖;本地工程战役目标 = 把 compB 血统扩展到 BS&gt;1:<b>对 gvr_pr 全 BS 段均值 ≥2.0×、逐点 ≥1.2×、零回退</b>,全行 exactness,nsys 冷-L2 同卡配对。</p>

<p><b>Shipped operator / 交付算子.</b> <code>run_batch_auto(logits[BS,W], n, k)</code> — a multi-arm batched exact top-K with a <b>measured (N,BS) dispatch table</b>
(branch <a href="https://github.com/longcheng-nv/TensorRT-LLM/tree/kf/compb-bs-ext"><code>kf/compb-bs-ext</code></a> @4daeefed1f, code-only; stages dd9cd928ef → f102594ba0 → 4daeefed1f):</p>
<div class="arm"><b>A · grid.y batching</b> (n ≤ 16896) — compB single-CTA tiers with row = blockIdx.y, one launch for the whole batch. Wins 1.61–1.80× at <i>every</i> BS 1–1024. / 小 n:单 CTA 梯队按 grid.y 批化,一次发射。</div>
<div class="arm"><b>B · ext_v4 row-teams</b> (large n, BS ≤ co-residency cap) — one team per row, per-row scratch slices, fence-less generation barrier; register diet <code>__launch_bounds__(512,4)</code> → 32 regs, rows_per_wave≈9; chunked waves beyond one wave (B&prime; persistent queue FALSIFIED @4b3914a4b4: chases ≤2% launch gap, pays 5–30% floor). / 大 n 小 BS:row-teams 单波,寄存器瘦身,多波 chunked(persistent queue 证伪)。</div>
<div class="arm"><b>C · tp4 exact-hist fused 2-pass</b> (mid BS: BS≤32 &amp; n/4≤32768, or BS≤64 &amp; n/4≤16384) — phase A sliced full read → exact per-row 2048-bin MSB hist, slice head cached in smem (1504×float4); barrier; phase B mostly-smem re-scan with <b>smem-staged collect + one bulk cursor reservation per CTA</b> (kills the per-row same-address atomic serialization that caused both the mid-BS valley and the flash@1024 anomaly); CTA0 tie refine. / 中 BS:精确直方图融合双遍 + smem 行缓存 + smem-staged 收集(根除同地址原子串行化)。</div>
<div class="arm"><b>D · tp3 fused sampled single-pass</b> (high BS ≤ occupancy cap) — whole-row uniform 1/16 sample → budget-driven b_safe (expected candidates ≤ CAP2/2, no δ hyper-param) → single full-read candidate collect → CTA0 exact finish; superset invariant (cand ≥ k) else exact fallback. / 高 BS:融合采样单遍,预算驱动阈值,候选超集不变量 + 精确兜底。</div>
<div class="arm"><b>E · tp2 3-kernel pipeline</b> (BS above occupancy cap) — same algorithm as D but sample/collect/finish as three launches, barrier-free (launch boundary = sync). / 极高 BS:三核流水,无核内屏障。</div>

<p><b>Final verdict / 终判</b> (final_bs.csv, 80/80 exact, nsys cold-L2, @650a8c740e):</p>
<span class="gate ok"><b>avg ≥ 2.0×</b> — TARGET envelope gm <b>2.083×</b> PASS (all-4-cell pooled {pooled:.3f}×)</span>
<span class="gate warn"><b>min ≥ 1.2×</b> — MISS at BS=32 only: 1.105–1.187× (4/40 pts; other 36 all ≥1.215)</span>
<span class="gate ok"><b>zero regression</b> — PASS, 0/40 below 1.0×</span>
<p></p>{TBL}
<p style="font-size:0.85em;color:#52514e">Speedup vs gvr_pr per cell × BS. Green ≥2.0×, amber &lt;1.2× (the BS=32 residual), red would be &lt;1.0× (none). 绿 ≥2.0×,橙 &lt;1.2×(BS=32 残余),无红格(零回退)。</p>
{SVG}
<p style="font-size:0.85em;color:#52514e">Fig. BSX — speedup vs gvr_pr over BS, toggle series with the chips; hover any point for its value; the per-cell chip reveals the 4 thin per-cell lines (flash 256k/512k K512, pro 512k/1024k K1024). Bold = pooled geomean (labels on ours). Dashed = 2.0× target / 1.2× floor / parity. <b>Caveat:</b> sglang v2 / flashinfer curves are within-REPORT ratios (numerator and denominator both measured on b200-027, real-capture BS-scaling sweep <code>rival_bs_layers.csv</code>); ours is a local b200-039 pair — the cuteDSL gvr arm drifts ~9% (med) across these nodes (§7.8), so cross-arm gaps carry that uncertainty; the vs-PR verdict itself is same-node paired.
/ 图 BSX — chips 勾选切换系列,悬停任意点看数值,"分 cell 细线"chip 展开 4 条 cell 曲线;粗线=池化几何均值。注意:sglang v2 / flashinfer 为 REPORT(b200-027)节点内比值,我方为本机(b200-039)配对;跨节点 gvr 臂漂移 ~9%(§7.8),跨臂对比含此不确定度,vs-PR 终判本身为同节点配对不受影响。</p>

<p><b>BS=32 residual / 残余谷(结构性).</b> gvr_pr&rsquo;s latency-flat plateau ends exactly at BS 32–64 (gvr@64 ≈ 2×gvr@32): BS=32 is the PR arm&rsquo;s best operating point, while any exact full-read arm still pays ≥1 full pass + 2 barriers + tail; deficit 1–9%. Identified next lever (out of scope): a preIdx-hint arm (the same legitimate input gvr consumes) skipping the histogram pass in the high-hit regime. / gvr 延迟平台恰在 BS 32–64 结束;差距 1–9%;下一杠杆 = preIdx hint 臂高命中区跳直方图遍。</p>

<p><b>Falsified & reverted / 证伪回退</b> (all measured cold): parallel finish emit; arena-drop + bigger cache; warp-aggregated candidate atomics (ballot tax, 3rd hit); B&prime; persistent queue (post-close, @4b3914a4b4).
<b>Hard rules / 硬规则</b>: (1) data crossing a fence-less spin barrier = <code>atomicExch</code> store + consumer <code>fence.acq_rel.gpu</code> (plain store loses in-flight entries — ~2 wrong tie members per bad row, stochastic, randn-adversarial rows caught it); (2) same-kernel plain-load-then-rezero of a shared global needs <code>__syncthreads</code> in between (hit twice).</p>
<p style="font-size:0.9em">Artifacts (local only): <code>kf_bs_scaling/ext/</code> — REPORT_BSEXT.html (bilingual), RESULTS.md, final_bs.csv; ledger R3_LEDGER.md close-out.</p>
</div>
<!-- KF-BSEXT:END -->'''

# ---------------- master timeline ----------------
TL_ITEMS = [
 # (date, lane, lane-color, title, target/request, outcome, perf, ship)
 ('07-21 01:19', 'C1 · bs1-real', '#2a78d6',
  'Campaign-1 launched <code>tfb91bvwm…</code> (KF managed B200, 6 agents/round)',
  'REQUEST: DSv4 indexer top-K decode BS=1, 865 real capture cells; <b>target +20% avg vs PR#16457 GVR, zero regression, exact</b>. / 原始请求:865 真实格,+20%,零回退,精确。',
  '', ''),
 ('07-21 ~09:15', 'C1 · bs1-real', '#2a78d6',
  'Campaign-1 SHIP: <b>c74f_sbx</b> = round-2 winner c74fb3c0 + engineer dispatch graft (1024-thread single-CTA rung)',
  '',
  '<b>gm 1.6828×</b> vs PR head · 865/865 exact · 0 cold reg · vs sglang v2 1.119× / radix 1.622× / flashinfer 1.492× · ~$751',
  'branch <code>kf/gvr-topk-c74fsbx</code> → §6'),
 ('07-21 13:40', 'R3 · beyond-champion', '#070',
  'R3 launched <code>e5q1zgrf…</code> — beat the campaign-1 champion',
  'TARGET: gm ≥1.60× vs <i>current</i> PR head, zero-reg, exact. / 目标 ≥1.60× 对最新 head。',
  '', ''),
 ('07-22', 'R3 · beyond-champion', '#070',
  'R3 SHIP: <b>compB</b> = harvested composite (aef3 − mid&lt;1&gt; ⊕ 30e7): fence-less gen-barrier skeleton + fast-tail finishes',
  '',
  '<b>gm 1.8267×</b> vs head b14ec40e1b · min 1.140, 0 reg · 865/865 exact · +8.1% over c74f_sbx · vs sglang v2 1.215× / radix 1.760× · $764.66',
  'branch <code>kf/gvr-topk-compB</code> → §7'),
 ('07-22', 'BS study', '#8b8a85',
  '§7.8 BS-scaling supplement (75 rungs × BS 1–1024, 5 arms local-paired, b200-019)',
  '',
  'compB = <b>BS=1 latency specialist by construction</b>: wins ALL arms at BS=1 (vs PR 1.874×), crossover already at BS=2, batched arms 45–125× faster at BS=1024; one-off spin-barrier livelock marker. → motivates BS-ext',
  '→ §7.8'),
 ('07-22', 'BS-ext · local eng.', '#b2478f',
  'BS-ext campaign (local, b200-039): batched multi-arm operator <code>run_batch_auto</code> — grid.y / ext_v4 row-teams / tp4 / tp3 / tp2 + measured (N,BS) dispatch',
  'TARGET: avg ≥2.0× vs gvr_pr over BS 2–1024, each pt ≥1.2×, zero reg. / 目标 全段均值 2.0×,逐点 1.2×,零回退。',
  '<b>TARGET gm 2.083× PASS</b> · 0/40 reg · 80/80 exact · min 1.105× @BS=32 (only 4/40 pts &lt;1.2) · ladder: D1 1.489 → D2 1.597 → tp3/tp4 2.083 · B&prime; persistent-queue falsified post-close',
  'branch <code>kf/compb-bs-ext</code> → §7.9'),
 ('07-22', 'R4 · coldstart lineage-2', '#b26a00',
  'R4 coldstart <code>pra6srbd…</code> (gvr-topk-cold60): fresh lineage under skeleton hard-lock, denominator pinned head 04a0900ff7',
  'TARGET: gm ≥1.60×, zero-reg, exact — reproducibility probe of lineage-1. / 冷启动复现性探针。',
  'SHIP champion <b>28dc11f6</b> (r3 perK-dispatch): <b>gm 1.6531×</b> · 865/865 exact · 0 real reg (L38 adjudicated 1.0129 @60rep) · vs sglang v2 1.099× · $1110.62 (budget-cut) — 3 rounds reach 1.65, below lineage-1 accumulated 1.83 (as expected)',
  'branches <code>kf/r4-champion-final-bs1</code> / <code>…-r3v11-bs1</code>'),
 ('07-22 14:55', 'R5 · BS batch, lineage-2', '#b26a00',
  'R5 v1 <code>rngnxv95…</code> (gvr-topk-bs2x) — platform submit path scored 0% on custom_inputs (bugs 26583–26602)',
  'TARGET: §7b 750 cells (75 × BS 2–1024) avg ≥2.0× vs head native batch, all ≥1.0, BS=1 holds champion level.',
  'v1 CANCELLED at $9 (agents produced 4 correct kernels, platform mis-scored). Lesson: platform workloads = safetensors only.',
  ''),
 ('07-22 21:06', 'R5 · BS batch, lineage-2', '#b26a00',
  'R5 v2 <b>RUNNING</b> <code>vk9m3tet…</code> (gvr-topk-bs2x-v2): 30 materialized [BS,npad] safetensors workloads (210 MB), BS ∈ {1,4,32,128,256,1024}',
  'Same bars; extreme corners kept for the local 750-cell verdict. Harvest loop: kernel show → compliance → nsys probe → 750-grid + BS=1 865 guard.',
  'round 1/10, $800 budget — <i>in flight at time of writing</i>',
  ''),
]
tl = ['''<!-- KF-MASTERTL:START -->
<h2 id="sec-mtl">T · Optimization master timeline / 优化全程时间线</h2>
<style>#mtl{margin:10px 0}#mtl .ev{display:flex;gap:12px;margin:0}#mtl .rail{display:flex;flex-direction:column;align-items:center;width:16px;flex:none}
#mtl .dot{width:12px;height:12px;border-radius:6px;flex:none;margin-top:6px;border:2.5px solid}#mtl .bar{width:2.5px;flex:1;background:#e4e3df}
#mtl .bd{padding:0 0 18px 0;min-width:0}#mtl .when{font-size:0.82em;color:#52514e}#mtl .lane{display:inline-block;border-radius:10px;padding:1px 9px;font-size:0.78em;color:#fff;margin-left:6px}
#mtl .tgt{background:#f2f6fc;border-left:3px solid #9db8d8;padding:3px 9px;margin:4px 0;font-size:0.88em}
#mtl .res{background:#f4faf4;border-left:3px solid #7cb87c;padding:3px 9px;margin:4px 0;font-size:0.88em}
#mtl .ship{font-size:0.85em;color:#52514e}</style>
<p>One line per milestone: the original request/targets, each KF campaign, and the operator + performance it produced. All speedups = nsys cold-L2 geomean vs the PR#16457 GVR arm on real capture data (865-cell BS=1 grid, or the stated BS envelope).
/ 每个里程碑一行:原始请求与目标、各 KF campaign、以及产出的算子与性能;加速比均为 nsys 冷-L2 对 PR#16457 GVR 臂的几何均值。</p>
<div id="mtl">''']
for i, (when, lane, col, title, tgt, res, ship) in enumerate(TL_ITEMS):
    last = i == len(TL_ITEMS) - 1
    tl.append(f'<div class="ev"><div class="rail"><div class="dot" style="border-color:{col};background:{"#fff" if not res else col}"></div>'
              f'{"" if last else chr(60)+"div class=bar"+chr(62)+chr(60)+"/div"+chr(62)}</div><div class="bd">'
              f'<span class="when">2026-{when}</span><span class="lane" style="background:{col}">{lane}</span>'
              f'<div>{title}</div>')
    if tgt:  tl.append(f'<div class="tgt">{tgt}</div>')
    if res:  tl.append(f'<div class="res">{res}</div>')
    if ship: tl.append(f'<div class="ship">{ship}</div>')
    tl.append('</div></div>')
tl.append('''</div>
<p style="font-size:0.88em;color:#52514e">Trajectory vs PR#16457 GVR (BS=1, 865 cells): 1.00× (baseline) → <b>1.6828×</b> (C1 c74f_sbx) → <b>1.8267×</b> (R3 compB, shipped BS=1 form) ‖ lineage-2 coldstart 1.6531× (R4). BS&gt;1 envelope (4 cells × BS 2–1024): <b>2.083×</b> (BS-ext) vs gvr_pr host-loop; R5 (vs head <i>native batch</i> denominator) in flight.
/ 轨迹:BS=1 三级台阶 1.68 → 1.83(交付形态);第二血统冷启动 1.65;BS&gt;1 包络 2.083(BS-ext),R5 进行中(分母为 head 原生批)。</p>
<!-- KF-MASTERTL:END -->''')
MTL = ''.join(tl)

# ---------------- splice ----------------
h = open(REPORT).read()
def splice(h, start, end, block, anchor, before=True):
    if start in h:
        return re.sub(re.escape(start) + '.*?' + re.escape(end), lambda m: block, h, flags=re.S)
    i = h.find(anchor)
    assert i >= 0, anchor
    if not before:
        i += len(anchor)
    return h[:i] + block + '\n' + h[i:]

h = splice(h, '<!-- KF-MASTERTL:START -->', '<!-- KF-MASTERTL:END -->', MTL,
           '<h2>0 · Harness at a glance', before=True)
h = splice(h, '<!-- KF-BSEXT:START -->', '<!-- KF-BSEXT:END -->', BSEXT,
           '<!-- KF-R3:END -->', before=True)

# banner pointer after the R3 banner (idempotent)
BAN = ('<!-- KF-BSEXTBANNER:START --><div style="background:#faf0f7;border:1.5px solid #b2478f;border-radius:8px;'
       'padding:10px 16px;margin:12px 0;font-size:0.95em"><b>2026-07-22 BS-ext / 批式扩展战役收官:</b> '
       'batched operator <code>run_batch_auto</code> (grid.y / ext_v4 / tp4 / tp3 / tp2 + measured dispatch) — '
       '<b>TARGET gm 2.083× vs gvr_pr over BS 2–1024</b>, 0/40 regressions, 80/80 exact, min 1.105× @BS=32. '
       'Branch <code>kf/compb-bs-ext</code>. See <a href="#sec-mtl">timeline</a> &amp; <a href="#sec-7bsext">§7.9</a>. '
       '/ BS&gt;1 批式算子收官:全段 2.083×,零回退;详见时间线与 §7.9。</div><!-- KF-BSEXTBANNER:END -->')
if '<!-- KF-BSEXTBANNER:START -->' in h:
    h = re.sub(r'<!-- KF-BSEXTBANNER:START -->.*?<!-- KF-BSEXTBANNER:END -->', lambda m: BAN, h, flags=re.S)
else:
    h = h.replace('<!-- KF-R3BANNER:END -->', '<!-- KF-R3BANNER:END -->\n' + BAN)

# §7.8 tail pointer to §7.9 (idempotent)
PTR = ('<!-- KF-78PTR:START --><p><b>Follow-up / 后续:</b> the BS&gt;1 cliff quantified here was addressed by the '
       'BS-ext campaign — batched multi-arm operator, gm 2.083× vs gvr_pr over BS 2–1024, see <a href="#sec-7bsext">§7.9</a>. '
       '/ 本节量化的悬崖已由 §7.9 的 BS-ext 批式算子解决(全段 2.083×)。</p><!-- KF-78PTR:END -->')
if '<!-- KF-78PTR:START -->' in h:
    h = re.sub(r'<!-- KF-78PTR:START -->.*?<!-- KF-78PTR:END -->', lambda m: PTR, h, flags=re.S)
else:
    h = h.replace('<!-- KF-R3BS:END -->', PTR + '\n<!-- KF-R3BS:END -->')

open(REPORT, 'w').write(h)
print(f'OK: pooled gm {pooled:.3f}, per-BS gm ' + ' '.join(f'{b}:{gm_bs[b]:.2f}' for b in BSL))
print('report bytes:', len(h))
