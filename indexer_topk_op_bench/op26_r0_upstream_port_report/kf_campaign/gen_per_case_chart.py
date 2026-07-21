#!/usr/bin/env python3
"""Generate KF_PER_CASE_CHART.html + inject section 7 of KF_PROCESS_LOG.html.

Per-case (865 cells, NO layer averaging) comparison of the shipped KF champion
c74f_sbx vs the PR#16457 GVR baseline, SGLang v2 and Radix-cuteDSL.

Sources:
  - kf_campaign/grid_c74fsbx.csv        champion + PR arm, nsys cold-L2 paired (07-21)
  - ../rival_layers_full.csv            REPORT §4b external arms, absolute µs (07-15/16)
  - ../real_3arm_layers_full.csv        REPORT §4b PR arm (anchor for cross-session calib)
Rival latencies are normalized per-cell to this campaign's session via the PR arm
measured in both (us × pr_kf/pr_report — same method as compare_rivals.py, med 1.010);
speedup-vs-PR ratios are computed WITHIN each arm's own session (paired, no calib).

Zero <script>: all interactivity is CSS-only (checkbox :checked ~ sibling).
Case selection = model × layer × ISL checkboxes, each dimension with ALL/NONE
override chips; lines are drawn as per-ISL segments so hiding an ISL removes
exactly those points/segments.
"""
import csv, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'KF_PER_CASE_CHART.html')

ISL_ORDER = ['4k','8k','16k','32k','64k','128k','256k','512k','1024k']
ISL_LBL = {'1024k':'1m'}
MODELS = [('flash','V4 Flash · K512 · cr=4'), ('pro','V4 Pro · K1024 · cr=4'), ('v32','V3.2 · K2048 · cr=1')]
DASHES = ['', '7 3', '2 2', '9 3 2 3', '1 3', '12 4', '4 2 1 2']

# ---- load ------------------------------------------------------------------
kf = {r['uuid']: r for r in csv.DictReader(open(os.path.join(HERE, 'grid_c74fsbx.csv')))}
riv = {'sglang_v2': {}, 'radix_cutedsl': {}}
for r in csv.DictReader(open(os.path.join(REPORT_DIR, 'rival_layers_full.csv'))):
    if r['op'] in riv:
        riv[r['op']][f"{r['model']}_{r['isl']}_L{int(r['L']):02d}"] = float(r['us'])
rep_pr = {}
for r in csv.DictReader(open(os.path.join(REPORT_DIR, 'real_3arm_layers_full.csv'))):
    rep_pr[f"{r['model']}_{r['isl']}_L{int(r['layer']):02d}"] = float(r['pr'])

cases = list(kf.values())
layers = {m: sorted({int(r['layer']) for r in cases if r['model'] == m}) for m, _ in MODELS}
isls = {m: [i for i in ISL_ORDER if any(r['isl'] == i for r in cases if r['model'] == m)] for m, _ in MODELS}
N_of = {(r['model'], r['isl']): int(r['N']) for r in cases}
uuid_of = {(r['model'], r['isl'], int(r['layer'])): r['uuid'] for r in cases}

def gmean(vs):
    return math.exp(sum(math.log(v) for v in vs) / len(vs))

def cal(u):                       # cross-session anchor calibration for rival latencies
    return float(kf[u]['pr_cold']) / rep_pr[u]

# per-arm accessors: lat(u) -> µs in this campaign's session frame; sp(u) -> vs-PR paired
ARMS = [
    ('aPR',  'PR#16457 GVR (baseline)', 'PR 基线',        '#55534e',
     lambda u: float(kf[u]['pr_cold']),                    None),
    ('aCH',  'KF champion c74f_sbx',    'KF 出货冠军',     '#d62f2f',
     lambda u: float(kf[u]['cand_cold']),                  lambda u: float(kf[u]['speedup_cold'])),
    ('aSGL', 'SGLang v2 (span)',        'SGLang v2',      '#008300',
     lambda u: riv['sglang_v2'][u] * cal(u),               lambda u: rep_pr[u] / riv['sglang_v2'][u]),
    ('aRDX', 'Radix cuteDSL',           'Radix cuteDSL',  '#e87ba4',
     lambda u: riv['radix_cutedsl'][u] * cal(u),           lambda u: rep_pr[u] / riv['radix_cutedsl'][u]),
]
GM = {tag: (gmean([spf(u) for u in kf]) if spf else None) for tag, _, _, _, _, spf in ARMS}

# ---- geometry ---------------------------------------------------------------
W, H, ML, MR, MT, MB = 1240, 430, 52, 14, 18, 34
PW, PH = W - ML - MR, H - MT - MB
_all_lat = [lf(u) for _, _, _, _, lf, _ in ARMS for u in kf]
LAT_LO, LAT_HI = min(_all_lat) * 0.92, max(_all_lat) * 1.06
SP_LO, SP_HI = 0.45, 3.0

def xpos(m, isl):
    xs = isls[m]
    return ML + PW * (xs.index(isl) / max(1, len(xs) - 1))

def ylat(v):
    t = (math.log10(v) - math.log10(LAT_LO)) / (math.log10(LAT_HI) - math.log10(LAT_LO))
    return MT + PH * (1 - t)

def ysp(v):
    return MT + PH * (1 - (v - SP_LO) / (SP_HI - SP_LO))

def shade(col, f):
    r, g, b = (int(col[i:i+2], 16) for i in (1, 3, 5))
    if f >= 0:
        r, g, b = (round(c + (255 - c) * f) for c in (r, g, b))
    else:
        r, g, b = (round(c * (1 + f)) for c in (r, g, b))
    return f'#{r:02x}{g:02x}{b:02x}'

def axes(m, kind):
    p = [f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="none" stroke="#c9c8c2"/>']
    for isl in isls[m]:
        x = xpos(m, isl)
        p.append(f'<line x1="{x:.0f}" y1="{MT}" x2="{x:.0f}" y2="{MT+PH}" stroke="#e4e3df"/>')
        p.append(f'<text x="{x:.0f}" y="{MT+PH+15}" text-anchor="middle" font-size="11" fill="#52514e">{ISL_LBL.get(isl,isl)}</text>')
        p.append(f'<text x="{x:.0f}" y="{MT+PH+28}" text-anchor="middle" font-size="9" fill="#8b8a85">N={N_of[(m,isl)]}</text>')
    if kind == 'lat':
        for v in [3, 4, 5, 7, 10, 15, 20, 30, 40, 60]:
            if not (LAT_LO <= v <= LAT_HI):
                continue
            y = ylat(v)
            p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="#e4e3df"/>')
            p.append(f'<text x="{ML-6}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#52514e">{v}</text>')
        p.append(f'<text x="14" y="{MT+PH/2:.0f}" font-size="11" fill="#52514e" transform="rotate(-90 14 {MT+PH/2:.0f})" text-anchor="middle">µs (cold-L2, log)</text>')
    else:
        for v in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
            y = ysp(v)
            w = ('#8b8a85' if v == 1.0 else '#e4e3df')
            p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="{w}"{" stroke-dasharray=\"5 3\"" if v==1.0 else ""}/>')
            p.append(f'<text x="{ML-6}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#52514e">{v:g}</text>')
        p.append(f'<text x="14" y="{MT+PH/2:.0f}" font-size="11" fill="#52514e" transform="rotate(-90 14 {MT+PH/2:.0f})" text-anchor="middle">speedup vs PR (same-session paired)</text>')
    return ''.join(p)

def series(m, kind):
    out = []
    nl = len(layers[m])
    for li, L in enumerate(layers[m]):
        dash = DASHES[li % len(DASHES)]
        f = -0.28 + 0.62 * (li / max(1, nl - 1))
        lcls = f'L{m}{L}'
        for tag, lab, labc, col, latf, spf in ARMS:
            if kind == 'sp' and spf is None:
                continue
            lcol = shade(col, f)
            xs, ys, vals, dots, per_isl = [], [], [], [], []
            for isl in isls[m]:
                u = uuid_of.get((m, isl, L))
                if u is None:
                    continue
                r = kf[u]
                if kind == 'lat':
                    v = latf(u)
                    y = ylat(v)
                    if tag == 'aPR':
                        det = f'PR {v:.2f}µs'
                    elif tag == 'aCH':
                        det = f'{lab}: {v:.2f}µs vs PR {float(r["pr_cold"]):.2f}µs = {float(r["speedup_cold"]):.3f}×'
                    else:
                        raw = riv['sglang_v2' if tag == 'aSGL' else 'radix_cutedsl'][u]
                        det = f'{lab}: {v:.2f}µs (raw {raw:.2f}µs × anchor-cal {cal(u):.3f}) vs PR {float(r["pr_cold"]):.2f}µs'
                    per_isl.append(f'{ISL_LBL.get(isl,isl)}={v:.1f}µs')
                else:
                    v = spf(u)
                    y = ysp(min(max(v, SP_LO), SP_HI))
                    det = f'{lab} vs PR: {v:.3f}×' + ('' if tag == 'aCH' else ' (report-session paired)')
                    per_isl.append(f'{ISL_LBL.get(isl,isl)}={v:.2f}×')
                tip = f'{m} L{L} · ISL {ISL_LBL.get(isl,isl)} (N={r["N"]}) · hit {r["hit"]} · {det}'
                x = xpos(m, isl)
                xs.append(x); ys.append(y); vals.append(v)
                dots.append(f'<circle class="I{isl}" cx="{x:.0f}" cy="{y:.1f}" r="3" fill="{lcol}" stroke="#fff" stroke-width="0.6"><title>{tip}</title></circle>')
            unit = 'µs' if kind == 'lat' else '×'
            i_mn, i_mx = vals.index(min(vals)), vals.index(max(vals))
            gsum = (f'{lab} · {m} L{L} — geomean {gmean(vals):.2f}{unit}, '
                    f'min {min(vals):.2f}{unit} @{ISL_LBL.get(isls[m][i_mn], isls[m][i_mn])}, '
                    f'max {max(vals):.2f}{unit} @{ISL_LBL.get(isls[m][i_mx], isls[m][i_mx])}'
                    f'\n{" · ".join(per_isl)}')
            da = f' stroke-dasharray="{dash}"' if dash else ''
            segs = []
            for i in range(len(xs) - 1):
                icls = f'I{isls[m][i]} I{isls[m][i+1]}'
                seg = f'x1="{xs[i]:.0f}" y1="{ys[i]:.1f}" x2="{xs[i+1]:.0f}" y2="{ys[i+1]:.1f}"'
                segs.append(f'<line class="{icls}" {seg} stroke="#fff" stroke-width="5" opacity="0" pointer-events="stroke"/>')
                segs.append(f'<line class="vis {icls}" {seg} stroke="{lcol}" stroke-width="1.7"{da}/>')
            out.append(f'<g class="ser {tag} {lcls}"><title>{gsum}</title>{"".join(segs)}{"".join(dots)}</g>')
    return ''.join(out)

# ---- controls + css ----------------------------------------------------------
inputs, css = [], []
chips_arm, chips_model, chips_isl, chips_layer = [], [], [], {}
css.append('.vizP svg .ser .vis{transition:stroke-width .08s,opacity .08s}')
css.append('.vizP svg .ser:hover .vis{stroke-width:3.8}')
css.append('.vizP svg .ser:hover circle{r:4.6}')
css.append('.vizP svg:has(.ser:hover) .ser:not(:hover){opacity:.15}')

def ovr_pair(scope, classes):
    """Emit ALL/NONE override inputs+rules for a list of togglable classes."""
    inputs.append(f'<input type="checkbox" id="ck-ALL{scope}">')
    inputs.append(f'<input type="checkbox" id="ck-NONE{scope}">')
    for c in classes:
        css.append(f'#ck-ALL{scope}:checked ~ * .{c}{{display:inline}}')
    for c in classes:
        css.append(f'#ck-NONE{scope}:checked ~ * .{c}{{display:none}}')
    for t in ('ALL', 'NONE'):
        css.append(f'#ck-{t}{scope}:checked ~ .ctl label[for=ck-{t}{scope}]{{border-color:#b3541e;color:#0b0b0b;box-shadow:inset 0 0 0 1px #b3541e;background:#fdf3ec}}')
    return (f'<label for="ck-ALL{scope}" class="ovr">ALL 全选</label>'
            f'<label for="ck-NONE{scope}" class="ovr">NONE 全不选</label>')

for tag, lab, labc, col, latf, spf in ARMS:
    inputs.append(f'<input type="checkbox" id="ck-{tag}" checked>')
    css.append(f'#ck-{tag}:not(:checked) ~ * .{tag}{{display:none}}')
    css.append(f'#ck-{tag}:checked ~ .ctl label[for=ck-{tag}]{{border-color:{col};color:#0b0b0b;box-shadow:inset 0 0 0 1px {col}}}')
    g = f' — gm {GM[tag]:.3f}× vs PR' if GM[tag] else ' (anchor)'
    chips_arm.append(f'<label for="ck-{tag}"><i style="background:{col}"></i><b>{lab}</b> / {labc}{g}</label>')

for m, mt in MODELS:
    inputs.append(f'<input type="checkbox" id="ck-m{m}" checked>')
    css.append(f'#ck-m{m}:not(:checked) ~ * .m-{m}{{display:none}}')
    css.append(f'#ck-m{m}:checked ~ .ctl label[for=ck-m{m}]{{border-color:#2a78d6;color:#0b0b0b}}')
    chips_model.append(f'<label for="ck-m{m}"><b>{mt}</b> · {len(layers[m])}L × {len(isls[m])} ISL = {len(layers[m])*len(isls[m])} cases</label>')

for isl in ISL_ORDER:
    inputs.append(f'<input type="checkbox" id="ck-I{isl}" checked>')
    css.append(f'#ck-I{isl}:not(:checked) ~ * .I{isl}{{display:none}}')
    css.append(f'#ck-I{isl}:checked ~ .ctl label[for=ck-I{isl}]{{border-color:#2a78d6;color:#0b0b0b}}')
    chips_isl.append(f'<label for="ck-I{isl}">{ISL_LBL.get(isl,isl)}</label>')
chips_isl.insert(0, ovr_pair('isl', [f'I{i}' for i in ISL_ORDER]))
for t in ('ALL', 'NONE'):
    css.append(f'#ck-{t}isl:checked ~ .ctl label[for^="ck-I"]{{opacity:.35}}')

for m, mt in MODELS:
    row = []
    for L in layers[m]:
        dflt = True if m != 'v32' else (L in (14, 34, 54))
        ck = 'checked' if dflt else ''
        inputs.append(f'<input type="checkbox" id="ck-L{m}{L}" {ck}>')
        css.append(f'#ck-L{m}{L}:not(:checked) ~ * .L{m}{L}{{display:none}}')
        css.append(f'#ck-L{m}{L}:checked ~ .ctl label[for=ck-L{m}{L}]{{border-color:#2a78d6;color:#0b0b0b}}')
        row.append(f'<label for="ck-L{m}{L}">L{L}</label>')
    ov = ovr_pair(m, [f'L{m}{L}' for L in layers[m]])
    for t in ('ALL', 'NONE'):
        css.append(f'#ck-{t}{m}:checked ~ .ctl label[for^="ck-L{m}"]{{opacity:.35}}')
    chips_layer[m] = ov + ''.join(row)

panels = []
for m, mt in MODELS:
    ncase = len(layers[m]) * len(isls[m])
    panels.append(f'''<div class="m-{m} panel">
<h3>{mt} — {ncase} cases ({len(layers[m])} layers × {len(isls[m])} ISL)</h3>
<div class="chartlbl">latency per case / 逐 case 延迟 (nsys cold-L2 µs, log; 外部臂已按逐格 PR 锚校准到本会话)</div>
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="background:#fcfcfb">{axes(m,'lat')}{series(m,'lat')}</svg>
<div class="chartlbl">speedup per case vs PR baseline (each arm paired within its own session) / 逐 case 对 PR 加速比(各臂自身会话内配对)</div>
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="background:#fcfcfb">{axes(m,'sp')}{series(m,'sp')}</svg>
</div>''')

FRAG_CSS = f'''.vizP{{margin:10px 0}}
.vizP .note{{color:#555;font-size:0.9em}}
.vizP h3{{color:#333;margin:20px 0 4px;font-size:1.05em}}
.vizP input{{position:absolute;opacity:0;pointer-events:none}}
.vizP .ctl{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px 12px;margin:8px 0;display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
.vizP .ctl label{{border:1.5px solid #c9c8c2;border-radius:14px;padding:2px 10px;cursor:pointer;font-size:12px;color:#52514e;user-select:none;white-space:nowrap}}
.vizP .ctl label i{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px;vertical-align:-1px}}
.vizP .ctl label.ovr{{border-style:dashed;font-weight:600;color:#b3541e}}
.vizP .ctl .hd{{font-weight:600;color:#1a1a2e;font-size:12.5px;margin-right:4px}}
.vizP .panel svg{{width:100%;height:auto;border:1px solid #ddd;border-radius:8px;margin:2px 0 10px}}
.vizP .chartlbl{{font-size:12px;color:#555;margin-top:6px}}
{''.join(css)}'''

INTRO = f'''<p class="note">Shipped <b>KF champion c74f_sbx</b> vs <b>PR#16457 GVR baseline</b>, <b>SGLang v2</b> and <b>Radix-cuteDSL</b>
on all <b>865 per-layer cases</b> (V4 Flash 21L×9 ISL + V4 Pro 30L×9 ISL + V3.2 58L×7 ISL), BS=1 fp32, nsys cold-L2, no layer averaging.
Champion+PR: this campaign's paired grids (07-21). External arms: REPORT §4b per-layer sweep (07-15/16,
<code>rival_layers_full.csv</code>), latencies normalized per-cell to this session via the PR arm measured in both
(<code>us × pr_kf/pr_report</code>, median 1.010 — same method as <code>compare_rivals.py</code>); speedup-vs-PR is paired
within each arm's own session (no calibration needed). Geomeans: champion 1.683× · SGLang v2 1.504× · Radix 1.038×.
冠军与 PR 为本战役同轮配对计时;SGLang v2 / Radix cuteDSL 取自 REPORT §4b 逐层扫描,延迟经逐格 PR 锚校准到本会话,加速比各自会话内配对、无需校准。
<b>Case 选择</b>:model × layer × ISL 三个维度的复选框交集唯一定位任意单个 case(如只勾 flash + L28 + 64k),各维度均有 ALL/NONE 覆盖开关。
<b>悬停数据点</b>=该 case 精确数值;<b>悬停线条</b>=整条线摘要(geomean/极值/逐 ISL 值),其余线自动淡出。</p>'''

OUTRO = f'''<p class="note">Speedup chart clips at {SP_HI}× / {SP_LO}× (hover shows true values). Exactness caveat: champion and Radix are
unconditionally exact on all 865 cases; SGLang v2 is conditionally exact (kMaxNumTie=2048 — real V3.2 L52 exceeds it at 128k/256k, see REPORT §8).
加速比图纵轴截断于 [{SP_LO}, {SP_HI}]×,悬停可见真值;精确性注记:冠军与 Radix 全 865 case 无条件精确,SGLang v2 受 kMaxNumTie=2048 条件限制(真实 V3.2 L52 在 128k/256k 越限)。</p>'''

FRAG_BODY = f'''{INTRO}
<div class="vizP">
{''.join(inputs)}
<div class="ctl"><span class="hd">arms / 对比臂</span>{''.join(chips_arm)}</div>
<div class="ctl"><span class="hd">models / 模型</span>{''.join(chips_model)}</div>
<div class="ctl"><span class="hd">ISL</span>{''.join(chips_isl)}<span class="note">(取消勾选即隐藏该 ISL 的点与相邻线段)</span></div>
<div class="ctl"><span class="hd">Flash layers</span>{chips_layer['flash']}</div>
<div class="ctl"><span class="hd">Pro layers</span>{chips_layer['pro']}</div>
<div class="ctl"><span class="hd">V3.2 layers</span><span class="note">(默认只勾 bench 层 14/34/54;ALL/NONE 为覆盖开关,勾上后强制全显/全隐、逐项勾选暂被忽略并置灰,取消覆盖即恢复;两者同勾 NONE 优先)</span>{chips_layer['v32']}</div>
{''.join(panels)}
{OUTRO}
</div>'''

# ---- standalone page ---------------------------------------------------------
html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>KF champion vs PR / SGLang v2 / Radix — per-case comparison (865 cases)</title>
<style>
body{{font:14px/1.55 -apple-system,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;background:#fafaf7;color:#1a1a2e;margin:0;padding:22px 26px;max-width:1330px}}
h1{{font-size:20px;margin:0 0 4px}}
code{{font-family:"SF Mono",Consolas,monospace;background:#f0f0ea;border-radius:4px;padding:1px 5px;font-size:0.9em}}
{FRAG_CSS}
</style></head><body>
<h1>KF champion c74f_sbx vs PR / SGLang v2 / Radix-cuteDSL — per-case comparison / 逐 case 性能对比(不做层间平均)</h1>
{FRAG_BODY}
</body></html>'''
open(OUT, 'w').write(html)
print(f'wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)')

# ---- idempotent injection into KF_PROCESS_LOG.html ---------------------------
LOG = os.path.join(HERE, 'KF_PROCESS_LOG.html')
MARK_S, MARK_E = '<!-- KF-PERCASE:START -->', '<!-- KF-PERCASE:END -->'
ANCHOR = '<!-- KF-FINAL:END -->'
section = f'''{MARK_S}
<h2 id="sec-percase">7 · Per-case: champion vs PR / SGLang v2 / Radix / 逐 case 对比(不做层间平均)</h2>
<style>{FRAG_CSS}</style>
{FRAG_BODY}
{MARK_E}'''
doc = open(LOG).read()
if MARK_S in doc and MARK_E in doc:
    pre, rest = doc.split(MARK_S, 1)
    _, post = rest.split(MARK_E, 1)
    doc = pre + section + post
    print('replaced existing KF-PERCASE section in KF_PROCESS_LOG.html')
else:
    assert ANCHOR in doc, 'anchor KF-FINAL:END not found'
    doc = doc.replace(ANCHOR, ANCHOR + '\n\n' + section, 1)
    print('inserted new KF-PERCASE section after KF-FINAL:END')
open(LOG, 'w').write(doc)
print(f'KF_PROCESS_LOG.html now {os.path.getsize(LOG)/1e6:.2f} MB')
