#!/usr/bin/env python3
"""Generate KF_PER_CASE_CHART.html — interactive per-case (no layer averaging)
performance comparison across all CLEAN KernelFactory full-grid verdicts.

Zero <script>: all interactivity is CSS-only (checkbox :checked ~ sibling),
per the report-viewer constraint. Data = the 865-case nsys cold-L2 paired
grids in this directory. Contaminated verdicts (grid_r2a, grid_r2b) excluded.
"""
import csv, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'KF_PER_CASE_CHART.html')

# ---- arms: (tag, csv, label EN, label CN, color, default-checked, is-baseline)
ARMS = [
    ('aPR',  'grid_c74fsbx.csv', 'PR head e6fdbfa (baseline)', 'PR 基线',        '#55534e', True,  True),
    ('aR1A', 'grid_r1a.csv',     'r1 41a94aaa gm1.316',        '第1轮 41a94aaa', '#b07fe0', False, False),
    ('aR1C', 'grid_r1c.csv',     'r1 winner ba1020ce gm1.366', '第1轮冠军 ba1020ce', '#2a78d6', True, False),
    ('aSBX', 'grid_sbx.csv',     'sbx (ba1020ce+sb17b) gm1.605','sbx 工程师嫁接', '#00a5c0', False, False),
    ('aR2A', 'grid_r2a2_fixed.csv','r2 0260cee7 gm1.642',      '第2轮 0260cee7', '#e8912d', False, False),
    ('aR2C', 'grid_r2c2g.csv',   'r2 c74fb3c0 gm1.671',        '第2轮 c74fb3c0', '#008300', False, False),
    ('aCH',  'grid_c74fsbx.csv', 'CHAMPION c74f_sbx gm1.683',  '出货冠军 c74f_sbx', '#d62f2f', True, False),
]
ISL_ORDER = ['4k','8k','16k','32k','64k','128k','256k','512k','1024k']
ISL_LBL = {'1024k':'1m'}
MODELS = [('flash','V4 Flash · K512 · cr=4'), ('pro','V4 Pro · K1024 · cr=4'), ('v32','V3.2 · K2048 · cr=1')]
DASHES = ['', '7 3', '2 2', '9 3 2 3', '1 3', '12 4', '4 2 1 2']

# ---- load ------------------------------------------------------------------
def load(fn):
    d = {}
    for r in csv.DictReader(open(os.path.join(HERE, fn))):
        d[r['uuid']] = r
    return d

grids = {tag: load(fn) for tag, fn, *_ in ARMS}
champ = grids['aCH']
cases = list(champ.values())
layers = {m: sorted({int(r['layer']) for r in cases if r['model'] == m}) for m, _ in MODELS}
isls = {m: [i for i in ISL_ORDER if any(r['isl'] == i for r in cases if r['model'] == m)] for m, _ in MODELS}
N_of = {(r['model'], r['isl']): int(r['N']) for r in cases}

def by_case(m, isl, L):
    for r in cases:
        if r['model'] == m and r['isl'] == isl and int(r['layer']) == L:
            return r['uuid']
    return None

# ---- geometry ---------------------------------------------------------------
W, H, ML, MR, MT, MB = 1240, 430, 52, 14, 18, 34
PW, PH = W - ML - MR, H - MT - MB
LAT_LO, LAT_HI = 2.5, 40.0          # µs, log scale
SP_LO, SP_HI = 0.6, 4.0             # speedup, linear

def xpos(m, isl):
    xs = isls[m]
    return ML + PW * (xs.index(isl) / max(1, len(xs) - 1))

def ylat(v):
    t = (math.log10(v) - math.log10(LAT_LO)) / (math.log10(LAT_HI) - math.log10(LAT_LO))
    return MT + PH * (1 - t)

def ysp(v):
    return MT + PH * (1 - (v - SP_LO) / (SP_HI - SP_LO))

def axes(m, kind):
    p = [f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="none" stroke="#c9c8c2"/>']
    for isl in isls[m]:
        x = xpos(m, isl)
        p.append(f'<line x1="{x:.0f}" y1="{MT}" x2="{x:.0f}" y2="{MT+PH}" stroke="#e4e3df"/>')
        p.append(f'<text x="{x:.0f}" y="{MT+PH+15}" text-anchor="middle" font-size="11" fill="#52514e">{ISL_LBL.get(isl,isl)}</text>')
        p.append(f'<text x="{x:.0f}" y="{MT+PH+28}" text-anchor="middle" font-size="9" fill="#8b8a85">N={N_of[(m,isl)]}</text>')
    if kind == 'lat':
        for v in [3, 4, 5, 7, 10, 15, 20, 30, 40]:
            y = ylat(v)
            p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="#e4e3df"/>')
            p.append(f'<text x="{ML-6}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#52514e">{v}</text>')
        p.append(f'<text x="14" y="{MT+PH/2:.0f}" font-size="11" fill="#52514e" transform="rotate(-90 14 {MT+PH/2:.0f})" text-anchor="middle">µs (cold-L2, log)</text>')
    else:
        for v in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            y = ysp(v)
            w = ('#8b8a85' if v == 1.0 else '#e4e3df')
            p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="{w}"{" stroke-dasharray=\"5 3\"" if v==1.0 else ""}/>')
            p.append(f'<text x="{ML-6}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#52514e">{v:g}</text>')
        p.append(f'<text x="14" y="{MT+PH/2:.0f}" font-size="11" fill="#52514e" transform="rotate(-90 14 {MT+PH/2:.0f})" text-anchor="middle">speedup = PR / cand (paired)</text>')
    return ''.join(p)

def shade(col, f):
    """Lighten (f>0) / darken (f<0) a #rrggbb color; keeps the arm hue readable per layer."""
    r, g, b = (int(col[i:i+2], 16) for i in (1, 3, 5))
    if f >= 0:
        r, g, b = (round(c + (255 - c) * f) for c in (r, g, b))
    else:
        r, g, b = (round(c * (1 + f)) for c in (r, g, b))
    return f'#{r:02x}{g:02x}{b:02x}'

def gmean(vs):
    return math.exp(sum(math.log(v) for v in vs) / len(vs))

def series(m, kind):
    out = []
    nl = len(layers[m])
    for li, L in enumerate(layers[m]):
        dash = DASHES[li % len(DASHES)]
        # per-layer lightness ramp within the arm hue: shallow layers darker, deep lighter
        f = -0.28 + 0.62 * (li / max(1, nl - 1))
        lcls = f'L{m}{L}'
        for tag, fn, lab, labc, col, dflt, is_base in ARMS:
            if kind == 'sp' and is_base:
                continue
            lcol = shade(col, f)
            arm_name = lab.split(' gm')[0]
            pts, dots, vals, per_isl = [], [], [], []
            for isl in isls[m]:
                u = by_case(m, isl, L)
                r = grids[tag].get(u)
                if r is None:
                    continue
                if kind == 'lat':
                    v = float(r['pr_cold']) if is_base else float(r['cand_cold'])
                    y = ylat(v)
                    tip = (f'{m} L{L} · ISL {ISL_LBL.get(isl,isl)} (N={r["N"]}) · hit {r["hit"]} · '
                           + (f'PR {float(r["pr_cold"]):.2f}µs' if is_base else
                              f'{arm_name}: {float(r["cand_cold"]):.2f}µs vs PR {float(r["pr_cold"]):.2f}µs = {float(r["speedup_cold"]):.3f}×'))
                    per_isl.append(f'{ISL_LBL.get(isl,isl)}={v:.1f}µs')
                else:
                    v = float(r['speedup_cold'])
                    y = ysp(min(v, SP_HI))
                    tip = (f'{m} L{L} · ISL {ISL_LBL.get(isl,isl)} (N={r["N"]}) · hit {r["hit"]} · '
                           f'{arm_name}: {v:.3f}× (PR {float(r["pr_cold"]):.2f}µs → {float(r["cand_cold"]):.2f}µs)')
                    per_isl.append(f'{ISL_LBL.get(isl,isl)}={v:.2f}×')
                vals.append(v)
                x = xpos(m, isl)
                pts.append(f'{x:.0f},{y:.1f}')
                dots.append(f'<circle cx="{x:.0f}" cy="{y:.1f}" r="3" fill="{lcol}" stroke="#fff" stroke-width="0.6"><title>{tip}</title></circle>')
            unit = 'µs' if kind == 'lat' else '×'
            gsum = (f'{arm_name} · {m} L{L} — geomean {gmean(vals):.2f}{unit}, '
                    f'min {min(vals):.2f}{unit} @{ISL_LBL.get(isls[m][vals.index(min(vals))], isls[m][vals.index(min(vals))])}, '
                    f'max {max(vals):.2f}{unit} @{ISL_LBL.get(isls[m][vals.index(max(vals))], isls[m][vals.index(max(vals))])}'
                    f'\n{" · ".join(per_isl)}')
            da = f' stroke-dasharray="{dash}"' if dash else ''
            out.append(f'<g class="ser {tag} {lcls}"><title>{gsum}</title>'
                       f'<polyline points="{" ".join(pts)}" fill="none" stroke="#fff" stroke-width="5" opacity="0" pointer-events="stroke"/>'
                       f'<polyline class="vis" points="{" ".join(pts)}" fill="none" stroke="{lcol}" stroke-width="1.7"{da}/>'
                       f'{"".join(dots)}</g>')
    return ''.join(out)

# ---- controls + css ----------------------------------------------------------
inputs, css, chips_arm, chips_model, chips_layer = [], [], [], [], {}
# hover affordances: line thickens + dots grow; all OTHER series fade so the
# hovered line pops (progressive enhancement via :has; harmless if unsupported).
css.append('.vizP svg .ser .vis{transition:stroke-width .08s,opacity .08s}')
css.append('.vizP svg .ser:hover .vis{stroke-width:3.8}')
css.append('.vizP svg .ser:hover circle{r:4.6}')
css.append('.vizP svg:has(.ser:hover) .ser:not(:hover){opacity:.15}')
for tag, fn, lab, labc, col, dflt, _ in ARMS:
    ck = 'checked' if dflt else ''
    inputs.append(f'<input type="checkbox" id="ck-{tag}" {ck}>')
    css.append(f'#ck-{tag}:not(:checked) ~ * .{tag}{{display:none}}')
    css.append(f'#ck-{tag}:checked ~ .ctl label[for=ck-{tag}]{{border-color:{col};color:#0b0b0b;box-shadow:inset 0 0 0 1px {col}}}')
    chips_arm.append(f'<label for="ck-{tag}"><i style="background:{col}"></i><b>{lab}</b> / {labc}</label>')
for m, mt in MODELS:
    inputs.append(f'<input type="checkbox" id="ck-m{m}" checked>')
    css.append(f'#ck-m{m}:not(:checked) ~ * .m-{m}{{display:none}}')
    css.append(f'#ck-m{m}:checked ~ .ctl label[for=ck-m{m}]{{border-color:#2a78d6;color:#0b0b0b}}')
    chips_model.append(f'<label for="ck-m{m}"><b>{mt}</b> · {len(layers[m])}L × {len(isls[m])} ISL = {len(layers[m])*len(isls[m])} cases</label>')
    row = []
    for L in layers[m]:
        dflt = True if m != 'v32' else (L in (14, 34, 54))
        ck = 'checked' if dflt else ''
        inputs.append(f'<input type="checkbox" id="ck-L{m}{L}" {ck}>')
        css.append(f'#ck-L{m}{L}:not(:checked) ~ * .L{m}{L}{{display:none}}')
        css.append(f'#ck-L{m}{L}:checked ~ .ctl label[for=ck-L{m}{L}]{{border-color:#2a78d6;color:#0b0b0b}}')
        row.append(f'<label for="ck-L{m}{L}">L{L}</label>')
    # ALL / NONE overrides: pure-CSS "select all / deselect all". They do not flip the
    # individual checkboxes — the rules below are emitted AFTER the per-layer rules
    # (same specificity, later source order wins), so ALL forces every layer of the
    # model visible and NONE forces them hidden; NONE beats ALL if both are checked.
    inputs.append(f'<input type="checkbox" id="ck-ALL{m}">')
    inputs.append(f'<input type="checkbox" id="ck-NONE{m}">')
    for L in layers[m]:
        css.append(f'#ck-ALL{m}:checked ~ * .L{m}{L}{{display:inline}}')
    for L in layers[m]:
        css.append(f'#ck-NONE{m}:checked ~ * .L{m}{L}{{display:none}}')
    for tag in ('ALL', 'NONE'):
        css.append(f'#ck-{tag}{m}:checked ~ .ctl label[for=ck-{tag}{m}]{{border-color:#b3541e;color:#0b0b0b;box-shadow:inset 0 0 0 1px #b3541e;background:#fdf3ec}}')
        css.append(f'#ck-{tag}{m}:checked ~ .ctl label[for^="ck-L{m}"]{{opacity:.35}}')
    chips_layer[m] = (f'<label for="ck-ALL{m}" class="ovr">ALL 全选</label>'
                      f'<label for="ck-NONE{m}" class="ovr">NONE 全不选</label>' + ''.join(row))

panels = []
for m, mt in MODELS:
    ncase = len(layers[m]) * len(isls[m])
    panels.append(f'''<div class="m-{m} panel">
<h3>{mt} — {ncase} cases ({len(layers[m])} layers × {len(isls[m])} ISL)</h3>
<div class="chartlbl">latency per case / 逐 case 延迟 (nsys cold-L2 µs, log)</div>
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="background:#fcfcfb">{axes(m,'lat')}{series(m,'lat')}</svg>
<div class="chartlbl">speedup per case vs PR (same-run paired) / 逐 case 对 PR 加速比(同轮配对)</div>
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

INTRO = f'''<p class="note">All <b>865 per-layer cases</b> (V4 Flash 21L×9 ISL + V4 Pro 30L×9 ISL + V3.2 58L×7 ISL), BS=1 fp32, nsys cold-L2 <b>paired</b>
(cand vs PR head <code>e6fdbfac3d</code> same-run), 8×B200 sharded grids, 2026-07-21, umbriel-b200-027.
Only <b>CLEAN full-grid verdicts</b> are shown — the contaminated r2a/r2b runs (double-driver PR-arm inflation, later invalidated
and re-measured as r2a2_fixed / r2c2g) are excluded. Source: <code>grid_*.csv</code> in <code>kf_campaign/</code>; generator
<code>gen_per_case_chart.py</code> (idempotent). 全部 865 个逐层 case,冷 L2 nsys 同轮配对计时,不做任何层间平均;仅展示通过锚检的干净终审网格,污染作废的 r2a/r2b 已排除。
x 轴 = ISL,色相 = 候选臂(同臂内明度随层号渐变,浅层深色/深层浅色),线型 = 层(循环虚线);用复选框选择候选臂/模型/层。
<b>悬停数据点</b>=该 case 精确数值;<b>悬停线条</b>=整条线摘要(几何均值 / 极值 / 逐 ISL 数值),同时其余线自动淡出以突出当前线。
Hover a point for the exact per-case numbers; hover a line for its full summary (geomean / min / max / per-ISL values) — other lines fade while hovering.</p>'''

OUTRO = f'''<p class="note">Speedup chart clips at {SP_HI}× (a handful of sbx small-N cells reach 3.75×; hover shows the true value).
加速比图纵轴截断于 {SP_HI}×,悬停可见真实值。Champion ship verdict: c74f_sbx geomean 1.6828×, 865/865 exact, zero cold regressions.</p>'''

FRAG_BODY = f'''{INTRO}
<div class="vizP">
{''.join(inputs)}
<div class="ctl"><span class="hd">arms / 候选臂</span>{''.join(chips_arm)}</div>
<div class="ctl"><span class="hd">models / 模型</span>{''.join(chips_model)}</div>
<div class="ctl"><span class="hd">Flash layers</span>{chips_layer['flash']}</div>
<div class="ctl"><span class="hd">Pro layers</span>{chips_layer['pro']}</div>
<div class="ctl"><span class="hd">V3.2 layers</span><span class="note">(默认只勾 bench 层 14/34/54;ALL/NONE 为覆盖开关——勾上后强制全显/全隐该模型所有层,逐层勾选暂被忽略并置灰,取消覆盖即恢复;两者同时勾选时 NONE 优先)</span>{chips_layer['v32']}</div>
{''.join(panels)}
{OUTRO}
</div>'''

# ---- standalone page ---------------------------------------------------------
html = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>KF campaign — per-case performance comparison (865 cases, no layer averaging)</title>
<style>
body{{font:14px/1.55 -apple-system,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;background:#fafaf7;color:#1a1a2e;margin:0;padding:22px 26px;max-width:1330px}}
h1{{font-size:20px;margin:0 0 4px}}
code{{font-family:"SF Mono",Consolas,monospace;background:#f0f0ea;border-radius:4px;padding:1px 5px;font-size:0.9em}}
{FRAG_CSS}
</style></head><body>
<h1>KernelFactory campaign — per-case performance comparison / 逐 case 性能对比(不做层间平均)</h1>
{FRAG_BODY}
</body></html>'''
open(OUT, 'w').write(html)
print(f'wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)')

# ---- idempotent injection into KF_PROCESS_LOG.html ---------------------------
LOG = os.path.join(HERE, 'KF_PROCESS_LOG.html')
MARK_S, MARK_E = '<!-- KF-PERCASE:START -->', '<!-- KF-PERCASE:END -->'
ANCHOR = '<!-- KF-FINAL:END -->'
section = f'''{MARK_S}
<h2 id="sec-percase">7 · Per-case interactive comparison / 逐 case 交互性能对比(不做层间平均)</h2>
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
