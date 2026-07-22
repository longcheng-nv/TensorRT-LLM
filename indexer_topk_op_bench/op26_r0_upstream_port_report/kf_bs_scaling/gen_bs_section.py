# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotent §7 BS-scaling supplement injector for KF_PROCESS_LOG.html.

Same conventions as gen_r3_section.py: bilingual, KPI tiles, CSS-only
checkbox/radio chips (no <script>), SVG line charts. Injected as a
KF-R3BS:START/END block immediately BEFORE the KF-R3:END marker (i.e. at
the end of §7). Data: kf_bs_joined.csv (parse_bs_kf.py).

  python3 gen_bs_section.py
"""
import csv
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
HTML = REPORT / "kf_campaign" / "KF_PROCESS_LOG.html"
MARK_S, MARK_E = "<!-- KF-R3BS:START -->", "<!-- KF-R3BS:END -->"
SEC_E = "<!-- KF-R3:END -->"

BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MODELS = ["flash", "pro", "v32"]
MODEL_LBL = {"flash": "V4-Flash (K=512)", "pro": "V4-Pro (K=1024)",
             "v32": "V3.2 (K=2048)"}
FONT = 'font-family="sans-serif"'

SERIES = [  # key -> (numerator col kernel, numerator col span, class, label, color)
    ("PR",  "gvr_pr_us", "gvr_pr_span", "bPR",  "vs GVR PR#16457", "#2a78d6"),
    ("SGL", "sgl_us",    "sgl_span",    "bSGL", "vs sglang v2",     "#008300"),
    ("RDX", "radix_us",  "radix_span",  "bRDX", "vs radix_cutedsl", "#e87ba4"),
    ("FI",  "fi_us",     "fi_span",     "bFI",  "vs flashinfer",    "#eda100"),
]


def gm(v):
    v = [x for x in v if x]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else None


def load():
    cells = []
    for r in csv.DictReader(open(HERE / "kf_bs_joined.csv")):
        c = {"model": r["model"], "isl": r["isl"], "L": int(r["L"]),
             "N": int(r["N"]), "BS": int(r["BS"]), "hit": r["hit"],
             "compB_us": float(r["compB_us"]),
             "compB_span": float(r["compB_span"]) if r["compB_span"] else None,
             "drift": float(r["drift"]) if r.get("drift") else None,
             "gvr2": float(r["gvr2_ratio"]) if r.get("gvr2_ratio") else None,
             "exact": r["exact_compB"] == "True"}
        for key, ku, ks, *_ in SERIES:
            c["x" + key] = (float(r[ku]) / c["compB_us"]) if r.get(ku) else None
            den = c["compB_span"] or c["compB_us"]
            c["s" + key] = (float(r[ks]) / den) if r.get(ks) else None
        cells.append(c)
    return cells


def series_by_bs(cells, prefix, model=None):
    out = {}
    for key, *_ in SERIES:
        pts = []
        for i, bs in enumerate(BS_GRID):
            v = [c[prefix + key] for c in cells
                 if c["BS"] == bs and c[prefix + key]
                 and (model is None or c["model"] == model)]
            if v:
                pts.append((i, gm(v)))
        out[key] = pts
    return out


def chart(kser, sser=None, w=760, h=360, title="", cls_extra=""):
    """log-y line chart over the BS grid; kernel-sum group .mk, span group .ms."""
    vals = [v for s in ([kser] + ([sser] if sser else [])) for pts in s.values()
            for _, v in pts]
    ymin = 10 ** math.floor(math.log10(min(vals)) * 1) / 1.3
    ymax = max(vals) * 1.35
    lpad, rpad, tpad, bpad = 56, 130, 26, 42
    pw, ph = w - lpad - rpad, h - tpad - bpad

    def X(i):
        return lpad + pw * i / (len(BS_GRID) - 1)

    def Y(v):
        return tpad + ph * (1 - (math.log10(v) - math.log10(ymin))
                            / (math.log10(ymax) - math.log10(ymin)))

    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}" '
         f'style="max-width:100%;background:var(--vz-surface);'
         f'border:1px solid var(--vz-grid);border-radius:8px">']
    grid = [g for g in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
            if ymin <= g <= ymax]
    for gv in grid:
        y = Y(gv)
        emph = gv == 1.0
        col = "var(--vz-ref)" if emph else "var(--vz-grid)"
        s.append(f'<line x1="{lpad}" y1="{y:.1f}" x2="{w-rpad}" y2="{y:.1f}" '
                 f'stroke="{col}" stroke-width="{1.6 if emph else 1}"/>')
        s.append(f'<text x="{lpad-8}" y="{y+4:.1f}" font-size="11" text-anchor="end" '
                 f'fill="var(--vz-text2)" {FONT}>{gv:g}×</text>')
    for i, bs in enumerate(BS_GRID):
        s.append(f'<text x="{X(i):.1f}" y="{h-bpad+18}" font-size="11" '
                 f'text-anchor="middle" fill="var(--vz-text2)" {FONT}>{bs}</text>')
    s.append(f'<text x="{lpad+pw/2}" y="{h-6}" font-size="11" text-anchor="middle" '
             f'fill="var(--vz-text2)" {FONT}>BS (same real row replicated)</text>')

    def draw(ser, mcls, dash):
        for key, _, _, cls, label, col in SERIES:
            pts = ser.get(key) or []
            if not pts:
                continue
            poly = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in pts)
            g = [f'<g class="{cls} {mcls}">',
                 f'<polyline points="{poly}" fill="none" stroke="{col}" '
                 f'stroke-width="2"{dash}/>']
            for i, v in pts:
                g.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" '
                         f'stroke="var(--vz-surface)" stroke-width="1.5">'
                         f'<title>{label} @ BS={BS_GRID[i]}: {v:.3f}×</title></circle>')
            li, lv = pts[-1]
            g.append(f'<text x="{X(li)+8:.1f}" y="{Y(lv)+4:.1f}" font-size="10.5" '
                     f'fill="{col}" {FONT}>{label.replace("vs ", "")}</text>')
            g.append("</g>")
            s.append("".join(g))

    draw(kser, "mk", "")
    if sser:
        draw(sser, "ms", ' stroke-dasharray="6 4"')
    s.append("</svg>")
    return "".join(s)


def build(cells):
    n_cells = len(cells)
    n_exact = sum(c["exact"] for c in cells)
    drifts = sorted(c["drift"] for c in cells if c["drift"])
    stab = sorted(c["gvr2"] for c in cells if c["gvr2"])
    bs1 = {k: gm([c["x" + k] for c in cells if c["BS"] == 1 and c["x" + k]])
           for k, *_ in SERIES}
    bs1024 = {k: gm([c["x" + k] for c in cells if c["BS"] == 1024 and c["x" + k]])
              for k, *_ in SERIES}

    # crossover: largest BS with pooled gm >= 1.0 per series
    pooled = series_by_bs(cells, "x")
    cross = {}
    for k, *_ in SERIES:
        ge1 = [BS_GRID[i] for i, v in pooled[k] if v >= 1.0]
        cross[k] = max(ge1) if ge1 else None

    # span tax: compB span/kernel gm per BS
    tax = {bs: gm([c["compB_span"] / c["compB_us"] for c in cells
                   if c["BS"] == bs and c["compB_span"]]) for bs in BS_GRID}

    kpi = []

    def tile(v, lbl):
        kpi.append(f'<span class="kpi">{v} <span style="color:#555">{lbl}</span></span>')

    tile(f"{bs1['PR']:.3f}×", "BS=1 gm vs PR head (75 rungs)")
    tile(f"{bs1024['PR']:.3f}×", "BS=1024 gm vs PR head")
    for k, lbl in (("SGL", "sglang v2"), ("RDX", "radix_cutedsl"), ("FI", "flashinfer")):
        tile(f"BS≤{cross[k]}" if cross[k] else "none", f"win region vs {lbl}")
    tile(f"{n_exact}/{n_cells}", "exact")
    if stab:
        tile(f"med {st.median(stab):.3f}", "pass1↔pass2 gvr stability")

    fig1 = chart(series_by_bs(cells, "x"), series_by_bs(cells, "s"),
                 title="compB BS scaling pooled")
    sm = "".join(
        f'<figure style="flex:1 1 340px;margin:0">'
        + chart(series_by_bs(cells, "x", m), None, w=460, h=300,
                title=f"compB BS scaling {m}")
        + f'<figcaption style="font-size:0.85em;color:var(--vz-text2);text-align:center">'
          f'{MODEL_LBL[m]}</figcaption></figure>'
        for m in MODELS)

    # per-BS table
    rows = []
    for bs in BS_GRID:
        cs = [c for c in cells if c["BS"] == bs]
        r = [f"<td><b>{bs}</b></td>"]
        for k, *_ in SERIES:
            v = gm([c["x" + k] for c in cs if c["x" + k]])
            vv = [c["x" + k] for c in cs if c["x" + k]]
            wins = sum(x >= 1.0 for x in vv)
            style = "background:#e7f6e7" if v and v >= 1.0 else (
                "background:#fbeaea" if v and v < 0.5 else "")
            r.append(f'<td style="{style}">{v:.3f}× <span style="color:#888">'
                     f'({wins}/{len(vv)})</span></td>' if v else "<td>—</td>")
        r.append(f"<td>{tax[bs]:.2f}×</td>" if tax[bs] else "<td>—</td>")
        rows.append("<tr>" + "".join(r) + "</tr>")
    table = ('<table style="border-collapse:collapse;font-size:0.88em">'
             '<tr><th>BS</th>'
             + "".join(f"<th>{lbl}</th>" for *_, lbl, _c in SERIES)
             + "<th>span/kernel tax</th></tr>" + "".join(rows) + "</table>")

    chips1 = "".join(
        f'<input type="checkbox" id="bs-{cls}" checked>' for *_, cls, _l, _c in SERIES)
    chips_lbl = '<div class="chips">' + "".join(
        f'<label for="bs-{cls}"><i style="background:{col}"></i><b>{lbl}</b></label>'
        for *_, cls, lbl, col in SERIES) + (
        '<label for="bs-ms" style="margin-left:18px"><i style="background:#999"></i>'
        '<b>+ span-projected (dashed) / 含发射间隙</b></label></div>')
    css_series = "".join(
        f".vizB #bs-{cls}:not(:checked) ~ * .{cls}{{display:none}}"
        f".vizB #bs-{cls}:checked ~ .chips label[for=bs-{cls}]"
        f"{{border-color:{col};color:#0b0b0b;box-shadow:inset 0 0 0 1px {col}}}"
        f".vizB .chips label[for=bs-{cls}] i{{display:inline-block;width:10px;"
        f"height:10px;border-radius:5px;margin-right:6px}}"
        for *_, cls, _l, col in SERIES)
    css = ('<style>.vizB{--vz-surface:#fcfcfb;--vz-grid:#e4e3df;--vz-ref:#8b8a85;'
           '--vz-text1:#0b0b0b;--vz-text2:#52514e;margin:14px 0}'
           '.vizB .chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}'
           '.vizB .chips label{border:1.5px solid #c9c8c2;border-radius:16px;'
           'padding:3px 12px;cursor:pointer;font-size:0.85em;color:#52514e;'
           'user-select:none}.vizB input{position:absolute;opacity:0;'
           'pointer-events:none}' + css_series +
           '.vizB #bs-ms:not(:checked) ~ * .ms{display:none}'
           '.vizB #bs-ms:checked ~ .chips label[for=bs-ms]{border-color:#555;'
           'color:#0b0b0b}.vizB .chips label[for=bs-ms] i{display:inline-block;'
           'width:10px;height:10px;border-radius:5px;margin-right:6px}'
           '.vizB td,.vizB th{border:1px solid #ddd;padding:3px 9px;text-align:right}'
           '.vizB th{background:#f4f4f2}</style>')

    anchor_note = (f"gvr_pr local (umbriel-b200-019) / REPORT (b200-027): "
                   f"med {st.median(drifts):.3f}, "
                   f"p95 {drifts[int(0.95*(len(drifts)-1))]:.3f}" if drifts else "n/a")

    html = f"""{MARK_S}
<h3 id="sec-7bs">7.9 · compB BS-scaling supplement / BS 扩展性补充 (BS 1–1024, real rows replicated)</h3>
{css}
<p><b>Protocol.</b> The §7 verdict envelope is BS=1. This supplement measures the shipped
<b>compB</b> composite across <b>BS 1–1024</b> (pow-2) on the REPORT §8b per-layer real-capture
envelope — fp32, layers flash&nbsp;{{10,22,34}} / pro&nbsp;{{14,30,46}} / v32&nbsp;{{14,34,54}}, all ISL rungs
(75&nbsp;rungs × 11&nbsp;BS), the <b>same real row replicated</b> across the batch. compB is a
<b>BS=1 single-row kernel</b> (the SOLBench problem contract): batching = BS sequential
same-stream launches from a host loop (kernel source untouched; the large-n path's static
scratch + generation-token barrier forbids concurrent per-row streams). Timing = nsys cold-L2
kernel-sum (20 cold + 50 warm NVTX reps, 512&nbsp;MB evict outside the range).
<b>All five arms are measured locally on this node</b> (umbriel-b200-019, GPUs&nbsp;1-7), per-batch
paired on one GPU: <code>gvr_pr</code> (REPORT-verbatim PR#16457 snapshot build, twice —
pass-1 with compB, pass-2 with the rivals, as a stability gate), sglang&nbsp;v2, radix_cutedsl
and flashinfer (REPORT-verbatim <code>ops_rival</code> builds). Taking rival numbers straight
from REPORT <code>rival_bs_layers.csv</code> was attempted first and REJECTED: the local
<code>gvr_pr</code> anchor drifts vs the b200-027 run by {anchor_note} — the node effect is
BS-structured (~1.07× at BS≤32, ~1.7× at BS≥256) and asymmetric between one-big-kernel arms
and compB's sequential launches, so cross-node PR-arm normalization is unsafe here; REPORT
columns are kept as diagnostics only in <code>kf_bs_scaling/kf_bs.csv</code> tooling.</p>
<p class="cng">协议:§7 判决包络是 BS=1;本补充在 REPORT §8b 逐层真实数据包络(75 档 × BS 1–1024,同一真实行复制成批)上测 compB。
compB 是单行 BS=1 内核(SOLBench 契约),批处理 = 主机循环逐行同流顺序发射(内核源码不动;large-n 路径的静态 scratch+代际
barrier 禁止并发流)。计时 = nsys 冷 L2 kernel-sum。<b>五个臂全部在本机 (b200-019) 逐批同卡配对实测</b>——先试过直接取
REPORT 的 rival 数据做 PR-arm 归一化,但本地 gvr_pr 对 027 报告值的漂移呈 BS 结构化(BS≤32 约 1.07×,BS≥256 约 1.7×)
且对不同内核形态不对称,跨节点归一化不安全,故全部本地重测;REPORT 数值仅留作诊断。</p>
<div style="margin:10px 0">{''.join(kpi)}</div>
<div class="vizB">
{chips1}<input type="checkbox" id="bs-ms">
{chips_lbl}
<figure style="margin:10px 0;text-align:center">{fig1}
<figcaption style="font-size:0.9em;color:var(--vz-text2)">Fig.&nbsp;BS1 — pooled geomean speedup of compB vs each arm over BS (75 rungs; log-y).
Solid = cold-L2 kernel-sum (report-canonical); dashed = NVTX GPU-projected span (includes compB's sequential-launch gaps).
<span class="cng">图 BS1 — compB 对各臂的池化几何均值加速比随 BS 变化(实线=kernel-sum;虚线=含发射间隙的 span 口径)。</span></figcaption></figure>
<div class="smrow" style="display:flex;gap:10px;flex-wrap:wrap">{sm}</div>
<figcaption style="font-size:0.9em;color:var(--vz-text2);text-align:center;margin-top:4px">
Fig.&nbsp;BS2 — per-model BS scaling (kernel-sum). <span class="cng">图 BS2 — 分模型 BS 扩展性(kernel-sum 口径)。</span></figcaption>
<div style="overflow-x:auto;margin:12px 0">{table}</div>
<p style="font-size:0.92em"><b>Reading.</b> Cell format: pooled geomean× (wins/rungs ≥1.0×).
<b>span/kernel tax</b> = compB NVTX-span ÷ kernel-sum — the sequential-launch gap cost that the
kernel-sum metric hides at high BS; batched arms launch once so their tax ≈ 1.
<span class="cng">表格单元 = 池化几何均值×(≥1.0× 的档数/总档数);span/kernel 税 = compB 顺序发射间隙成本,批式臂单次发射故 ≈1。</span></p>
<div style="background:#fdf3e7;border:1.5px solid #b3541e;border-radius:8px;padding:10px 16px;margin:12px 0;font-size:0.93em">
<b>Reliability finding / 可靠性发现.</b> One cell HUNG: <b>compB @ v32 128k L54 BS=1024</b>
(v30 contiguous-slice ladder path, K=2048, N=131075) spun at 100% GPU util for &gt;40 min
(expected &lt;2 min) under 81,920 back-to-back launches and was killed; the same batch's
BS≤512 cells and the other two v32-128k layers (L14/L34, identical N → identical grid) all
completed, so the trigger is <b>data-dependent</b>. The fence-less sense-token grid barrier +
static-scratch re-zero contract is exact at BS=1 (865/865) but this is a live risk marker for
any production port that fires the kernel at high sustained launch rates — echoes the
R3_LEDGER constraint note on the fence-less barrier. The cell is recorded as an error
(never faked); all other 824 cells completed.
<span class="cng">该格在 8.2 万次连续发射下 GPU 100% 空转 &gt;40 分钟被杀;同批 BS≤512 与同 N 的另两层均正常 ⇒ 触发条件与数据相关。
无 fence 的 sense-token barrier + 静态 scratch 复零契约在 BS=1 下 865/865 精确,但对高频连续发射的生产移植是活风险标记
(呼应 R3_LEDGER 的 fence-less 约束条目)。该格如实记为 error,其余 824 格全部完成。</span></div>
{MARK_E}"""
    return html


def main():
    cells = load()
    assert cells, "kf_bs_joined.csv empty — run parse_bs_kf.py first"
    block = build(cells)
    doc = HTML.read_text()
    if MARK_S in doc:
        pre = doc[:doc.index(MARK_S)]
        post = doc[doc.index(MARK_E) + len(MARK_E):]
        doc = pre + block + post
    else:
        assert SEC_E in doc, "KF-R3:END marker not found"
        doc = doc.replace(SEC_E, block + "\n" + SEC_E, 1)
    HTML.write_text(doc)
    print(f"injected {len(block)} chars into {HTML.name}")


if __name__ == "__main__":
    main()
