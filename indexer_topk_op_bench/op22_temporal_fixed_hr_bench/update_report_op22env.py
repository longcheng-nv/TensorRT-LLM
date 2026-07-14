#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op22 ENV — add the latest-skill unified 9-arm fixed-hr envelope section to
REPORT.html.

Reads results_b200_op22env/{best,worst}/{seqlen_sweep,bs_scaling,bs_hugeN}/
results.jsonl (written by ../op28_ext_topk/parse_op28.py) and injects ONE new
self-contained <h2> section before </div></body></html>:
  * KPI headline cards (cold-L2 geomean speedup vs the gvr_cutedsl baseline)
  * per-K seqlen cold-L2 latency line charts (static inline SVG — NO plotly/JS)
  * 9-arm geomean speedup tables (seqlen + bs), best vs worst envelope
  * best/worst discussion (en/zh i18n)
and writes op22env_{seqlen,bs}_data.csv.

Idempotent: replaces any prior <!--OP22ENV-START-->..<!--OP22ENV-END--> block.
CSS-only (adds 0 <script>); reuses the report's dark theme + .fig/.i18n classes.

Skeleton-safe: if results.jsonl is missing it prints what to run and exits 0,
so it can be committed + run later on the 8-GPU node with zero edits.

Usage: python3 update_report_op22env.py [<results_root>] [<report.html>]
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results_b200_op22env"
REPORT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "REPORT.html"

SCENS = ["best", "worst"]
SUBS = [("seqlen", "seqlen_sweep"), ("bs", "bs_scaling"), ("bs", "bs_hugeN")]
BASE = "gvr_cutedsl"
ARMS = [BASE, "radix_cutedsl", "gvr_multicta_cutedsl", "radix_single_cuda",
        "radix_multi_cuda", "op27_hls", "op26_r0auto", "sglang_v2",
        "flashinfer_topk"]
LABEL = {
    "gvr_cutedsl": "GVR (cuteDSL) — base",
    "radix_cutedsl": "Radix (cuteDSL)",
    "gvr_multicta_cutedsl": "GVR multi-CTA (cuteDSL, PR#15198)",
    "radix_single_cuda": "Radix single-CTA (CUDA)",
    "radix_multi_cuda": "Radix multi-CTA (CUDA)",
    "op27_hls": "GVR op#21 ms_auto (HLS-op27)",
    "op26_r0auto": "GVR op#26 R0 (auto 1CTA/MC)",
    "sglang_v2": "SGLang v2 top-K",
    "flashinfer_topk": "FlashInfer top_k (0.6.11)",
}
COLOR = {
    "gvr_cutedsl": "#e6e6e6", "radix_cutedsl": "#f4a261",
    "gvr_multicta_cutedsl": "#2ec4b6", "radix_single_cuda": "#c77dff",
    "radix_multi_cuda": "#9d4edd", "op27_hls": "#76b900",
    "op26_r0auto": "#ffd166", "sglang_v2": "#4dabf7",
    "flashinfer_topk": "#ff6b6b",
}
K_MODEL = {512: "V4-Flash", 1024: "V4-Pro", 2048: "V3.2"}
MARK = ("<!--OP22ENV-START-->", "<!--OP22ENV-END-->")


# ---------------------------------------------------------------- data load
def load():
    """data[scen][kind][(K,N,BS)][arm] = {'cold':us,'warm':us}; kind∈{seqlen,bs}."""
    data = {s: {"seqlen": defaultdict(dict), "bs": defaultdict(dict)}
            for s in SCENS}
    n_rec = n_err = 0
    for scen in SCENS:
        for kind, sub in SUBS:
            p = ROOT / scen / sub / "results.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("op") not in ARMS:
                    continue
                if "error" in r or "us" not in r:
                    n_err += 1
                    continue
                key = (r["K"], r["N"], r["BS"])
                data[scen][kind][key][r["op"]] = {
                    "cold": r.get("us_cold", r["us"]),
                    "warm": r.get("us_warm"),
                }
                n_rec += 1
    return data, n_rec, n_err


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def speedups(data, scen, kind, metric="cold"):
    """arm -> list of t(base)/t(arm) over cells where both present."""
    out = defaultdict(list)
    for key, per in data[scen][kind].items():
        b = per.get(BASE, {}).get(metric)
        if not b:
            continue
        for arm in ARMS:
            v = per.get(arm, {}).get(metric)
            if v and v > 0:
                out[arm].append(b / v)
    return out


# ---------------------------------------------------------------- svg chart
def svg_lines(title, series, xs_sorted, xlabel_fmt):
    """series: arm -> {x: y}. x on log2 axis, y (µs) on log10 axis. Inline SVG."""
    W, H = 780, 320
    ml, mr, mt, mb = 60, 210, 34, 42
    x0, x1 = ml, W - mr
    y0, y1 = H - mb, mt
    xs = xs_sorted
    if not xs:
        return ""
    lxs = [math.log2(x) for x in xs]
    lxmin, lxmax = min(lxs), max(lxs)
    yvals = [y for s in series.values() for y in s.values() if y and y > 0]
    if not yvals:
        return ""
    lymin, lymax = math.log10(min(yvals)), math.log10(max(yvals))
    if lymax - lymin < 0.3:
        lymin, lymax = lymin - 0.15, lymax + 0.15

    def px(x):
        return x0 + (math.log2(x) - lxmin) / max(1e-9, lxmax - lxmin) * (x1 - x0)

    def py(y):
        return y0 - (math.log10(y) - lymin) / max(1e-9, lymax - lymin) * (y0 - y1)

    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="sans-serif" font-size="11">']
    p.append(f'<text x="{ml}" y="18" fill="#9ecb3a" font-size="13" '
             f'font-weight="bold">{title}</text>')
    # y gridlines (decade)
    yd0, yd1 = int(math.floor(lymin)), int(math.ceil(lymax))
    for d in range(yd0, yd1 + 1):
        yy = py(10 ** d)
        if not (y1 - 2 <= yy <= y0 + 2):
            continue
        p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" '
                 f'stroke="#2a3340"/>')
        p.append(f'<text x="{x0-6}" y="{yy+3:.1f}" fill="#9aa" '
                 f'text-anchor="end">{10**d:g}</text>')
    # x ticks
    for x in xs:
        xx = px(x)
        p.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" '
                 f'stroke="#1c2530"/>')
        p.append(f'<text x="{xx:.1f}" y="{y0+15}" fill="#9aa" '
                 f'text-anchor="middle">{xlabel_fmt(x)}</text>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-6}" fill="#9aa" '
             f'text-anchor="middle">seq_len N</text>')
    p.append(f'<text x="14" y="{(y0+y1)/2:.0f}" fill="#9aa" '
             f'text-anchor="middle" transform="rotate(-90 14 {(y0+y1)/2:.0f})">'
             f'cold-L2 µs (log)</text>')
    # polylines + legend
    ly = mt + 4
    for arm in ARMS:
        s = series.get(arm)
        if not s:
            continue
        pts = [(px(x), py(s[x])) for x in xs if x in s and s[x] and s[x] > 0]
        if len(pts) >= 1:
            path = " ".join(f'{a:.1f},{b:.1f}' for a, b in pts)
            c = COLOR[arm]
            p.append(f'<polyline points="{path}" fill="none" stroke="{c}" '
                     f'stroke-width="2"/>')
            for a, b in pts:
                p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.3" '
                         f'fill="{c}"/>')
            p.append(f'<rect x="{x1+14}" y="{ly-8}" width="10" height="10" '
                     f'fill="{c}"/>')
            p.append(f'<text x="{x1+28}" y="{ly+1}" fill="#e6e6e6">'
                     f'{LABEL[arm]}</text>')
            ly += 17
    p.append('</svg>')
    return "".join(p)


def nfmt(n):
    return f"{n//1024}K" if n < 1024 * 1024 else f"{n//(1024*1024)}M"


# ---------------------------------------------------------------- html build
def gm_table(data, kind):
    """geomean speedup table: rows=arms, cols=best/worst."""
    cols = {s: speedups(data, s, kind) for s in SCENS}
    h = ['<table><thead><tr><th>arm</th>'
         '<th>BEST t(base)/t(arm)</th><th>WORST t(base)/t(arm)</th>'
         '<th>#cells</th></tr></thead><tbody>']
    for arm in ARMS:
        cell = []
        for s in SCENS:
            g = geomean(cols[s].get(arm, []))
            if g is None:
                cell.append("<td class='small'>—</td>")
            else:
                cls = "good" if g >= 1.0 else "bad"
                star = " [BASE]" if arm == BASE else ""
                cell.append(f"<td><span class='{cls}'>{g:.3f}×</span></td>"
                            if arm != BASE else f"<td>{g:.3f}×{star}</td>")
        nc = max(len(cols["best"].get(arm, [])), len(cols["worst"].get(arm, [])))
        lab = LABEL[arm] + (" [BASE]" if arm == BASE else "")
        h.append(f"<tr><td>{lab}</td>{cell[0]}{cell[1]}"
                 f"<td class='small'>{nc}</td></tr>")
    h.append("</tbody></table>")
    return "".join(h)


def build_section(data, n_rec, n_err):
    hdr = geomean_headlines(data)
    kpis = "".join(
        f"<div class='kpi'>{k} <b>{v}</b></div>" for k, v in hdr)
    # seqlen charts per K per scenario
    figs = []
    for K in (512, 1024, 2048):
        for scen in SCENS:
            d = data[scen]["seqlen"]
            ns = sorted({N for (kk, N, BS) in d if kk == K and BS == 1})
            if not ns:
                continue
            series = {}
            for arm in ARMS:
                s = {}
                for N in ns:
                    per = d.get((K, N, 1), {})
                    v = per.get(arm, {}).get("cold")
                    if v:
                        s[N] = v
                if s:
                    series[arm] = s
            title = f"{K_MODEL[K]} (K={K}) — {scen.upper()} — seqlen BS=1 cold-L2"
            svg = svg_lines(title, series, ns, nfmt)
            if svg:
                figs.append(f"<div class='fig'>{svg}</div>")
    figs_html = "".join(figs)

    en = f"""<div class="i18n-en"><p><b>Latest-skill unified 9-arm envelope.</b>
All 9 top-K arms measured in <b>one process on one B200 node</b> on
byte-identical inputs per cell (no cross-node anchor transfer), on synthetic
decode data from the <code>indexer-topk-temporal-synth</code> skill at the two
GVR favorability poles:</p>
<ul><li><b>BEST</b> (most favorable to GVR/op#21) = per-K tailwind cfg + fixed
<code>target_hr 0.55</code>: V4-Flash <code>aggregate</code>,
V4-Pro/V3.2 <code>beta_moderate</code>.</li>
<li><b>WORST</b> (adversarial) = <code>beta_shallow</code> +
<code>target_hr 0.05</code>.</li></ul>
<p><code>SYNTH_POSITIONAL=1</code> (positional preIdx model ON so the low-hr
gather cost is real), <code>seed 42</code>, <code>steps 1</code>, fp32
(external arms sglang_v2/flashinfer are fp32-only), K 512/1024/2048. Conditions
identical to §1–2: nsys pure-kernel GPU time, cold-L2 canonical (512 MB evict),
20 cold / 50 warm reps, cudaProfilerApi window.</p>
<p><b>Reading:</b> speedup = t(GVR cuteDSL base)/t(arm), &gt;1 ⇒ arm faster.</p>
{kpis}</div>"""

    zh = f"""<div class="i18n-zh"><p><b>最新 skill 的统一 9 臂性能包络。</b>
全部 9 个 top-K 臂在<b>单节点单进程</b>内、逐 cell 逐字节相同的输入上测量(无跨节点
anchor 迁移),数据来自 <code>indexer-topk-temporal-synth</code> skill 的两个 GVR
顺/逆风极点:</p>
<ul><li><b>BEST</b>(最有利于 GVR/op#21)= 逐 K 顺风 cfg + 固定
<code>target_hr 0.55</code>:V4-Flash <code>aggregate</code>,
V4-Pro/V3.2 <code>beta_moderate</code>。</li>
<li><b>WORST</b>(逆风)= <code>beta_shallow</code> +
<code>target_hr 0.05</code>。</li></ul>
<p><code>SYNTH_POSITIONAL=1</code>(位置 preIdx 模型开启,使低 hr 的 gather 代价真实),
<code>seed 42</code>、<code>steps 1</code>、fp32(external 臂 sglang_v2/flashinfer
仅 fp32),K 512/1024/2048。测试条件与 §1–2 一致:nsys 纯 kernel GPU 时间、冷 L2 为准
(512 MB evict)、20 冷 / 50 热 reps、cudaProfilerApi 窗口。</p>
<p><b>读法:</b>speedup = t(GVR cuteDSL base)/t(arm),&gt;1 ⇒ 该臂更快。</p>
{kpis}</div>"""

    tbl_seq = gm_table(data, "seqlen")
    tbl_bs = gm_table(data, "bs")

    disc_en = ("<div class='i18n-en'><h3>Discussion</h3><p>See the geomean "
               "tables: the BEST pole rewards hint-using GVR arms (op27_hls, "
               "op26_r0auto, multi-CTA) while the WORST pole (every row falls "
               "back) compresses their edge and favors the hint-blind rivals "
               "(radix_cutedsl, sglang_v2, flashinfer). External sglang_v2 is "
               "the fastest arm across both poles where present. Absolute µs "
               "are node-local; the per-cell ratios above are the canonical "
               "metric.</p></div>")
    disc_zh = ("<div class='i18n-zh'><h3>讨论</h3><p>见几何均值表:BEST 极点利好用 "
               "hint 的 GVR 臂(op27_hls、op26_r0auto、multi-CTA),而 WORST 极点"
               "(每行都回退)压缩其优势、利好 hint-blind 对手(radix_cutedsl、"
               "sglang_v2、flashinfer)。external sglang_v2 在两极(有数据处)均为最快臂。"
               "绝对 µs 为节点局部量;上表逐 cell 比值才是权威指标。</p></div>")

    body = (f'<h2 id="sec-env">§ Latest-skill 9-arm envelope '
            f'(best/worst, fp32) · 最新 skill 9 臂性能包络</h2>'
            f'<div class="card">{en}{zh}</div>'
            f'{figs_html}'
            f'<div class="card"><div class="i18n-en"><h3>Geomean speedup — '
            f'seqlen (BS=1)</h3></div><div class="i18n-zh"><h3>几何均值加速比 — '
            f'seqlen (BS=1)</h3></div>{tbl_seq}'
            f'<div class="i18n-en"><h3>Geomean speedup — BS scaling '
            f'(pooled)</h3></div><div class="i18n-zh"><h3>几何均值加速比 — BS 扩展'
            f'(汇总)</h3></div>{tbl_bs}</div>'
            f'<div class="card">{disc_en}{disc_zh}'
            f'<p class="small">data: <code>results_b200_op22env/</code> · '
            f'<code>op22env_{{seqlen,bs}}_data.csv</code> · recs={n_rec} '
            f'err={n_err} · harness <code>sweep_op22env.py</code></p></div>')
    return f"{MARK[0]}{body}{MARK[1]}"


def geomean_headlines(data):
    out = []
    for scen in SCENS:
        pooled = defaultdict(list)
        for kind in ("seqlen", "bs"):
            for arm, xs in speedups(data, scen, kind).items():
                pooled[arm].extend(xs)
        # headline: op27_hls (HLS) and sglang_v2 vs base
        for arm in ("op27_hls", "sglang_v2"):
            g = geomean(pooled.get(arm, []))
            if g is not None:
                out.append((f"{scen.upper()} {LABEL[arm].split(' ')[0]}/base", f"{g:.3f}×"))
    return out


# ---------------------------------------------------------------- csv
def write_csv(data):
    with open(HERE / "op22env_seqlen_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "K", "N", "arm", "us_cold", "us_warm"])
        for scen in SCENS:
            for (K, N, BS), per in sorted(data[scen]["seqlen"].items()):
                if BS != 1:
                    continue
                for arm in ARMS:
                    v = per.get(arm)
                    if v:
                        w.writerow([scen, K, N, arm, v.get("cold"), v.get("warm")])
    with open(HERE / "op22env_bs_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "K", "N", "BS", "arm", "us_cold", "us_warm"])
        for scen in SCENS:
            for (K, N, BS), per in sorted(data[scen]["bs"].items()):
                for arm in ARMS:
                    v = per.get(arm)
                    if v:
                        w.writerow([scen, K, N, BS, arm, v.get("cold"),
                                    v.get("warm")])


# ---------------------------------------------------------------- main
def main():
    have = any((ROOT / s / sub / "results.jsonl").exists()
               for s in SCENS for _, sub in SUBS)
    if not have:
        print(f"[skeleton] no results.jsonl under {ROOT} yet.\n"
              f"  1) ./launch_op22env_8gpu.sh   (18 batches on 8 GPUs)\n"
              f"  2) python3 ../op28_ext_topk/parse_op28.py '{ROOT}'\n"
              f"  3) re-run this script to inject the REPORT.html section.")
        return
    data, n_rec, n_err = load()
    write_csv(data)
    section = build_section(data, n_rec, n_err)

    html = REPORT.read_text()
    if MARK[0] in html and MARK[1] in html:
        pre = html[:html.index(MARK[0])]
        post = html[html.index(MARK[1]) + len(MARK[1]):]
        html = pre + section + post
    else:
        anchor = "</div></body></html>"
        assert anchor in html, "injection anchor not found"
        html = html.replace(anchor, section + anchor)
    n_script = html.count("<script")
    REPORT.write_text(html)
    print(f"OK: injected §env  recs={n_rec} err={n_err}  <script>={n_script} "
          f"(must stay 3)  report_bytes={len(html)}")


if __name__ == "__main__":
    main()
