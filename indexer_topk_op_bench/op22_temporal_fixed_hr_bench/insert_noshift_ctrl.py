# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Insert the v32 preIdx no-shift CONTROL-EXPERIMENT card into REPORT.html.

Idempotent: bounded by <!--NOSHIFT-CTRL-BEGIN-->..<!--NOSHIFT-CTRL-END-->,
re-running replaces the block. Data = op22real_v32_noshift_ab.csv (same-GPU
paired A/B produced by sweep_op22_real.py with OP22REAL_V32_NOSHIFT=0/1).
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REP = HERE / "REPORT.html"
BEGIN = "<!--NOSHIFT-CTRL-BEGIN-->"
END = "<!--NOSHIFT-CTRL-END-->"

rows = list(csv.DictReader(open(HERE / "op22real_v32_noshift_ab.csv")))
by = {(r["dtype"], r["arm"]): r for r in rows}
GVR = ["gvr_cutedsl", "op26_r0auto", "op27_hls", "op26_1cta", "op26_mc",
       "gvr_multicta_cutedsl"]
INDEP = ["radix_cutedsl", "radix_single_cuda", "sglang_v2", "flashinfer_topk"]
LBL = {"gvr_cutedsl": "gvr_cutedsl [BASE]", "op26_r0auto": "GVR op#26 R0 (auto)",
       "op27_hls": "GVR op#21 ms_auto (HLS-op27)", "op26_1cta": "op26_1cta",
       "op26_mc": "op26_mc", "gvr_multicta_cutedsl": "gvr_multicta_cutedsl",
       "radix_cutedsl": "radix_cutedsl", "radix_single_cuda": "radix_single_cuda",
       "sglang_v2": "sglang_v2", "flashinfer_topk": "flashinfer_topk"}


def spd_span(v):
    cls = "good" if v >= 1.03 else ("bad" if v <= 0.97 else "")
    return f"<span class='{cls}'>{v:.3f}×</span>"


def rowhtml(arm, dt="fp32"):
    r = by.get((dt, arm))
    if not r:
        return ""
    return (f"<tr><td>{LBL[arm]}</td><td>{float(r['us_plus1']):.2f}</td>"
            f"<td>{float(r['us_noshift']):.2f}</td>"
            f"<td>{spd_span(float(r['paired_noshift_speedup']))}</td>"
            f"<td class='muted'>{float(r['per_layer_min']):.2f}–"
            f"{float(r['per_layer_max']):.2f}</td></tr>")


hc = float(by[("fp32", "gvr_cutedsl")]["hit_plus1"])
hn = float(by[("fp32", "gvr_cutedsl")]["hit_noshift"])
r0 = float(by[("fp32", "op26_r0auto")]["paired_noshift_speedup"])
hls = float(by[("fp32", "op27_hls")]["paired_noshift_speedup"])

gvr_rows = "".join(rowhtml(a) for a in GVR)
indep_rows = "".join(rowhtml(a) for a in INDEP)

TABLE = f"""
<table class="tbl"><thead><tr>
<th>arm (fp32, v32/K2048, BS=1, 9 layers)</th>
<th>µs +1 (hit {hc:.2f})</th><th>µs raw (hit {hn:.2f})</th>
<th>raw-align speedup</th><th class='muted'>per-layer range</th></tr></thead>
<tbody>
<tr class='sub'><td colspan=5><b>hint-driven (GVR) arms</b></td></tr>
{gvr_rows}
<tr class='sub'><td colspan=5><b>hint-independent arms (control — must ≈1.00)</b></td></tr>
{indep_rows}
</tbody></table>"""

EN = f"""<div class="i18n-en"><p>The report's real rows feed the v32 (cr=1) kernel
the recomputed previous-step top-K as <code>preIdx</code>; the kernel internally
reads <code>logits[(preIdx+1) mod N]</code> (the validated production +1 offset),
giving a median kernel-read hit-rate of <b>{hc:.2f}</b>. This control passes
<code>(preIdx−1) mod N</code> so the kernel's +1 recovers RAW current-frame
alignment, lifting the median hit-rate to <b>{hn:.2f}</b> (shallow layers L0/L1
recover from ≈0.01–0.02 to 0.23/0.89). Same-GPU paired A/B (fp32 clean: all four
hint-independent arms land at 0.997–1.002×, confirming zero cross-run bias),
exactness preserved (vdiff=0, recall=1, n_neg=0 on every cell).</p>
{TABLE}
<p><b>Verdict — preIdx alignment is NOT the driver of the real-row collapse.</b>
Raising the hit-rate 0.44→0.66 speeds up only <b>op26_r0auto (+14%,
{r0:.3f}×)</b> and <b>op27_hls (+{(hls-1)*100:.0f}%, {hls:.3f}×)</b>; the base and
the 1cta/mc/multi-CTA GVR arms do not improve (0.81–0.92×, high per-layer
scatter). Even the r0auto gain is far short of the ~1.7–1.8× the §"sequence-length
sweep" headline showed — that headline is a geomean dominated by N≥256K cells,
whereas the real captures sit at moderate N (14K–70K) where the dispatch/HLS
optimizations are latency-bound (op32 structural wall). op26_r0auto's raw-aligned
ratio-vs-base does rise into the synth-at-matched-N band, but that is partly the
base regressing, not a clean win. Corroborated across dtypes: op26_r0auto raw-align
speedup = 1.144/1.131/1.118× (fp32/bf16/fp16). <span class="muted">Data:
<code>op22real_v32_noshift_ab.csv</code>; runner
<code>sweep_op22_real.py OP22REAL_V32_NOSHIFT=1</code>. bf16/fp16
<code>radix_cutedsl</code> drifted to 0.81× from a cuteDSL compile-warmth artifact
across the sequential processes — <code>radix_single_cuda</code>=1.00 confirms the
GPU was stable, so fp32 is the clean headline.</span></p></div>"""

ZH = f"""<div class="i18n-zh"><p>报告的真实行给 v32(cr=1) kernel 传入重算的上一步 top-K
作为 <code>preIdx</code>；kernel 内部读 <code>logits[(preIdx+1) mod N]</code>（已验证
合理的生产 +1 偏移），kernel-read 命中率中位 <b>{hc:.2f}</b>。本对照传入
<code>(preIdx−1) mod N</code>，让 kernel 的 +1 恢复到 RAW 当前帧对齐，命中率中位升到
<b>{hn:.2f}</b>（浅层 L0/L1 从 ≈0.01–0.02 回到 0.23/0.89）。同 GPU 配对 A/B（fp32
干净：四个不依赖 hint 的臂全部落在 0.997–1.002×，证明无跨轮偏置），正确性保持
（每格 vdiff=0、recall=1、n_neg=0）。</p>
{TABLE}
<p><b>结论 —— preIdx 对齐不是真实行塌陷的主因。</b>把命中率 0.44→0.66 只提速了
<b>op26_r0auto（+14%，{r0:.3f}×）</b>和 <b>op27_hls（+{(hls-1)*100:.0f}%，
{hls:.3f}×）</b>；base 及 1cta/mc/multi-CTA 各臂并未变快（0.81–0.92×，逐层散度大）。
即便 r0auto 的增益也远不及§"序列长度扫描"headline 的 ~1.7–1.8×——那个 headline 是被
N≥256K 单元主导的几何均值，而真实采集处于中短 N（14K–70K），此处 dispatch/HLS 优化受
latency 限制（op32 结构墙）。op26_r0auto raw 对齐后的 ratio-vs-base 确实升入"合成同 N"
区间，但部分是 base 退化所致，并非干净胜出。跨 dtype 佐证：op26_r0auto raw 对齐提速 =
1.144/1.131/1.118×（fp32/bf16/fp16）。<span class="muted">数据：
<code>op22real_v32_noshift_ab.csv</code>；运行
<code>sweep_op22_real.py OP22REAL_V32_NOSHIFT=1</code>。bf16/fp16 的
<code>radix_cutedsl</code> 因 cuteDSL 跨进程编译热度漂移到 0.81×——
<code>radix_single_cuda</code>=1.00 证明 GPU 本身稳定，故以 fp32 为干净 headline。</span>
</p></div>"""

CARD = (f'{BEGIN}<div class="card" id="realcap-noshift-ctrl">'
        f'<h3><span class="i18n-en">Control — v32 preIdx raw-alignment '
        f'(no-shift) sensitivity</span>'
        f'<span class="i18n-zh">对照 —— v32 preIdx raw 对齐（去 +1）敏感性</span>'
        f'</h3>{EN}{ZH}</div>{END}')

html = REP.read_text()
# strip any prior insertion
html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", html, flags=re.S)
anchor = "<!-- REALCAP-CHAPTER-END -->"
assert anchor in html, "realcap end anchor missing"
html = html.replace(anchor, anchor + CARD, 1)
bak = REP.with_suffix(".html.bak_pre_noshift")
if not bak.exists():
    bak.write_text(REP.read_text())
REP.write_text(html)
print(f"inserted control card ({len(CARD)} bytes) after {anchor}")
print(f"fp32 headline: r0auto {r0:.3f}x  op27_hls {hls:.3f}x  hit {hc:.2f}->{hn:.2f}")
