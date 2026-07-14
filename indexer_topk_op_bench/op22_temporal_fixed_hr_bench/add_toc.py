#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Structure REPORT.html into content chapters and inject a bilingual, grouped
# Table of Contents (目录) synced to those chapters.
#
# Idempotent:
#   * adds id="sec-N" to the 8 numbered <h2> (only if still bare);
#   * replaces any prior <!--TOC-START-->..<!--TOC-END--> block;
#   * injects TOC CSS once (guarded by /*TOC-CSS*/ marker);
#   * pure HTML/CSS — adds NO <script> (keeps the report's <script> count).
import re
from pathlib import Path

REPORT = Path(__file__).parent / "REPORT.html"
h = REPORT.read_text(encoding="utf-8")
scripts_before = h.count("<script")

# --- 1) give the 8 numbered sections stable ids (bare <h2> only) --------------
h = re.sub(r'<h2>(<span class="i18n-en">(\d+)\.)',
           lambda m: f'<h2 id="sec-{m.group(2)}">{m.group(1)}', h)

# --- 2) TOC model: (id, en, zh) grouped into content chapters -----------------
PARTS = [
    ("Part I · Synthetic-data sweeps &amp; mechanism",
     "第一部分 · 合成数据扫描与机理", [
        ("sec-1", "1. Seq-len sweep (BS=1)", "1. 序列长度扫描（BS=1）"),
        ("sec-2", "2. BS-scaling (BS 1→2048)", "2. BS 扩展性（BS 1→2048）"),
        ("sec-3", "3. Full data — legacy dataset", "3. 全量数据（legacy 数据集）"),
        ("sec-4", "4. Geomean &amp; hit-rate sensitivity — legacy",
                  "4. 几何均值与 hit-rate 敏感性（legacy）"),
        ("sec-5", "5. Findings — extreme cells (op21 vs Radix)",
                  "5. 极端 cell 发现（op21 对 Radix）"),
        ("sec-6", "6. Mechanism — why hr=0.90 slows the GVR family",
                  "6. 机制 — 为何 hr=0.90 拖慢整个 GVR 家族"),
     ]),
    ("Part II · Analytical bounds", "第二部分 · 解析上下界", [
        ("sec-7", "7. op23 — deterministic UB/LB bounds",
                  "7. op23 — 确定性上/下界"),
        ("sec-8", "8. op24 — synthesis-parameter favorability bounds",
                  "8. op24 — 顺/逆风参数定界"),
     ]),
    ("Part III · Real captured data &amp; cross-arm envelope",
     "第三部分 · 真实采集数据与跨臂包络", [
        ("realcap", "9. Real captured-data — production inference logits",
                    "9. 真实采集数据 — 生产推理 logits"),
        ("sec-env", "§ Latest-skill 9-arm envelope (best/worst, fp32)",
                    "§ 最新 skill 9 臂性能包络（best/worst, fp32）"),
        ("v4cap", "10. V4 decode-capture — seq-len scan &amp; BS scaling",
                  "10. V4 decode 采集 — 序列长度扫描与 BS 扩展"),
     ]),
]

def li(sid, en, zh):
    return (f'<li><a href="#{sid}">'
            f'<span class="i18n-en">{en}</span>'
            f'<span class="i18n-zh">{zh}</span></a></li>')

blocks = []
for pen, pzh, items in PARTS:
    lis = "".join(li(*it) for it in items)
    blocks.append(
        f'<div class="toc-part"><span class="i18n-en">{pen}</span>'
        f'<span class="i18n-zh">{pzh}</span></div><ul class="toc-list">{lis}</ul>')
toc = (
    '<!--TOC-START--><nav class="toc card" id="toc">'
    '<h2 style="margin-top:0;border:none">'
    '<span class="i18n-en">Table of Contents</span>'
    '<span class="i18n-zh">目录</span></h2>'
    + "".join(blocks) + '</nav><!--TOC-END-->')

# --- 3) inject / replace TOC right after the title's meta paragraph -----------
h = re.sub(r"\n?<!--TOC-START-->.*?<!--TOC-END-->", "", h, flags=re.S)
m = re.search(r"</h1>\s*<p class=\"meta\">.*?</p>", h, flags=re.S)
if not m:
    raise SystemExit("!! could not locate <h1>+meta anchor for TOC insertion")
h = h[:m.end()] + "\n" + toc + h[m.end():]

# --- 4) TOC CSS (once) --------------------------------------------------------
if "/*TOC-CSS*/" not in h:
    css = ("/*TOC-CSS*/\n"
           ".toc{margin:20px 0}\n"
           ".toc>h2{color:#76b900;font-size:20px}\n"
           ".toc-part{color:#9ecb3a;font-weight:700;margin:12px 0 4px;"
           "font-size:14px;letter-spacing:.02em}\n"
           ".toc-list{list-style:none;margin:0 0 6px;padding-left:14px;"
           "columns:2;column-gap:32px}\n"
           "@media(max-width:800px){.toc-list{columns:1}}\n"
           ".toc-list li{margin:3px 0;break-inside:avoid}\n"
           ".toc-list a{color:#e6e6e6;text-decoration:none;"
           "border-bottom:1px dotted #3a4652}\n"
           ".toc-list a:hover{color:#76b900;border-bottom-color:#76b900}\n"
           "h2[id]{scroll-margin-top:12px}\n")
    h = h.replace("</style>", css + "</style>", 1)

scripts_after = h.count("<script")
assert scripts_after == scripts_before, \
    f"<script> count changed {scripts_before}->{scripts_after}"

REPORT.write_text(h, encoding="utf-8")
ids = re.findall(r'<h2 id="(sec-\d+|realcap|sec-env|v4cap)"', h)
print(f"OK: TOC injected. section-ids present={len(set(ids))}/11  "
      f"<script>={scripts_after} (unchanged)  bytes={len(h)}")
print("ids:", sorted(set(ids)))
