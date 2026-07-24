# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full per-case data report (NO averaging) -> R4_R5_FULL_DATA.html.

Tables: R4 champion 865 BS=1 cells; R5 champion 750 (cell x BS) cases;
R5 champion BS=1 guard 865 cells. Bilingual narrative via CSS-only toggle;
tables are shared (bilingual column headers).
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ISL_ORD = {s: i for i, s in enumerate(
    ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"])}


def sp_class(v):
    return "hi" if v >= 1.5 else ("reg" if v < 1.0 else "")


def table_r4(path, cand_label):
    rows = [r for r in csv.DictReader(open(path)) if r["speedup_cold"]]
    rows.sort(key=lambda r: (r["model"], ISL_ORD[r["isl"]], int(r["layer"])))
    out = [f'<table><tr><th>cell</th><th>N</th><th>K</th><th>hit</th>'
           f'<th>PR head µs</th><th>{cand_label} µs</th><th>speedup 加速比</th></tr>']
    cur = None
    for r in rows:
        key = (r["model"], r["isl"])
        if key != cur:
            cur = key
            out.append(f'<tr class="grp"><td colspan="7">{r["model"]} · ISL {r["isl"]}</td></tr>')
        v = float(r["speedup_cold"])
        out.append(f'<tr><td>{r["uuid"]}</td><td>{r["N"]}</td><td>{r["K"]}</td>'
                   f'<td>{r["hit"]}</td><td>{float(r["pr_cold"]):.2f}</td>'
                   f'<td>{float(r["cand_cold"]):.2f}</td>'
                   f'<td class="{sp_class(v)}">{v:.3f}</td></tr>')
    out.append("</table>")
    return "\n".join(out), len(rows)


def table_r5(path):
    rows = [r for r in csv.DictReader(open(path)) if r["speedup_cold"]]
    rows.sort(key=lambda r: (r["model"], ISL_ORD[r["isl"]], int(r["layer"]), int(r["bs"])))
    out = ['<table><tr><th>cell</th><th>BS</th><th>N</th><th>K</th><th>hit</th>'
           '<th>PR head batch µs</th><th>R5 champ µs</th><th>speedup 加速比</th></tr>']
    cur = None
    for r in rows:
        key = (r["model"], r["isl"], r["layer"])
        if key != cur:
            cur = key
            out.append(f'<tr class="grp"><td colspan="8">{r["model"]} · ISL {r["isl"]} · L{int(r["layer"]):02d} (N={r["N"]}, K={r["K"]}, hit={r["hit"]})</td></tr>')
        v = float(r["speedup_cold"])
        out.append(f'<tr><td>{r["uuid"]}</td><td>{r["bs"]}</td><td>{r["N"]}</td>'
                   f'<td>{r["K"]}</td><td>{r["hit"]}</td>'
                   f'<td>{float(r["pr_cold"]):.2f}</td><td>{float(r["cand_cold"]):.2f}</td>'
                   f'<td class="{sp_class(v)}">{v:.3f}</td></tr>')
    out.append("</table>")
    return "\n".join(out), len(rows)


t_r4, n_r4 = table_r4(HERE / "grid_r4r3cg.csv", "R4 champ")
t_r5, n_r5 = table_r5(HERE / "r5_bs" / "grid_r5g_final.csv")
t_g, n_g = table_r4(HERE / "grid_r5bs1guard.csv", "R5 champ")

EN = f"""<h1>Full per-case data — KF champions vs GVR PR head (no averaging)</h1>
<p class="meta">Denominator: PR#16457 pinned head <code>04a0900ff7</code>. nsys cold-L2
pure kernel time, same-GPU paired arms, B200. All rows exact (tie-robust per-row).
Raw CSVs: <code>grid_r4r3cg.csv</code>, <code>r5_bs/grid_r5g_final.csv</code>,
<code>grid_r5bs1guard.csv</code>.</p>
<ul><li>§1 — R4 champion <code>28dc11f6</code>, BS=1, {n_r4} cells</li>
<li>§2 — R5 champion <code>156ab438</code> (batched), BS 2–1024, {n_r5} cases</li>
<li>§3 — R5 champion at BS=1 (guard), {n_g} cells</li></ul>
<p>Green = ≥1.5×, orange = &lt;1.0×.</p>"""
ZH = f"""<h1>全量逐格数据 — KF 冠军算子 vs GVR PR head(不做平均)</h1>
<p class="meta">分母:PR#16457 锁定 head <code>04a0900ff7</code>。nsys cold-L2 纯 kernel
时间,同 GPU 配对双臂,B200。所有行均 tie-robust 精确。
原始 CSV:<code>grid_r4r3cg.csv</code>、<code>r5_bs/grid_r5g_final.csv</code>、
<code>grid_r5bs1guard.csv</code>。</p>
<ul><li>§1 — R4 冠军 <code>28dc11f6</code>,BS=1,{n_r4} 格</li>
<li>§2 — R5 冠军 <code>156ab438</code>(批式),BS 2–1024,{n_r5} 格</li>
<li>§3 — R5 冠军在 BS=1(守门),{n_g} 格</li></ul>
<p>绿色 = ≥1.5×,橙色 = &lt;1.0×。</p>"""

PAGE = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>KF champions — full per-case data</title>
<style>
body {{ font: 13px/1.5 -apple-system, "Segoe UI", Roboto, "Noto Sans CJK SC", sans-serif;
       max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #1a2233; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 26px; }}
code {{ background: #f0f3f8; padding: 1px 5px; border-radius: 4px; }}
.meta {{ color: #55627a; }}
table {{ border-collapse: collapse; margin: 10px 0; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid #d6dde8; padding: 3px 9px; text-align: right; }}
th {{ background: #f0f3f8; position: sticky; top: 36px; }}
td:first-child {{ text-align: left; }}
tr.grp td {{ background: #e7eef9; font-weight: 600; text-align: left; }}
td.hi {{ background: #e7f5ec; font-weight: 600; }}
td.reg {{ background: #fdeee3; }}
.langbar {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #d6dde8; z-index: 5; }}
.langbar label {{ cursor: pointer; padding: 4px 14px; border: 1px solid #b9c4d6; border-radius: 6px; margin-right: 8px; }}
#lang-en, #lang-zh, #pane-en, #pane-zh {{ display: none; }}
#lang-en:checked ~ #pane-en {{ display: block; }}
#lang-zh:checked ~ #pane-zh {{ display: block; }}
#lang-en:checked ~ .langbar label[for=lang-en],
#lang-zh:checked ~ .langbar label[for=lang-zh] {{ background: #1a2233; color: #fff; }}
</style></head><body>
<input type="radio" name="lang" id="lang-en" checked>
<input type="radio" name="lang" id="lang-zh">
<div class="langbar"><label for="lang-en">English</label><label for="lang-zh">中文</label></div>
<div id="pane-en">{EN}</div>
<div id="pane-zh">{ZH}</div>
<h2>§1 · R4 champion 28dc11f6 — BS=1, 865 cells / 格</h2>
{t_r4}
<h2>§2 · R5 champion 156ab438 — BS 2–1024, 750 cases / 格</h2>
{t_r5}
<h2>§3 · R5 champion at BS=1 — guard, 865 cells / 格</h2>
{t_g}
</body></html>"""
out = HERE / "R4_R5_FULL_DATA.html"
out.write_text(PAGE)
print(f"wrote {out} ({len(PAGE)/1e6:.2f} MB, rows {n_r4}+{n_r5}+{n_g})")
