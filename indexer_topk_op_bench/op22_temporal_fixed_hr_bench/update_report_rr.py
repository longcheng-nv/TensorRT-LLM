# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr — surgical REPORT.html section 1/2 refresh from the radix-relative
re-test (results_b200_op22rr, parse_op22.py output).

Replaces ONLY: the interactive D data blob, the JS op constants + speedup
base (now t(gvr_cutedsl)/t(op), >1 = op faster than the GVR-1CTA baseline),
the section 1/2 op/scenario controls and intro prose, and adds a re-test
notice + new CSV links. Sections 3-8 (incl. op23 section 7 and op24 section
8) are untouched except a one-line ub/lb dataset note in section 7.

Usage: python3 update_report_rr.py [<out_root>]  (default ../results_b200_op22rr)
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op22rr"
REPORT = HERE / "REPORT.html"

SCENS = ["real", "best", "worst"]
SUBS = [("seqlen", "seqlen_sweep", "seq"), ("bs", "bs_scaling", "bs"),
        ("bs_hugeN", "bs_hugeN", "bs")]
OPS = ["gvr_cutedsl", "op21_legacy", "op21_hls", "radix_cutedsl",
       "sglang_streaming"]
BASE = "gvr_cutedsl"
REV = "51be558e77"  # iter16 HEAD measured


def load():
    rows = []
    for scen in SCENS:
        for _sw, sub, w in SUBS:
            f = OUT_ROOT / scen / sub / "results.jsonl"
            if not f.exists():
                print(f"MISSING {f}")
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if "error" in r or "us_cold" not in r:
                    continue
                rows.append({"s": scen, "w": w, "K": r["K"], "d": r["dtype"],
                             "N": r["N"], "B": r["BS"], "o": r["op"],
                             "c": round(r["us_cold"], 3),
                             "h": round(r.get("us_warm", r["us_cold"]), 3)})
    return rows


def write_csvs(rows):
    by = {}
    for r in rows:
        by.setdefault((r["w"], r["s"], r["K"], r["d"], r["N"], r["B"]),
                      {})[r["o"]] = r
    for w, name in (("seq", "op22rr_seqlen_data.csv"),
                    ("bs", "op22rr_bs_data.csv")):
        head = ["scenario", "K", "dtype", "N", "BS"]
        for o in OPS:
            head += [f"{o}_cold_us", f"{o}_warm_us"]
        head += [f"speedup_vs_base_{o}" for o in OPS if o != BASE]
        out = [head]
        for key in sorted(k for k in by if k[0] == w):
            _w, s, K, d, N, B = key
            ops = by[key]
            row = [s, K, d, N, B]
            for o in OPS:
                r = ops.get(o)
                row += [r["c"] if r else "", r["h"] if r else ""]
            base = ops.get(BASE)
            for o in OPS:
                if o == BASE:
                    continue
                r = ops.get(o)
                row.append(round(base["c"] / r["c"], 4)
                           if (base and r and r["c"]) else "")
            out.append(row)
        with open(HERE / name, "w", newline="") as f:
            csv.writer(f).writerows(out)
        print(f"wrote {name} ({len(out) - 1} rows)")


def sub1(t, old, new, label):
    assert old in t, f"anchor missing: {label}"
    assert t.count(old) == 1, f"anchor not unique: {label}"
    return t.replace(old, new, 1)


def patch_report(rows):
    t = REPORT.read_text()

    # ---- 1. data blob ----
    i = t.find("const D=[")
    j = t.find("];\n", i)
    assert i > 0 and j > i
    t = t[:i] + "const D=" + json.dumps(rows, separators=(",", ":")) + ";" \
        + t[j + 2:]

    # ---- 2. JS constants (colors / labels / baseline) ----
    i = t.find("const COL=")
    j = t.find(";\n", i)
    assert i > 0 and j > i
    consts = (
        'const COL={"gvr_cutedsl":"#b3e05a","op21_legacy":"#c9a227",'
        '"op21_hls":"#ffd700","radix_cutedsl":"#4ea8de",'
        '"sglang_streaming":"#d62728"},'
        'DASH={"real":"solid","best":"dash","worst":"dot"},'
        'SHORT={"gvr_cutedsl":"GVR-1CTA","op21_legacy":"op21-legacy",'
        '"op21_hls":"HLS","radix_cutedsl":"Radix",'
        '"sglang_streaming":"SGLang"},MAIN="gvr_cutedsl"')
    t = t[:i] + consts + ";" + t[j + 1:]

    # ---- 3. axis/title labels ----
    t = sub1(t, "spd:'Speedup rival/op21 (>1 = op21 faster)'",
             "spd:'Speedup t(GVR-1CTA)/t(op) (>1 = faster than GVR-1CTA)'",
             "en spd")
    t = sub1(t, "spdb:'Speedup rival/op21 vs BS'",
             "spdb:'Speedup t(GVR-1CTA)/t(op) vs BS'", "en spdb")
    t = sub1(t, "spd:'加速比 rival/op21（>1 = op21 更快）'",
             "spd:'加速比 t(GVR-1CTA)/t(算子)（>1 = 快于 GVR-1CTA）'",
             "zh spd")
    t = sub1(t, "spdb:'加速比 rival/op21 对 BS'",
             "spdb:'加速比 t(GVR-1CTA)/t(算子) 对 BS'", "zh spdb")

    # ---- 4. JS speedup ratio: base_time / op_time ----
    t = sub1(t,
             "spd.push({x:xs2,y:xs2.map(N=>c[N][rg]/t21[N][rg]),"
             "name:s+'·'+SHORT[o]+'/op21',",
             "spd.push({x:xs2,y:xs2.map(N=>t21[N][rg]/c[N][rg]),"
             "name:s+'·'+SHORT[o],", "seq ratio")
    t = sub1(t,
             "spd.push({x:bs2,y:bs2.map(B=>c[B+'|'+N][rg]/t21[B+'|'+N][rg]),\n"
             "name:s+'·'+SHORT[o]+'/op21',",
             "spd.push({x:bs2,y:bs2.map(B=>t21[B+'|'+N][rg]/c[B+'|'+N][rg]),\n"
             "name:s+'·'+SHORT[o],", "bs ratio")

    # ---- 5. section 1/2 op + scenario controls ----
    ops_ck = lambda cls: (
        f'<label class="ck"><input type="checkbox" class="{cls}" '
        f'value="gvr_cutedsl" checked>GVR-1CTA (base)</label> '
        f'<label class="ck"><input type="checkbox" class="{cls}" '
        f'value="op21_legacy" checked>op21-legacy</label> '
        f'<label class="ck"><input type="checkbox" class="{cls}" '
        f'value="op21_hls" checked>HLS</label> '
        f'<label class="ck"><input type="checkbox" class="{cls}" '
        f'value="radix_cutedsl" checked>Radix</label> '
        f'<label class="ck"><input type="checkbox" class="{cls}" '
        f'value="sglang_streaming" checked>SGLang</label>')
    old1 = ('<label class="ck"><input type="checkbox" class="sck1" '
            'value="real" checked>REAL</label> <label class="ck">'
            '<input type="checkbox" class="sck1" value="ub" checked>UB'
            '</label> <label class="ck"><input type="checkbox" class="sck1" '
            'value="lb" checked>LB</label>')
    t = sub1(t, old1,
             '<label class="ck"><input type="checkbox" class="sck1" '
             'value="real" checked>REAL</label>', "sck1 ub/lb removal")
    for cls in ("ock1", "ock2"):
        i = t.find(f'<label class="ck"><input type="checkbox" class="{cls}" '
                   'value="gvr_ms_auto" checked>op21</label>')
        j = t.find('sglang_streaming" checked>SGLang</label>', i)
        assert i > 0 and j > i, cls
        j += len('sglang_streaming" checked>SGLang</label>')
        t = t[:i] + ops_ck(cls) + t[j:]

    # ---- 6. re-test notice + intro prose ----
    notice = (
        '<div class="card" style="border-color:#76b900">'
        '<div class="i18n-en"><p><b>Sections 1-2 re-measured 2026-07-08</b> '
        '(B200, mixed nodes — same SKU: real seq-len fp32/bf16 on '
        'umbriel-b200-047; fp32/bf16 real+best and fp16 real-K512/1024 on '
        'umbriel-b200-037; worst fp32/bf16, fp16 K2048 and fp16 '
        'best/worst-K512/1024 on umbriel-b200-044; per-batch host in the '
        'driver logs; kernel HEAD <code>' + REV + '</code> = op21 '
        'iter16, nsys cold-L2 protocol unchanged, 20 cold / 50 warm reps). '
        'Scenario definitions are the op24 RADIX-RELATIVE grid-average poles '
        '(REPORT §8.3b): <b>best</b> = per-K favorable cfg + fixed hr 0.55 '
        '(K512 aggregate / K1024 beta_moderate / K2048 beta_shallow), '
        '<b>worst</b> = beta_shallow + hr 0.05, <b>real</b> = aggregate + '
        'sampled hr (unchanged, identical bundles). Operator set: GVR-1CTA '
        '(<code>gvr_cutedsl</code>, the BASELINE — speedup = t(GVR-1CTA)/'
        't(op), &gt;1 ⇒ op faster), op21-legacy (pre-HLS: FALSI=0+DIST=0), '
        'HLS (op21 ship defaults @HEAD), Radix (<code>radix_cutedsl</code>), '
        'SGLang (fp32 &amp; K≤1024 only). All ops read the SAME bundle row '
        'per test point (one generation, shared inputs). Per-cell exactness '
        'validated at BS=1 for every arm. The previous section-1/2 dataset '
        '(op21-relative scenarios, rival/op21 ratios, incl. the ub/lb '
        'curves) is preserved in <a href="op22_seqlen_data.csv">'
        'op22_seqlen_data.csv</a> / <a href="op22_bs_data.csv">'
        'op22_bs_data.csv</a>; sections 3-7 still describe it.</p></div>'
        '<div class="i18n-zh"><p><b>第 1-2 节已于 2026-07-08 重测</b>'
        '（B200，跨节点混合——同 SKU：real seq-len fp32/bf16 采于 '
        'umbriel-b200-047；fp32/bf16 real+best 与 fp16 real-K512/1024 在 '
        'umbriel-b200-037；worst fp32/bf16、fp16 K2048 及 fp16 '
        'best/worst-K512/1024 在 umbriel-b200-044；每批主机见 driver 日志；'
        'kernel HEAD <code>' + REV + '</code> = '
        'op21 iter16，nsys cold-L2 协议不变，20 cold / 50 warm reps）。'
        '场景定义采用 op24 的 <b>radix-相对全网格平均</b>两极（本报告 '
        '§8.3b）：<b>best</b> = 各 K 的顺风 cfg + 固定 hr 0.55（K512 '
        'aggregate / K1024 beta_moderate / K2048 beta_shallow），'
        '<b>worst</b> = beta_shallow + hr 0.05，<b>real</b> = aggregate + '
        '采样 hr（定义不变，bundle 逐字节复用）。算子集合：GVR-1CTA'
        '（<code>gvr_cutedsl</code>，<b>基线</b> —— 加速比 = t(GVR-1CTA)/'
        't(算子)，&gt;1 ⇒ 该算子更快）、op21-legacy（HLS 前：FALSI=0+'
        'DIST=0）、HLS（op21 当前 ship 默认 @HEAD）、Radix'
        '（<code>radix_cutedsl</code>）、SGLang（仅 fp32 且 K≤1024）。'
        '每个测试点所有算子读取<b>同一份</b> bundle 行（一次生成、共享输入）'
        '；BS=1 处每臂逐格精确性校验。旧的第 1-2 节数据集（op21-相对场景、'
        'rival/op21 比值、含 ub/lb 曲线）保留于 '
        '<a href="op22_seqlen_data.csv">op22_seqlen_data.csv</a> / '
        '<a href="op22_bs_data.csv">op22_bs_data.csv</a>；第 3-7 节仍描述'
        '旧数据集。</p></div></div>\n')
    anchor1 = ('<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span>'
               '<span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>')
    t = sub1(t, anchor1, notice + anchor1, "notice before s1")

    # ---- 7. section 3: add new CSV links ----
    old3 = ("<a href='op22_bs_data.csv'>op22_bs_data.csv</a> (both regimes + "
            "op21 cold speedups; first column = scenario)")
    if old3 in t:
        t = t.replace(
            old3,
            old3 + ". <b>Radix-relative re-test (2026-07-08, sections 1-2)"
            "</b>: <a href='op22rr_seqlen_data.csv'>op22rr_seqlen_data.csv"
            "</a> · <a href='op22rr_bs_data.csv'>op22rr_bs_data.csv</a>", 1)

    # ---- 8. section 7 ub/lb reference note ----
    old7 = ("Select the <b>ub</b>/<b>lb</b> scenarios in the §1 interactive "
            "panel to see the per-N curves.")
    if old7 in t:
        t = t.replace(old7,
                      "The ub/lb per-N curves belong to the pre-2026-07-08 "
                      "§1 dataset (op21-relative); they were removed from "
                      "the re-tested §1 panel and are preserved in "
                      "<a href='op22_seqlen_data.csv'>op22_seqlen_data.csv"
                      "</a>.", 1)
    old7z = "在 §1 交互面板勾选 <b>ub</b>/<b>lb</b> 场景即可查看各 N 曲线。"
    if old7z in t:
        t = t.replace(old7z,
                      "ub/lb 各 N 曲线属于 2026-07-08 之前的 §1 数据集"
                      "（op21-相对），重测后已从 §1 面板移除，数据保留在 "
                      "<a href='op22_seqlen_data.csv'>op22_seqlen_data.csv"
                      "</a>。", 1)

    REPORT.write_text(t)
    print(f"REPORT.html patched: D={len(rows)} rows")


def main():
    rows = load()
    assert rows, "no parsed rows — run parse_op22.py first"
    n_scen = {s: sum(r["s"] == s for r in rows) for s in SCENS}
    print("rows per scenario:", n_scen)
    write_csvs(rows)
    patch_report(rows)


if __name__ == "__main__":
    main()
