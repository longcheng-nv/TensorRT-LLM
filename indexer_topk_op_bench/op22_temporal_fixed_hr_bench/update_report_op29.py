#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr+op29 — add the gvr29_hbe arm (op29 iter12 ship state) to
REPORT.html §1-2:

  gvr29_hbe : GVR op#29 HBE-noB — fork of sglang_v2 (register/streaming/
              cluster dispatch verbatim) + a 4th tier at N>=65536 streaming
              rows: hint-free chunked row sample -> ONE cand-targeted
              column (rank 2*rS_K) -> 1-cmp/elem fused single DRAM pass ->
              count-validity (cnt_a>=K) -> candidate-only mini-hist b* ->
              boundary-exact resolve; miss falls back to the stock 2-pass
              (fail-soft, never observed on the grid).

APPEND-ON-TOP LAST-WRITER (same contract as update_report_op28.py): patches
the CURRENT REPORT.html state (op28-final, 15 arms) idempotently — D-blob
rows for gvr29_hbe dropped-then-appended, COL/SHORT upserted, note/method
rows skip-if-present. If an OLDER updater is re-run afterwards it erases
this arm — re-run this script after it.

Data: ../results_b200_op29 (fp32-only, 3 arms same-batch: gvr_cutedsl
anchor + sglang_v2 + gvr29_hbe), anchor-transferred per cell onto the
op22rr µs scale via the co-located gvr_cutedsl arm.
Exactness: op29_gvr_hbe/gate_op29.log (324/324, tie-aware value-multiset,
both colB arms, forced HBE engagement everywhere incl N=32768).

Usage: python3 update_report_op29.py [<op29_root>]
"""
import importlib.util
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP29_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op29"
ORIG_ROOT = HERE.parents[0] / "results_b200_op22rr"
GATE_LOG = HERE.parents[0] / "op29_gvr_hbe" / "gate_op29.log"
REPORT = HERE / "REPORT.html"
BAK = HERE / "REPORT.html.bak_pre_op29_20260713"

G29 = "gvr29_hbe"
SGLV2 = "sglang_v2"
BASE = "gvr_cutedsl"
COL29 = "#06d6a0"
NAME29 = "GVR op#29 HBE-noB (1-pass sample-column)"
NOTE_ID = "op29-note-2026-07-13"

# reuse load()/transfer_ref()/sub1()/subn() from the op27 updater
spec = importlib.util.spec_from_file_location(
    "update_report_op27", HERE / "update_report_op27.py")
_u27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_u27)
load, transfer_ref = _u27.load, _u27.transfer_ref
sub1, subn = _u27.sub1, _u27.subn


def geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def headline(local_g29, local_sgl):
    """Same-batch sglang_v2/gvr29 ratios: per-(scen,K) grid geomean (ship-rule
    slice) + engaged-tier (N>=65536, BS>=1024) per-cell range."""
    sgl = {(r["s"], r["w"], r["K"], r["d"], r["N"], r["B"]): r["c"]
           for r in local_sgl}
    slice_r = defaultdict(list)
    engaged = []
    for r in local_g29:
        k = (r["s"], r["w"], r["K"], r["d"], r["N"], r["B"])
        if k not in sgl:
            continue
        ratio = sgl[k] / r["c"]
        slice_r[(r["s"], r["K"])].append(ratio)
        if r["N"] >= 65536 and r["B"] >= 1024:
            engaged.append(ratio)
    gm = {sk: geo(v) for sk, v in sorted(slice_r.items())}
    return gm, (min(engaged), max(engaged)) if engaged else (None, None)


def _note(gm, eng_rng):
    gm_txt = ", ".join(f"{s}/K{k} {v:.3f}" for (s, k), v in gm.items())
    gm_min = min(gm.values())
    en = (
        f'<div class="card" id="{NOTE_ID}" style="border-color:{COL29}">'
        '<h3><span class="i18n-en">Method note — op29 GVR HBE-noB arm '
        '(2026-07-13): 1-pass sample-column top-K, first in-tree arm to '
        'beat SGLang v2</span>'
        '<span class="i18n-zh">方法注记 —— op29 GVR HBE-noB 臂'
        '（2026-07-13）：单遍采样定列 top-K，首个反超 SGLang v2 的树内臂'
        '</span></h3>'
        '<div class="i18n-en"><p><b>Design:</b> fork of the vendored '
        'sglang_v2 (register/streaming/8-CTA-cluster dispatch verbatim) '
        'plus a 4th dispatch tier for streaming rows with N&ge;65536: a '
        '64&times;64-chunk coalesced row sample places ONE candidate-'
        'targeted collect column (sample rank 2&middot;rS_K); a single '
        'fused DRAM pass (1 cmp/element) collects {val,idx} above it '
        '(smem buf + global spill); validity = count&ge;K proves top-K '
        'containment; the exact threshold bin is then recovered from a '
        'candidate-only smem mini-histogram and resolved with the stock '
        'boundary-exact tie machinery. Miss &rArr; fail-soft to the stock '
        '2-pass (never fired on this grid). Exactness is UNCONDITIONAL '
        '(hint/sample quality moves only speed); gate 324/324 '
        '(<code>op29_gvr_hbe/gate_op29.log</code>, tie-aware '
        'value-multiset, HBE force-engaged everywhere incl N=32768). '
        'The tier-B insurance column of earlier iterations was REMOVED '
        'after NCU attribution (~16 inst per band element in an '
        'issue-bound pass, fired 0/18) — see '
        '<code>op29_gvr_hbe/ITERATIONS.md</code> iter11/12.</p>'
        '<p><b>Protocol:</b> byte-identical op22rr bundles, same nsys '
        'cold/warm-L2 protocol (20/50 reps), fp32; 3 arms SAME-BATCH '
        '(gvr_cutedsl anchor + sglang_v2 + gvr29_hbe); plotted µs '
        'anchor-transferred onto the original op22rr scale. Plan kernel '
        'untimed (production amortization, same convention as the '
        'sglang_v2 arm); outside the HBE tier the arm is the sglang_v2 '
        'fork itself, so those cells read as parity by construction.</p>'
        f'<p><b>Headline (same-batch, cold-L2):</b> engaged tier '
        f'(N&ge;65536, BS&ge;1024) per-cell sglang_v2/gvr29 = '
        f'{eng_rng[0]:.2f}-{eng_rng[1]:.2f}&times;; full-grid '
        f'(scenario&times;K) geomeans all &ge;{gm_min:.3f} '
        f'({gm_txt}).</p></div>'
        '<div class="i18n-zh"><p><b>设计：</b>基于逐字 vendor 的 sglang_v2 '
        '分发（register/streaming/8-CTA cluster 不变）新增第 4 层：'
        'N&ge;65536 的流式行走 HBE-noB —— 64&times;64 分块合并采样放置'
        '单条候选目标列（采样秩 2&middot;rS_K），单遍融合 DRAM 扫描'
        '（每元素 1 次比较）收集候选（smem + 全局溢出），计数判据 '
        'count&ge;K 证明 top-K 包含，再从候选 mini 直方图恢复精确阈值 '
        'bin 并用原 tie 机制边界精确求解；miss 则 fail-soft 回退原 2 遍'
        '（本网格从未触发）。精确性无条件（gate 324/324，见 '
        '<code>op29_gvr_hbe/gate_op29.log</code>）。早期迭代的 tier-B '
        '保险列经 NCU 归因后已删除（issue-bound 扫描中每候选带元素 '
        '~16 条指令、18 次采样 0 触发）—— 详见 '
        '<code>op29_gvr_hbe/ITERATIONS.md</code> iter11/12。</p>'
        '<p><b>协议：</b>op22rr bundle 逐字节复用，同 nsys 冷/热 L2 协议'
        '（20/50 reps），仅 fp32；三臂同批（gvr_cutedsl 锚 + sglang_v2 + '
        'gvr29_hbe），绘图 µs 已锚换算回 op22rr 原尺度。plan kernel 不计时'
        '（与 sglang_v2 臂同口径）；HBE 层之外本臂即 sglang_v2 fork，'
        '相应 cell 按构造为持平。</p>'
        f'<p><b>结论（同批冷 L2）：</b>引擎化层（N&ge;65536、BS&ge;1024）'
        f'单 cell sglang_v2/gvr29 = {eng_rng[0]:.2f}-{eng_rng[1]:.2f}'
        f'&times;；全网格（场景&times;K）几何均值全部 &ge;{gm_min:.3f}'
        f'（{gm_txt}）。</p></div></div>')
    return en


def method_row():
    return (f"<tr><td style='color:{COL29}'>"
            "GVR op#29 HBE-noB — sglang_v2-fork dispatch + 1-pass "
            "sample-column tier @ N&ge;65536 streaming (fail-soft, "
            "op29_gvr_hbe/src/gvr29)</td><td>CUDA C++</td>"
            "<td>exact (sample-guided, hint-free)</td>"
            "<td>fp32 &times; 512/1024/2048 — op29 2026-07-13, "
            "anchor-transferred (same-batch vs sglang_v2)</td></tr>")


def main():
    gate = GATE_LOG.read_text()
    assert "fails=0 errs=0" in gate, "op29 exactness gate not green"

    orig_rows = load(ORIG_ROOT, {BASE})
    new_raw = load(OP29_ROOT, {G29})
    base_local = load(OP29_ROOT, {BASE})
    sgl_local = load(OP29_ROOT, {SGLV2})
    print(f"orig_base={len(orig_rows)} new={len(new_raw)} "
          f"base_local={len(base_local)} sgl_local={len(sgl_local)}")
    assert orig_rows and new_raw and base_local and sgl_local

    gm, eng_rng = headline(new_raw, sgl_local)
    print("ship-rule slice geomeans (sglang_v2/gvr29, same-batch):")
    for sk, v in gm.items():
        print(f"  {sk}: {v:.3f}")
    print(f"engaged-tier per-cell range: {eng_rng[0]:.3f}-{eng_rng[1]:.3f}")

    adj = transfer_ref(orig_rows, new_raw, base_local, "op29")
    assert adj, "no anchor-transferred rows"

    if not BAK.exists():
        shutil.copy2(REPORT, BAK)
        print(f"backup -> {BAK.name}")

    t = REPORT.read_text()

    # ---- 1. D blob: drop existing gvr29 rows, append adj ----
    i = t.find("const D=[")
    j = t.find("];", i)
    assert i > 0 and j > i
    d_rows = json.loads(t[i + len("const D="):j + 1])
    d_rows = [r for r in d_rows if r["o"] != G29]
    d_rows += [{k: v for k, v in r.items() if k in
                ("s", "w", "K", "d", "N", "B", "o", "c", "h")} for r in adj]
    t = t[:i] + "const D=" + json.dumps(d_rows, separators=(",", ":")) \
        + ";" + t[j + 2:]

    # ---- 2. COL / SHORT upsert ----
    m = re.search(r'const COL=(\{.*?\}),DASH=(\{.*?\}),SHORT=(\{.*?\}),'
                  r'MAIN="gvr_cutedsl";', t, re.S)
    assert m, "COL/SHORT consts not found"
    col = json.loads(m.group(1))
    short = json.loads(m.group(3))
    col[G29] = COL29
    short[G29] = NAME29
    consts = ("const COL=" + json.dumps(col, separators=(",", ":"))
              + ",DASH=" + m.group(2)
              + ",SHORT=" + json.dumps(short, separators=(",", ":"))
              + ',MAIN="gvr_cutedsl";')
    t = t[:m.start()] + consts + t[m.end():]

    # ---- 3. checkboxes after sglang_v2 (both panels) ----
    for cls in ("ock1", "ock2"):
        mm = re.search(f'<label class="ck"><input type="checkbox" '
                       f'class="{cls}" value="sglang_v2" checked>'
                       f'[^<]*</label>', t)
        assert mm, f"{cls} sglang_v2 label not found"
        anchor = mm.group(0)
        assert t.count(anchor) == 1
        if f'class="{cls}" value="{G29}"' not in t:
            t = sub1(t, anchor,
                     anchor + (f' <label class="ck"><input type="checkbox" '
                               f'class="{cls}" value="{G29}" checked>'
                               f'{NAME29}</label>'),
                     f"{cls} op29 anchor")

    # ---- 4. bilingual method note before section 1 ----
    anchor1 = ('<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span>'
               '<span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>')
    if NOTE_ID not in t:
        t = sub1(t, anchor1, _note(gm, eng_rng) + anchor1,
                 "op29 note before s1")

    # ---- 5. methodology-table rows (en+zh; before the radix row) ----
    rad_row = ("<tr><td style='color:#4ea8de'>Radix (cuteDSL)</td>"
               "<td>CuTe DSL</td><td>exact (hint-blind)</td>"
               "<td>full — strongest hint-blind rival</td></tr>")
    if f"<tr><td style='color:{COL29}'>" not in t:
        assert t.count(rad_row) == 2, "methodology radix rows not found"
        t = subn(t, rad_row, method_row() + rad_row, 2,
                 "op29 methodology rows")

    REPORT.write_text(t)
    n29 = sum(1 for r in d_rows if r["o"] == G29)
    print(f"REPORT.html patched: D={len(d_rows)} rows  gvr29_hbe={n29}")


if __name__ == "__main__":
    main()
