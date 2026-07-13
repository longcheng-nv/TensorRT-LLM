#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr+op28 — add the two EXTERNAL latest top-K arms to REPORT.html §1-2:

  sglang_v2       : sglang@main (2026-07-13) DSv4 top-K v2 (vendored
                    ops/sglang_v2, kernels verbatim; register/streaming/
                    8-CTA-persistent-cluster dispatch + PDL epilogue stitch)
  flashinfer_topk : flashinfer.top_k public API (0.6.11 == main on the B200
                    clusters path)

APPEND-ON-TOP LAST-WRITER: unlike update_report_op27.py this does NOT
re-derive prior arms — it patches the CURRENT REPORT.html state (op27-final,
13 arms incl op26_r0auto) idempotently: D-blob rows for the two new ops are
dropped-then-appended, COL/SHORT keys upserted, note/method rows
skip-if-present. If an OLDER updater is re-run afterwards it will erase
these arms — re-run this script after it.

Data: ../results_b200_op28 (node umbriel-b200-027, fp32-only, GPUs 2-7,
2026-07-13), anchor-transferred per cell onto the op22rr µs scale via the
co-located gvr_cutedsl arm:  us_adj = us * us_base(orig)/us_base(local).
Exactness: standalone gate op28_ext_topk/gate_op28.log (459/459).

Usage: python3 update_report_op28.py [<op28_root>]
"""
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP28_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op28"
ORIG_ROOT = HERE.parents[0] / "results_b200_op22rr"
GATE_LOG = HERE.parents[0] / "op28_ext_topk" / "gate_op28.log"
REPORT = HERE / "REPORT.html"
BAK = HERE / "REPORT.html.bak_pre_op28_20260713"

SGLV2 = "sglang_v2"
FI = "flashinfer_topk"
BASE = "gvr_cutedsl"
NEW = {
    SGLV2: {"col": "#ff5d8f", "name": "SGLang v2 top-K (main 2026-07)"},
    FI: {"col": "#8ac926", "name": "FlashInfer top_k (0.6.11)"},
}
NOTE_ID = "op28-note-2026-07-13"

# reuse load()/transfer_ref()/sub1()/subn() from the op27 updater
spec = importlib.util.spec_from_file_location(
    "update_report_op27", HERE / "update_report_op27.py")
_u27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_u27)
load, transfer_ref = _u27.load, _u27.transfer_ref
sub1, subn = _u27.sub1, _u27.subn


def _op28_note():
    en = (
        f'<div class="card" id="{NOTE_ID}" style="border-color:#ff5d8f">'
        '<h3><span class="i18n-en">Method note — op28 EXTERNAL latest '
        'top-K arms (2026-07-13): SGLang v2 &amp; FlashInfer</span>'
        '<span class="i18n-zh">方法注记 —— op28 外部最新 top-K 臂'
        '（2026-07-13）：SGLang v2 与 FlashInfer</span></h3>'
        '<div class="i18n-en"><p><b>SGLang v2 top-K</b> is the CURRENT '
        'sglang@main DeepSeek-V4 top-K (<code>topk_impl.cuh</code> + '
        '<code>topk_v2.cuh</code>, credits vLLM persistent_topk + '
        'FlashInfer topk) — a complete rewrite that DELETED the '
        '<code>top512::StreamingTopK</code> benchmarked as the '
        '"SGLang StreamingTopK" arm. Vendored with kernels VERBATIM '
        '(<code>ops/sglang_v2</code>; only the tvm-ffi host layer replaced, '
        'dispatch constants identical). Runtime-K single-pass design: '
        'register-resident path (seq&le;16K: one global read, threshold '
        'from a 12-bit fp16-coarse smem histogram, winners scattered from '
        'REGISTERS), TMA-style streaming path (&le;64K), and an 8-CTA '
        'cluster path for long rows (chunked scan + one-shot DSMEM '
        'histogram all-reduce). <b>FlashInfer top_k</b> is the public '
        '<code>flashinfer.top_k</code> (installed 0.6.11; python + B200 '
        'clusters-path kernel byte-identical to main), non-deterministic '
        'default = <code>fast_topk_clusters_exact</code>.</p>'
        '<p><b>Protocol:</b> byte-identical op22rr bundles, same nsys '
        'cold/warm-L2 protocol (20/50 reps), fp32 only (both kernels are '
        'fp32-scores; K=2048 has no sglang_streaming reference). Measured '
        '2026-07-13 on umbriel-b200-027 (GPUs 2-7) with a co-located '
        'GVR (cuteDSL) anchor; plotted µs are anchor-transferred onto the '
        'original scale (median drift 0.998). Exactness gate 459/459 '
        '(sorted value-multiset, incl. N=1M and the 2-kernel path).</p>'
        '<p><b>Two-kernel caveat:</b> for seq&gt;cluster_threshold '
        '(&ge;131K here) and 30&lt;BS&le;512, sglang v2 is TWO stitched '
        'kernels (persistent 8-CTA cluster pool + a main kernel doing the '
        'short rows and the PDL-synchronized page-transform epilogue). '
        'The canonical metric (per-range kernel-time SUM) COUNTS BOTH; '
        'under PDL overlap the sum can overstate wall-clock (worst '
        'observed: span = 0.56&times;sum at N=262144 BS=64) and elsewhere '
        'understate the inter-kernel gap (span up to 1.21&times;sum at '
        'N=131072) — honest wall-clock spans are in '
        '<code>op28_ext_topk/op28_*_data.csv</code> (<code>*_span_us</code>). '
        'The <code>topk_plan</code> kernel (~7µs, once per step, reused '
        'across the ~61 indexer layers &rArr; ~0.11µs/layer) runs untimed, '
        'matching production amortization.</p>'
        '<p><b>Headline (cold-L2 geomean, seqlen+bs):</b> sglang_v2 beats '
        'EVERY in-tree arm — vs GVR op#21 ms_auto (HLS) 1.38-1.77&times;, '
        'vs GVR op#26 R0 auto 1.34-1.64&times;, vs Radix (cuteDSL) '
        '1.68-1.81&times;, vs the old SGLang StreamingTopK '
        '~1.97&times;. flashinfer_topk is roughly at parity with the best '
        'in-tree arms (vs HLS 0.95-1.21&times;) and ~1.37&times; slower '
        'than sglang_v2. Full tables: '
        '<code>op28_ext_topk/RESULTS_SUMMARY.md</code>.</p></div>'
        '<div class="i18n-zh"><p><b>SGLang v2 top-K</b> 为当前 sglang@main '
        '的 DSv4 top-K（<code>topk_impl.cuh</code>+<code>topk_v2.cuh</code>，'
        '致谢 vLLM persistent_topk 与 FlashInfer topk）—— 完全重写并已删除'
        '本报告 "SGLang StreamingTopK" 臂所测的 <code>top512</code> 旧版。'
        'kernel 逐字 vendor（<code>ops/sglang_v2</code>，仅替换 tvm-ffi '
        'host 层，分发常数一致）。运行时 K 单遍设计：寄存器驻留路径'
        '（seq&le;16K：单次全局读，12-bit fp16 粗直方图求阈值，胜者直接从'
        '寄存器 scatter）、流式路径（&le;64K）、长行 8-CTA cluster 路径'
        '（分块扫描 + DSMEM 一次性直方图 all-reduce）。'
        '<b>FlashInfer top_k</b> 为公开 <code>flashinfer.top_k</code>'
        '（本机 0.6.11 与 main 在 B200 clusters 路径逐字节一致），默认非'
        '确定模式 = <code>fast_topk_clusters_exact</code>。</p>'
        '<p><b>协议：</b>op22rr bundle 逐字节复用，同 nsys 冷/热 L2 协议'
        '（20/50 reps），仅 fp32。2026-07-13 于 umbriel-b200-027（GPU2-7）'
        '与同址 GVR (cuteDSL) 锚共测，绘图 µs 已锚换算回原尺度（漂移中位 '
        '0.998）。exactness 459/459（含 N=1M 与双 kernel 路径）。</p>'
        '<p><b>双 kernel 口径：</b>seq&gt;cluster_threshold（本网格 '
        '&ge;131K）且 30&lt;BS&le;512 时 sglang v2 为两个 PDL 拼接 kernel'
        '（持久 cluster 池 + 主 kernel 负责短行与 epilogue 变换）；规范'
        '指标（区间内 kernel 时间求和）两者都计入 —— PDL 重叠时求和会高估'
        '墙钟（最差 N=262144 BS=64 处 span=0.56&times;sum），其余处则漏掉'
        'kernel 间隙（N=131072 处 span 至 1.21&times;sum）；诚实墙钟见 '
        '<code>op28_ext_topk/op28_*_data.csv</code> 的 '
        '<code>*_span_us</code> 列。<code>topk_plan</code>（约 7µs/步，'
        '61 层复用 &rArr; ~0.11µs/层）不计时，与生产摊销一致。</p>'
        '<p><b>结论（冷 L2 几何均值，seqlen+bs）：</b>sglang_v2 胜过全部'
        '树内臂 —— 对 GVR op#21 ms_auto (HLS) 1.38-1.77&times;、对 GVR '
        'op#26 R0 auto 1.34-1.64&times;、对 Radix (cuteDSL) '
        '1.68-1.81&times;、对旧版 SGLang StreamingTopK 约 1.97&times;。'
        'flashinfer_topk 与树内最强臂大致持平（对 HLS 0.95-1.21&times;），'
        '比 sglang_v2 慢约 1.37&times;。完整表格：'
        '<code>op28_ext_topk/RESULTS_SUMMARY.md</code>。</p></div></div>')
    return en


def method_rows():
    r1 = (f"<tr><td style='color:{NEW[SGLV2]['col']}'>"
          "SGLang v2 top-K (main 2026-07) — register/streaming/8-CTA-"
          "cluster dispatch + PDL epilogue (vendored verbatim, "
          "ops/sglang_v2)</td><td>CUDA C++ (external)</td>"
          "<td>exact (hint-blind)</td>"
          "<td>fp32 &times; 512/1024/2048 — op28 2026-07-13, "
          "anchor-transferred (node 027)</td></tr>")
    r2 = (f"<tr><td style='color:{NEW[FI]['col']}'>"
          "FlashInfer top_k (0.6.11==main) — fast_topk_clusters_exact "
          "(public API, values+int64 indices)</td>"
          "<td>CUDA C++ (external)</td><td>exact (hint-blind)</td>"
          "<td>fp32 &times; 512/1024/2048 — op28 2026-07-13, "
          "anchor-transferred (node 027)</td></tr>")
    return r1 + r2


def main():
    gate = GATE_LOG.read_text()
    assert "fails=0 errs=0" in gate, "op28 exactness gate not green"

    orig_rows = load(ORIG_ROOT, {BASE})
    new_raw = load(OP28_ROOT, set(NEW))
    base_local = load(OP28_ROOT, {BASE})
    print(f"orig_base={len(orig_rows)} new={len(new_raw)} "
          f"base_local={len(base_local)}")
    assert orig_rows and new_raw and base_local

    adj = transfer_ref(orig_rows, new_raw, base_local, "op28-027")
    assert adj, "no anchor-transferred rows"

    if not BAK.exists():
        shutil.copy2(REPORT, BAK)
        print(f"backup -> {BAK.name}")

    t = REPORT.read_text()

    # ---- 1. D blob: drop existing rows of the new ops, append adj ----
    i = t.find("const D=[")
    j = t.find("];", i)
    assert i > 0 and j > i
    d_rows = json.loads(t[i + len("const D="):j + 1])
    d_rows = [r for r in d_rows if r["o"] not in NEW]
    d_rows += [{k: v for k, v in r.items() if k in
                ("s", "w", "K", "d", "N", "B", "o", "c", "h")} for r in adj]
    t = t[:i] + "const D=" + json.dumps(d_rows, separators=(",", ":")) \
        + ";" + t[j + 2:]

    # ---- 2. COL / SHORT: upsert the two new keys ----
    m = re.search(r'const COL=(\{.*?\}),DASH=(\{.*?\}),SHORT=(\{.*?\}),'
                  r'MAIN="gvr_cutedsl";', t, re.S)
    assert m, "COL/SHORT consts not found"
    col = json.loads(m.group(1))
    short = json.loads(m.group(3))
    for op, meta in NEW.items():
        col[op] = meta["col"]
        short[op] = meta["name"]
    consts = ("const COL=" + json.dumps(col, separators=(",", ":"))
              + ",DASH=" + m.group(2)
              + ",SHORT=" + json.dumps(short, separators=(",", ":"))
              + ',MAIN="gvr_cutedsl";')
    t = t[:m.start()] + consts + t[m.end():]

    # ---- 3. checkboxes after op26_r0auto (both panels) ----
    for cls in ("ock1", "ock2"):
        mm = re.search(f'<label class="ck"><input type="checkbox" '
                       f'class="{cls}" value="op26_r0auto" checked>'
                       f'[^<]*</label>', t)
        assert mm, f"{cls} op26_r0auto label not found"
        anchor = mm.group(0)
        assert t.count(anchor) == 1
        add = ""
        for op, meta in NEW.items():
            if f'class="{cls}" value="{op}"' not in t:
                add += (f' <label class="ck"><input type="checkbox" '
                        f'class="{cls}" value="{op}" checked>'
                        f'{meta["name"]}</label>')
        if add:
            t = sub1(t, anchor, anchor + add, f"{cls} op28 anchor")

    # ---- 4. bilingual method note before section 1 ----
    anchor1 = ('<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span>'
               '<span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>')
    if NOTE_ID not in t:
        t = sub1(t, anchor1, _op28_note() + anchor1, "op28 note before s1")

    # ---- 5. methodology-table rows (en+zh; before the radix row) ----
    rad_row = ("<tr><td style='color:#4ea8de'>Radix (cuteDSL)</td>"
               "<td>CuTe DSL</td><td>exact (hint-blind)</td>"
               "<td>full — strongest hint-blind rival</td></tr>")
    if f"<tr><td style='color:{NEW[SGLV2]['col']}'>" not in t:
        assert t.count(rad_row) == 2, "methodology radix rows not found"
        t = subn(t, rad_row, method_rows() + rad_row, 2,
                 "op28 methodology rows")

    REPORT.write_text(t)
    counts = {o: sum(1 for r in d_rows if r["o"] == o) for o in NEW}
    print(f"REPORT.html patched: D={len(d_rows)} rows  op28={counts}")


if __name__ == "__main__":
    main()
