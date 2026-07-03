#!/usr/bin/env python3
"""Generate the interactive HTML indexer-top-K evaluation report from sweep JSONL.

Each cell carries cold-L2 (flushed, canonical) and warm-L2 (logits-hot, fused-path)
kernel µs. Report has a cold/warm toggle, GVR-vs-others speedup curves, full table,
CSV exports, and auto-analysis. Timing = CUDA-graph replay + cudaEvent, validated
against nsys (~11%).

Hardware dimension: the SAME synthetic inputs (seed=42, unified preIdx hit-rate
0.6) are swept on every machine, so all cross-machine differences are pure
hardware. The report carries a B300/B200 ``view`` selector (per-machine or a
``compare`` overlay) and a per-op hardware-ratio table. B300 = sm_103, B200 =
sm_100; both runs share the prebuilt CUDA-op .so and identical cuteDSL kernels.

Bilingual: the report is fully English/中文 switchable via a header toggle. Every
prose block is emitted twice (``i18n-en`` / ``i18n-zh``) and CSS swaps them; the
Plotly charts and the JS-built data table re-render in the active language.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

OPS = ["gvr_cuda", "gvr_cutedsl", "gvr_cutedsl_rs", "gvr_multicta_cutedsl", "gvr_op8", "gvr_port", "gvr_mt", "gvr_sandwich", "radix_single_cuda", "radix_multi_cuda", "radix_cutedsl", "radix_cutedsl_single", "radix_cutedsl_multi", "sglang_streaming"]
OP_LABEL = {"gvr_cuda": "GVR (CUDA)", "gvr_cutedsl": "GVR (cuteDSL)",
            "gvr_cutedsl_rs": "GVR (cuteDSL, rank-scatter P4)",
            "gvr_multicta_cutedsl": "GVR multi-CTA (cuteDSL, PR#15198)",
            "gvr_op8": "GVR op#8 (multi-CTA + rank-scatter P4)",
            "gvr_port": "GVR op#17 threshold-portfolio (cluster, auto-G)",
            "gvr_mt": "GVR op#18 multi-threshold (single-CTA)",
            "gvr_sandwich": "GVR op#19 sandwich two-threshold (dispatch)",
            "radix_single_cuda": "Radix single-CTA (CUDA)", "radix_multi_cuda": "Radix multi-CTA (CUDA)",
            "radix_cutedsl": "Radix (cuteDSL)",
            "radix_cutedsl_single": "Radix (cuteDSL)-single", "radix_cutedsl_multi": "Radix (cuteDSL)-multi",
            "sglang_streaming": "SGLang StreamingTopK (fp32, K≤1024)"}
COL = {"gvr_cuda": "#76b900", "gvr_cutedsl": "#b3e05a", "gvr_cutedsl_rs": "#5fd35f",
       "gvr_multicta_cutedsl": "#2ec4b6", "gvr_op8": "#9b5de5",
       "gvr_port": "#ff6ec7", "gvr_mt": "#00e5ff", "gvr_sandwich": "#ffffff",
       "radix_single_cuda": "#ff8c42", "radix_multi_cuda": "#e84855", "radix_cutedsl": "#4ea8de",
       "radix_cutedsl_single": "#f6c453", "radix_cutedsl_multi": "#1f6feb",
       "sglang_streaming": "#d62728"}
# Overlay ops (#17/#18/#19) are anchored onto the nsys scale via per-cell ratio
# transfer against their in-run gvr_cutedsl baseline — see merge_anchored_ops().
ANCHOR = "gvr_cutedsl"
BASELINE = "gvr_cuda"
# B300 first => hardware ratio is B200/B300 (>1 ⇒ B300 faster). B300 solid, B200 dashed.
DASH = {"B300": "solid", "B200": "dash"}


def bi(en, zh, tag="div"):
    """Wrap a block of HTML in both languages; CSS shows the active one."""
    return (f'<{tag} class="i18n-en">{en}</{tag}>'
            f'<{tag} class="i18n-zh">{zh}</{tag}>')


def load(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def index_cells(rows, sweep):
    """cells[(K,dtype,N,BS)][op] = {'c': cold_us, 'w': warm_us}"""
    cells = {}
    for r in rows:
        if r.get("sweep") != sweep or "us" not in r:
            continue
        key = (r["K"], r["dtype"], r["N"], r["BS"])
        cells.setdefault(key, {})[r["op"]] = {"c": r.get("us_cold", r["us"]), "w": r.get("us_warm", r["us"])}
    return cells


def load_machine(root):
    """Read both the 5-op sweep and the 6th-op (cluster) sweep for one machine.

    The cluster op (gvr_multicta_cutedsl) has a distinct op name, so its records
    merge into the SAME cells dict keyed by (K,dtype,N,BS); pilot files (if any)
    are merged too for completeness.
    """
    # NOTE: pilot_*.jsonl deliberately NOT read — pilots may carry stray rows from
    # a different machine (the op#7 bring-up pilot ran on B200 but wrote into the
    # default results/ dir = the B300 view). Only the canonical sweep outputs.
    root = Path(root)
    seq = index_cells(load([
        root / "seqlen_sweep" / "results.jsonl",
        root / "cluster_sweep" / "seqlen.jsonl",
        root / "op8_sweep" / "seqlen.jsonl",
    ]), "seqlen")
    bs = index_cells(load([
        root / "bs_scaling" / "results.jsonl",
        root / "cluster_sweep" / "bs.jsonl",
        root / "op8_sweep" / "bs.jsonl",
    ]), "bs")
    return seq, bs


def merge_anchored_ops(mdata, bucket):
    """Overlay ops #17/#18/#19 onto the nsys machine data (ratio transfer).

    Each op bucket ran a back-to-back A/B against the SAME single-CTA
    ``gvr_cutedsl`` baseline, on the SAME synthetic report bundles (seed=42,
    hit-rate 0.6), with CUDA-graph + cudaEvent cold-L2 timing. Absolute event µs
    are not comparable to this report's nsys pure-kernel scale, so we transfer
    the per-cell RATIO (op/base) onto this report's nsys ``gvr_cutedsl`` µs.
    The overlay preserves each op's measured relative speedup exactly and puts
    it on the common nsys absolute axis.

    Coverage: op#17 = B200+B300 full 720-cell grid, cold+warm; op#19 = B200 full
    grid, cold only; op#18 = B200 BS=1 grid (20 K×N × 3 dtypes) + a BS sweep at
    (K512, fp32, N65536), cold only. Missing warm entries are stored as None and
    render as gaps in the warm-L2 view. Non-exact cells (op#17: 2/720 16-bit
    D0-bug cells) are dropped.
    """
    def put(hw, K, dt, N, BS, op, rc, rw=None):
        if hw not in mdata:
            return
        for cells in mdata[hw]:  # (seq, bs) — seq only has BS=1 keys
            d = cells.get((K, dt, N, BS))
            if not d or ANCHOR not in d:
                continue
            a = d[ANCHOR]
            d[op] = {"c": a["c"] * rc,
                     "w": a["w"] * rw if (rw is not None and a.get("w") is not None) else None}

    # ---- op#17 threshold-portfolio: 3-op fullgrid (base/port/mc) per arch ----
    # Prefer the v2 (D0-fixed, count>kC exactness) fullgrid when present; the
    # 2 v1-inexact cells (K2048 N16384 BS16 bf16/fp16) are exact in v2, so the
    # v2 file contributes all 720 cells instead of 718.
    for hw, fn in (("B200", "fullgrid_b200.jsonl"), ("B300", "fullgrid_b300.jsonl")):
        p2 = (bucket / "op17_gvr_portfolio" / "v2" / "results"
              / fn.replace(".jsonl", "_v2.jsonl"))
        p = p2 if p2.exists() else bucket / "op17_gvr_portfolio" / "results" / fn
        if hw not in mdata or not p.exists():
            continue
        by = {}
        for r in load([p]):
            by.setdefault((r["K"], r["dtype"], r["N"], r["BS"]), {})[r["op"]] = r
        for (K, dt, N, BS), d in by.items():
            b, po = d.get("base"), d.get("port")
            if not b or not po or not po.get("exact", True):
                continue
            rw = (po["us_warm"] / b["us_warm"]) if (po.get("us_warm") and b.get("us_warm")) else None
            put(hw, K, dt, N, BS, "gvr_port", po["us_cold"] / b["us_cold"], rw)

    # ---- op#19 sandwich: fullgrid (fp32+bf16, fp16 in a 2nd file), B200, cold ----
    for fn in ("fullgrid_b200.jsonl", "fullgrid_b200_fp16.jsonl"):
        p = bucket / "op19_gvr_sandwich" / "results" / fn
        if not p.exists():
            continue
        for r in load([p]):
            if not r.get("exact", True):
                continue
            put("B200", r["K"], r["dtype"], r["N"], r["BS"], "gvr_sandwich",
                r["sw_us"] / r["base_us"])

    # ---- op#18 multi-threshold: BS=1 grid + one-cell BS sweep, B200, cold ----
    p = bucket / "op18_gvr_1cta_multithresh" / "results" / "validate_x3.jsonl"
    if p.exists():
        for r in load([p]):
            if not r.get("exact", True):
                continue
            put("B200", r["K"], r["dtype"], r["N"], 1, "gvr_mt", r["mt_us"] / r["base_us"])
    p = bucket / "op18_gvr_1cta_multithresh" / "results" / "bs_sweep_k512_n65536.csv"
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("exact", "OK") != "OK":
                    continue
                # bs-sweep dict only (seq has no BS>1 keys; its BS=1 stays validate_x3's)
                bs_cells = mdata.get("B200", (None, {}))[1]
                d = bs_cells.get((512, "fp32", 65536, int(row["BS"])))
                if d and ANCHOR in d:
                    a = d[ANCHOR]
                    d["gvr_mt"] = {"c": a["c"] * float(row["mt_us"]) / float(row["base_us"]), "w": None}


def to_csv(mdata, sweep_key, out_csv):
    """Flat CSV across all machines for one sweep; first column = hardware."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        head = ["hw", "K", "dtype", "N", "BS"]
        for o in OPS:
            head += [f"{o}_cold_us", f"{o}_warm_us"]
        head += [f"gvr_cold_speedup_vs_{o}" for o in OPS if o != BASELINE]
        w.writerow(head)
        for hw, (seq, bs) in mdata.items():
            cells = seq if sweep_key == "seqlen" else bs
            for (K, dt, N, BS), d in sorted(cells.items()):
                row = [hw, K, dt, N, BS]
                for o in OPS:
                    row += [round(d[o]["c"], 3) if o in d else "",
                            round(d[o]["w"], 3) if o in d and d[o]["w"] is not None else ""]
                base = d.get(BASELINE, {}).get("c")
                for o in OPS:
                    if o == BASELINE:
                        continue
                    row.append(round(d[o]["c"] / base, 3) if (base and o in d and base > 0) else "")
                w.writerow(row)


def analysis(mdata):
    """GVR-vs-others speedup summary, computed per hardware. Returns (en, zh) pairs."""
    out = []
    for hw, (seq, bs) in mdata.items():
        allc = list(seq.values()) + list(bs.values())
        for o in OPS:
            if o == BASELINE:
                continue
            sp = [d[o]["c"] / d[BASELINE]["c"] for d in allc if BASELINE in d and o in d and d[BASELINE]["c"] > 0]
            if sp:
                med, mn, mx = st.median(sp), min(sp), max(sp)
                faster, total = sum(x > 1 for x in sp), len(sp)
                en = (f"[{hw}] GVR(CUDA) vs {OP_LABEL[o]} (cold-L2): median={med:.2f}× "
                      f"(min={mn:.2f}×, max={mx:.2f}×); GVR faster in {faster}/{total} cells.")
                zh = (f"[{hw}] GVR(CUDA) 对 {OP_LABEL[o]}（冷 L2）：中位 {med:.2f}× "
                      f"（最小 {mn:.2f}×，最大 {mx:.2f}×）；GVR 更快的 cell 数 {faster}/{total}。")
                out.append((en, zh))
    return out


def cells_to_json(cells, sweep):
    return [{"sw": sweep, "K": K, "dtype": dt, "N": N, "BS": BS, "t": d} for (K, dt, N, BS), d in cells.items()]


def compare_table(mdata):
    """Per-op median cold-L2 µs per machine over each machine's full cell set."""
    labels = list(mdata.keys())
    out = {}
    for op in OPS:
        row = {}
        for lab in labels:
            seq, bs = mdata[lab]
            allc = list(seq.values()) + list(bs.values())
            vals = [d[op]["c"] for d in allc if op in d and d[op]["c"] > 0]
            row[lab] = st.median(vals) if vals else None
        out[op] = row
    return labels, out


def build_compare_block(mdata):
    if len(mdata) < 2:
        return ""
    labels, tbl = compare_table(mdata)
    ref = labels[0]  # denominator (B300)

    def table_html():
        h = "<table><tr><th>operator</th>"
        for lab in labels:
            h += f"<th>{lab} median µs</th>"
        h += f"<th>{labels[-1]}/{ref}</th></tr>"
        for op in OPS:
            h += f"<tr><td>{OP_LABEL[op]}</td>"
            vals = tbl[op]
            for lab in labels:
                v = vals[lab]
                h += f"<td>{v:.2f}</td>" if v is not None else "<td>–</td>"
            a, b = vals[ref], vals[labels[-1]]
            h += f"<td>{(b / a):.3f}×</td>" if (a and b) else "<td>–</td>"
            h += "</tr>"
        h += "</table>"
        return h

    t = table_html()
    en = ("<h3>Hardware comparison — per-op median cold-L2 µs by machine</h3>"
          "<p>Identical synthetic inputs (seed=42, unified preIdx hit-rate 0.6) on every machine, so "
          f"differences are pure hardware. Ratio = {labels[-1]}/{ref} (&gt;1 ⇒ {ref} faster). Median "
          "taken over each machine's full cell set (seqlen + BS-scaling).</p>" + t)
    zh = ("<h3>硬件对比 —— 各算子按机器的冷 L2 中位 µs</h3>"
          "<p>每台机器使用完全相同的合成输入（seed=42，统一 preIdx 命中率 0.6），故差异纯属硬件。"
          f"比值 = {labels[-1]}/{ref}（&gt;1 ⇒ {ref} 更快）。中位取自每台机器的全部 cell（seqlen + BS 扫描）。</p>" + t)
    return '<div class="card">' + bi(en, zh) + '</div>'


HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Indexer Top-K Operator Evaluation</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1419;color:#e6e6e6}
.wrap{max-width:1300px;margin:0 auto;padding:24px} h1{color:#76b900}
h2{color:#76b900;border-bottom:1px solid #2a3340;padding-bottom:6px;margin-top:36px} h3{color:#9ecb3a}
.card{background:#161b22;border:1px solid #2a3340;border-radius:10px;padding:18px;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid #2a3340;padding:5px 8px;text-align:right}
th{background:#1c2530;color:#9ecb3a} td:first-child,th:first-child{text-align:left}
select{background:#1c2530;color:#e6e6e6;border:1px solid #2a3340;padding:6px;border-radius:6px;margin:4px}
.chart{height:640px} .row{display:flex;gap:16px;flex-wrap:wrap} .row>div{flex:1;min-width:480px}
.kpi{display:inline-block;background:#1c2530;border-radius:8px;padding:10px 16px;margin:6px} .kpi b{color:#76b900;font-size:20px}
li{margin:4px 0} .reg{font-weight:bold;color:#9ecb3a}
/* --- i18n: English default; body.zh swaps to Chinese --- */
.i18n-zh{display:none}
body.zh .i18n-en{display:none}
body.zh .i18n-zh{display:revert}
#langbtn{position:fixed;top:14px;right:16px;z-index:99;background:#76b900;color:#0f1419;border:none;
  font-weight:bold;padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px}
#langbtn:hover{background:#8fd400}
</style></head><body><div class="wrap">
<button id="langbtn" onclick="toggleLang()">中文</button>
<h1><span class="i18n-en">DeepSeek-V4 Indexer Top-K — Single-Operator Evaluation</span><span class="i18n-zh">DeepSeek-V4 Indexer Top-K —— 单算子评测</span></h1>
{intro}
{banner}
<div class="card"><h3><span class="i18n-en">Test environment</span><span class="i18n-zh">测试环境</span></h3>{env}</div>
<div class="card"><h3><span class="i18n-en">Operators &amp; methodology</span><span class="i18n-zh">算子与方法学</span></h3>{ops}</div>
<div class="card"><h3><span class="i18n-en">Algorithmic comparison — SGLang StreamingTopK vs the in-tree families (B300 cold-L2; B200 hardware-invariant)</span><span class="i18n-zh">算法对比 —— SGLang StreamingTopK vs 树内算子族（B300 冷 L2；B200 硬件无关）</span></h3>{algo}</div>
{compare_block}
<div class="card"><h3><span class="i18n-en">Synthetic input generation (random data characteristics)</span><span class="i18n-zh">合成输入生成（随机数据特征）</span></h3>{synth}</div>
<div class="card"><h3><span class="i18n-en">Input logit distribution — CCDF of the benchmarked data</span><span class="i18n-zh">输入 logit 分布 —— 被测数据的 CCDF</span></h3>{input_ccdf}</div>
<div class="card"><h3><span class="i18n-en">Deterministic reproduction &amp; SKILL invocation</span><span class="i18n-zh">确定性复现与 SKILL 调用</span></h3>{repro}</div>
<div class="card"><h3><span class="i18n-en">Auto-analysis (cold-L2, per hardware)</span><span class="i18n-zh">自动分析（冷 L2，按硬件）</span></h3><ul>{analysis}</ul></div>
<div class="card"><h3><span class="i18n-en">Timing validation — nsys vs CUDA-event (cold-L2 spot-check, B300)</span><span class="i18n-zh">计时验证 —— nsys vs CUDA-event（冷 L2 抽查，B300）</span></h3>
{nsys_note}{nsys}</div>

<div class="card">
<label><span class="i18n-en">view</span><span class="i18n-zh">视图</span> <select id="view"></select></label>
<label><span class="i18n-en">L2 regime</span><span class="i18n-zh">L2 状态</span> <select id="reg"><option value="c" data-en="cold-L2 (flushed)" data-zh="冷 L2（已清空）">cold-L2 (flushed)</option><option value="w" data-en="warm-L2" data-zh="热 L2">warm-L2</option></select></label>
<span style="color:#888;font-size:12px">&nbsp;<span class="i18n-en">(<b>view</b>: a single machine, or <b>compare</b> = B300 solid + B200 dashed)</span><span class="i18n-zh">（<b>视图</b>：单台机器，或 <b>compare</b> = B300 实线 + B200 虚线）</span></span>
</div>

<div class="card"><b><span class="i18n-en">Show operators:</span><span class="i18n-zh">显示算子：</span></b>
<label style="color:#76b900"><input type="checkbox" class="opck" value="gvr_cuda" checked> GVR (CUDA)</label>
<label style="color:#b3e05a"><input type="checkbox" class="opck" value="gvr_cutedsl" checked> GVR (cuteDSL)</label>
<label style="color:#5fd35f"><input type="checkbox" class="opck" value="gvr_cutedsl_rs" checked> GVR (cuteDSL, rank-scatter P4)</label>
<label style="color:#2ec4b6"><input type="checkbox" class="opck" value="gvr_multicta_cutedsl" checked> GVR multi-CTA (cuteDSL, PR#15198)</label>
<label style="color:#9b5de5"><input type="checkbox" class="opck" value="gvr_op8" checked> GVR op#8 (multi-CTA + rank-scatter P4)</label>
<label style="color:#ff6ec7"><input type="checkbox" class="opck" value="gvr_port" checked> GVR op#17 threshold-portfolio (cluster, auto-G)</label>
<label style="color:#00e5ff"><input type="checkbox" class="opck" value="gvr_mt" checked> GVR op#18 multi-threshold (single-CTA)</label>
<label style="color:#ffffff"><input type="checkbox" class="opck" value="gvr_sandwich" checked> GVR op#19 sandwich two-threshold (dispatch)</label>
<label style="color:#ff8c42"><input type="checkbox" class="opck" value="radix_single_cuda" checked> Radix single-CTA (CUDA)</label>
<label style="color:#e84855"><input type="checkbox" class="opck" value="radix_multi_cuda" checked> Radix multi-CTA (CUDA)</label>
<label style="color:#4ea8de"><input type="checkbox" class="opck" value="radix_cutedsl" checked> Radix (cuteDSL)</label>
<label style="color:#f6c453"><input type="checkbox" class="opck" value="radix_cutedsl_single" checked> Radix (cuteDSL)-single</label>
<label style="color:#1f6feb"><input type="checkbox" class="opck" value="radix_cutedsl_multi" checked> Radix (cuteDSL)-multi</label>
<label style="color:#d62728"><input type="checkbox" class="opck" value="sglang_streaming" checked> SGLang StreamingTopK (fp32, K&le;1024)</label>
<span style="color:#888;font-size:12px">&nbsp;<span class="i18n-en">(toggles latency &amp; speedup curves; GVR(CUDA) stays the speedup baseline regardless)</span><span class="i18n-zh">（切换延迟与加速比曲线；无论如何 GVR(CUDA) 始终是加速比基准）</span></span></div>

<h2><span class="i18n-en">1. Seq-len sweep (BS=1)</span><span class="i18n-zh">1. 序列长度扫描（BS=1）</span></h2>
<div class="card"><label><span class="i18n-en">dtype</span><span class="i18n-zh">数据类型</span> <select id="s_dt"></select></label><label>K <select id="s_k"></select></label>
<div class="row"><div id="s_lat" class="chart"></div><div id="s_spd" class="chart"></div></div></div>

<h2><span class="i18n-en">2. BS-scaling</span><span class="i18n-zh">2. BS 扩展性</span></h2>
<div class="card"><label><span class="i18n-en">dtype</span><span class="i18n-zh">数据类型</span> <select id="b_dt"></select></label><label>K <select id="b_k"></select></label><label>N <select id="b_n"></select></label>
<div class="row"><div id="b_lat" class="chart"></div><div id="b_spd" class="chart"></div></div></div>

<h2><span class="i18n-en">3. Full data</span><span class="i18n-zh">3. 完整数据</span></h2>
<div class="card"><p><span class="i18n-en">Download:</span><span class="i18n-zh">下载：</span> <a href="seqlen_data.csv" style="color:#76b900">seqlen_data.csv</a> ·
<a href="bs_data.csv" style="color:#76b900">bs_data.csv</a> &nbsp;<span class="i18n-en">(both regimes + cold speedups; first column = hardware)</span><span class="i18n-zh">（两种 L2 状态 + 冷加速比；第一列 = 硬件）</span></p>
<div id="tbl" style="max-height:520px;overflow:auto"></div></div>

<h2><span class="i18n-en">4. GVR P2 / P4-snap iteration counts (BS=1)</span><span class="i18n-zh">4. GVR P2 / P4-snap 迭代次数（BS=1）</span></h2>
<div class="card">{iters}</div>

<script>
const OPS={ops_json}, OPL={opl_json}, BASE="{baseline}", COL={col_json};
const DATA={data_json};              // {{hw: {{seq:[...], bs:[...]}}}}
const LABELS={labels_json};          // hardware labels in order (B300, B200)
const DASH={dash_json};              // {{hw: 'solid'|'dash'}}
let LANG='en';
const I18N={{
  en:{{latseq:'Latency vs seq-len',spd1:'GVR(CUDA) speedup (>1 ⇒ GVR faster)',latbs:'Latency vs BS',
       spd2:'GVR(CUDA) speedup',xseq:'seq-len N',xbs:'batch size',us:'µs',spd:'speedup ×',
       cold:'cold-L2',warm:'warm-L2',nodata:'no data',
       cols:['hw','sweep','K','dtype','N','BS']}},
  zh:{{latseq:'延迟 vs 序列长度',spd1:'GVR(CUDA) 加速比 (>1 ⇒ GVR 更快)',
       latbs:'延迟 vs BS',spd2:'GVR(CUDA) 加速比',xseq:'序列长度 N',xbs:'batch size',
       us:'微秒(µs)',spd:'加速比 ×',cold:'冷 L2',warm:'热 L2',nodata:'无数据',
       cols:['硬件','扫描','K','dtype','N','BS']}}
}};
function L(k){{return I18N[LANG][k]}}
function regL(){{return R()=='c'?L('cold'):L('warm')}}
function R(){{return reg.value}}
function AOPS(){{return OPS.filter(o=>{{let e=document.querySelector('.opck[value="'+o+'"]');return !e||e.checked}})}}
function curViews(){{return view.value==='compare'?LABELS:[view.value]}}
function isCmp(){{return view.value==='compare'}}
const LAYOUT=(t,xt,yt)=>({{title:{{text:t,font:{{color:'#e6e6e6'}}}},paper_bgcolor:'#161b22',plot_bgcolor:'#0f1419',
  font:{{color:'#ccc'}},xaxis:{{title:xt,gridcolor:'#2a3340',type:'log'}},yaxis:{{title:yt,gridcolor:'#2a3340'}},
  legend:{{orientation:'h',y:-0.3}},margin:{{t:40,r:10}}}});
function uniq(a){{return [...new Set(a)].sort((x,y)=>x-y)}}
function fill(sel,vals){{let cur=sel.value;sel.innerHTML='';vals.forEach(v=>{{let o=document.createElement('option');o.value=v;o.text=v;sel.appendChild(o)}});if(vals.includes(+cur)||vals.includes(cur))sel.value=cur;}}
function val(c,op){{return c&&c.t[op]?c.t[op][R()]:null}}
function seqDraw(){{let dt=s_dt.value,K=+s_k.value,lat=[],spd=[];
 for(let lab of curViews()){{let cs=DATA[lab].seq.filter(c=>c.dtype==dt&&c.K==K),Ns=uniq(cs.map(c=>c.N));
  for(let op of AOPS()){{lat.push({{x:Ns,y:Ns.map(n=>val(cs.find(c=>c.N==n),op)),name:OPL[op]+(isCmp()?' ['+lab+']':''),mode:'lines+markers',line:{{color:COL[op],dash:DASH[lab]}}}});}}
  for(let op of AOPS().filter(o=>o!=BASE)){{spd.push({{x:Ns,y:Ns.map(n=>{{let c=cs.find(c=>c.N==n);let a=val(c,op),b=val(c,BASE);return(a&&b)?a/b:null}}),name:OPL[op]+(isCmp()?' ['+lab+']':'')+' / GVR',mode:'lines+markers',line:{{color:COL[op],dash:DASH[lab]}}}});}}
 }}
 Plotly.newPlot('s_lat',lat,LAYOUT(L('latseq')+' (K='+K+', '+dt+', BS=1, '+regL()+')',L('xseq'),L('us')),{{responsive:true}});
 Plotly.newPlot('s_spd',spd,LAYOUT(L('spd1'),L('xseq'),L('spd')),{{responsive:true}});}}
function bsDraw(){{let dt=b_dt.value,K=+b_k.value,N=+b_n.value,lat=[],spd=[];
 for(let lab of curViews()){{let cs=DATA[lab].bs.filter(c=>c.dtype==dt&&c.K==K&&c.N==N),Bs=uniq(cs.map(c=>c.BS));
  for(let op of AOPS()){{lat.push({{x:Bs,y:Bs.map(b=>val(cs.find(c=>c.BS==b),op)),name:OPL[op]+(isCmp()?' ['+lab+']':''),mode:'lines+markers',line:{{color:COL[op],dash:DASH[lab]}}}});}}
  for(let op of AOPS().filter(o=>o!=BASE)){{spd.push({{x:Bs,y:Bs.map(b=>{{let c=cs.find(c=>c.BS==b);let a=val(c,op),bb=val(c,BASE);return(a&&bb)?a/bb:null}}),name:OPL[op]+(isCmp()?' ['+lab+']':'')+' / GVR',mode:'lines+markers',line:{{color:COL[op],dash:DASH[lab]}}}});}}
 }}
 Plotly.newPlot('b_lat',lat,LAYOUT(L('latbs')+' (K='+K+', '+dt+', N='+N+', '+regL()+')',L('xbs'),L('us')),{{responsive:true}});
 Plotly.newPlot('b_spd',spd,LAYOUT(L('spd2'),L('xbs'),L('spd')),{{responsive:true}});}}
function bsFillN(){{let dt=b_dt.value,K=+b_k.value,ns=[];for(let lab of curViews())ns=ns.concat(DATA[lab].bs.filter(c=>c.dtype==dt&&c.K==K).map(c=>c.N));fill(b_n,uniq(ns));bsDraw();}}
function refillDims(){{let seq=[],bs=[];for(let lab of curViews()){{seq=seq.concat(DATA[lab].seq);bs=bs.concat(DATA[lab].bs);}}
 let dts=[...new Set(seq.concat(bs).map(c=>c.dtype))],ks=uniq(seq.concat(bs).map(c=>c.K));
 fill(s_dt,dts);fill(s_k,ks);fill(b_dt,dts);fill(b_k,ks);bsFillN();seqDraw();buildTbl();}}
fill(view,LABELS.concat(LABELS.length>1?['compare']:[]));
if(LABELS.length>1)view.value='compare';
s_dt.onchange=s_k.onchange=seqDraw;b_dt.onchange=b_k.onchange=bsFillN;b_n.onchange=bsDraw;
reg.onchange=()=>{{seqDraw();bsDraw();buildTbl();}};view.onchange=refillDims;
document.querySelectorAll('.opck').forEach(c=>c.onchange=()=>{{seqDraw();bsDraw();}});
refillDims();
function buildTbl(){{let rows=[];for(let lab of LABELS){{for(let c of DATA[lab].seq.concat(DATA[lab].bs))rows.push([lab,c]);}}
 if(!rows.length){{tbl.innerHTML='<p>'+L('nodata')+'</p>';return}}
 let H=L('cols');
 let h='<table><tr><th>'+H[0]+'</th><th>'+H[1]+'</th><th>'+H[2]+'</th><th>'+H[3]+'</th><th>'+H[4]+'</th><th>'+H[5]+'</th>'+OPS.map(o=>'<th>'+OPL[o]+' µs ('+regL()+')</th>').join('')+'</tr>';
 rows.sort((a,b)=>a[0].localeCompare(b[0])||a[1].K-b[1].K||a[1].dtype.localeCompare(b[1].dtype)||a[1].N-b[1].N||a[1].BS-b[1].BS);
 for(let [lab,c] of rows){{h+='<tr><td>'+lab+'</td><td>'+c.sw+'</td><td>'+c.K+'</td><td>'+c.dtype+'</td><td>'+c.N+'</td><td>'+c.BS+'</td>'+OPS.map(o=>'<td>'+((c.t[o]&&c.t[o][R()]!=null)?c.t[o][R()].toFixed(2):'–')+'</td>').join('')+'</tr>'}}
 tbl.innerHTML=h+'</table>';}}
buildTbl();
function setRegOptionText(){{document.querySelectorAll('#reg option').forEach(o=>{{o.text=o.getAttribute('data-'+LANG)||o.text;}});}}
setRegOptionText();
function toggleLang(){{let on=document.body.classList.toggle('zh');LANG=on?'zh':'en';
 document.getElementById('langbtn').textContent=on?'English':'中文';
 setRegOptionText();seqDraw();bsDraw();buildTbl();}}
</script></div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    bucket = here.parent
    ap.add_argument("--machine", action="append", default=None,
                    help="LABEL:results_dir, e.g. B300:results_b300_nsys (repeatable). "
                         "Default: B300:results_b300_nsys B200:results_b200_nsys.")
    ap.add_argument("--out", default=str(here / "report.html"))
    args = ap.parse_args()

    # Default to the nsys pure-kernel re-test roots (full-grid, all 11 ops,
    # warm+cold L2). The legacy CUDA-event roots were B300:results B200:results_b200.
    specs = args.machine or ["B300:results_b300_nsys", "B200:results_b200_nsys"]
    mdata = {}
    for spec in specs:
        lab, root = spec.split(":", 1)
        root_p = root if Path(root).is_absolute() else bucket / root
        seq, bs = load_machine(root_p)
        # Only include a machine that actually has data (e.g. B200 may be mid-sweep).
        if seq or bs:
            mdata[lab] = (seq, bs)

    # Overlay ops #17/#18/#19 (anchored ratio transfer onto the nsys scale).
    merge_anchored_ops(mdata, bucket)

    to_csv(mdata, "seqlen", here / "seqlen_data.csv")
    to_csv(mdata, "bs", here / "bs_data.csv")

    # ---- intro paragraph (bilingual) ----
    intro_en = (
        "<p>Apples-to-apples comparison of 11 indexer top-K operators extracted from TensorRT-LLM and "
        "benchmarked standalone (the 11th, <b>SGLang StreamingTopK</b>, is the external cross-reference — "
        "see the algorithmic comparison card below). Primary objective: <b>speedup of GVR top-K vs the other operators</b>. "
        "Timing = <b>nsys pure-kernel GPU time</b> (per-cell NVTX range, kernel time summed from "
        "<code>nvtx_kern_sum</code>; the L2-evict kernel is excluded). This supersedes the earlier "
        "CUDA-event timing (which over-charged by the graph-launch + single-replay overhead, ~11%); "
        "every cell here is the true kernel GPU time. Two L2 regimes: "
        "<span class=\"reg\">cold-L2</span> (512 MB flush before each launch — canonical, isolated worst case) and "
        "<span class=\"reg\">warm-L2</span> (logits hot in L2 — models the fused indexer→top-K path). "
        "This report adds a <b>hardware dimension</b>: the identical sweep is run on "
        "<span class=\"reg\">B300</span> (sm_103) and <span class=\"reg\">B200</span> (sm_100); use the "
        "<b>view</b> selector below for a single machine or a <b>compare</b> overlay (B300 solid, B200 dashed). "
        "Three later optimization ops — <b>#17 threshold-portfolio</b>, <b>#18 multi-threshold</b> and "
        "<b>#19 sandwich two-threshold</b> — are additionally overlaid as <b>anchored series</b> "
        "(per-cell op/base ratio from their in-bucket A/B, transferred onto this report's nsys "
        "<code>gvr_cutedsl</code> axis; coverage &amp; method in the methodology card below).</p>")
    intro_zh = (
        "<p>对从 TensorRT-LLM 中抽取并独立基准测试的 11 个 indexer top-K 算子做同口径对比"
        "（第 11 个 <b>SGLang StreamingTopK</b> 是外部交叉参考 —— 见下方算法对比卡片）。"
        "首要目标：<b>GVR top-K 相对其他算子的加速比</b>。计时 = <b>nsys 纯 kernel GPU 时间</b>"
        "（按 cell 的 NVTX 区间，从 <code>nvtx_kern_sum</code> 求和 kernel 时间；L2 清空 kernel 已排除）。"
        "这取代了早前的 CUDA-event 计时（后者因 graph 启动 + 单次 replay 开销高估约 11%）；"
        "此处每个 cell 都是真实 kernel GPU 时间。两种 L2 状态："
        "<span class=\"reg\">冷 L2</span>（每次启动前清空 512 MB —— 规范化、隔离的最坏情况）与 "
        "<span class=\"reg\">热 L2</span>（logits 驻留 L2 —— 模拟融合 indexer→top-K 路径）。"
        "本报告还增加了<b>硬件维度</b>：相同 sweep 同时在 <span class=\"reg\">B300</span>(sm_103) 与 "
        "<span class=\"reg\">B200</span>(sm_100) 上运行；用下方 <b>view</b> 选择器查看单机或 <b>compare</b> 叠加"
        "（B300 实线，B200 虚线）。另外叠加了三个后续优化算子 —— <b>#17 threshold-portfolio</b>、"
        "<b>#18 multi-threshold</b>、<b>#19 sandwich two-threshold</b> —— 作为<b>锚定序列</b>"
        "（用各自 bucket 内 A/B 的按-cell op/base 比值，迁移到本报告的 nsys <code>gvr_cutedsl</code> 刻度上；"
        "覆盖范围与方法见下方方法学卡片）。</p>")
    intro_html = bi(intro_en, intro_zh)

    # ---- env (KPIs + provenance note) ----
    kpis = ("<div class='kpi'>GPUs <b>8× B300 SXM6</b> (sm_103) <span style='color:#888'>+</span> <b>8× B200</b> (sm_100)</div>"
            "<div class='kpi'>CUDA <b>13.1</b></div>"
            "<div class='kpi'>torch <b>2.11</b></div><div class='kpi'>cutlass-dsl <b>4.5.0</b></div>"
            "<div class='kpi'>TRT-LLM <b>1.3.0rc15</b></div>")
    kpi_timing_en = "<div class='kpi'>timing <b>nsys pure-kernel GPU time, cold+warm L2</b> (NVTX-range projection)</div>"
    kpi_timing_zh = "<div class='kpi'>计时 <b>nsys 纯 kernel GPU 时间，冷+热 L2</b>（NVTX 区间投影）</div>"
    cells_en = ", ".join(f"<b>{lab}</b> {len(s)} seqlen + {len(b)} bs" for lab, (s, b) in mdata.items())
    cells_zh = "，".join(f"<b>{lab}</b> {len(s)} 个 seqlen + {len(b)} 个 bs" for lab, (s, b) in mdata.items())
    env_en = (kpis + kpi_timing_en +
              "<p style='color:#9aa;margin-top:8px'>Both machines run the <b>identical</b> sweep "
              "(same synthetic inputs, seed=42, unified preIdx hit-rate 0.6, same prebuilt CUDA-op .so "
              "and cuteDSL kernels), so every cross-machine difference is pure hardware. Cells loaded: "
              + cells_en + ".</p>")
    env_zh = (kpis + kpi_timing_zh +
              "<p style='color:#9aa;margin-top:8px'>两台机器运行<b>完全相同</b>的 sweep"
              "（相同合成输入，seed=42，统一 preIdx 命中率 0.6，相同的预编译 CUDA-op .so 与 cuteDSL kernel），"
              "因此所有跨机差异纯属硬件。已加载 cells："
              + cells_zh + "。</p>")
    env = bi(env_en, env_zh)

    # ---- operators & methodology ----
    ops_table_en = ("<table><tr><th>Operator</th><th>Kind</th><th>exact?</th><th>dtypes×K</th></tr>"
        "<tr><td>GVR (CUDA) — <b>baseline</b></td><td>CUDA C++</td><td>exact (heuristic pre_idx threshold-seed)</td><td>fp32/bf16/fp16 × 512/1024/2048</td></tr>"
        "<tr><td>GVR (cuteDSL) — single-CTA</td><td>CuTe DSL</td><td>exact (heuristic seed)</td><td>full</td></tr>"
        "<tr><td>GVR (cuteDSL, rank-scatter P4)</td><td>CuTe DSL</td><td>exact (heuristic seed)</td><td>full; op#7 exact rank-scatter P4 variant</td></tr>"
        "<tr><td>GVR multi-CTA (cuteDSL, <a href='https://github.com/NVIDIA/TensorRT-LLM/pull/15198' style='color:#76b900'>PR#15198</a>)</td>"
        "<td>CuTe DSL (cluster / DSMEM)</td><td>exact (heuristic seed)</td><td>full; cluster_size auto-dispatched</td></tr>"
        "<tr><td><b>GVR op#8 (multi-CTA cluster + rank-scatter-exact P4)</b></td>"
        "<td>CuTe DSL (cluster / DSMEM)</td><td>exact (heuristic seed)</td><td>full; stacks the two best GVR levers (B300 op#8 study)</td></tr>"
        "<tr><td><b>GVR op#17 threshold-portfolio</b> (cluster, auto-G) — <i>anchored overlay</i></td>"
        "<td>CuTe DSL (cluster / DSMEM)</td><td>exact (2/720 16-bit D0-bug cells dropped)</td>"
        "<td>full grid, B200+B300, cold+warm; G thresholds evaluated in parallel across the cluster, auto-G per (N,BS)</td></tr>"
        "<tr><td><b>GVR op#18 multi-threshold</b> (single-CTA) — <i>anchored overlay</i></td>"
        "<td>CuTe DSL</td><td>exact</td>"
        "<td>B200, BS=1 grid (all K×dtype×N) + BS sweep at (K512, fp32, N65536), cold only; CDF-aware multi-threshold round-1</td></tr>"
        "<tr><td><b>GVR op#19 sandwich two-threshold</b> (dispatch) — <i>anchored overlay</i></td>"
        "<td>CuTe DSL (single-CTA + cluster, per-cell dispatch table)</td><td>exact (720/720)</td>"
        "<td>full grid, B200 only, cold only; sandwich thr-pair from one M-ary ladder pass, 240-key dispatch per dtype</td></tr>"
        "<tr><td>Radix single-CTA (CUDA)</td><td>CUDA C++</td><td>exact</td><td>full</td></tr>"
        "<tr><td>Radix multi-CTA (CUDA)</td><td>CUDA C++</td><td>exact</td><td>full</td></tr>"
        "<tr><td>Radix (cuteDSL)</td><td>CuTe DSL</td><td>exact</td><td>full; production auto chunk heuristic (single- or multi-CTA per cell)</td></tr>"
        "<tr><td>Radix (cuteDSL)-single — forced single-CTA</td><td>CuTe DSL</td><td>exact</td>"
        "<td>1 CTA scans the whole row; only N that fit one CTA's smem (~57K fp32 / ~115K bf16-fp16)</td></tr>"
        "<tr><td>Radix (cuteDSL)-multi — forced multi-CTA</td><td>CuTe DSL</td><td>exact</td>"
        "<td>full; &ge;2 CTAs cooperate per row via global-histogram merge (production 8192 chunk floor)</td></tr>"
        "<tr><td><b>SGLang StreamingTopK</b> (external)</td><td>CUDA C++ (real SGLang <code>device::top512::"
        "StreamingTopK&lt;K&gt;</code> SASS)</td><td>exact (no hint)</td>"
        "<td><b>fp32 only</b> (TMA <code>float</code> loads) &times; <b>K&isin;{512,1024}</b> only "
        "(<code>kMaxTies=kMaxTopK=1024</code>; K=2048 is outside the design envelope &mdash; verified "
        "non-exact, so absent from the K=2048 / V3.2 real-data rows)</td></tr></table>")
    ops_para_en = ("<p><b>The 10 in-tree ops compute exact top-K</b> across all dtypes&times;K — they solve the same "
        "problem and are directly interchangeable. <b>SGLang StreamingTopK is also exact</b> but only over "
        "its faithful envelope (fp32, K&le;1024); it is included as an <b>external reference kernel</b>, not "
        "an interchangeable in-tree path. GVR is a threshold-based exact selector: the <code>pre_idx</code> hint only "
        "<b>seeds</b> its Phase-2 threshold search (Guess); the full-N Verify/Refine phases guarantee "
        "correctness regardless of hint quality (a hit-rate of 0 only adds secant iterations, not error). "
        "All ops consume <b>identical</b> logits / pre_idx per cell. Correctness vs torch.topk = "
        "value-equivalence (vdiff=0); CUDA-op parity vs in-tree ≤3%. Input generation detailed below.</p>"
        "<p><b>Anchored overlays (ops #17/#18/#19)</b>: these three ops were benchmarked in their own buckets "
        "as back-to-back A/Bs against the same single-CTA <code>gvr_cutedsl</code> baseline, on the same "
        "synthetic report bundles (seed=42, hit-rate 0.6), with CUDA-graph + cudaEvent cold-L2 timing. "
        "Their curves here are <b>ratio-transferred</b>: per cell, µs = (this report's nsys "
        "<code>gvr_cutedsl</code> µs) × (op/base event ratio measured in-run). This preserves each op's "
        "measured relative speedup exactly while placing it on the common nsys absolute axis "
        "(spot cells were independently nsys-validated in each bucket: op#17 1.21–1.67×, op#18 1.10–1.45×, "
        "op#19 fp32 8/8 + 16-bit 6/6 positive). Coverage limits: op#18/#19 are B200 + cold-L2 only "
        "(gaps in the warm view / B300 view); op#18 has BS&gt;1 only at (K512, fp32, N65536).</p>")
    ops_table_zh = ("<table><tr><th>算子</th><th>类型</th><th>精确?</th><th>dtype×K</th></tr>"
        "<tr><td>GVR (CUDA) —— <b>基准</b></td><td>CUDA C++</td><td>精确（启发式 pre_idx 阈值种子）</td><td>fp32/bf16/fp16 × 512/1024/2048</td></tr>"
        "<tr><td>GVR (cuteDSL) —— 单 CTA</td><td>CuTe DSL</td><td>精确（启发式种子）</td><td>全部</td></tr>"
        "<tr><td>GVR (cuteDSL, rank-scatter P4)</td><td>CuTe DSL</td><td>精确（启发式种子）</td><td>全部；op#7 精确 rank-scatter P4 变体</td></tr>"
        "<tr><td>GVR multi-CTA (cuteDSL, <a href='https://github.com/NVIDIA/TensorRT-LLM/pull/15198' style='color:#76b900'>PR#15198</a>)</td>"
        "<td>CuTe DSL（cluster / DSMEM）</td><td>精确（启发式种子）</td><td>全部；cluster_size 自动派发</td></tr>"
        "<tr><td><b>GVR op#8（multi-CTA cluster + 精确 rank-scatter P4）</b></td>"
        "<td>CuTe DSL（cluster / DSMEM）</td><td>精确（启发式种子）</td><td>全部；叠加两个最佳 GVR 手段（B300 op#8 研究）</td></tr>"
        "<tr><td><b>GVR op#17 threshold-portfolio</b>（cluster，auto-G）—— <i>锚定叠加</i></td>"
        "<td>CuTe DSL（cluster / DSMEM）</td><td>精确（2/720 个 16-bit D0-bug cell 已剔除）</td>"
        "<td>全网格，B200+B300，冷+热；G 个阈值在 cluster 内并行评估，按 (N,BS) 自动选 G</td></tr>"
        "<tr><td><b>GVR op#18 multi-threshold</b>（单 CTA）—— <i>锚定叠加</i></td>"
        "<td>CuTe DSL</td><td>精确</td>"
        "<td>B200，BS=1 网格（全部 K×dtype×N）+ (K512, fp32, N65536) 的 BS 扫描，仅冷 L2；CDF 感知的多阈值第一轮</td></tr>"
        "<tr><td><b>GVR op#19 sandwich two-threshold</b>（调度表）—— <i>锚定叠加</i></td>"
        "<td>CuTe DSL（单 CTA + cluster，按 cell 调度表）</td><td>精确（720/720）</td>"
        "<td>全网格，仅 B200，仅冷 L2；一次 M 叉阶梯扫描免费取得阈值对，每 dtype 240 键调度表</td></tr>"
        "<tr><td>Radix single-CTA (CUDA)</td><td>CUDA C++</td><td>精确</td><td>全部</td></tr>"
        "<tr><td>Radix multi-CTA (CUDA)</td><td>CUDA C++</td><td>精确</td><td>全部</td></tr>"
        "<tr><td>Radix (cuteDSL)</td><td>CuTe DSL</td><td>精确</td><td>全部；生产环境自动 chunk 启发式（按 cell 单/多 CTA）</td></tr>"
        "<tr><td>Radix (cuteDSL)-single —— 强制单 CTA</td><td>CuTe DSL</td><td>精确</td>"
        "<td>1 个 CTA 扫描整行；仅适用于能放进单 CTA smem 的 N（fp32 约 57K / bf16-fp16 约 115K）</td></tr>"
        "<tr><td>Radix (cuteDSL)-multi —— 强制多 CTA</td><td>CuTe DSL</td><td>精确</td>"
        "<td>全部；每行 &ge;2 个 CTA 经全局直方图归并协作（生产环境 8192 chunk 下限）</td></tr>"
        "<tr><td><b>SGLang StreamingTopK</b>（外部）</td><td>CUDA C++（真实 SGLang <code>device::top512::"
        "StreamingTopK&lt;K&gt;</code> SASS）</td><td>精确（无提示）</td>"
        "<td><b>仅 fp32</b>（TMA <code>float</code> 加载）&times; <b>仅 K&isin;{512,1024}</b>"
        "（<code>kMaxTies=kMaxTopK=1024</code>；K=2048 超出设计范围 —— 已验证非精确，故在 K=2048 / V3.2 真实数据行中缺席）</td></tr></table>")
    ops_para_zh = ("<p><b>10 个树内算子在全部 dtype&times;K 下都计算精确 top-K</b> —— 它们解决同一问题，可直接互换。"
        "<b>SGLang StreamingTopK 同样精确</b>，但仅在其忠实范围内（fp32，K&le;1024）；它作为<b>外部参考 kernel</b> 纳入，"
        "并非可互换的树内路径。GVR 是基于阈值的精确选择器：<code>pre_idx</code> 提示只<b>为</b>其 Phase-2 阈值搜索（Guess）"
        "<b>提供种子</b>；全 N 的 Verify/Refine 阶段无论提示质量如何都保证正确性（命中率为 0 只增加割线迭代，不产生误差）。"
        "所有算子每个 cell 消费<b>完全相同</b>的 logits / pre_idx。相对 torch.topk 的正确性 = 值等价（vdiff=0）；"
        "CUDA-op 与树内算子的一致性 ≤3%。输入生成见下文。</p>"
        "<p><b>锚定叠加算子（op#17/#18/#19）</b>：这三个算子在各自 bucket 内与同一个单 CTA "
        "<code>gvr_cutedsl</code> 基准做背靠背 A/B，使用相同的合成 report 数据包（seed=42，命中率 0.6）、"
        "CUDA-graph + cudaEvent 冷 L2 计时。此处曲线为<b>比值迁移</b>：每个 cell 的 µs =（本报告 nsys "
        "<code>gvr_cutedsl</code> µs）×（bucket 内实测的 op/base event 比值）。这在保持各算子实测相对加速比"
        "不变的前提下，将其放到统一的 nsys 绝对刻度上（各 bucket 均有独立 nsys 抽查验证：op#17 1.21–1.67×，"
        "op#18 1.10–1.45×，op#19 fp32 8/8 + 16-bit 6/6 正收益成立）。覆盖范围限制：op#18/#19 仅 B200 + 冷 L2"
        "（热 L2 视图 / B300 视图中为空缺）；op#18 的 BS&gt;1 数据仅在 (K512, fp32, N65536)。</p>")
    ops_html = bi(ops_table_en + ops_para_en, ops_table_zh + ops_para_zh)

    # ---- synthetic input generation ----
    synth_table_en = ("<table><tr><th>K</th><th>generator skill</th><th>compress_ratio</th><th>fit source</th>"
        "<th>pre_idx hit-rate (preIdx∩topK)</th></tr>"
        "<tr><td>512</td><td><code>swebench-temporal-synth-v4flash</code></td><td>4</td>"
        "<td>real V4-Flash swe-bench 64K B300 captures — 21 GVR-active layers (even 2..42)</td><td>0.60 (vertical)</td></tr>"
        "<tr><td>1024</td><td><code>swebench-temporal-synth-v4pro</code></td><td>4</td>"
        "<td>real V4-Pro swe-bench 64K B300 captures — 30 GVR-active layers (even 2..60)</td><td>0.60 (vertical)</td></tr>"
        "<tr><td>2048</td><td><code>swebench-temporal-synth</code> (true V3.2, cr=1)</td><td>1</td>"
        "<td>real SWE-Bench L20/L22/L42 beta fits</td><td>0.60 (diagonal)</td></tr></table>")
    synth_table_zh = ("<table><tr><th>K</th><th>生成器 skill</th><th>compress_ratio</th><th>拟合来源</th>"
        "<th>pre_idx 命中率 (preIdx∩topK)</th></tr>"
        "<tr><td>512</td><td><code>swebench-temporal-synth-v4flash</code></td><td>4</td>"
        "<td>真实 V4-Flash swe-bench 64K B300 captures —— 21 个 GVR 活跃层（偶数 2..42）</td><td>0.60（纵向）</td></tr>"
        "<tr><td>1024</td><td><code>swebench-temporal-synth-v4pro</code></td><td>4</td>"
        "<td>真实 V4-Pro swe-bench 64K B300 captures —— 30 个 GVR 活跃层（偶数 2..60）</td><td>0.60（纵向）</td></tr>"
        "<tr><td>2048</td><td><code>swebench-temporal-synth</code>（真 V3.2，cr=1）</td><td>1</td>"
        "<td>真实 SWE-Bench L20/L22/L42 beta 拟合</td><td>0.60（对角）</td></tr></table>")
    synth_en = (
        "<p>Inputs are <b>not uniform random</b> — they reproduce the statistics of real DeepSeek-V4 "
        "indexer captures so that the GVR path (whose Phase-2 threshold search is seeded by temporal "
        "locality via <code>pre_idx</code> — exact output) is exercised realistically. One bundle per (K, dtype, N): "
        "<code>logits[1,&nbsp;N]</code> + <code>pre_idx[1,&nbsp;K]</code>, then replicated across BS rows "
        "(no VarLen) so every op sees the same data.</p>" + synth_table_en +
        "<p><b>logits distribution:</b> temporally-coherent values drawn from 3 fitted <b>Beta</b> "
        "distributions (shallow / moderate / deep buckets by layer-mean magnitude); this sweep uses the "
        "<b>moderate</b> bucket (<code>cfg=beta_moderate</code>) across all cells, in fp32/bf16/fp16.<br>"
        "<b>pre_idx (temporal hint, required by GVR):</b> the prev-step top-K, built as Gaussian noise + "
        "a <b>binary-searched coefficient</b> calibrated to a <b>unified realised hit-rate of 0.6</b> for "
        "every K / dtype / N (<code>synth_data.TARGET_HR=0.6</code>, overriding each skill's per-cfg "
        "default). Propagation depends on compress_ratio: <b>vertical</b> for K=512/1024 (cr=4 — caller "
        "stores prev_topk, kernel offset 0) and <b>diagonal</b> for K=2048 (cr=1 — caller stores "
        "prev_topk−1, kernel preIdxOffset=+1 ⇒ net read = prev_topk). The radix ops ignore pre_idx and "
        "reach exact top-K by per-bit histogram select; GVR also returns <b>exact</b> top-K and uses "
        "pre_idx only to seed its threshold search.<br>"
        "<b>compress_ratio convention:</b> <code>radix_cutedsl</code> takes <code>seq_lens=N</code>; all "
        "other ops take <code>seq_lens=N&times;cr</code> (kernel recomputes N=kv_len/cr).<br>"
        "<b>Determinism:</b> fixed <code>seed=42</code> per (K, dtype, N) — fully reproducible; N grid "
        "4K..256K (cells with N&gt;2K only). The <b>same bundles</b> are used on every machine, so the "
        "B300/B200 comparison isolates hardware.</p>")
    synth_zh = (
        "<p>输入<b>不是均匀随机</b> —— 它们复现真实 DeepSeek-V4 indexer 捕获的统计特征，"
        "使 GVR 路径（其 Phase-2 阈值搜索由 <code>pre_idx</code> 的时间局部性提供种子 —— 输出精确）得到真实测试。"
        "每个 (K, dtype, N) 一个 bundle：<code>logits[1,&nbsp;N]</code> + <code>pre_idx[1,&nbsp;K]</code>，"
        "再跨 BS 行复制（无 VarLen），使每个算子看到相同数据。</p>" + synth_table_zh +
        "<p><b>logits 分布：</b>时间相干的值，取自 3 个拟合的 <b>Beta</b> 分布"
        "（按层均值大小分为 shallow / moderate / deep 三档）；本 sweep 所有 cell 都用 <b>moderate</b> 档"
        "（<code>cfg=beta_moderate</code>），dtype 为 fp32/bf16/fp16。<br>"
        "<b>pre_idx（时间提示，GVR 必需）：</b>上一步的 top-K，构造为高斯噪声 + 一个<b>二分搜索系数</b>，"
        "对每个 K / dtype / N 校准到<b>统一实现命中率 0.6</b>"
        "（<code>synth_data.TARGET_HR=0.6</code>，覆盖每个 skill 的 per-cfg 默认值）。"
        "传播方式取决于 compress_ratio：K=512/1024 为<b>纵向</b>（cr=4 —— 调用方存 prev_topk，kernel 偏移 0），"
        "K=2048 为<b>对角</b>（cr=1 —— 调用方存 prev_topk−1，kernel preIdxOffset=+1 ⇒ 净读取 = prev_topk）。"
        "radix 算子忽略 pre_idx，通过逐位直方图选择达到精确 top-K；GVR 同样返回<b>精确</b> top-K，"
        "仅用 pre_idx 为其阈值搜索提供种子。<br>"
        "<b>compress_ratio 约定：</b><code>radix_cutedsl</code> 取 <code>seq_lens=N</code>；"
        "其余算子取 <code>seq_lens=N&times;cr</code>（kernel 重新计算 N=kv_len/cr）。<br>"
        "<b>确定性：</b>每个 (K, dtype, N) 固定 <code>seed=42</code> —— 完全可复现；"
        "N 网格 4K..256K（仅 N&gt;2K 的 cell）。每台机器使用<b>相同 bundle</b>，故 B300/B200 对比仅隔离硬件。</p>")
    synth_html = bi(synth_en, synth_zh)

    # ---- deterministic reproduction (code blocks stay English) ----
    code_a = (
        "<pre style='background:#0f1419;border:1px solid #2a3340;border-radius:6px;padding:10px;"
        "overflow:auto;white-space:pre;color:#cfe8a0;font-size:12px'>"
        "# K=512 (V4 Flash)\n"
        "from synth_temporal_data import synthesize   # swebench-temporal-synth-v4flash\n"
        "b = synthesize(N=N, BS=1, cfg_name=&quot;beta_moderate&quot;, K=512,  compress_ratio=4, "
        "seed=42, dtype=DT, target_hr=0.6)\n\n"
        "# K=1024 (V4 Pro)\n"
        "from synth_temporal_data import synthesize   # swebench-temporal-synth-v4pro\n"
        "b = synthesize(N=N, BS=1, cfg_name=&quot;beta_moderate&quot;, K=1024, compress_ratio=4, "
        "seed=42, dtype=DT, target_hr=0.6)\n\n"
        "# K=2048 (true V3.2, cr=1) — generator is fp32-only; cast logits afterwards\n"
        "from synth_temporal_data import synthesize   # swebench-temporal-synth\n"
        "b = synthesize(N=N, BS=1, cfg_name=&quot;beta_moderate&quot;, K=2048, seed=42, target_hr=0.6)\n"
        "b[&quot;logits&quot;] = b[&quot;logits&quot;].to(DT)</pre>")
    code_b = (
        "<pre style='background:#0f1419;border:1px solid #2a3340;border-radius:6px;padding:10px;"
        "overflow:auto;white-space:pre;color:#cfe8a0;font-size:12px'>"
        "# K=512  (V4 Flash)\n"
        "python .claude/skills/swebench-temporal-synth-v4flash/src/synth_temporal_data.py \\\n"
        "    --K 512  --compress_ratio 4 --N &lt;N&gt; --cfg beta_moderate --target_hr 0.6 \\\n"
        "    --seed 42 --bs 1 --dtype &lt;DT&gt; --outdir &lt;dir&gt;\n\n"
        "# K=1024 (V4 Pro)\n"
        "python .claude/skills/swebench-temporal-synth-v4pro/src/synth_temporal_data.py \\\n"
        "    --K 1024 --compress_ratio 4 --N &lt;N&gt; --cfg beta_moderate --target_hr 0.6 \\\n"
        "    --seed 42 --bs 1 --dtype &lt;DT&gt; --outdir &lt;dir&gt;\n\n"
        "# K=2048 (true V3.2, cr=1, fp32-only)\n"
        "python .claude/skills/swebench-temporal-synth/src/synth_temporal_data.py \\\n"
        "    --K 2048 --N &lt;N&gt; --cfg beta_moderate --target_hr 0.6 --seed 42 --bs 1 --outdir &lt;dir&gt;</pre>")
    repro_en = (
        "<p>The per-cell logits / pre_idx are generated on-the-fly and held only in RAM during the "
        "sweep (<b>not</b> persisted to disk — only the timing JSONL is saved). Generation is fully "
        "deterministic, so any cell can be reproduced exactly. <b>Fixed conditions for every cell:</b></p>"
        "<ul>"
        "<li><code>seed=42</code>, <code>target_hr=0.6</code>, <code>cfg=beta_moderate</code> (moderate "
        "bucket only), <code>BS=1</code> bundle replicated across rows (no VarLen)</li>"
        "<li>dtype &isin; {fp32, bf16, fp16}; N &isin; {4096, 8192, 16384, 32768, 65536, 131072, 262144}, "
        "restricted to N &gt; 2&middot;K</li>"
        "<li>hit-rate calibration: binary search on the pre_idx Gaussian-noise coefficient, "
        "<code>calib_iters=20</code>, <code>calib_tol=0.005</code>, <code>max_c=5.0</code> (defaults)</li>"
        "<li>K=512 &rarr; <code>compress_ratio=4</code> (vertical); K=1024 &rarr; <code>compress_ratio=4</code> "
        "(vertical); K=2048 &rarr; <code>compress_ratio=1</code> (diagonal, V3.2 generator is fp32-only, "
        "logits cast to dtype afterwards)</li>"
        "</ul>"
        "<p><b>(a) Exact call used by the harness</b> (<code>harness/synth_data.py:get_bundle</code> "
        "&rarr; each skill's <code>synthesize()</code>):</p>" + code_a +
        "<p><b>(b) Equivalent standalone SKILL CLI</b> (writes <code>logits.pt</code> / "
        "<code>preIdx.pt</code> / <code>seq_lens.pt</code> + <code>meta.json</code> to <code>--outdir</code>; "
        "<code>DT</code> &isin; {fp32,bf16,fp16}, <code>N</code> from the grid above):</p>" + code_b +
        "<p><b>(c) Natural-language SKILL prompt</b> (what triggers each generator skill in Claude Code):</p>"
        "<blockquote style='border-left:3px solid #76b900;margin:6px 0;padding:4px 12px;color:#bcd'>"
        "&ldquo;Use the <code>swebench-temporal-synth-v4{flash|pro}</code> skill (or "
        "<code>swebench-temporal-synth</code> for K=2048) to generate V4 "
        "{Flash K=512 | Pro K=1024 | V3.2 K=2048} indexer top-K synthetic data: "
        "<code>cfg=beta_moderate</code>, <code>target_hr=0.6</code>, <code>seed=42</code>, "
        "<code>compress_ratio=4</code> (K=2048 &rarr; cr=1), <code>BS=1</code>, "
        "<code>dtype</code> in {fp32,bf16,fp16}, for each N in {4K,8K,16K,32K,64K,128K,256K} with "
        "N&gt;2&middot;K. Emit <code>logits</code>, <code>preIdx</code>, <code>seq_lens</code> + "
        "<code>meta.json</code> (report the realised <code>kernel_side_hit_rate</code> and "
        "<code>calibrated_noise_c</code>).&rdquo;</blockquote>"
        "<p style='color:#9aa'>Generator sources are committed under "
        "<code>.claude/skills/&lt;skill&gt;/src/synth_temporal_data.py</code>; the unified "
        "<code>target_hr=0.6</code> override lives in <code>harness/synth_data.py:TARGET_HR</code>. "
        "See <code>report/PREIDX_SEMANTICS.md</code> for the vertical/diagonal pre_idx contract.</p>")
    repro_zh = (
        "<p>每个 cell 的 logits / pre_idx 在 sweep 过程中即时生成、仅驻留 RAM"
        "（<b>不</b>落盘 —— 只保存计时 JSONL）。生成完全确定，故任意 cell 都可精确复现。"
        "<b>每个 cell 的固定条件：</b></p>"
        "<ul>"
        "<li><code>seed=42</code>，<code>target_hr=0.6</code>，<code>cfg=beta_moderate</code>（仅 moderate 档），"
        "<code>BS=1</code> bundle 跨行复制（无 VarLen）</li>"
        "<li>dtype &isin; {fp32, bf16, fp16}；N &isin; {4096, 8192, 16384, 32768, 65536, 131072, 262144}，"
        "限制为 N &gt; 2&middot;K</li>"
        "<li>命中率校准：对 pre_idx 高斯噪声系数做二分搜索，"
        "<code>calib_iters=20</code>，<code>calib_tol=0.005</code>，<code>max_c=5.0</code>（默认值）</li>"
        "<li>K=512 &rarr; <code>compress_ratio=4</code>（纵向）；K=1024 &rarr; <code>compress_ratio=4</code>"
        "（纵向）；K=2048 &rarr; <code>compress_ratio=1</code>（对角，V3.2 生成器仅 fp32，logits 之后再转 dtype）</li>"
        "</ul>"
        "<p><b>(a) harness 使用的精确调用</b>（<code>harness/synth_data.py:get_bundle</code> "
        "&rarr; 每个 skill 的 <code>synthesize()</code>）：</p>" + code_a +
        "<p><b>(b) 等价的独立 SKILL CLI</b>（向 <code>--outdir</code> 写 <code>logits.pt</code> / "
        "<code>preIdx.pt</code> / <code>seq_lens.pt</code> + <code>meta.json</code>；"
        "<code>DT</code> &isin; {fp32,bf16,fp16}，<code>N</code> 取自上方网格）：</p>" + code_b +
        "<p><b>(c) 自然语言 SKILL 提示</b>（在 Claude Code 中触发各生成器 skill 的说法）：</p>"
        "<blockquote style='border-left:3px solid #76b900;margin:6px 0;padding:4px 12px;color:#bcd'>"
        "“用 <code>swebench-temporal-synth-v4{flash|pro}</code> skill（K=2048 用 "
        "<code>swebench-temporal-synth</code>）生成 V4 "
        "{Flash K=512 | Pro K=1024 | V3.2 K=2048} indexer top-K 合成数据："
        "<code>cfg=beta_moderate</code>，<code>target_hr=0.6</code>，<code>seed=42</code>，"
        "<code>compress_ratio=4</code>（K=2048 &rarr; cr=1），<code>BS=1</code>，"
        "<code>dtype</code> 取 {fp32,bf16,fp16}，对每个 N &isin; {4K,8K,16K,32K,64K,128K,256K} 且 "
        "N&gt;2&middot;K。输出 <code>logits</code>、<code>preIdx</code>、<code>seq_lens</code> + "
        "<code>meta.json</code>（报告实现的 <code>kernel_side_hit_rate</code> 与 "
        "<code>calibrated_noise_c</code>）。”</blockquote>"
        "<p style='color:#9aa'>生成器源码提交于 "
        "<code>.claude/skills/&lt;skill&gt;/src/synth_temporal_data.py</code>；统一的 "
        "<code>target_hr=0.6</code> 覆盖位于 <code>harness/synth_data.py:TARGET_HR</code>。"
        "纵向/对角 pre_idx 约定见 <code>report/PREIDX_SEMANTICS.md</code>。</p>")
    repro_html = bi(repro_en, repro_zh)

    # ---- algorithmic comparison (SGLang) ----
    algo_measured_en = (
        "<table><tr><th>regime</th><th>SGLang</th><th>GVR(CUDA)</th><th>GVR rank-scatter P4</th>"
        "<th>Radix single (cuteDSL)</th><th>Radix multi (CUDA)</th><th>verdict</th></tr>"
        "<tr><td>K512 N=4K BS=1 (short)</td><td><b>12.4</b></td><td>15.9</td><td>14.9</td><td>13.5</td>"
        "<td>43.4</td><td><b>fastest</b> — edges out smem-resident radix-single (13.5)</td></tr>"
        "<tr><td>K512 N=65K BS=1 (mid)</td><td>29.0</td><td>28.7</td><td><b>22.8</b></td>"
        "<td>&mdash; (smem cap)</td><td>47.6</td><td>rank-scatter (hint) leads; SGLang &asymp; GVR-CUDA; radix-single can't run</td></tr>"
        "<tr><td>K512 N=256K BS=1 (long)</td><td>80.4</td><td>66.9</td><td><b>39.8</b></td>"
        "<td>&mdash;</td><td>53.8</td><td>loses — single-CTA stream grows ~linearly; hint &amp; multi-CTA win</td></tr>"
        "<tr><td>K512 N=65K BS=512 (high batch)</td><td><b>82.0</b></td><td>103.3</td><td>99.6</td>"
        "<td>&mdash;</td><td>189.5</td><td><b>fastest</b> — lean 1-CTA/row fills the GPU at high BS</td></tr>"
        "<tr><td>K512 N=65K BS=2048</td><td><b>249.3</b></td><td>260.5</td><td>263.0</td><td>&mdash;</td>"
        "<td>655.1</td><td><b>fastest</b> — 2.6&times; vs radix-multi</td></tr></table>")
    algo_measured_zh = (
        "<table><tr><th>场景</th><th>SGLang</th><th>GVR(CUDA)</th><th>GVR rank-scatter P4</th>"
        "<th>Radix single (cuteDSL)</th><th>Radix multi (CUDA)</th><th>结论</th></tr>"
        "<tr><td>K512 N=4K BS=1（短）</td><td><b>12.4</b></td><td>15.9</td><td>14.9</td><td>13.5</td>"
        "<td>43.4</td><td><b>最快</b> —— 险胜 smem 驻留的 radix-single（13.5）</td></tr>"
        "<tr><td>K512 N=65K BS=1（中）</td><td>29.0</td><td>28.7</td><td><b>22.8</b></td>"
        "<td>&mdash;（smem 上限）</td><td>47.6</td><td>rank-scatter（带提示）领先；SGLang &asymp; GVR-CUDA；radix-single 无法运行</td></tr>"
        "<tr><td>K512 N=256K BS=1（长）</td><td>80.4</td><td>66.9</td><td><b>39.8</b></td>"
        "<td>&mdash;</td><td>53.8</td><td>落败 —— 单 CTA 流式随 N 近线性增长；提示与多 CTA 胜出</td></tr>"
        "<tr><td>K512 N=65K BS=512（高批）</td><td><b>82.0</b></td><td>103.3</td><td>99.6</td>"
        "<td>&mdash;</td><td>189.5</td><td><b>最快</b> —— 精简的 1 CTA/行在高 BS 下填满 GPU</td></tr>"
        "<tr><td>K512 N=65K BS=2048</td><td><b>249.3</b></td><td>260.5</td><td>263.0</td><td>&mdash;</td>"
        "<td>655.1</td><td><b>最快</b> —— 相对 radix-multi 2.6&times;</td></tr></table>")
    algo_en = (
        "<p>SGLang's <code>device::top512::StreamingTopK&lt;K&gt;</code> is a <b>single-CTA-per-row, "
        "TMA double-buffered streaming bucket-select</b>: stream the row once through a 4096-bin "
        "<b>fp16 coarse histogram</b>, walk bins high&rarr;low to find the rank-K boundary bin, then do an "
        "exact 4-round 8-bit radix refine on the &le;<code>kMaxTies=1024</code> boundary candidates. It is "
        "<b>exact and hint-free</b> — unlike GVR it needs <b>no <code>pre_idx</code></b> and has no "
        "threshold-search convergence behaviour; unlike radix-single-CTA it streams, so it has <b>no "
        "smem-capacity ceiling</b>. Constraints: <b>fp32-only</b> (TMA <code>float</code>) and "
        "<b>K&le;1024</b> (the tie buffer is sized <code>kMaxTies=kMaxTopK=1024</code>).</p>"
        "<h4 style='color:#9ecb3a;margin:14px 0 4px'>How it maps onto the report's three algorithm families</h4>"
        "<table><tr><th>vs family</th><th>StreamingTopK advantage</th><th>StreamingTopK disadvantage</th></tr>"
        "<tr><td><b>GVR</b> (CUDA / cuteDSL / rank-scatter / cluster / op#8)</td>"
        "<td>Hint-free &amp; unconditionally exact: no <code>pre_idx</code> to build/propagate, no secant "
        "iteration count or P3 over-collect, no real-data undershoot risk. Simpler launch, fewer phases.</td>"
        "<td>GVR's <code>pre_idx</code> lets it <b>skip most of N</b> (scan only a candidate band), so at "
        "<b>long context</b> GVR pays far less than a full single-CTA stream of all N.</td></tr>"
        "<tr><td><b>Radix single-CTA</b> (CUDA / cuteDSL-single)</td>"
        "<td>No smem ceiling — StreamingTopK runs the <b>entire</b> N grid (radix-single dies above "
        "~57K fp32 / one CTA's smem). TMA streaming hides memory latency that the smem-resident radix "
        "cannot once the row no longer fits.</td>"
        "<td>When the row <b>does</b> fit one CTA's smem (small N), radix-single is the leanest path and "
        "edges StreamingTopK out (it never re-reads from global).</td></tr>"
        "<tr><td><b>Radix multi-CTA</b> (CUDA / cuteDSL-multi) &amp; GVR cluster/op#8</td>"
        "<td>Far lower fixed overhead at small N and much better BS scaling — one lean CTA/row vs a "
        "multi-CTA global-histogram merge / cluster barrier per row.</td>"
        "<td>At <b>long context + low BS</b> the multi-CTA kernels split one row across many CTAs and stay "
        "~flat in N, while StreamingTopK's lone CTA streams serially and grows ~linearly with N &mdash; so "
        "they overtake it once one CTA can no longer hide the stream.</td></tr></table>"
        "<h4 style='color:#9ecb3a;margin:14px 0 4px'>Measured position on this B300 (cold-L2, fp32, "
        "<code>beta_moderate</code>; median &micro;s) &mdash; B200 reproduces within ~3% (hardware-invariant)</h4>"
        + algo_measured_en +
        "<h4 style='color:#9ecb3a;margin:14px 0 4px'>Verdict</h4>"
        "<ul>"
        "<li><b>Sweet spot:</b> small&ndash;to&ndash;medium N (&le;~65K) and <b>high batch</b> — there "
        "StreamingTopK is the fastest or co-fastest kernel measured (on B300 it is the fastest even at the "
        "shortest N=4K/BS=1 cell, edging out the smem-resident radix-single), and it does it <b>without a "
        "<code>pre_idx</code> hint</b>, which is its key structural advantage over GVR.</li>"
        "<li><b>Where it loses:</b> <b>long context at low BS</b> (N&ge;128K, BS=1). GVR's hint lets it "
        "skip most of N (rank-scatter P4 = 39.8&micro;s vs 80.4&micro;s at 256K), and multi-CTA radix / GVR "
        "cluster split the row across SMs and stay flat — the lone streaming CTA cannot.</li>"
        "<li><b>Envelope limits:</b> fp32-only and K&le;1024 — so it is <b>not</b> applicable to the V3.2 "
        "K=2048 real-data comparison (and is non-exact if forced there), and cannot serve the bf16/fp16 "
        "cells the in-tree ops cover.</li>"
        "<li><b>Net:</b> an excellent <b>hint-free, exact, no-smem-cap</b> reference that validates the "
        "in-tree kernels in the short/mid/high-BS regime, but the production GVR path retains a real "
        "long-context edge precisely because it exploits <code>pre_idx</code> temporal locality.</li>"
        "</ul>"
        "<p style='color:#9aa'>Numbers are this-B300 cold-L2 medians (same synthetic bundles, seed=42, "
        "unified preIdx hit-rate 0.6, same prebuilt SASS as B200; op is set-exact 512/512 &amp; 1024/1024 on "
        "sm_103). The identical B200 sweep reproduces every cell within ~3% — the GVR(CUDA)-vs-SGLang verdict "
        "is hardware-invariant (0.84&times; median on both; SGLang faster in 138/182 B300, 139/182 B200 cells). "
        "Toggle <span style='color:#d62728'>SGLang StreamingTopK</span> "
        "in the operator selector below to overlay its latency/speedup curves on every chart.</p>")
    algo_zh = (
        "<p>SGLang 的 <code>device::top512::StreamingTopK&lt;K&gt;</code> 是一个<b>每行单 CTA、"
        "TMA 双缓冲流式桶选择</b>：将整行流式经过一个 4096 桶的 <b>fp16 粗直方图</b>，"
        "从高到低遍历桶找到第 K 名的边界桶，再对 &le;<code>kMaxTies=1024</code> 个边界候选做精确的 4 轮 8 位 radix 细化。"
        "它<b>精确且无需提示</b> —— 与 GVR 不同，它<b>不需要 <code>pre_idx</code></b>，没有阈值搜索收敛行为；"
        "与 radix-single-CTA 不同，它是流式的，所以<b>没有 smem 容量上限</b>。"
        "约束：<b>仅 fp32</b>（TMA <code>float</code>）与 <b>K&le;1024</b>"
        "（tie 缓冲区大小为 <code>kMaxTies=kMaxTopK=1024</code>）。</p>"
        "<h4 style='color:#9ecb3a;margin:14px 0 4px'>它如何映射到报告的三个算法族</h4>"
        "<table><tr><th>对比族</th><th>StreamingTopK 优势</th><th>StreamingTopK 劣势</th></tr>"
        "<tr><td><b>GVR</b>（CUDA / cuteDSL / rank-scatter / cluster / op#8）</td>"
        "<td>无提示且无条件精确：无需构造/传播 <code>pre_idx</code>，没有割线迭代次数或 P3 过采集，没有真实数据欠采风险。"
        "启动更简单、阶段更少。</td>"
        "<td>GVR 的 <code>pre_idx</code> 让它能<b>跳过大部分 N</b>（只扫描候选带），故在<b>长上下文</b>下 "
        "GVR 远比对全 N 做单 CTA 流式便宜。</td></tr>"
        "<tr><td><b>Radix single-CTA</b>（CUDA / cuteDSL-single）</td>"
        "<td>无 smem 上限 —— StreamingTopK 可跑<b>整个</b> N 网格（radix-single 在约 57K fp32 / 单 CTA smem 以上崩溃）。"
        "TMA 流式隐藏了行不再放得下时 smem 驻留 radix 无法隐藏的访存延迟。</td>"
        "<td>当行<b>确实</b>能放进单 CTA smem（小 N）时，radix-single 是最精简路径，险胜 StreamingTopK"
        "（它从不重读全局）。</td></tr>"
        "<tr><td><b>Radix multi-CTA</b>（CUDA / cuteDSL-multi）与 GVR cluster/op#8</td>"
        "<td>小 N 时固定开销低得多，BS 扩展性也好得多 —— 每行一个精简 CTA，相对每行的多 CTA 全局直方图归并 / cluster 屏障。</td>"
        "<td>在<b>长上下文 + 低 BS</b> 下，多 CTA kernel 把一行拆到许多 CTA 上、随 N 近似持平，"
        "而 StreamingTopK 的单 CTA 串行流式随 N 近线性增长 —— 故当单 CTA 不再能隐藏流式时它们会反超。</td></tr></table>"
        "<h4 style='color:#9ecb3a;margin:14px 0 4px'>本 B300 上的实测位置（冷 L2，fp32，"
        "<code>beta_moderate</code>；中位 &micro;s）—— B200 在约 3% 内复现（硬件无关）</h4>"
        + algo_measured_zh +
        "<h4 style='color:#9ecb3a;margin:14px 0 4px'>结论</h4>"
        "<ul>"
        "<li><b>最佳区间：</b>中小 N（&le;约 65K）与<b>高批</b> —— 此处 StreamingTopK 是实测最快或并列最快的 kernel"
        "（在 B300 上即便最短的 N=4K/BS=1 cell 它也最快，险胜 smem 驻留的 radix-single），"
        "而且<b>不需要 <code>pre_idx</code> 提示</b>，这是它相对 GVR 的关键结构优势。</li>"
        "<li><b>落败之处：</b><b>低 BS 的长上下文</b>（N&ge;128K，BS=1）。GVR 的提示让它跳过大部分 N"
        "（rank-scatter P4 在 256K 为 39.8&micro;s vs 80.4&micro;s），多 CTA radix / GVR cluster 把行拆到各 SM 上保持持平 ——"
        "单个流式 CTA 做不到。</li>"
        "<li><b>范围限制：</b>仅 fp32 且 K&le;1024 —— 故<b>不</b>适用于 V3.2 K=2048 真实数据对比"
        "（强行使用则非精确），也无法服务树内算子覆盖的 bf16/fp16 cell。</li>"
        "<li><b>总体：</b>一个优秀的<b>无提示、精确、无 smem 上限</b>参考，"
        "在短/中/高 BS 区间验证了树内 kernel，但生产 GVR 路径在长上下文仍保有真实优势，正因为它利用了 "
        "<code>pre_idx</code> 时间局部性。</li>"
        "</ul>"
        "<p style='color:#9aa'>数字为本 B300 冷 L2 中位（相同合成 bundle，seed=42，统一 preIdx 命中率 0.6，"
        "与 B200 相同的预编译 SASS；该算子在 sm_103 上集合精确 512/512 与 1024/1024）。"
        "完全相同的 B200 sweep 在约 3% 内复现每个 cell —— GVR(CUDA) 对 SGLang 的结论硬件无关"
        "（两机中位均 0.84&times;；SGLang 更快的 cell 数：B300 138/182，B200 139/182）。"
        "点选下方算子选择器中的 <span style='color:#d62728'>SGLang StreamingTopK</span> "
        "可在每张图上叠加其延迟/加速比曲线。</p>")
    algo_html = bi(algo_en, algo_zh)

    ana = analysis(mdata)
    # Stale-data banner: shown only while any machine's results/.pending_rerun exists.
    pending = (bucket / "results" / ".pending_rerun").exists()
    if pending:
        banner_en = ("<h3 style=\"color:#ff6b6b\">⚠️ Data pending full re-run</h3>"
            "<p>A <code>results/.pending_rerun</code> flag is present — some numbers shown may be stale. "
            "See <code>B300_RERUN_PROMPT.md</code> / <code>PREIDX_SEMANTICS.md</code>.</p>")
        banner_zh = ("<h3 style=\"color:#ff6b6b\">⚠️ 数据待全量重跑</h3>"
            "<p>存在 <code>results/.pending_rerun</code> 标志 —— 部分显示数字可能过时。"
            "参见 <code>B300_RERUN_PROMPT.md</code> / <code>PREIDX_SEMANTICS.md</code>。</p>")
        banner_html = ("<div class=\"card\" style=\"border-color:#e84855;background:#2a1a1d\">"
                       + bi(banner_en, banner_zh) + "</div>")
    else:
        banner_html = ""

    # nsys validation table (cold-L2 spot-check, B300), if present
    nsys_note_en = ("<p>Per-op pure GPU kernel µs from nsys <code>cuda_gpu_kern_sum</code> (L2-evict kernel excluded) "
        "vs the sweep's event-timed cold-L2 µs. Event timing is a consistent slight over-estimate "
        "(includes graph-launch + single-replay overhead); op ordering preserved.</p>")
    nsys_note_zh = ("<p>各算子的纯 GPU kernel µs 取自 nsys <code>cuda_gpu_kern_sum</code>（已排除 L2 清空 kernel），"
        "与 sweep 的 event 计时冷 L2 µs 对比。event 计时是一致的轻微高估"
        "（含 graph 启动 + 单次 replay 开销）；算子排序保持不变。</p>")
    nsys_note = bi(nsys_note_en, nsys_note_zh)
    nsys_csv = here / "nsys_vs_event.csv"
    nsys_html = bi("<p>(nsys spot-check pending)</p>", "<p>（nsys 抽查待补）</p>")
    if nsys_csv.exists():
        rr = [r for r in csv.DictReader(nsys_csv.open()) if r.get("nsys/event")]
        if rr:
            def nsys_table(hdr, foot_fmt):
                t = ("<table><tr>" + "".join(f"<th>{h}</th>" for h in hdr) + "</tr>")
                for r in sorted(rr, key=lambda x: (int(x["K"]), x["dtype"], int(x["N"]), int(x["BS"]), x["op"])):
                    t += (f"<tr><td>{OP_LABEL.get(r['op'], r['op'])}</td><td>{r['K']}</td><td>{r['dtype']}</td>"
                          f"<td>{r['N']}</td><td>{r['BS']}</td><td>{r['nsys_kernel_us']}</td>"
                          f"<td>{r['event_cold_us']}</td><td>{r['nsys/event']}×</td></tr>")
                rs = [float(r["nsys/event"]) for r in rr]
                t += "</table>" + foot_fmt(st.median(rs), min(rs), max(rs), len(rs))
                return t
            en = nsys_table(
                ["op", "K", "dtype", "N", "BS", "nsys µs", "event µs", "nsys/event"],
                lambda m, lo, hi, n: (f"<p>nsys/event ratio: median <b>{m:.3f}</b>, "
                                      f"range [{lo:.3f}, {hi:.3f}], n={n} — event sweep numbers are "
                                      "accurate kernel-time proxies (slightly conservative).</p>"))
            zh = nsys_table(
                ["算子", "K", "dtype", "N", "BS", "nsys µs", "event µs", "nsys/event"],
                lambda m, lo, hi, n: (f"<p>nsys/event 比值：中位 <b>{m:.3f}</b>，"
                                      f"范围 [{lo:.3f}, {hi:.3f}]，n={n} —— event sweep 数字是"
                                      "准确的 kernel 时间代理（略偏保守）。</p>"))
            nsys_html = bi(en, zh)

    # GVR P2/P4 iteration-count supplement (BS=1). Regenerates the standalone
    # interactive page and embeds a static summary + link. Graceful no-op if the
    # sweep hasn't been run (report/iters_data.csv absent).
    iters_html = bi("<p>(iteration-count sweep pending — run <code>python harness/sweep_iters.py</code>)</p>",
                    "<p>（迭代次数 sweep 待补 —— 运行 <code>python harness/sweep_iters.py</code>）</p>")
    iters_csv = here / "iters_data.csv"
    if iters_csv.exists():
        try:
            import gen_iters_report as _gir
            _df = _gir.pd.read_csv(iters_csv)
            _gir.main()  # (re)write iters_report.html
            iters_html = _gir.fragment_html(_df)
        except Exception as e:  # pragma: no cover - report is best-effort
            iters_html = f"<p>(iters report build failed: {type(e).__name__}: {e})</p>"

    # Input-data CCDF figure (generated by gen_input_ccdf.py from the same
    # synth_data bundles the sweep consumes). Inline the SVG so the report stays
    # a single portable file; graceful note if the artifact hasn't been built.
    ccdf_svg = here / "input_ccdf.svg"
    if ccdf_svg.exists():
        svg = ccdf_svg.read_text()
        svg = svg[svg.find("<svg"):]  # drop the XML/DOCTYPE preamble for clean inlining
        svg_box = f'<div style="background:#0f1419;border:1px solid #2a3340;border-radius:8px;padding:8px">{svg}</div>'
        ccdf_en = (
            "<p>CCDF <b>P[ logit &ge; x ]</b> of the actual logit arrays fed to every operator "
            "(<code>harness/synth_data.get_bundle</code>, <code>beta_moderate</code>, "
            "N=65536, fp32). The marginal logit distribution is N-invariant, so one representative "
            "N captures it; bf16/fp16 only quantise these values, not reshape them. <b>Left:</b> "
            "full-range CCDF (linear) — the overall <code>beta_moderate</code> shape per generator. "
            "<b>Right:</b> the same CCDF on a log y-axis, exposing the rare upper tail that top-K "
            "actually selects; each dot marks the <b>top-K selection boundary</b> (the K-th largest "
            "logit, at CCDF=K/N) — the operators keep everything to its right. The three generators "
            "(V4-Flash K=512, V4-Pro K=1024, V3.2 K=2048) differ in their fitted moderate-bucket "
            "Beta, so the curves are distinct but all heavy-left (most mass at low logits).</p>"
            + svg_box +
            '<p style="color:#9aa;font-size:12px;margin-top:8px">Regenerate: '
            '<code>python report/gen_input_ccdf.py</code> → <code>report/input_ccdf.svg</code> + '
            '<a href="input_ccdf.csv" style="color:#76b900">input_ccdf.csv</a> (decimated curve data).</p>')
        ccdf_zh = (
            "<p>喂给每个算子的真实 logit 数组的 CCDF <b>P[ logit &ge; x ]</b>"
            "（<code>harness/synth_data.get_bundle</code>，<code>beta_moderate</code>，"
            "N=65536，fp32）。边缘 logit 分布与 N 无关，故一个代表性 N 即可刻画；"
            "bf16/fp16 只量化这些值，不改变形状。<b>左：</b>全量程 CCDF（线性）—— 每个生成器整体的 "
            "<code>beta_moderate</code> 形状。<b>右：</b>同一 CCDF 取对数 y 轴，"
            "暴露 top-K 实际选中的稀有上尾；每个点标记 <b>top-K 选择边界</b>"
            "（第 K 大的 logit，位于 CCDF=K/N）—— 算子保留其右侧的全部。三个生成器"
            "（V4-Flash K=512、V4-Pro K=1024、V3.2 K=2048）拟合的 moderate 档 Beta 不同，"
            "故曲线各异但都重左（大部分质量在低 logit）。</p>"
            + svg_box +
            '<p style="color:#9aa;font-size:12px;margin-top:8px">重新生成：'
            '<code>python report/gen_input_ccdf.py</code> → <code>report/input_ccdf.svg</code> + '
            '<a href="input_ccdf.csv" style="color:#76b900">input_ccdf.csv</a>（抽稀后的曲线数据）。</p>')
        input_ccdf_html = bi(ccdf_en, ccdf_zh)
    else:
        input_ccdf_html = bi('<p>(input CCDF pending — run <code>python report/gen_input_ccdf.py</code> to build '
                             '<code>report/input_ccdf.svg</code>)</p>',
                             '<p>（输入 CCDF 待补 —— 运行 <code>python report/gen_input_ccdf.py</code> 生成 '
                             '<code>report/input_ccdf.svg</code>）</p>')

    # Build per-hw DATA + labels for the JS.
    data_json = {lab: {"seq": cells_to_json(seq, "seqlen"), "bs": cells_to_json(bs, "bs")}
                 for lab, (seq, bs) in mdata.items()}
    # Analysis <li> pairs (bilingual).
    if ana:
        analysis_html = "".join(f'<li class="i18n-en">{en}</li><li class="i18n-zh">{zh}</li>' for en, zh in ana)
    else:
        analysis_html = bi("(awaiting data)", "（等待数据）", tag="li")

    # Use token replacement (NOT str.format) — the CSS/JS contain many literal
    # braces that str.format would misparse.
    repl = {
        "{intro}": intro_html,
        "{env}": env, "{ops}": ops_html, "{algo}": algo_html, "{synth}": synth_html, "{repro}": repro_html,
        "{input_ccdf}": input_ccdf_html,
        "{nsys_note}": nsys_note, "{nsys}": nsys_html, "{banner}": banner_html, "{iters}": iters_html,
        "{compare_block}": build_compare_block(mdata),
        "{analysis}": analysis_html,
        "{ops_json}": json.dumps(OPS), "{opl_json}": json.dumps(OP_LABEL), "{baseline}": BASELINE,
        "{col_json}": json.dumps(COL), "{data_json}": json.dumps(data_json),
        "{labels_json}": json.dumps(list(mdata.keys())), "{dash_json}": json.dumps(DASH),
    }
    # Collapse the doubled JS braces ({{ }}) to single (template was authored for
    # str.format); placeholders are single-brace and untouched by this.
    html = HTML.replace("{{", "{").replace("}}", "}")
    for k, v in repl.items():
        html = html.replace(k, v)
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  (machines={list(mdata.keys())})")
    for lab, (seq, bs) in mdata.items():
        print(f"  {lab}: {len(seq)} seqlen, {len(bs)} bs cells")
    for en, _zh in ana:
        print("  -", en)


if __name__ == "__main__":
    main()
