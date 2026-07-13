#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 — add chapter "9. Real captured-data" to REPORT.html.

Data = results_b200_op22real/realcap_sweep/results.jsonl (parse_op22real.py
output): the 13 report arms measured on PRODUCTION-CAPTURED indexer logits
(V4 Flash SWE-100K / V4 Pro SWE-64K K1024 / V3.2 SWE-64K; every layer's
LAST decode step, preIdx = previous step's top-K), layers x BS 1..2048,
nsys cold/warm-L2, single node (no anchor transfer).

APPEND-ON-TOP LAST-WRITER: patches the CURRENT REPORT.html idempotently —
the chapter lives between REALCAP sentinel comments and is dropped-then-
re-inserted, so re-running after new parses is safe. Prior updaters
(update_report_op28.py etc.) do not touch this block; if one is re-run it
leaves this chapter intact (it only edits the §1-2 D blob).

Also writes op22real_layer_data.csv (BS=1 per-layer) and
op22real_bs_data.csv (per-BS layer-geomeans + ratios).

Usage: python3 update_report_realcap.py [<out_root>]
"""
import json
import math
import shutil
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    HERE.parents[0] / "results_b200_op22real"
RESULTS = OUT_ROOT / "realcap_sweep" / "results.jsonl"
REPORT = HERE / "REPORT.html"
BAK = HERE / f"REPORT.html.bak_pre_realcap_{date.today():%Y%m%d}"

BEGIN = "<!-- REALCAP-CHAPTER-BEGIN (update_report_realcap.py) -->"
END = "<!-- REALCAP-CHAPTER-END -->"

BASE = "gvr_cutedsl"
ARMS = ["gvr_cutedsl", "op21_legacy", "op27_hls", "gvr_multicta_cutedsl",
        "op26_1cta", "op26_mc", "op26_r0auto",
        "radix_cutedsl", "radix_single_cuda", "radix_multi_cuda",
        "sglang_streaming", "sglang_v2", "flashinfer_topk", "gvr29_hbe"]
COL = {"gvr_cutedsl": "#b3e05a", "op21_legacy": "#c9a227",
       "op27_hls": "#c77dff", "gvr_multicta_cutedsl": "#2ec4b6",
       "op26_1cta": "#9b5de5", "op26_mc": "#f15bb5",
       "op26_r0auto": "#00bbf9", "radix_cutedsl": "#4ea8de",
       "radix_single_cuda": "#ff8c42", "radix_multi_cuda": "#e84855",
       "sglang_streaming": "#d62728", "sglang_v2": "#ff5d8f",
       "flashinfer_topk": "#8ac926", "gvr29_hbe": "#06d6a0"}
SHORT = {"gvr_cutedsl": "GVR (cuteDSL)",
         "op21_legacy": "GVR op#21 ms_auto (pre-HLS)",
         "op27_hls": "GVR op#21 ms_auto (HLS-op27, HEAD)",
         "gvr_multicta_cutedsl": "GVR multi-CTA (cuteDSL, PR#15198)",
         "op26_1cta": "GVR op#26 logP2+RS (single-CTA)",
         "op26_mc": "GVR op#26 logP2 (multi-CTA, PR#15198)",
         "op26_r0auto": "GVR op#26 R0 (auto 1CTA/MC dispatch)",
         "radix_cutedsl": "Radix (cuteDSL)",
         "radix_single_cuda": "Radix single-CTA (CUDA)",
         "radix_multi_cuda": "Radix multi-CTA (CUDA)",
         "sglang_streaming": "SGLang StreamingTopK",
         "sglang_v2": "SGLang v2 top-K (main 2026-07)",
         "flashinfer_topk": "FlashInfer top_k (0.6.11)",
         "gvr29_hbe": "GVR op#29 HBE-noB (1-pass sample-column)"}
MODELS = ["flash", "pro", "v32"]
MODEL_LABEL = {
    "flash": ("V4 Flash · K=512 · SWE-100K", "V4 Flash · K=512 · SWE-100K"),
    "pro": ("V4 Pro · K=1024 · SWE-64K", "V4 Pro · K=1024 · SWE-64K"),
    "v32": ("V3.2 · K=2048 · SWE-64K", "V3.2 · K=2048 · SWE-64K")}
DTS = ["fp32", "bf16", "fp16"]
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def bi(en, zh, tag="div"):
    return (f'<{tag} class="i18n-en">{en}</{tag}>'
            f'<{tag} class="i18n-zh">{zh}</{tag}>')


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load():
    rows = []
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["op"] not in ARMS:
            continue
        # the gvr29 supplement batches (results_*o29.jsonl) carry a
        # co-located gvr_cutedsl anchor for scale-drift verification only —
        # drop it here so the baseline stays the original fleet's rows.
        if "o29" in r.get("batch", "") and r["op"] == BASE:
            continue
        rows.append(r)
    ok = [r for r in rows if "error" not in r and r.get("us_cold")
          and r.get("us_warm")]
    err = [r for r in rows if "error" in r]
    return rows, ok, err


def cell_map(ok):
    """(model, dtype, arm, layer, BS) -> rec"""
    m = {}
    for r in ok:
        m[(r["model"], r["dtype"], r["op"], r["layer"], r["BS"])] = r
    return m


def layer_gm(cm, model, dt, arm, BS, metric):
    xs = [cm[k][metric] for k in cm
          if k[0] == model and k[1] == dt and k[2] == arm and k[4] == BS
          and metric in cm[k]]
    return gm(xs)


def ratio_cells(cm, model, dt, arm, bss, metric):
    """gm over (layer, BS in bss) of per-cell arm/BASE."""
    rs = []
    for k, r in cm.items():
        if (k[0], k[1], k[2]) != (model, dt, arm) or k[4] not in bss:
            continue
        b = cm.get((model, dt, BASE, k[3], k[4]))
        if b and metric in r and metric in b and b[metric] > 0:
            rs.append(r[metric] / b[metric])
    return gm(rs)


def fmt_r(r, invert_good=False):
    if r is None:
        return "—"
    cls = ""
    if abs(r - 1) >= 0.02:
        good = (r < 1)
        cls = "good" if good else "bad"
    return f'<span class="{cls}">{r:.3f}</span>' if cls else f"{r:.3f}"


def meta_table(ok):
    """model -> dict(N, Npad, K, cr, s_last, layers, hr min/med/max per dt)"""
    out = {}
    for m in MODELS:
        rs = [r for r in ok if r["model"] == m]
        if not rs:
            continue
        layers = sorted({r["layer"] for r in rs})
        d = {"N": rs[0]["N"], "Npad": rs[0].get("Npad", rs[0]["N"]),
             "K": rs[0]["K"], "cr": rs[0]["cr"], "s_last": rs[0]["s_last"],
             "layers": layers, "hr": {}}
        for dt in DTS:
            hs = sorted({r["hit_rate"] for r in rs if r["dtype"] == dt})
            if hs:
                d["hr"][dt] = (hs[0], hs[len(hs) // 2], hs[-1])
        out[m] = d
    return out


def exactness_summary(rows):
    """per (model, arm): counts + max vdiff + min recall + sum n_neg at BS=1."""
    agg = {}
    for r in rows:
        if r.get("BS") != 1 or "vdiff" not in r:
            continue
        k = (r["model"], r["op"])
        a = agg.setdefault(k, {"n": 0, "vd": 0.0, "rc": 1.0, "nn": 0})
        a["n"] += 1
        a["vd"] = max(a["vd"], r["vdiff"])
        a["rc"] = min(a["rc"], r["recall"])
        a["nn"] += r["n_neg"]
    return agg


def js_records(ok):
    """Compact arrays [mi, di, oi, L, B, cold, warm] with index maps —
    ~55% smaller than keyed objects at this record count (23.6k)."""
    recs = []
    for r in ok:
        recs.append([MODELS.index(r["model"]), DTS.index(r["dtype"]),
                     ARMS.index(r["op"]), r["layer"], r["BS"],
                     round(r["us_cold"], 2), round(r["us_warm"], 2)])
    return recs


def write_csvs(ok, cm):
    p1 = HERE / "op22real_layer_data.csv"
    with open(p1, "w") as f:
        f.write("model,K,dtype,layer,hit_rate,op,us_cold,us_warm,"
                "vdiff,recall,n_neg\n")
        for r in sorted(ok, key=lambda r: (r["model"], r["dtype"],
                                           r["layer"], r["op"])):
            if r["BS"] != 1:
                continue
            f.write(f'{r["model"]},{r["K"]},{r["dtype"]},{r["layer"]},'
                    f'{r.get("hit_rate", "")},{r["op"]},'
                    f'{r["us_cold"]:.3f},{r["us_warm"]:.3f},'
                    f'{r.get("vdiff", "")},{r.get("recall", "")},'
                    f'{r.get("n_neg", "")}\n')
    p2 = HERE / "op22real_bs_data.csv"
    with open(p2, "w") as f:
        f.write("model,K,dtype,op,BS,gm_us_cold,gm_us_warm,"
                "ratio_vs_base_cold,ratio_vs_base_warm\n")
        for m in MODELS:
            K = {r["model"]: r["K"] for r in ok}.get(m, "")
            for dt in DTS:
                for arm in ARMS:
                    for BS in BS_GRID:
                        gc_ = layer_gm(cm, m, dt, arm, BS, "us_cold")
                        gh = layer_gm(cm, m, dt, arm, BS, "us_warm")
                        if gc_ is None:
                            continue
                        rc = ratio_cells(cm, m, dt, arm, [BS], "us_cold")
                        rh = ratio_cells(cm, m, dt, arm, [BS], "us_warm")
                        f.write(f"{m},{K},{dt},{arm},{BS},{gc_:.3f},"
                                f"{gh:.3f},"
                                f"{rc if rc is None else round(rc, 4)},"
                                f"{rh if rh is None else round(rh, 4)}\n")
    return p1, p2


# --------------------------------------------------------------------------
# HTML assembly (structure mirrors the existing chapters; classes from the
# report CSS: card / i18n-en / i18n-zh / ctl / ck / plt / cold / warm)
# --------------------------------------------------------------------------

def method_card(meta):
    src_rows = []
    SRC = {
        "flash": ("DeepSeek-V4 Flash", 512, 4,
                  "Q9j real-loop capture, SWE-bench <b>100K</b> prompt "
                  "(<code>capture_20260520T083843_alllayers_swe100k</code>)",
                  "21 (even 2..42)"),
        "pro": ("DeepSeek-V4 Pro", 1024, 4,
                "Q9k real-loop capture, SWE-bench <b>64K</b> prompt, native "
                "K=1024 (<code>capture_20260520T164146Z_v4pro_K1024_64k</code>)",
                "30 (even 2..60)"),
        "v32": ("DeepSeek-V3.2", 2048, 1,
                "SWE-bench-64K per-layer decode logging "
                "(<code>SWE_Bench_64K_decode_logits</code>)",
                "9 ({0,1,20,21,22,40,41,42,60})"),
    }
    for m in MODELS:
        s = SRC[m]
        d = meta.get(m)
        if not d:
            continue
        src_rows.append(
            f"<tr><td>{s[0]}</td><td>{s[1]}</td><td>{s[2]}</td>"
            f"<td style='text-align:left'>{s[3]}</td><td>{s[4]}</td>"
            f"<td>{d['s_last']}</td><td>{d['N']}</td><td>{d['Npad']}</td>"
            f"</tr>")
    tbl = ("<table><tr><th>model</th><th>K</th><th>cr</th>"
           "<th>capture (ISL per user spec)</th><th>layers</th>"
           "<th>last step s</th><th>N (real valid)</th>"
           "<th>N<sub>pad</sub> (row stride)</th></tr>"
           + "".join(src_rows) + "</table>")
    en = f"""<div class="card" id="realcap-method">
<h3>{bi("Method — REAL production-captured inputs (2026-07-13)",
        "方法 —— 真实生产采集输入（2026-07-13）", "span")}</h3>
<div class="i18n-en">
<p>This chapter re-runs the report arm set on <b>real inference-captured
indexer logits</b> (REAL_DATA_INVENTORY §B/C/D; loader
<code>harness/real_data_v2.py</code>) instead of synthetic bundles. Per
user spec: K=512 → V4 Flash <b>ISL=100K</b>; K=1024 → V4 Pro
<b>ISL=64K</b> (native-K1024 capture); K=2048 → V3.2
<b>SWE_Bench_64K_decode_logits</b>. For every captured layer, the input is
the <b>LAST decode step's</b> logits row; <b>preIdx = the PREVIOUS decode
step's top-K</b> (V4: captured <code>topk.out[(L, s−1)]</code> verbatim;
V3.2: exact same-dtype <code>torch.topk</code> on the previous row —
production <code>dsa.py</code> conventions). BS scaling replicates the
same row to BS (identical to the synthetic report methodology; no VarLen).
16-bit cells use the fp32 capture truncated to bf16/fp16; the correctness
reference is <code>torch.topk</code> on the SAME truncated dtype. Rows are
stride-padded to 64 elements (pad = <code>finfo(dtype).min</code>,
<code>seq_lens</code> stay at the real N) — production logits buffers are
stride-padded too.</p>{tbl}
<p><b>Protocol identical to §1–2</b>: nsys pure-kernel NVTX-range
projection, cold-L2 (512 MB evict, canonical) + warm-L2, 20/50 reps,
eager+sync in range, B200, one GPU per (model,dtype) batch, single node —
all arm ratios are co-located (no cross-node anchor transfer).
Caveats: (1) <code>flashinfer_topk</code> has no varlen argument → it
scans the full padded row (≤63 extra elements, pad can never enter the
top-K); (2) <code>sglang_v2</code> gets real varlen semantics
(<code>seq_lens</code>=N on the padded buffer); (3) <code>op21_hls</code>
/ <code>op25_hls</code> are historical binaries not separately
re-measurable at HEAD — <code>op27_hls</code> is that lineage's tip
(K512/K1024 bit-identical to the op25 ship; K2048 adds the op27 tail
ladder); (4) real N is fixed per capture (no seq-len axis): 25154 / 14478
/ 70690 — between the 16K and 65K grid poles of §1 for V4 and near 65K
for V3.2; (5) <code>gvr29_hbe</code> (fp32-only) was measured in
same-node supplement batches with a co-located <code>gvr_cutedsl</code>
scale anchor — its HBE tier guard is N≥65536, so it engages only on V3.2
(N=70690); on Flash/Pro it runs its stock sglang_v2-fork dispatch.</p></div>
<div class="i18n-zh">
<p>本章把报告臂集换到<b>真实推理采集的 indexer logits</b> 上重测
（REAL_DATA_INVENTORY §B/C/D；加载器 <code>harness/real_data_v2.py</code>），
不再使用合成 bundle。按用户规格：K=512 → V4 Flash <b>ISL=100K</b>；
K=1024 → V4 Pro <b>ISL=64K</b>（原生 K1024 采集）；K=2048 → V3.2
<b>SWE_Bench_64K_decode_logits</b>。对每个采集层，输入为该层<b>最后一个
decode step</b> 的 logits 行；<b>preIdx = 上一 decode step 的 top-K</b>
（V4：直接用采集的 <code>topk.out[(L, s−1)]</code>；V3.2：对上一行做同
dtype 精确 <code>torch.topk</code> 重算 —— 与生产 <code>dsa.py</code>
约定一致）。BS 扩展 = 同一行复制到 BS（与合成报告方法一致；无 VarLen）。
16-bit 单元使用 fp32 采集截断到 bf16/fp16；正确性参考为<b>同截断
dtype</b> 的 <code>torch.topk</code>。行按 64 元素对齐补边
（补值 = <code>finfo(dtype).min</code>，<code>seq_lens</code> 保持真实 N）——
生产 logits 缓冲同样是 stride 补边的。</p>{tbl}
<p><b>协议与 §1–2 完全一致</b>：nsys 纯 kernel NVTX 投影、冷 L2
（512 MB 清空，规范口径）+ 热 L2、20/50 次重复、区间内 eager+sync、
B200、每 (model,dtype) 批独占一张 GPU、单节点 —— 所有臂比值同机同批
（无跨节点锚点转移）。注意：(1) <code>flashinfer_topk</code> 无 varlen
参数 → 扫全补边行（≤63 个额外元素，补值不可能进 top-K）；
(2) <code>sglang_v2</code> 使用真实 varlen 语义（补边缓冲上
<code>seq_lens</code>=N）；(3) <code>op21_hls</code>/<code>op25_hls</code>
是历史二进制，HEAD 无法单独复测 —— <code>op27_hls</code> 即该谱系尖端
（K512/K1024 与 op25 ship 位相同；K2048 多 op27 尾梯）；(4) 真实 N 由采集
固定（无 seq-len 轴）：25154 / 14478 / 70690；(5) <code>gvr29_hbe</code>
（仅 fp32）在同节点补测批中测量，带同批 <code>gvr_cutedsl</code> 尺度锚 ——
其 HBE 层守卫为 N≥65536，只在 V3.2（N=70690）触发；Flash/Pro 上走其
sglang_v2-fork 基础分发。</p></div>
</div>"""
    return en


def hr_exact_card(meta, ex):
    hr_rows = []
    for m in MODELS:
        d = meta.get(m)
        if not d:
            continue
        for dt in DTS:
            if dt not in d["hr"]:
                continue
            lo, med, hi = d["hr"][dt]
            hr_rows.append(f"<tr><td>{MODEL_LABEL[m][0]}</td><td>{dt}</td>"
                           f"<td>{lo:.3f}</td><td>{med:.3f}</td>"
                           f"<td>{hi:.3f}</td></tr>")
    hr_tbl = ("<table><tr><th>model</th><th>dtype</th>"
              "<th>hit-rate min</th><th>median</th><th>max</th></tr>"
              + "".join(hr_rows) + "</table>")
    ex_rows = []
    for m in MODELS:
        for arm in ARMS:
            a = ex.get((m, arm))
            if not a:
                continue
            ok = a["vd"] == 0 and a["nn"] == 0
            ex_rows.append(
                f"<tr><td>{SHORT[arm]}</td><td>{MODEL_LABEL[m][0]}</td>"
                f"<td>{a['n']}</td>"
                f"<td>{a['vd']:.1e}</td><td>{a['rc']:.4f}</td>"
                f"<td>{a['nn']}</td>"
                f"<td>{'<span class=good>EXACT</span>' if ok else '<span class=bad>CHECK</span>'}</td></tr>")
    ex_tbl = ("<table><tr><th>arm</th><th>model</th>"
              "<th># (layer,dtype) cells</th><th>max vdiff</th>"
              "<th>min recall</th><th>Σ n_neg</th><th>verdict</th></tr>"
              + "".join(ex_rows) + "</table>")
    return f"""<div class="card" id="realcap-hr">
<h3>{bi("Kernel-read hit rate (per layer) & exactness",
        "kernel 读取 hit-rate（逐层）与精确性", "span")}</h3>
{bi(f"<p>Real temporal-hint quality varies per layer — this is what the "
    f"synthetic scenarios controlled for. Per-model spread across layers "
    f"(same-dtype kernel-read hit rate of preIdx against the reference "
    f"top-K):</p>{hr_tbl}",
    f"<p>真实时序提示质量随层变化 —— 这正是合成场景所控制的变量。各模型"
    f"逐层展布（preIdx 相对参考 top-K 的同 dtype kernel 读取 hit-rate）："
    f"</p>{hr_tbl}")}
{bi(f"<p>Exactness at BS=1, EVERY (arm, layer, dtype): sorted-value "
    f"equivalence vs same-dtype <code>torch.topk</code> (tie-order "
    f"agnostic — GVR row order is atomicAdd-nondeterministic):</p>{ex_tbl}"
    f"<p class='small'>The two CHECK cells share ONE root cause: on V4 Pro "
    f"layer 56 (fp32) the real row's K-th/(K+1)-th boundary gap is only "
    f"2.742e-6 (~20 ulps at |logit|≈1.13) and the op26 single-CTA "
    f"threshold refinement deterministically admits the (K+1)-th element "
    f"instead — 1/1024 indices, |Δvalue| = 2.7e-6, reproducible 3/3 "
    f"(<code>op26_r0auto</code> inherits it via its <code>plain</code> "
    f"route). All other 1969/1971 cells are bit-exact. Logged as an op26 "
    f"boundary-precision follow-up; the op26 synth gates never produced a "
    f"boundary gap this small.</p>",
    f"<p>BS=1 下对每个 (臂, 层, dtype) 的精确性：与同 dtype "
    f"<code>torch.topk</code> 的排序值等价（对并列顺序不敏感 —— GVR 输出"
    f"行序为 atomicAdd 非确定）：</p>{ex_tbl}"
    f"<p class='small'>两个 CHECK cell 同一根因：V4 Pro 层 56（fp32）真实"
    f"行的 K/(K+1) 名边界间隙仅 2.742e-6（|logit|≈1.13 处约 20 ulps），"
    f"op26 单 CTA 阈值精化确定性地收进第 (K+1) 名 —— 1024 个索引错 1 个，"
    f"|Δ值| = 2.7e-6，3/3 复现（<code>op26_r0auto</code> 经 "
    f"<code>plain</code> 路由继承）。其余 2029/2031 个 BS=1 cell 全部位精确。"
    f"已记为 op26 边界精度跟进项；op26 合成 gate 从未出现过这么小的边界"
    f"间隙。</p>")}
</div>"""


def geomean_card(cm):
    grp = [("BS=1", [1]), ("BS 2–64", [2, 4, 8, 16, 32, 64]),
           ("BS 128–2048", [128, 256, 512, 1024, 2048]),
           ("ALL", BS_GRID)]
    blocks = []
    for m in MODELS:
        rows = []
        for arm in ARMS:
            if arm == BASE:
                continue
            tds = []
            any_ = False
            for dt in DTS:
                for glab, bss in grp:
                    rc = ratio_cells(cm, m, dt, arm, bss, "us_cold")
                    rh = ratio_cells(cm, m, dt, arm, bss, "us_warm")
                    if rc is not None:
                        any_ = True
                    cell = (f'<span class="cold">{fmt_r(rc)}</span>'
                            f'<span class="warm">{fmt_r(rh)}</span>')
                    tds.append(f"<td>{cell}</td>")
            if any_:
                rows.append(f"<tr><td style='color:{COL[arm]}'>"
                            f"{SHORT[arm]}</td>" + "".join(tds) + "</tr>")
        head1 = "".join(f'<th colspan="4">{dt}</th>' for dt in DTS)
        head2 = "".join(f"<th>{g}</th>" for _ in DTS for g, _b in grp)
        blocks.append(
            f"<h3>{MODEL_LABEL[m][0]}</h3>"
            f'<div class="scrolltbl"><table>'
            f'<tr><th rowspan="2">arm</th>{head1}</tr>'
            f"<tr>{head2}</tr>" + "".join(rows) + "</table></div>")
    return f"""<div class="card" id="realcap-gm">
<h3>{bi("Geomean µs ratio vs GVR (cuteDSL) baseline — real captured rows",
        "对 GVR (cuteDSL) 基线的几何均值 µs 比值 —— 真实采集行", "span")}</h3>
{bi("<p>Per-cell ratio arm/baseline geomeaned over (layer × BS-group). "
    "<b>&lt;1 (green) = arm faster than the GVR cuteDSL baseline</b>; "
    "cold-L2 shown under the cold toggle, warm-L2 under warm.</p>",
    "<p>逐 cell 比值 臂/基线，在（层 × BS 组）上取几何均值。"
    "<b>&lt;1（绿）= 该臂快于 GVR cuteDSL 基线</b>；冷/热 L2 随页面"
    "顶部的 metric 开关切换。</p>")}
{"".join(blocks)}
</div>"""


def findings_card(cm, meta):
    def r(m, dt, arm, bss=BS_GRID, metric="us_cold"):
        return ratio_cells(cm, m, dt, arm, bss, metric)

    items_en, items_zh = [], []
    for m in MODELS:
        lab = MODEL_LABEL[m][0]
        best_arm, best_v = None, None
        for arm in ARMS:
            if arm == BASE:
                continue
            v = r(m, "fp32", arm)
            if v is not None and (best_v is None or v < best_v):
                best_arm, best_v = arm, v
        if best_arm is None:
            continue
        v27 = r(m, "fp32", "op27_hls")
        vrad = r(m, "fp32", "radix_cutedsl")
        vsgl = r(m, "fp32", "sglang_v2")
        def inv(x):
            return None if x is None else 1.0 / x
        items_en.append(
            f"<li><b>{lab}</b> (fp32, all BS, cold): fastest arm = "
            f"<b style='color:{COL[best_arm]}'>{SHORT[best_arm]}</b> at "
            f"{best_v:.3f}× the baseline µs ({1/best_v:.2f}× speedup). "
            f"op27_hls {fmt_r(v27)} · radix_cutedsl {fmt_r(vrad)} · "
            f"sglang_v2 {fmt_r(vsgl)}.</li>")
        items_zh.append(
            f"<li><b>{lab}</b>（fp32，全 BS，冷）：最快臂 = "
            f"<b style='color:{COL[best_arm]}'>{SHORT[best_arm]}</b>，"
            f"µs 为基线的 {best_v:.3f}×（提速 {1/best_v:.2f}×）。"
            f"op27_hls {fmt_r(v27)} · radix_cutedsl {fmt_r(vrad)} · "
            f"sglang_v2 {fmt_r(vsgl)}。</li>")
    items_en.append(
        "<li><b>Synth-vs-real anchor check</b> (vs §2 REAL-scenario at the "
        "nearest grid N; <code>sanity_realcap_vs_rr.py</code>): the "
        "hint-blind arms (radix, sglang_v2, multi-CTA GVR) agree within "
        "~±20% gm ratio. The synth-tuned GVR arms (op27_hls, op26_r0auto) "
        "run consistently WORSE on real rows than the synthetic REAL "
        "scenario predicted — on Flash and V3.2, op27_hls flips from "
        "faster-than-baseline (synth) to slower (real). Two causes: real "
        "N (25154/14478/70690) falls between the dispatch tables' grid "
        "poles, and real per-layer hit rates span poles the aggregate "
        "synth hr distribution under-weights (V3.2 layers 0/1 sit at "
        "hr≈0.02). Same direction as the report.html §5 finding that real "
        "data flips synth-tuned verdicts.</li>")
    items_zh.append(
        "<li><b>合成-真实锚点对照</b>（对 §2 REAL 场景最近网格 N；"
        "<code>sanity_realcap_vs_rr.py</code>）：对提示不敏感的臂"
        "（radix、sglang_v2、multi-CTA GVR）几何均值比值一致在 ~±20% 内；"
        "而按合成数据调优的 GVR 臂（op27_hls、op26_r0auto）在真实行上"
        "系统性差于合成 REAL 场景的预测 —— Flash 与 V3.2 上 op27_hls 从"
        "合成的快于基线翻转为真实的慢于基线。两个原因：真实 N"
        "（25154/14478/70690）落在 dispatch 表网格极点之间；真实逐层 "
        "hit-rate 覆盖聚合合成分布低估的极点（V3.2 层 0/1 的 hr≈0.02）。"
        "方向与 report.html §5 『真实数据翻转合成调优结论』一致。</li>")
    return f"""<div class="card" id="realcap-findings">
<h3>{bi("Findings — real captured rows", "Findings —— 真实采集行", "span")}</h3>
{bi("<ul>" + "".join(items_en) + "</ul>",
    "<ul>" + "".join(items_zh) + "</ul>")}
</div>"""


def panels(ok):
    js_data = json.dumps(js_records(ok), separators=(",", ":"))
    model_items = "".join(
        f'<label class="ck"><input type="radio" name="rcm" value="{m}"'
        f'{" checked" if m == "v32" else ""}>{MODEL_LABEL[m][0]}</label>'
        for m in MODELS)
    dt_items = "".join(
        f'<label class="ck"><input type="radio" name="rcd" value="{d}"'
        f'{" checked" if d == "fp32" else ""}>{d}</label>'
        for d in DTS)
    op_items = "".join(
        f'<label class="ck"><input type="checkbox" class="rco" value="{o}"'
        f'{" checked"}>{SHORT[o]}</label>' for o in ARMS)
    ctl = (f'<div class="ctl">{bi("<b>model</b>", "<b>模型</b>", "span")} '
           f"{model_items}<br>"
           f'{bi("<b>operators</b>", "<b>算子</b>", "span")} {op_items}'
           f"<br>{dt_items}</div>")
    figs = ('<div class="row">'
            '<div class="plt" id="rc_lay"></div>'
            '<div class="plt" id="rc_layr"></div></div>'
            '<div class="row">'
            '<div class="plt" id="rc_bs"></div>'
            '<div class="plt" id="rc_bsr"></div></div>')
    js = """
<script>
const RCA=%s;
const RCM=%s,RCD=%s,RCO=%s;
const RC=RCA.map(a=>({m:RCM[a[0]],d:RCD[a[1]],o:RCO[a[2]],L:a[3],B:a[4],c:a[5],h:a[6]}));
const RCCOL=%s,RCSHORT=%s,RCBASE=%s;
const RCT={en:{lay:'Per-layer latency (BS=1)',layr:'Per-layer ratio arm/base (BS=1)',
bs:'Latency vs BS (geomean over layers)',bsr:'Ratio arm/base vs BS (geomean over layers)',
us:'µs (log)',lyr:'layer index',bsx:'batch size (log)',r:'ratio × (<1 = faster than base)'},
zh:{lay:'逐层延迟（BS=1）',layr:'逐层比值 臂/基线（BS=1）',
bs:'延迟 对 BS（层几何均值）',bsr:'比值 臂/基线 对 BS（层几何均值）',
us:'µs（对数）',lyr:'层号',bsx:'batch size（对数）',r:'比值 ×（<1 = 快于基线）'}};
function rcLang(){return document.getElementById('lang-zh').checked?'zh':'en'}
function rcReg(){return document.getElementById('m-cold').checked?'c':'h'}
function rcGm(a){const v=a.filter(x=>x>0);if(!v.length)return null;
return Math.exp(v.reduce((s,x)=>s+Math.log(x),0)/v.length)}
function rcLAY(t,xt,yt,ylog){return {title:{text:t,font:{size:15}},
paper_bgcolor:'#161b22',plot_bgcolor:'#0f1419',font:{color:'#e6e6e6',size:12},
margin:{t:42,r:10,b:48,l:56},showlegend:true,
legend:{orientation:'h',y:-0.22,font:{size:10.5}},
xaxis:{title:xt,gridcolor:'#2a3340'},
yaxis:{title:yt,type:ylog?'log':'linear',gridcolor:'#2a3340'}}}
function rcDraw(){const L=RCT[rcLang()],rg=rcReg(),
m=document.querySelector('input[name=rcm]:checked').value,
d=document.querySelector('input[name=rcd]:checked').value,
ops=[...document.querySelectorAll('.rco:checked')].map(x=>x.value),
R=RC.filter(r=>r.m==m&&r.d==d);
const lay=[],layr=[],bs=[],bsr=[];
const layers=[...new Set(R.map(r=>r.L))].sort((a,b)=>a-b);
const bss=[...new Set(R.map(r=>r.B))].sort((a,b)=>a-b);
const get=(o,Lr,B)=>{const x=R.find(r=>r.o==o&&r.L==Lr&&r.B==B);return x?x[rg]:null};
for(const o of ops){
const y1=layers.map(Lr=>get(o,Lr,1));
if(y1.some(v=>v)){lay.push({x:layers,y:y1,name:RCSHORT[o],mode:'lines+markers',
marker:{size:5},line:{color:RCCOL[o]}});}
if(o!=RCBASE){const yr=layers.map(Lr=>{const a=get(o,Lr,1),b=get(RCBASE,Lr,1);
return a&&b?a/b:null});
if(yr.some(v=>v)){layr.push({x:layers,y:yr,name:RCSHORT[o],mode:'lines+markers',
marker:{size:5},line:{color:RCCOL[o]}});}}
const yb=bss.map(B=>rcGm(layers.map(Lr=>get(o,Lr,B)).filter(Boolean)));
if(yb.some(v=>v)){bs.push({x:bss,y:yb,name:RCSHORT[o],mode:'lines+markers',
marker:{size:5},line:{color:RCCOL[o]}});}
if(o!=RCBASE){const ybr=bss.map(B=>{
const rs=layers.map(Lr=>{const a=get(o,Lr,B),b=get(RCBASE,Lr,B);
return a&&b?a/b:null}).filter(Boolean);return rcGm(rs)});
if(ybr.some(v=>v)){bsr.push({x:bss,y:ybr,name:RCSHORT[o],mode:'lines+markers',
marker:{size:5},line:{color:RCCOL[o]}});}}}
layr.push({x:[layers[0],layers[layers.length-1]],y:[1,1],mode:'lines',
showlegend:false,line:{color:'#888',width:1}});
bsr.push({x:[1,2048],y:[1,1],mode:'lines',showlegend:false,line:{color:'#888',width:1}});
const reg=rg=='c'?'cold':'warm';
Plotly.newPlot('rc_lay',lay,rcLAY(L.lay+' ('+m+', '+d+', '+reg+'-L2)',L.lyr,L.us,true),{responsive:true});
Plotly.newPlot('rc_layr',layr,rcLAY(L.layr,L.lyr,L.r,false),{responsive:true});
const lb=rcLAY(L.bs+' ('+m+', '+d+', '+reg+'-L2)',L.bsx,L.us,true);lb.xaxis.type='log';
Plotly.newPlot('rc_bs',bs,lb,{responsive:true});
const lr=rcLAY(L.bsr,L.bsx,L.r,false);lr.xaxis.type='log';
Plotly.newPlot('rc_bsr',bsr,lr,{responsive:true});}
document.querySelectorAll('input[name=rcm],input[name=rcd],.rco')
.forEach(e=>e.addEventListener('change',rcDraw));
['lang-zh','lang-en','m-cold','m-warm'].forEach(id=>{const e=document.getElementById(id);
if(e)e.addEventListener('change',()=>setTimeout(rcDraw,0));});
rcDraw();
</script>""" % (js_data, json.dumps(MODELS), json.dumps(DTS),
                json.dumps(ARMS), json.dumps(COL), json.dumps(SHORT),
                json.dumps(BASE))
    return (f'<div class="card" id="realcap-charts">'
            f'<h3>{bi("Interactive charts — real captured rows",
                      "交互图表 —— 真实采集行", "span")}</h3>'
            f"{ctl}{figs}</div>{js}")


def build_chapter(rows, ok, err, cm):
    meta = meta_table(ok)
    ex = exactness_summary(rows)
    n_cells = len({(r['model'], r['dtype'], r['op'], r['layer'], r['BS'])
                   for r in ok})
    hdr = f"""
<h2 id="realcap">{bi("9. Real captured-data — production inference logits "
                     "(last decode step, all layers)",
                     "9. 真实采集数据 —— 生产推理 logits（最后 decode 步，全部层）",
                     "span")}</h2>
{bi(f"<p>{n_cells} measured cells, {len(err)} errors. Node "
    f"<code>umbriel-b200-072</code>, 8 GPUs (one per (model,dtype) batch), "
    f"2026-07-13. Per-cell data: <code>op22real_layer_data.csv</code> / "
    f"<code>op22real_bs_data.csv</code>.</p>",
    f"<p>{n_cells} 个实测 cell，{len(err)} 个错误。节点 "
    f"<code>umbriel-b200-072</code>，8 张 GPU（每 (model,dtype) 批一张），"
    f"2026-07-13。逐 cell 数据：<code>op22real_layer_data.csv</code> / "
    f"<code>op22real_bs_data.csv</code>。</p>")}"""
    return (BEGIN + hdr + method_card(meta) + hr_exact_card(meta, ex)
            + geomean_card(cm) + findings_card(cm, meta) + panels(ok) + END)


def main():
    rows, ok, err = load()
    print(f"loaded {len(rows)} rows: ok={len(ok)} err={len(err)}")
    if err:
        from collections import Counter
        c = Counter(r["error"].split(":")[0] for r in err)
        print("  errors:", dict(c))
    cm = cell_map(ok)
    p1, p2 = write_csvs(ok, cm)
    print(f"wrote {p1.name}, {p2.name}")

    t = REPORT.read_text(encoding="utf-8")
    if not BAK.exists():
        shutil.copy(REPORT, BAK)
        print(f"backup -> {BAK.name}")
    # drop existing chapter (idempotent re-run)
    i = t.find(BEGIN)
    if i >= 0:
        j = t.find(END)
        assert j > i, "corrupt REALCAP sentinels"
        tail = t[j + len(END):].lstrip("\n")
        t = t[:i].rstrip("\n") + "\n" + tail
        print("dropped existing REALCAP chapter")
    chapter = build_chapter(rows, ok, err, cm)
    anchor = "</div></body></html>"
    assert t.rstrip().endswith(anchor), "unexpected REPORT.html tail"
    k = t.rfind(anchor)
    t = t[:k] + chapter + "\n" + t[k:]
    REPORT.write_text(t, encoding="utf-8")
    print(f"REPORT.html updated (+{len(chapter)} chars)")


if __name__ == "__main__":
    main()
