"""Bilingual (CSS-only toggle, ZERO <script>) HTML report for the
multi-threshold block_count_ge micro-bench. Reads results.csv.

High-quality layout: soft gradient background, white cards, color-coded
heatmap tables, per-chart captions, <details> methodology (native, no JS)."""
import csv
import io
import math
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "results.csv")))
Ns = sorted({int(r["N"]) for r in rows})
Ms = sorted({int(r["M"]) for r in rows})
DTS = ["fp32", "fp16"]
data = defaultdict(lambda: defaultdict(dict))
cfg = {}
for r in rows:
    if not r["us_med"]:
        continue
    data[r["dtype"]][int(r["N"])][int(r["M"])] = float(r["us_med"])
    cfg[(r["dtype"], int(r["N"]))] = (r["TT"], r["VECW"])

# consistent color per M across all charts
MCOL = {1: "#4062bb", 2: "#3aa0a0", 4: "#e0a800", 6: "#e0662b", 8: "#c0392b"}


def nlabel(n):
    return f"{n//1024}K"


def ratio(dt, n, M):
    b = data[dt][n].get(1); v = data[dt][n].get(M)
    return v / b if b and v else None


def netfac(dt, n, M):
    r = ratio(dt, n, M)
    return r / math.log2(M + 1) if r is not None else None


def _svg(fig):
    buf = io.StringIO(); fig.savefig(buf, format="svg"); plt.close(fig)
    s = buf.getvalue(); return s[s.find("<svg"):]


def svg_lines(dt, title, mode):
    fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=110)
    if mode == "abs":
        for M in Ms:
            ax.plot([nlabel(n) for n in Ns], [data[dt][n].get(M) for n in Ns],
                    marker="o", color=MCOL[M], label=f"M={M}")
        ax.set_xlabel("N (elements)"); ax.set_ylabel("kernel time (µs), cold-L2")
        ax.legend(title="thresholds", fontsize=8)
    else:
        for n in Ns:
            if mode == "ratio":
                ys = [ratio(dt, n, M) for M in Ms]; yl = "time(M) / time(M=1)"
            elif mode == "amort":
                ys = [(ratio(dt, n, M) / M) if ratio(dt, n, M) else None for M in Ms]
                yl = "amortized cost / threshold  (÷ M=1)"
            else:
                ys = [netfac(dt, n, M) for M in Ms]; yl = "est. total P2-refine time  (÷ M=1)"
            ax.plot(Ms, ys, marker="o", label=nlabel(n))
        ax.axhline(1.0, color="#888", ls="--", lw=0.9)
        ax.set_xlabel("M (thresholds per scan)"); ax.set_ylabel(yl)
        ax.set_xticks(Ms); ax.legend(title="N", fontsize=7, ncol=2)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _svg(fig)


charts = {}
for dt in DTS:
    for m in ("abs", "ratio", "amort", "net"):
        charts[(dt, m)] = svg_lines(dt, f"{dt.upper()} — " + {
            "abs": "cold-L2 time vs N", "ratio": "overhead ÷ M=1",
            "amort": "amortized cost / threshold", "net": "est. total P2-refine time"}[m], m)


# ---------- tables ----------
def heat(v, lo, hi, good_low=True):
    """green→red scale; good_low: lower=green."""
    if v is None:
        return "#fff"
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    if not good_low:
        t = 1 - t
    # green (good) -> yellow -> red (bad)
    if t < 0.5:
        r = int(0xd9 + (0xff - 0xd9) * (t / 0.5)); g = int(0xf2 + (0xf3 - 0xf2) * (t / 0.5)); b = 0xdf
    else:
        r = 0xff; g = int(0xf3 - (0xf3 - 0xd0) * ((t - 0.5) / 0.5)); b = int(0xdf - (0xdf - 0xcf) * ((t - 0.5) / 0.5))
    return f"#{r:02x}{g:02x}{b:02x}"


def raw_table(dt):
    th = "".join(f"<th>M={M}</th>" for M in Ms)
    body = ""
    for n in Ns:
        tt, vw = cfg.get((dt, n), ("?", "?"))
        cells = ""
        for M in Ms:
            v = data[dt][n].get(M); r = ratio(dt, n, M)
            bg = heat(r, 1.0, 2.4)
            badge = f"<span class='bdg'>×{r:.2f}</span>" if r is not None else ""
            cells += f"<td style='background:{bg}'>{v:.2f}{badge}</td>"
        body += (f"<tr><td class='n'>{nlabel(n)}<span class='sub'>T{tt}·V{vw}</span></td>{cells}</tr>")
    return f"<table class='data'><thead><tr><th>N ↓ / M →</th>{th}</tr></thead><tbody>{body}</tbody></table>"


def net_table():
    th = "".join(f"<th>M={M}</th>" for M in Ms)
    body = ""
    for dt in DTS:
        for n in Ns:
            facs = {M: netfac(dt, n, M) for M in Ms}
            best = min((M for M in Ms if facs[M] is not None), key=lambda M: facs[M])
            cells = ""
            for M in Ms:
                f = facs[M]; bg = heat(f, 0.35, 1.0)
                mark = " ★" if M == best else ""
                cells += f"<td style='background:{bg}'>{f:.2f}{mark}</td>"
            body += f"<tr><td class='n'>{dt} · {nlabel(n)}</td>{cells}<td class='n best'>M={best}</td></tr>"
    return (f"<table class='data'><thead><tr><th>dtype · N</th>{th}<th>argmin</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def avg_ratio(dt, M):
    xs = [ratio(dt, n, M) for n in Ns if ratio(dt, n, M)]
    return sum(xs) / len(xs) if xs else 0


# ---------- CSS ----------
CSS = """
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:'Inter',-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;
 color:#1c2530;background:linear-gradient(160deg,#e8eef7 0%,#f4f7fb 40%,#eef3f8 100%);
 background-attachment:fixed;line-height:1.55;font-size:15px}
.toggle{position:sticky;top:0;z-index:20;background:linear-gradient(90deg,#2b3f7a,#4062bb);
 color:#fff;padding:11px 28px;box-shadow:0 2px 10px rgba(0,0,0,.18);font-size:14px;
 display:flex;align-items:center;gap:18px}
.toggle label{cursor:pointer;background:rgba(255,255,255,.15);padding:5px 14px;border-radius:20px;
 font-weight:600;transition:background .15s}
.toggle label:hover{background:rgba(255,255,255,.3)}
.wrap{max-width:1120px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:26px;font-weight:800;margin:.1em 0 .1em;letter-spacing:-.3px}
.subtitle{color:#5a6577;font-size:14px;margin-bottom:18px}
h2{font-size:19px;font-weight:700;margin:1.8em 0 .6em;color:#26324a}
h2 .num{display:inline-block;background:#4062bb;color:#fff;border-radius:6px;
 padding:1px 9px;font-size:14px;margin-right:9px;vertical-align:middle}
h3{font-size:14px;font-weight:600;margin:1.1em 0 .4em;color:#3a4658;text-transform:uppercase;letter-spacing:.4px}
.card{background:#fff;border-radius:12px;padding:18px 20px;margin:14px 0;
 box-shadow:0 1px 3px rgba(20,40,80,.08),0 6px 20px rgba(20,40,80,.04);border:1px solid #e6ebf2}
.hero{background:linear-gradient(120deg,#fff,#f2f6ff);border-left:5px solid #4062bb}
.hero b{color:#2b3f7a}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.kpi{flex:1;min-width:150px;background:#f6f9ff;border:1px solid #dce6f7;border-radius:10px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:800;color:#2b3f7a}
.kpi .l{font-size:12px;color:#5a6577;margin-top:2px}
table.data{border-collapse:separate;border-spacing:0;width:100%;margin:.5em 0;font-size:13px;
 border-radius:10px;overflow:hidden;box-shadow:0 1px 2px rgba(0,0,0,.05)}
table.data th,table.data td{border-bottom:1px solid #e3e8ef;border-right:1px solid #e3e8ef;padding:7px 9px;text-align:center}
table.data th{background:#2b3f7a;color:#fff;font-weight:600}
table.data td.n{font-weight:700;background:#f0f4fb!important;color:#26324a}
table.data td.best{color:#12683a}
.bdg{display:block;font-size:10px;color:#556;font-weight:600;margin-top:1px}
.sub{display:block;font-size:9.5px;color:#8a93a3;font-weight:400}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.fig{background:#fff;border:1px solid #e6ebf2;border-radius:10px;padding:8px}
.fig svg{width:100%;height:auto;display:block}
.cap{font-size:12px;color:#5a6577;padding:6px 4px 2px;text-align:center}
.note{border-left:4px solid #4062bb;padding:11px 15px;font-size:13.5px}
.key{border-left-color:#e0a800;background:#fffdf5}
.rec{border-left-color:#12683a;background:#f2fbf5}
code{background:#eef2f9;padding:1px 6px;border-radius:5px;font-size:12.5px;color:#2b3f7a}
pre{background:#0f1b33;color:#dce6f7;border-radius:9px;padding:13px 15px;overflow-x:auto;
 font-size:12.5px;line-height:1.5;font-family:'SF Mono',ui-monospace,Menlo,Consolas,monospace;margin:.5em 0}
pre .c{color:#7f93b8}pre .k{color:#8fd0ff}pre .m{color:#ffcf6b}
ol{margin:.3em 0 .3em 1.3em;padding:0}ol li{margin:.35em 0}
details{margin:.4em 0}summary{cursor:pointer;font-weight:600;color:#2b3f7a;padding:4px 0}
ul{margin:.3em 0 .3em 1.1em;padding:0}li{margin:.25em 0}
.legend{font-size:11.5px;color:#5a6577}.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin:0 3px 0 8px}
.zh{display:none}
#zh:checked ~ .wrap .en{display:none}
#zh:checked ~ .wrap .zh{display:revert}
#zh{position:absolute;opacity:0;pointer-events:none}
"""


def bi(en, zh):
    return f"<span class='en'>{en}</span><span class='zh'>{zh}</span>"


H = []
def add(s): H.append(s)

add(f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
    f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
    f"<title>block_count_ge multi-threshold micro-bench · B200</title>"
    f"<style>{CSS}</style></head><body>")
add("<input type='checkbox' id='zh'>")
add("<div class='toggle'><span>🌐</span>"
    f"<label for='zh'>{bi('Switch to 中文','切换 English')}</label>"
    f"<span style='opacity:.85'>{bi('Multi-threshold block_count_ge · NVIDIA B200','多阈值 block_count_ge · NVIDIA B200')}</span></div>")
add("<div class='wrap'>")

add(f"<h1>{bi('Multi-threshold <code>block_count_ge</code> micro-benchmark','多阈值 <code>block_count_ge</code> 微基准')}</h1>")
add(f"<div class='subtitle'>{bi('GVR Top-K Phase-2/3 threshold-count primitive · NVIDIA B200 · nsys cold-L2','GVR Top-K Phase-2/3 阈值计数原语 · NVIDIA B200 · nsys cold-L2')}</div>")

# hero TL;DR
r4_32, r4_16 = avg_ratio("fp32", 4), avg_ratio("fp16", 4)
r2_32 = avg_ratio("fp32", 2)
add("<div class='card hero'>")
add(f"<div class='note' style='border:0;padding:0'><b>{bi('TL;DR','速览')}</b> — "
    f"{bi('Fusing M thresholds into ONE scan is nearly free at M=2 and cheap at M=4 because <code>block_count_ge</code> is memory-bound: the row is read once and only M cheap predicated compares are added. Evaluating 4 thresholds costs ~1.25–1.40× a single scan instead of 4×, so an <b>M-ary Phase-2 refinement</b> reaches the K-th rank in far fewer total scans. <b>M=4 is the robust sweet spot</b> (small-N latency-bound cells even favor M=6–8; the largest fp16 cell favors M=2).',
  '把 M 个阈值融合进一次 scan,在 M=2 几乎免费、M=4 也很便宜——因为 <code>block_count_ge</code> 是 memory-bound:一行只读一次,只多做 M 次廉价谓词比较。评估 4 个阈值仅需 ~1.25–1.40× 单次 scan(而非 4×),故 <b>M-ary 的 Phase-2 细化</b>能以远少的总 scan 数找到第 K 名。<b>M=4 是稳健甜点</b>(小 N 的 latency-bound cell 甚至偏向 M=6–8;最大的 fp16 cell 偏向 M=2)。')}</div>")
add("<div class='kpis'>")
add(f"<div class='kpi'><div class='v'>×{r2_32:.2f}</div><div class='l'>{bi('fp32 M=2 mean cost (vs M=1)','fp32 M=2 平均开销(vs M=1)')}</div></div>")
add(f"<div class='kpi'><div class='v'>×{r4_32:.2f} / ×{r4_16:.2f}</div><div class='l'>{bi('fp32 / fp16 M=4 mean cost','fp32 / fp16 M=4 平均开销')}</div></div>")
add(f"<div class='kpi'><div class='v'>0.31–0.35</div><div class='l'>{bi('M=4 amortized cost per threshold (÷ scan)','M=4 每阈值摊销(÷ 单 scan)')}</div></div>")
add(f"<div class='kpi'><div class='v'>M=4</div><div class='l'>{bi('recommended for Phase-2 refine','Phase-2 细化推荐值')}</div></div>")
add("</div></div>")

# methodology (collapsible)
add(f"<h2><span class='num'>1</span>{bi('Method','方法')}</h2>")
add("<div class='card'><details open><summary>"
    f"{bi('Kernel, input, and timing (click to fold)','kernel、输入与计时(点击折叠)')}</summary>")
add(f"<div class='note'>{bi(
 '<b>Kernel</b> — a standalone CUDA kernel faithfully mirroring the device-side <code>block_count_ge</code> in <code>gvr_topk_decode.py</code>: one CTA per row, 128/256-bit vectorized loads, 4-way unroll (LSU ILP), <b>M static per-thread register GE-counters</b>, M unrolled predicated compares per element, M-wide warp-shuffle + block reduce. This isolates the primitive that Phase-2/3 calls repeatedly.',
 '<b>Kernel</b> — 一个独立 CUDA kernel,忠实复刻 <code>gvr_topk_decode.py</code> 里设备端的 <code>block_count_ge</code>:每行一个 CTA、128/256-bit 向量化 load、4-way 展开(LSU ILP)、<b>M 个静态每线程寄存器 GE 计数器</b>、每元素 M 路展开谓词比较、M 宽 warp-shuffle + block 归约。以此隔离 Phase-2/3 反复调用的这个原语。')}</div>")
add(f"<div class='note'>{bi(
 '<b>Input</b> — identical to <code>report.html</code>: <code>synth_data.get_bundle(K=1024, beta_moderate, cr=4, seed=42)</code> via the <code>swebench-temporal-synth-v4pro</code> skill (V4-Pro temporally-coherent logits), dumped to fp32 &amp; fp16 rows at N=4K…256K. Thresholds are placed around the K-th-rank neighborhood (cost is threshold-value-independent).',
 '<b>输入</b> — 与 <code>report.html</code> 完全一致:经 <code>swebench-temporal-synth-v4pro</code> skill 的 <code>synth_data.get_bundle(K=1024, beta_moderate, cr=4, seed=42)</code>(V4-Pro 时序相干 logits),dump 成 fp32/fp16 行,N=4K…256K。阈值取在第 K 名邻域(开销与阈值取值无关)。')}</div>")
add(f"<div class='note'>{bi(
 '<b>Timing</b> — nsys cold-L2: a 512 MB L2-evict memset before every launch (matches the report’s <code>_EVICT</code>); per-launch <code>cuda_gpu_kern_sum</code> <b>Med</b> (col5) over 60 launches, then the <b>median of 3 repeats</b> (guards against the Avg-inflating cold-launch jitter). BS=1 (the report’s N-sweep regime). Tuning matches production dispatch: <code>T=1024 if N≥65536 else 512</code>; fp32 uses 256-bit vec when N≥16384.',
 '<b>计时</b> — nsys cold-L2:每次 launch 前 512 MB 清 L2(对齐 report 的 <code>_EVICT</code>);取 60 次 launch 的 <code>cuda_gpu_kern_sum</code> <b>Med</b>(col5),再取 <b>3 次复现的中位数</b>(避免 Avg 被冷启动抖动抬高)。BS=1(report 的 N-sweep regime)。Tuning 对齐生产:<code>T=1024 if N≥65536 else 512</code>;fp32 在 N≥16384 用 256-bit 向量。')}</div>")
add("</details></div>")

# ---- §2 current block_count_ge compute logic (M=1) ----
add(f"<h2><span class='num'>2</span>{bi('Current <code>block_count_ge</code> compute logic (M=1)','当前 <code>block_count_ge</code> 计算逻辑(M=1)')}</h2>")
add("<div class='card'>")
add(f"<div class='note'>{bi(
 'The production <code>block_count_ge</code> counts <code>input[i] &gt;= threshold</code> over this CTA&#39;s row slice for ONE threshold, and is what Phase-2 (secant) and Phase-3 call repeatedly. It is a memory-bandwidth-bound streaming reduction, near-optimal for a single threshold:',
 '生产版 <code>block_count_ge</code> 针对<b>单个</b>阈值,统计本 CTA 行切片内 <code>input[i] &gt;= threshold</code> 的数量,是 Phase-2(secant)与 Phase-3 反复调用的原语。它是 memory-bandwidth-bound 的流式归约,对单阈值已近乎最优:')}</div>")
add("<ol>"
    f"<li>{bi('Each thread streams its strided slice with 128/256-bit vectorized loads (<code>vec_w = vec_bits/dtype.width</code>), <b>4-way LLVM-unrolled</b> so ~4 <code>LDG</code> stay in flight (LSU ILP).','每线程以 128/256-bit 向量化 load 扫描其跨步切片(<code>vec_w = vec_bits/dtype.width</code>),<b>4-way LLVM 展开</b>,~4 条 <code>LDG</code> 同时在飞(LSU ILP)。')}</li>"
    f"<li>{bi('Per element a single <b>predicated</b> increment <code>c += (v &gt;= thr)</code> — no branch divergence.','每元素一次<b>谓词化</b>自增 <code>c += (v &gt;= thr)</code> —— 无分支发散。')}</li>"
    f"<li>{bi('Scalar tail loop for the &lt;vec_w remainder; then <code>smem_ptcnt[tid]=c</code> caches the per-thread count for Phase-3 prefix-sum reuse.','标量尾循环处理 &lt;vec_w 余数;随后 <code>smem_ptcnt[tid]=c</code> 缓存每线程计数供 Phase-3 前缀和复用。')}</li>"
    f"<li>{bi('Warp-shuffle sum → <code>smem_wcnt[warp]</code> → block reduce (warp-parallel in warp 0, or serial tid0) → <code>s_iscalars[0] = cand_count</code>.','Warp-shuffle 求和 → <code>smem_wcnt[warp]</code> → block 归约(warp 0 并行或 tid0 串行)→ <code>s_iscalars[0] = cand_count</code>。')}</li>"
    f"<li>{bi('At <code>cluster_size&gt;1</code>: DSMEM <code>mapa.shared::cluster</code> all-reduce across peer CTAs (no GMEM atomics). An optional SMEM-slice cache lets the snap loop re-scan from SMEM instead of GMEM.','<code>cluster_size&gt;1</code> 时:通过 DSMEM <code>mapa.shared::cluster</code> 跨 peer CTA all-reduce(无 GMEM 原子)。可选的 SMEM-slice 缓存让 snap 循环从 SMEM 而非 GMEM 重扫。')}</li>"
    "</ol>")
add("<pre><span class='c'># current block_count_ge — ONE threshold</span>\n"
    "c = 0\n"
    "<span class='k'>for</span> each vec_w chunk (4-way unrolled):   <span class='c'># 128/256-bit vec load</span>\n"
    "    <span class='k'>for</span> j <span class='k'>in</span> vec_w: c += (v[j] &gt;= thr)      <span class='c'># 1 predicated add</span>\n"
    "scalar-tail; smem_ptcnt[tid] = c            <span class='c'># P3 reuse cache</span>\n"
    "c = warp_reduce_sum(c); smem_wcnt[warp] = c\n"
    "block_reduce -> s_iscalars[0] = count(v &gt;= thr)   <span class='c'># +DSMEM all-reduce if cluster</span></pre>")
add("</div>")

# ---- §3 best multi-threshold implementation + shortcomings ----
add(f"<h2><span class='num'>3</span>{bi('Multi-threshold: best implementation &amp; current gaps','多阈值:最佳实现思路与当前不足')}</h2>")
add("<div class='card'>")
add(f"<div class='note'>{bi(
 'To evaluate M thresholds in ONE scan, keep the identical vectorized/unrolled <b>memory path</b> (the row is read once) and add M cheap compares. Best shape: M sorted thresholds + M counters in registers, M unrolled predicated adds per element, an M-wide reduce.',
 '要在一次 scan 内评估 M 个阈值,保持完全相同的向量化/展开<b>内存路径</b>(行只读一次),只增加 M 次廉价比较。最佳形态:M 个有序阈值 + M 个计数器放寄存器,每元素 M 路展开谓词加,末尾 M 宽归约。')}</div>")
add("<pre><span class='c'># block_count_ge_multi&lt;M&gt; — M compile-time constant</span>\n"
    "<span class='m'>float</span> thr[M]; <span class='m'>int</span> c[M] = {0}\n"
    "<span class='k'>for</span> each vec_w chunk (4-way unrolled):     <span class='c'># SAME memory path as M=1</span>\n"
    "    <span class='k'>for</span> j <span class='k'>in</span> vec_w:\n"
    "        <span class='k'>for</span> m <span class='k'>in</span> M: c[m] += (v[j] &gt;= thr[m])  <span class='c'># M static predicated adds</span>\n"
    "M warp_reduce_sum; smem_wcnt[num_warps*M]; <span class='c'>1 barrier</span>; block-reduce M columns</pre>")
add(f"<h3>{bi('Design choices','设计要点')}</h3>")
add("<ol>"
    f"<li>{bi('<b>M static register counters, NOT a difference-array.</b> The “one increment per element” trick (compute <code>k = #(v&gt;=thr_m)</code>, then <code>d[k]++</code>, suffix-sum at the end) needs a <b>dynamically-indexed</b> register array → spills to local memory on GPU. M unrolled predicated adds (M FSETP + M IADD) are branch-free, spill-free, and cheaper for M≤8.',
 '<b>用 M 个静态寄存器计数器,而非 difference-array。</b>「每元素只加一次」的技巧(算 <code>k = #(v&gt;=thr_m)</code> 再 <code>d[k]++</code>,末尾后缀和)需要<b>动态下标</b>寄存器数组 → GPU 上 spill 到 local memory。M 路展开谓词加(M FSETP + M IADD)无分支、无 spill,M≤8 时更省。')}</li>"
    f"<li>{bi('<b>Sorted thresholds ⇒ M-ary search.</b> Evaluating M points/scan splits the bracket (M+1)-ways → ~log_(M+1) rounds vs log_2 for M=1. This is a continuum: <b>M=1 = secant</b>, <b>M→kNumBins = the Phase-4 histogram</b>.',
 '<b>阈值有序 ⇒ M-ary 搜索。</b>一次 scan 评估 M 个点,把区间分成 (M+1) 份 → 约 log_(M+1) 轮(M=1 为 log_2)。这是一个连续谱:<b>M=1=secant</b>,<b>M→kNumBins=Phase-4 直方图</b>。')}</li>"
    f"<li>{bi('<b>Reduction layout:</b> stage <code>smem_wcnt[num_warps*M]</code> and reduce M columns in warp 0 with a SINGLE barrier — do not do M serial block-reduces.',
 '<b>归约布局:</b>用 <code>smem_wcnt[num_warps*M]</code> 暂存,在 warp 0 用<b>一个</b> barrier 归约 M 列 —— 不要做 M 次串行 block-reduce。')}</li>"
    f"<li>{bi('<b>P3 cache:</b> don’t cache all M per-thread counts (M× SMEM). After the M-ary round picks one sub-bracket, recompute a single M=1 <code>block_count_ge</code> at the chosen threshold for the Phase-3 collect (1 extra scan, keeps SMEM flat).',
 '<b>P3 缓存:</b>不要缓存全部 M 份每线程计数(M× SMEM)。M-ary 一轮定出子区间后,在选定阈值上补跑一次 M=1 <code>block_count_ge</code> 供 Phase-3 collect(多 1 次 scan,SMEM 不膨胀)。')}</li>"
    "</ol>")
add(f"<div class='note key'><b>{bi('Current gaps / shortcomings','当前不足')}</b><ol>"
    f"<li>{bi('<b>Not yet wired into the real kernel</b> — this is a standalone micro-bench; the actual Phase-2 secant still evaluates one threshold/scan. The net-benefit below is a MODEL; the end-to-end P2-refine A/B has not been run.',
 '<b>尚未接入真实 kernel</b> —— 这是独立微基准;真实 Phase-2 secant 仍每次 scan 一个阈值。下文净收益是<b>模型</b>,端到端 P2 细化 A/B 未跑。')}</li>"
    f"<li>{bi('<b>M=1 baseline is modeled as bisection</b> (log_2); GVR’s real M=1 is a preIdx-seeded <b>superlinear secant</b>, so the modeled gain of M-ary over the <i>actual</i> secant is optimistic — must be confirmed by a real A/B.',
 '<b>M=1 基线按 bisection 建模</b>(log_2);GVR 实际 M=1 是 preIdx 播种的<b>超线性 secant</b>,故 M-ary 相对<i>真实</i> secant 的收益偏乐观 —— 需真实 A/B 确认。')}</li>"
    f"<li>{bi('<b>BS=1 only</b> (matches the report’s N-sweep). The throughput regime (large BS, GPU-saturated) is not measured; M-scaling there may become compute-bound sooner.',
 '<b>仅 BS=1</b>(对齐 report 的 N-sweep)。吞吐 regime(大 BS、GPU 打满)未测;那里 M-scaling 可能更早变 compute-bound。')}</li>"
    f"<li>{bi('<b>Register/occupancy at M=6/8 not validated in-kernel.</b> The isolated micro-bench lacks GVR’s ~40–70-reg pressure; adding M counters+thresholds inside the full kernel could cross an occupancy tier (esp. fp32 <code>min_blocks_per_mp=2</code>) and change the curve.',
 '<b>M=6/8 的寄存器/occupancy 未在完整 kernel 内验证。</b>独立微基准没有 GVR ~40–70 寄存器压力;在完整 kernel 里加 M 计数器+阈值可能跨过 occupancy 档位(尤其 fp32 <code>min_blocks_per_mp=2</code>),改变曲线。')}</li>"
    f"<li>{bi('<b>fp16/bf16 compare goes through an fp32 cvt</b> (threshold is fp32); a native-precision compare could shave a cvt — unmeasured.',
 '<b>fp16/bf16 比较经过一次 fp32 cvt</b>(阈值是 fp32);原生精度比较可省一次 cvt —— 未测。')}</li>"
    f"<li>{bi('<b><code>smem_ptcnt</code> is written every call</b> (existing TODO) — a wasted STS when P3 reuse isn’t needed, and it multiplies by M in a naive multi-threshold version unless guarded.',
 '<b><code>smem_ptcnt</code> 每次调用都写</b>(已有 TODO)—— 不需要 P3 复用时是浪费的 STS,且朴素多阈值版本若不加保护会乘以 M。')}</li>"
    "</ol></div>")
add("</div>")

# results per dtype
sec = 4
for dt in DTS:
    add(f"<h2><span class='num'>{sec}</span>{bi(dt.upper()+' results','结果 '+dt.upper())}</h2>"); sec += 1
    add("<div class='card'>")
    add(f"<h3>{bi('Per-launch time (µs) — cell color = overhead ÷ M=1','单次 launch 时间(µs)— 单元颜色 = 相对 M=1 倍率')}</h3>")
    add("<p class='legend'><span class='sw' style='background:#d9f2df'></span>"
        f"{bi('≈1× (free)','≈1×(免费)')}<span class='sw' style='background:#fff3b0'></span>"
        f"{bi('~1.7×','~1.7×')}<span class='sw' style='background:#ffd0cf'></span>{bi('≥2.4×','≥2.4×')}</p>")
    add(raw_table(dt))
    add("<div class='grid' style='margin-top:14px'>")
    add(f"<div class='fig'>{charts[(dt,'abs')]}<div class='cap'>{bi('Absolute cold-L2 time grows with N; curves fan out by M.','冷 L2 绝对时间随 N 增长;曲线按 M 展开。')}</div></div>")
    add(f"<div class='fig'>{charts[(dt,'ratio')]}<div class='cap'>{bi('Overhead vs M=1: flat till M=2, steeper at large N.','相对 M=1 的倍率:到 M=2 基本持平,大 N 处变陡。')}</div></div>")
    add(f"<div class='fig'>{charts[(dt,'amort')]}<div class='cap'>{bi('Amortized cost per threshold ≪ 1 → fusion beats separate scans.','每阈值摊销 ≪ 1 → 融合优于分开 scan。')}</div></div>")
    add(f"<div class='fig'>{charts[(dt,'net')]}<div class='cap'>{bi('Est. total P2 time (cost×rounds): a clear minimum near M=4–6.','估计 P2 总时间(cost×轮数):M=4–6 处明显最小。')}</div></div>")
    add("</div></div>")

# net-benefit table
add(f"<h2><span class='num'>{sec}</span>{bi('Net benefit — optimal M','净收益 — 最优 M')}</h2>"); sec += 1
add("<div class='card'>")
add(f"<div class='note'>{bi(
 'Model: an M-ary bisection round evaluates M thresholds in one scan and splits the bracket into (M+1) parts ⇒ rounds ∝ 1/log₂(M+1). <b>Estimated total refine time ÷ M=1</b> = <code>time(M)/time(1) ÷ log₂(M+1)</code>; &lt;1 means the fused M-ary scan finds the K-th rank in less total time. Cell color: green=faster, ★=row minimum. Caveat: the M=1 column models plain bisection; GVR’s real M=1 uses a preIdx-seeded superlinear secant, so read this as a within-bisection-family comparison.',
 '模型:一轮 M-ary bisection 在一次 scan 内评估 M 个阈值,把区间分成 (M+1) 份 ⇒ 轮数 ∝ 1/log₂(M+1)。<b>估计总细化时间 ÷ M=1</b> = <code>time(M)/time(1) ÷ log₂(M+1)</code>;&lt;1 表示融合的 M-ary scan 总时间更短。单元颜色:绿=更快,★=该行最小。注意:M=1 列按普通 bisection 建模;GVR 实际 M=1 用 preIdx 播种的超线性 secant,故应视为 bisection 族内比较。')}</div>")
add(net_table())
add("</div>")

# findings
add(f"<h2><span class='num'>{sec}</span>{bi('Key findings &amp; recommendation','关键结论与建议')}</h2>")
le, lz = [], []
for dt in DTS:
    le.append(f"<li><b>{dt}</b>: mean overhead M2 ×{avg_ratio(dt,2):.2f}, M4 ×{avg_ratio(dt,4):.2f}, M6 ×{avg_ratio(dt,6):.2f}, M8 ×{avg_ratio(dt,8):.2f}; amortized/threshold at M4 = {avg_ratio(dt,4)/4:.2f}.</li>")
    lz.append(f"<li><b>{dt}</b>:平均倍率 M2 ×{avg_ratio(dt,2):.2f}、M4 ×{avg_ratio(dt,4):.2f}、M6 ×{avg_ratio(dt,6):.2f}、M8 ×{avg_ratio(dt,8):.2f};M4 每阈值摊销 = {avg_ratio(dt,4)/4:.2f}。</li>")
add(f"<div class='card'><div class='note key'><b>{bi('Measured','实测')}</b><ul>"
    f"<span class='en'>{''.join(le)}</span><span class='zh'>{''.join(lz)}</span></ul></div>")
add(f"<div class='note'><b>{bi('Two regimes','两个 regime')}</b> — {bi(
 'Small N (4–8K) is <b>latency-bound</b>: the single CTA stalls on memory, the M compares are fully hidden, so even M=8 is ≤1.2× and the net-benefit favors M=6–8. Large N (128–256K) is <b>bandwidth/compute-exposed</b>: the M compares surface (M=8 up to 2.4× fp32 / 3.4× fp16), pulling the optimum back to M=4 (fp16 256K even to M=2). fp16 steepens earlier because it reads half the bytes → memory finishes sooner → compares dominate at smaller N.',
 '小 N(4–8K)为 <b>latency-bound</b>:单 CTA 卡在内存延迟,M 次比较被完全掩盖,故 M=8 也 ≤1.2×,净收益偏向 M=6–8。大 N(128–256K)为 <b>bandwidth/compute-exposed</b>:比较开始显现(M=8 达 fp32 2.4× / fp16 3.4×),把最优拉回 M=4(fp16 256K 甚至到 M=2)。fp16 更早变陡:读的字节减半 → 内存更早完成 → 比较在更小 N 就占主导。')}</div>")
add(f"<div class='note rec'><b>{bi('Recommendation','建议')}</b> — {bi(
 'Use <b>M=4</b> for the GVR Phase-2 refine: near-free at small/mid N and still net-positive at 256K, converging in ~log₅ rounds (~2.3× fewer than binary). Do NOT push M to 6–8 for large N — the added compares/registers erode the gain and you are effectively rebuilding a tiny histogram; hand the final K-th resolution to the existing Phase-4 histogram (all bins in one scan). Design a <code>block_count_ge_multi&lt;M&gt;</code> with M a compile-time constant so the M compares and M-wide reduce fully unroll.',
 '在 GVR Phase-2 细化用 <b>M=4</b>:小/中 N 近乎免费,256K 仍净赚,约 log₅ 轮收敛(比二分少 ~2.3×)。大 N 不要上到 6–8——额外比较/寄存器吃掉收益,且相当于在重造一个小直方图;最终第 K 名定位交给已有的 Phase-4 直方图(一遍出所有 bin)。实现 <code>block_count_ge_multi&lt;M&gt;</code> 时 M 用编译期常量,使 M 路比较与 M 宽归约完全展开。')}</div></div>")

add("</div></body></html>")
out = HERE / "REPORT.html"
out.write_text("".join(H), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} B, <script:{out.read_text().count('<script')})")
