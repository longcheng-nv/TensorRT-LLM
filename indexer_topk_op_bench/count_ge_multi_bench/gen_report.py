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

# results per dtype
sec = 2
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
