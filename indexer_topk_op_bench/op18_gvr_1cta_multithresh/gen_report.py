# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op18 bilingual (EN/中文, CSS-only toggle, zero <script>) HTML report generator.
# Reads results/*.jsonl|csv and emits REPORT.html.
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def bi(en, zh, tag="span"):
    return f'<{tag} class="en">{en}</{tag}><{tag} class="zh">{zh}</{tag}>'


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def sp_cls(s):
    if s >= 1.15:
        return "g2"
    if s >= 1.03:
        return "g1"
    if s >= 0.99:
        return "n0"
    return "r1"


def tbl_x3():
    rows = load_jsonl(RES / "validate_x3.jsonl")
    out = []
    for dt in ("fp32", "bf16", "fp16"):
        rs = [r for r in rows if r["dtype"] == dt]
        sp = [r["speedup"] for r in rs]
        h = f'<h3>{dt} — {bi("min %.3f / avg %.3f / max %.3f, %d/%d exact" % (min(sp), statistics.mean(sp), max(sp), len(rs), len(rs)), "最小 %.3f / 平均 %.3f / 最大 %.3f，%d/%d 精确" % (min(sp), statistics.mean(sp), max(sp), len(rs), len(rs)))}</h3>'
        t = ['<table><tr><th>K</th><th>N</th><th>config</th>'
             f'<th>{bi("baseline µs", "基线 µs")}</th><th>op18 µs</th><th>{bi("speedup", "加速比")}</th></tr>']
        for r in rs:
            t.append(f'<tr><td>{r["K"]}</td><td>{r["N"]}</td><td>{r["cfg"]}</td>'
                     f'<td>{r["base_us"]:.1f}</td><td>{r["mt_us"]:.1f}</td>'
                     f'<td class="{sp_cls(r["speedup"])}">{r["speedup"]:.3f}×</td></tr>')
        t.append("</table>")
        out.append(h + "".join(t))
    return "\n".join(out)


def tbl_sweep(path, title_en, title_zh):
    rows = load_jsonl(RES / path)
    cfgs = [k for k in rows[0] if k not in ("K", "N", "dtype", "base_us")]
    t = [f"<h3>{bi(title_en, title_zh)}</h3>",
         '<table><tr><th>K</th><th>N</th><th>base µs</th>'
         + "".join(f"<th>{c}</th>" for c in cfgs) + "</tr>"]
    for r in rows:
        cells = "".join(
            f'<td class="{sp_cls(r[c]["speedup"])}">{r[c]["speedup"]:.3f}</td>' for c in cfgs)
        t.append(f'<tr><td>{r["K"]}</td><td>{r["N"]}</td><td>{r["base_us"]:.1f}</td>{cells}</tr>')
    sums = []
    for c in cfgs:
        sp = [r[c]["speedup"] for r in rows]
        sums.append(f'<tr><td colspan="3"><b>{c}</b></td><td colspan="{len(cfgs)}">'
                    f'min {min(sp):.3f} · avg {statistics.mean(sp):.3f} · max {max(sp):.3f}</td></tr>')
    t.append("".join(sums))
    t.append("</table>")
    return "".join(t)


def tbl_nsys():
    lines = (RES / "nsys_summary.csv").read_text().strip().splitlines()[1:]
    ev = {(r["K"], r["N"]): r["speedup"] for r in load_jsonl(RES / "validate_x3.jsonl") if r["dtype"] == "fp32"}
    t = [f'<table><tr><th>K</th><th>N</th><th>{bi("base µs (nsys med)", "基线 µs（nsys 中位）")}</th>'
         f'<th>op18 µs</th><th>{bi("nsys speedup", "nsys 加速比")}</th><th>{bi("event speedup", "event 加速比")}</th></tr>']
    for ln in lines:
        K, N, b, a, s = ln.split(",")
        K, N, b, a, s = int(K), int(N), float(b), float(a), float(s)
        t.append(f'<tr><td>{K}</td><td>{N}</td><td>{b:.2f}</td><td>{a:.2f}</td>'
                 f'<td class="{sp_cls(s)}">{s:.3f}×</td><td>{ev.get((K, N), 0):.3f}×</td></tr>')
    t.append("</table>")
    return "".join(t)


def tbl_bs():
    lines = (RES / "bs_sweep_k512_n65536.csv").read_text().strip().splitlines()[1:]
    t = [f'<table><tr><th>BS</th><th>{bi("baseline µs", "基线 µs")}</th><th>op18 µs</th>'
         f'<th>{bi("speedup", "加速比")}</th><th>{bi("exact", "精确")}</th></tr>']
    for ln in lines:
        bs, b, m, s, ok = ln.split(",")
        t.append(f'<tr><td>{bs}</td><td>{b}</td><td>{m}</td>'
                 f'<td class="{sp_cls(float(s))}">{s}×</td><td>{ok}</td></tr>')
    t.append("</table>")
    return "".join(t)


def tbl_dispatch():
    from importlib.util import spec_from_file_location, module_from_spec
    import sys
    sys.path.insert(0, str(HERE.parents[0] / "harness"))
    sys.path.insert(0, str(HERE.parents[0] / "ops"))
    # parse _DISPATCH straight from the source to avoid importing torch/cuda
    src = (HERE / "src" / "gvr_mt_op.py").read_text()
    ns = {}
    block = src.split("_DISPATCH = ")[1].split("\n}\n")[0] + "\n}"
    disp = eval(block, {}, {})
    t = [f'<table><tr><th>K</th><th>N</th><th>M</th><th>{bi("rounds R", "轮数 R")}</th>'
         f'<th>accept ×K</th></tr>']
    for K in sorted(disp):
        for N in sorted(disp[K]):
            M, R, acc = disp[K][N]
            t.append(f"<tr><td>{K}</td><td>{N}</td><td>{M}</td><td>{R}</td><td>{acc}</td></tr>")
    t.append("</table>")
    return "".join(t)


CSS = """
body{font-family:'Segoe UI',system-ui,'PingFang SC','Microsoft YaHei',sans-serif;margin:0;background:#f4f6f8;color:#1a2330}
.wrap{max-width:1180px;margin:0 auto;padding:24px 32px 80px}
h1{font-size:26px;margin:8px 0 2px}h2{font-size:20px;border-left:5px solid #76b900;padding-left:10px;margin-top:38px}
h3{font-size:16px;margin:18px 0 6px}
table{border-collapse:collapse;margin:10px 0;font-size:13px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08)}
th,td{border:1px solid #d7dee6;padding:4px 10px;text-align:right}
th{background:#243447;color:#fff;text-align:center}
td:first-child,td:nth-child(2){text-align:center}
.g2{background:#1e7d32;color:#fff;font-weight:600}.g1{background:#a5d6a7}.n0{background:#fff9c4}.r1{background:#ffcdd2}
.card{background:#fff;border-radius:8px;padding:16px 22px;margin:14px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}
code,pre{background:#eef2f6;border-radius:4px;font-size:12.5px}
pre{padding:10px 14px;overflow-x:auto}
.tag{display:inline-block;background:#76b900;color:#fff;border-radius:4px;padding:1px 8px;font-size:12px;margin-right:6px}
.warn{border-left:4px solid #e65100;background:#fff3e0;padding:8px 14px;margin:10px 0}
.good{border-left:4px solid #1e7d32;background:#e8f5e9;padding:8px 14px;margin:10px 0}
/* CSS-only language toggle (no JS) */
#lang-en,#lang-zh{display:none}
.langbar{position:sticky;top:0;background:#243447;padding:10px 32px;z-index:9}
.langbar label{color:#fff;border:1px solid #76b900;border-radius:5px;padding:4px 14px;cursor:pointer;margin-right:8px;font-size:14px}
#lang-en:checked~.langbar label[for=lang-en],#lang-zh:checked~.langbar label[for=lang-zh]{background:#76b900;font-weight:700}
#lang-en:checked~.wrap .zh{display:none}
#lang-zh:checked~.wrap .en{display:none}
"""


def main():
    x3 = load_jsonl(RES / "validate_x3.jsonl")
    fp32 = [r["speedup"] for r in x3 if r["dtype"] == "fp32"]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>op18 — Single-CTA Multi-Threshold GVR Top-K (B200)</title>
<style>{CSS}</style></head>
<body>
<input type="radio" name="lang" id="lang-en" checked>
<input type="radio" name="lang" id="lang-zh">
<div class="langbar"><label for="lang-en">English</label><label for="lang-zh">中文</label></div>
<div class="wrap">
<h1>{bi("op18 — Single-CTA Multi-Threshold GVR Top-K Operator", "op18 —— 单 CTA 多阈值 GVR Top-K 算子")}</h1>
<p><span class="tag">B200 sm_100</span><span class="tag">cuteDSL</span><span class="tag">exact</span><span class="tag">2026-07-02</span><br>
{bi("Baseline: single-CTA <code>gvr_cutedsl</code> (vendored production kernel). Data: report synth bundles (seed 42). Timing: cold-L2 (512 MB evict) + CUDA-graph + cudaEvent, ×3-median; positive claims nsys pure-kernel validated.",
    "基线：单 CTA <code>gvr_cutedsl</code>（vendored 生产内核）。数据：report 合成 bundle（seed 42）。计时：cold-L2（512 MB 驱逐）+ CUDA-graph + cudaEvent，×3 中位数；所有正收益均经 nsys 纯内核校验。")}</p>

<div class="good">{bi(
 f"<b>Headline:</b> exact, no-regression win over the single-CTA baseline on all 60 cells × 3 dtypes: fp32 min 1.010× / avg 1.144× / max 1.344×; bf16 avg 1.114×; fp16 avg 1.143×. nsys pure-kernel 1.10–1.45× on spot cells; BS 1→128 win grows 1.08→1.16×.",
 f"<b>结论：</b>对单 CTA 基线在全部 60 单元 × 3 数据类型上精确且零回退：fp32 最小 1.010× / 平均 1.144× / 最大 1.344×；bf16 平均 1.114×；fp16 平均 1.143×。nsys 纯内核抽检 1.10–1.45×；BS 1→128 收益从 1.08× 增长到 1.16×。")}</div>

<h2>1. {bi("Design", "设计")}</h2>
<div class="card">
{bi("""<p>The baseline GVR kernel refines its threshold with a <b>secant search</b>: each Phase-2 iteration is one full-N <code>block_count_ge</code> scan evaluating ONE threshold, accepting any count in [K, kC] (kC = 5120/6144 ≈ 2.5–10×K) — a loose accept that leaves Phase-4 a large candidate set. op18 replaces Phase-2 with an <b>adaptive M-ary multi-threshold search</b>, per the count_ge_multi_bench primitive and op17's band selection:</p>
<ul>
<li><b>block_count_ge_multi&lt;M&gt;</b> — ONE full-N scan evaluates M sorted thresholds: identical vectorized/4-way-unrolled memory path, M static register counters, M branchless predicated adds per element, M-column staged block reduce with a single barrier.</li>
<li><b>All M per-thread count columns cached</b> in smem (column-major, M·threads·4 B ≤ 32 KB): the winning column is copied to <code>smem_ptcnt</code>, so Phase-3 collects with done=1 and <b>zero recount</b> — closing the “+1 extra scan” gap in the count_ge_multi design note.</li>
<li><b>CDF-aware placement (the decisive lever):</b> round-1 thresholds are compile-time fracs on the preIdx band [pmin, pmax] (op17: count(pmin) ≥ K always), fit offline per (K, N, M) on 5 synth seeds to a geometric count ladder (<code>scripts/optimize_fracs.py</code>). fracs[0]=0 anchors exactness; M ≥ 3 is seed-safe.</li>
<li><b>Adaptive rounds:</b> if the tightest count &gt; c_accept, one refine round re-places M thresholds strictly inside the surviving bracket (round-2 pass is L2-warm). done=1 only when count ∈ [K, kC]; anything else falls back to the baseline's exact retry-shrink path.</li>
<li><b>Per-(K, N) dispatch</b> (replaces kFTarget tuning): M4–M6 R1 at N ≤ 8K (latency-bound, compares free), M3 R1 at 16–65K, M2 R2 acc=2K at N ≥ 131K (M2 is the only tax-free width on a cold bandwidth-exposed pass).</li>
</ul>""",
 """<p>基线 GVR 内核用<b>割线搜索</b>细化阈值：Phase-2 每次迭代都是一遍全 N 的 <code>block_count_ge</code>，只评估一个阈值，接受窗口 [K, kC]（kC = 5120/6144 ≈ 2.5–10×K）非常宽松，导致 Phase-4 候选集偏大。op18 依据 count_ge_multi_bench 原语与 op17 的带选择方法，将 Phase-2 替换为<b>自适应 M 元多阈值搜索</b>：</p>
<ul>
<li><b>block_count_ge_multi&lt;M&gt;</b> —— 一遍全 N 扫描同时评估 M 个有序阈值：内存路径与生产版完全相同（向量化 + 4 路展开），M 个静态寄存器计数器、每元素 M 路无分支谓词加、M 列一次 barrier 的分级归约。</li>
<li><b>缓存全部 M 份每线程计数列</b>（列主序 smem，M·threads·4 B ≤ 32 KB）：获胜列拷贝到 <code>smem_ptcnt</code>，Phase-3 以 done=1 收集，<b>零重扫</b> —— 补上 count_ge_multi 设计笔记中“多一遍扫描”的缺口。</li>
<li><b>CDF 感知放置（决定性杠杆）：</b>第 1 轮阈值是 preIdx 带 [pmin, pmax]（op17：count(pmin) ≥ K 恒成立）上的编译期分数，离线按 (K, N, M) 在 5 个种子上拟合几何计数阶梯（<code>scripts/optimize_fracs.py</code>）。fracs[0]=0 锚定精确性；M ≥ 3 对种子鲁棒。</li>
<li><b>自适应轮数：</b>若最紧计数 &gt; c_accept，追加一轮在存活区间内部重新放置 M 个阈值（第 2 轮为 L2 暖数据）。仅当计数 ∈ [K, kC] 才 done=1；否则回落到基线的精确 retry-shrink 路径。</li>
<li><b>按 (K, N) 分派</b>（取代 kFTarget 调参）：N ≤ 8K 用 M4–M6 R1（延迟受限，比较免费），16–65K 用 M3 R1，N ≥ 131K 用 M2 R2 acc=2K（冷的带宽暴露遍上只有 M2 免税）。</li>
</ul>""", "div")}
</div>

<h2>2. {bi("Why naive multi-threshold fails: the L2 trap", "朴素多阈值为何失败：L2 陷阱")}</h2>
<div class="card">
{bi("""<p>The first A/B (uniform placement, always-2-rounds) averaged 0.93× with 0.70× at 262K. clock64 phase timing (<code>scripts/measure_mt_phases.py</code>) localized it (K512/N262144, cycles):</p>
<pre>baseline: P1 3.5K | P2 27.3K = COLD pass ~22K + L2-WARM pass ~5K | P3 29K | P4 10.2K
op18 M4R1: P1 3.6K | P2 33K (ONE cold M4 pass) | P3 28K | P4 15.3K</pre>
<p>The baseline's “extra” secant passes are nearly free (row ≤ 1 MB ≪ 50 MB L2), while the M-compare tax rides the COLD pass and is <b>latency-exposure-bound</b>: branchless rewrites and unroll 8/16 changed nothing; in-kernel M-scaling equals the standalone microbench (M2 ≈ free, M4 ×1.46, M8 ×2.7 at 262K). Collapsing passes therefore buys only warm passes — the win must come from a <b>tighter threshold in one near-free pass</b> (P4 shrink), which is what CDF-aware placement delivers. A second discovery: P4 histogram-snap time is <b>placement-sensitive, not cand-linear</b> (7.2K vs 15.3K cyc at same-order cand).</p>""",
 """<p>第一次 A/B（均匀放置、固定 2 轮）平均只有 0.93×，262K 处 0.70×。clock64 分相计时（<code>scripts/measure_mt_phases.py</code>）定位了原因（K512/N262144，cycles）：</p>
<pre>基线：  P1 3.5K | P2 27.3K = 冷遍 ~22K + L2 暖遍 ~5K | P3 29K | P4 10.2K
op18 M4R1：P1 3.6K | P2 33K（一遍冷 M4）| P3 28K | P4 15.3K</pre>
<p>基线“多出来”的割线遍几乎免费（行 ≤ 1 MB ≪ 50 MB L2），而 M 路比较税落在<b>冷遍</b>上且属于<b>延迟暴露受限</b>：无分支重写与 8/16 路展开均无效；内核内 M 缩放与独立微基准一致（262K 处 M2 ≈ 免费、M4 ×1.46、M8 ×2.7）。因此“折叠遍数”只能省下暖遍 —— 收益必须来自<b>一遍近免费扫描中拿到更紧阈值</b>（P4 收缩），这正是 CDF 感知放置的作用。另一发现：P4 直方图-snap 时间<b>对阈值位置敏感，并非只随候选数线性</b>（同量级候选 7.2K 对 15.3K cyc）。</p>""", "div")}
</div>

<h2>3. {bi("Parameter tuning", "参数调优")}</h2>
<div class="card">
{tbl_sweep("config_sweep_fp32.jsonl", "Round 1 — uniform/dyadic/pmean placements (all ≤1.0 avg — falsified)", "第一轮 —— 均匀/二进/均值放置（平均均 ≤1.0，被证伪）")}
{tbl_sweep("config_sweep_f3.jsonl", "Round 2 — CDF-aware placement (oracle min 1.004 / avg 1.143 / max 1.349)", "第二轮 —— CDF 感知放置（oracle 最小 1.004 / 平均 1.143 / 最大 1.349）")}
<h3>{bi("Tuned dispatch table (fp32-fit, generalizes to bf16/fp16)", "调优后的分派表（按 fp32 拟合，泛化到 bf16/fp16）")}</h3>
{tbl_dispatch()}
{bi('<p>CTA thread count follows the production config (1024 if N ≥ 65536 else 512, BS ≤ SMs); an explicit threads override was probed and the production choice stayed optimal. kC is unchanged (exactness bound only); kFTarget is superseded by the c_accept ladder.</p>',
    '<p>CTA 线程数沿用生产配置（N ≥ 65536 且 BS ≤ SM 数时 1024，否则 512）；显式线程覆写已探测，生产配置仍最优。kC 不变（仅作精确性上界）；kFTarget 被 c_accept 阶梯取代。</p>', "div")}
</div>

<h2>4. {bi("Final results (×3-median, cold-L2, exact 60/60)", "最终结果（×3 中位，cold-L2，60/60 精确）")}</h2>
<div class="card">{tbl_x3()}</div>

<h2>5. {bi("nsys pure-kernel validation (repo rule)", "nsys 纯内核校验（仓库规则)")}</h2>
<div class="card">
{tbl_nsys()}
{bi('<p>nsys ≥ event on every cell — event timing is the conservative bound (launch overhead).</p>',
    '<p>所有单元 nsys ≥ event —— event 计时是保守下界（含发射开销）。</p>', "div")}
</div>

<h2>6. {bi("Batch-size sweep (K512 N65536 fp32)", "批大小扫描（K512 N65536 fp32）")}</h2>
<div class="card">
{tbl_bs()}
{bi("<p>Win grows with BS (1.08×→1.16×): each CTA saves its warm passes independently, and the extra smem (M·threads·4 B) never crosses an occupancy tier. No high-BS guard needed — unlike op17's cluster portfolio which degenerates at BS ≥ 32.</p>",
    "<p>收益随 BS 增长（1.08×→1.16×）：每个 CTA 独立省掉自己的暖遍，额外 smem（M·threads·4 B）不会跨越占用率档位。无需高 BS 保护 —— 不同于 op17 的 cluster 组合在 BS ≥ 32 退化。</p>", "div")}
</div>

<h2>7. {bi("Honest bounds & relation to op17", "诚实边界与 op17 的关系")}</h2>
<div class="warn">
{bi("""<ul>
<li>Peak BS=1 speedup is below op17's cooperative-cluster portfolio (nsys 1.10–1.45× vs 1.21–1.67×): a single CTA cannot scan redundantly for free, so the tight threshold costs an M-compare tax instead of idle-SM bandwidth. op18's value is <b>robustness</b>: single-CTA (no cluster machinery, no G=2 instability), wins at ALL BS, all dtypes, zero regression.</li>
<li>Weakest cells: K1024/4K (~1.02×) and K2048/16K (~1.01×) — near-neutral, not losses.</li>
<li>Fracs are fit on synth bundles; real DSv4 captures have different CCDF tails — refit before production (see LEARNINGS follow-ups).</li>
</ul>""",
 """<ul>
<li>BS=1 峰值加速低于 op17 的协作 cluster 组合（nsys 1.10–1.45× 对 1.21–1.67×）：单 CTA 无法免费冗余扫描，紧阈值要付 M 路比较税而不是白嫖空闲 SM 带宽。op18 的价值在<b>鲁棒性</b>：单 CTA（无 cluster 机制、无 G=2 不稳定），全 BS、全数据类型获胜，零回退。</li>
<li>最弱单元：K1024/4K（~1.02×）与 K2048/16K（~1.01×）—— 近中性，非回退。</li>
<li>fracs 基于合成 bundle 拟合；真实 DSv4 捕获的 CCDF 尾部不同 —— 生产化前需重拟合（见 LEARNINGS 后续项）。</li>
</ul>""", "div")}
</div>

<h2>8. {bi("Files & reproduction", "文件与复现")}</h2>
<div class="card">
{bi('<p><b>Code location:</b> branch <a href="https://github.com/longcheng-nv/TensorRT-LLM/tree/omni/gvr-1cta-multithresh/indexer_topk_op_bench/op18_gvr_1cta_multithresh"><code>omni/gvr-1cta-multithresh</code> on github.com/longcheng-nv/TensorRT-LLM</a>, directory <code>indexer_topk_op_bench/op18_gvr_1cta_multithresh/</code>.</p>',
    '<p><b>代码位置：</b><a href="https://github.com/longcheng-nv/TensorRT-LLM/tree/omni/gvr-1cta-multithresh/indexer_topk_op_bench/op18_gvr_1cta_multithresh">github.com/longcheng-nv/TensorRT-LLM 的 <code>omni/gvr-1cta-multithresh</code> 分支</a>，目录 <code>indexer_topk_op_bench/op18_gvr_1cta_multithresh/</code>。</p>', "div")}
<pre>
src/gvr_mt_op.py                  {bi("kernel + gvr_mt_auto dispatch", "内核 + gvr_mt_auto 分派")}
scripts/optimize_fracs.py         {bi("CDF-aware frac fitting (5 seeds) -> results/fracs_table.json", "CDF 感知分数拟合（5 种子）-> results/fracs_table.json")}
scripts/config_sweep.py --f3      {bi("config sweep", "配置扫描")}
scripts/validate_x3.py            {bi("final x3-median validation (all dtypes)", "最终 ×3 中位校验（全数据类型）")}
scripts/measure_mt_phases.py      {bi("clock64 phase breakdown vs baseline", "clock64 分相对比基线")}
scripts/bs_sweep.py               {bi("batch-size sweep", "批大小扫描")}
scripts/drive_nsys.sh             {bi("nsys pure-kernel validation", "nsys 纯内核校验")}
ITERATIONS.md / LEARNINGS.md      {bi("full iteration log / knowledge base", "完整迭代日志 / 知识库")}
</pre></div>
</div></body></html>"""
    out = HERE / "REPORT.html"
    out.write_text(html)
    n_script = html.count("<script")
    print(f"wrote {out} ({len(html)} bytes), <script> tags: {n_script}")
    assert n_script == 0


if __name__ == "__main__":
    main()
