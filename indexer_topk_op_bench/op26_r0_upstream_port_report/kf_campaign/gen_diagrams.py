# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotent injector: inline-SVG flow diagrams -> KF_PROCESS_LOG.html.

Splices content between <!-- KF-DIAGRAMS:START --> / :END markers (inserted
after the KPI card on first run). Pure SVG + CSS, no <script>.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / "KF_PROCESS_LOG.html"

BOX = ('<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
       'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
TXT = ('<text x="{x}" y="{y}" font-size="{fs}" text-anchor="middle" '
       'fill="{fill}" font-family="-apple-system,Segoe UI,sans-serif"'
       '{extra}>{t}</text>')


def box(x, y, w, h, lines, fill="#ffffff", stroke="#0b6e4f", fs=12,
        tfill="#1a1a2e", bold_first=False):
    s = [BOX.format(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke)]
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * (fs + 3) / 2 + fs * 0.35
    for i, ln in enumerate(lines):
        extra = ' font-weight="700"' if (bold_first and i == 0) else ""
        s.append(TXT.format(x=x + w / 2, y=y0 + i * (fs + 3), fs=fs,
                            fill=tfill, extra=extra, t=ln))
    return "".join(s)


def arrow(x1, y1, x2, y2, label="", dash=False, color="#555"):
    d = ' stroke-dasharray="6 4"' if dash else ""
    s = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
         f'stroke-width="1.6" marker-end="url(#arr)"{d}/>')
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 6
        s += TXT.format(x=mx, y=my, fs=11, fill="#555", extra="", t=label)
    return s


DEFS = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/></marker></defs>')

G = "#eef7f2"   # green fill
Y = "#fdf3e7"   # amber fill
B = "#eef1f8"   # blue fill
RD = "#fbecec"  # red-ish


def fig(svg, cap_en, cap_cn, w):
    return (f'<figure style="margin:18px 0;text-align:center">'
            f'<svg viewBox="0 0 {w[0]} {w[1]}" style="max-width:100%;'
            f'background:#fff;border:1px solid #ddd;border-radius:8px">'
            f'{DEFS}{svg}</svg>'
            f'<figcaption style="font-size:0.9em;color:#444;margin-top:6px">'
            f'<b>{cap_en}</b><br>{cap_cn}</figcaption></figure>')


# ---------------- Diagram A: campaign lifecycle ----------------
a = []
a.append(box(20, 30, 190, 96, ["Problem bundle", "definition.json", "workload.jsonl ×28",
                               "baselines.jsonl · prompt.md", "assets ×28 (.safetensors)"],
             fill=B, fs=11, bold_first=True))
a.append(arrow(210, 78, 260, 78))
a.append(box(260, 52, 120, 52, ["kf campaign", "init"], fill=G, bold_first=True))
a.append(arrow(380, 78, 430, 78, "campaign.yaml"))
a.append(box(430, 52, 130, 52, ["prepare", "(validate +", "baseline stamp)"], fill=G, fs=11, bold_first=True))
a.append(arrow(560, 78, 610, 78))
a.append(box(610, 52, 100, 52, ["start"], fill=G, bold_first=True))
a.append(arrow(710, 78, 760, 78))
# round loop container
a.append('<rect x="760" y="16" width="380" height="330" rx="10" fill="#fafcf9" '
         'stroke="#0b6e4f" stroke-width="1.8" stroke-dasharray="8 5"/>')
a.append(TXT.format(x=950, y=38, fs=13, fill="#0b6e4f",
                    extra=' font-weight="700"', t="ROUND r (sequential, r ≤ 20)"))
a.append(box(790, 52, 320, 46, ["spawn 6 agents in parallel",
                                "2×Fable-5(max) · 2×GPT-5.6(xhigh) · 2×Opus-4.8"],
             fill=Y, fs=11, bold_first=True))
a.append(arrow(950, 98, 950, 120))
a.append(box(790, 120, 320, 40, ["agent loop: write CUDA kernel → compile → GPU eval",
                                 "(iterate freely inside 4 h soft budget)"], fill="#fff", fs=10.5))
a.append(arrow(950, 160, 950, 182))
a.append(box(790, 182, 152, 40, ["submit Solution"], fill="#fff", fs=11))
a.append(arrow(942, 202, 970, 202))
a.append(box(970, 182, 140, 40, ["LLM compliance", "judge (bans)"], fill=RD, fs=10.5))
a.append(arrow(950, 222, 950, 244))
a.append(box(790, 244, 320, 40, ["fitness = 40% perf + 30% correct + 20% eff + 10% novel"],
             fill="#fff", fs=10.5))
a.append(arrow(950, 284, 950, 306))
a.append(box(790, 306, 320, 30, ["round summary + shared insights DB → seed round r+1"],
             fill=Y, fs=10.5))
# terminal
a.append(arrow(1140, 180, 1190, 180))
a.append(box(1190, 130, 150, 100, ["terminate when:", "max_rounds 20", "stagnation ≥ 4",
                                   "converged", "8 h wall"], fill=B, fs=11, bold_first=True))
a.append(arrow(1265, 230, 1265, 262))
a.append(box(1190, 262, 150, 44, ["best kernel →", "local harvest"], fill=G, bold_first=True))
DIAG_A = fig("".join(a),
             "Fig. A — KernelFactory campaign lifecycle: evolutionary rounds of parallel LLM agents over a fixed SOLBench problem.",
             "图 A — KernelFactory 战役生命周期:固定 SOLBench 问题上,并行 LLM agent 的进化式轮次。",
             (1360, 360))

# ---------------- Diagram B: per-candidate eval pipeline ----------------
b = []
b.append(box(20, 40, 170, 56, ["Solution sources", "kernel.cu (+ .cpp)", "TVM-FFI entry"],
             fill=B, fs=11, bold_first=True))
b.append(arrow(190, 68, 240, 68))
b.append(box(240, 40, 170, 56, ["compile server", "tvm_ffi.cpp.build →", "benchmark_kernel.so"],
             fill=G, fs=11, bold_first=True))
b.append(arrow(410, 68, 460, 68))
# per-workload loop box
b.append('<rect x="460" y="14" width="700" height="300" rx="10" fill="#fafcf9" '
         'stroke="#0b6e4f" stroke-width="1.8" stroke-dasharray="8 5"/>')
b.append(TXT.format(x=810, y=36, fs=13, fill="#0b6e4f", extra=' font-weight="700"',
                    t="for each of the 28 workloads (GPU eval server, B200)"))
b.append(box(485, 50, 190, 62, ["load inputs", "logits[1,npad] fp32 (blob)",
                                "pre_idx[1,k] i32 · n_valid"], fill=Y, fs=10.5, bold_first=True))
b.append(arrow(580, 112, 580, 136))
b.append(box(485, 136, 190, 48, ["reference run()", "torch.topk ground truth"], fill="#fff", fs=10.5, bold_first=True))
b.append(arrow(675, 160, 715, 160))
b.append(box(715, 136, 190, 48, ["candidate run", "shape/dtype gate"], fill="#fff", fs=10.5, bold_first=True))
b.append(arrow(810, 184, 810, 208))
b.append(box(700, 208, 220, 56, ["custom check_topk", "all mandatory (> kth) present",
                                 "rest ⊆ tie (= kth) · unique · in-range"], fill=RD, fs=10.5, bold_first=True))
b.append(arrow(920, 236, 960, 236, "pass"))
b.append(box(960, 208, 180, 56, ["benchmark loop", "warm-up ≥10 reps,", "CUDA-event timing"], fill=G, fs=10.5, bold_first=True))
b.append(arrow(1050, 264, 1050, 288))
b.append(box(930, 288, 215, 20, [""], fill="#fff", stroke="#fff"))
b.append(TXT.format(x=1050, y=302, fs=11, fill="#1a1a2e", extra="",
                    t="latency_ms → speedup = baseline / candidate"))
b.append(arrow(1160, 236, 1210, 236))
b.append(box(1210, 190, 130, 92, ["Trace", "status · latency", "speedup ·", "logs"], fill=B, fs=11, bold_first=True))
DIAG_B = fig("".join(b),
             "Fig. B — CudaGym/SOLBench per-candidate evaluation harness (correctness gate before timing; speedup vs the stamped PR baseline).",
             "图 B — CudaGym/SOLBench 单候选评测流水线(先正确性门再计时;加速比对 prepare 盖章的 PR 基线)。",
             (1360, 330))

# ---------------- Diagram C: local <-> cloud dataflow ----------------
c = []
c.append('<rect x="14" y="14" width="600" height="420" rx="10" fill="#f4f8f4" stroke="#0b6e4f" stroke-width="1.6"/>')
c.append(TXT.format(x=314, y=38, fs=13, fill="#0b6e4f", extra=' font-weight="700"',
                    t="LOCAL · umbriel-b200-027 (8× B200, idle)"))
c.append('<rect x="680" y="14" width="600" height="420" rx="10" fill="#f4f6fb" stroke="#33518f" stroke-width="1.6"/>')
c.append(TXT.format(x=980, y=38, fs=13, fill="#33518f", extra=' font-weight="700"',
                    t="CLOUD · kernelfactory.nvidia.com (managed B200 pool)"))
c.append(box(40, 56, 250, 62, ["§4 real data (865 cells)", "real_3arm_layers_full.csv (PR baseline)",
                               "decode-capture slims (V4/V3.2)"], fill=B, fs=10.5, bold_first=True))
c.append(arrow(165, 118, 165, 142))
c.append(box(40, 142, 250, 56, ["export_cells.py", "28-cell stratified subset →",
                                "safetensors + tie/mandatory sets"], fill=G, fs=10.5, bold_first=True))
c.append(arrow(290, 170, 700, 90, "kf campaign init/prepare/start"))
c.append(box(700, 60, 260, 60, ["campaign", "tfb91bvwm972kfyf1bc1trj5e0",
                                "cuda_cpp · effort max · ≤20 rounds"], fill=Y, fs=10.5, bold_first=True))
c.append(arrow(830, 120, 830, 146))
c.append(box(700, 146, 260, 54, ["rounds: 6 agents evolve kernels", "(Fig. A) · eval harness (Fig. B)"],
             fill="#fff", fs=10.5, bold_first=True))
c.append(arrow(830, 200, 830, 226))
c.append(box(700, 226, 260, 48, ["candidates DB", "kernel list / results / insights"], fill=B, fs=10.5, bold_first=True))
c.append(arrow(700, 250, 320, 250, "harvest: kf campaign results"))
c.append(box(40, 226, 280, 48, ["quick_ab.py (exactness smoke)", "28-cell candidate vs PR-head"], fill="#fff", fs=10.5, bold_first=True))
c.append(arrow(180, 274, 180, 298))
c.append(box(40, 298, 280, 60, ["nsys_ab.py — house protocol", "cold-L2 512MB evict · NVTX GPU projection",
                                "paired same-GPU vs gvrpkg_head @e6fdbfac"], fill=G, fs=10.5, bold_first=True))
c.append(arrow(180, 358, 180, 382))
c.append(box(40, 382, 400, 40, ["VERDICT: 865-cell full grid · geomean ≥1.20 · zero regression · all exact"],
             fill=RD, fs=11, bold_first=True))
c.append(arrow(320, 320, 700, 320, "regressions found → fork --append-prompt", True))
c.append(box(700, 298, 260, 48, ["fork: rewind/extend rounds", "with corrective steering"], fill=Y, fs=10.5, bold_first=True))
c.append(arrow(960, 322, 1010, 322, "", True))
c.append(box(1010, 298, 250, 48, ["new rounds see the local", "regression evidence"], fill="#fff", fs=10.5))
DIAG_C = fig("".join(c),
             "Fig. C — End-to-end optimization loop: cloud campaign evolves kernels; local 8×B200 arm re-verifies with nsys cold-L2 and steers via fork.",
             "图 C — 端到端优化闭环:云端战役演化 kernel;本机 8×B200 臂用 nsys 冷-L2 复验并经 fork 回灌导向。",
             (1300, 450))

block = ("<!-- KF-DIAGRAMS:START -->\n"
         '<h2>0 · Harness at a glance / 流程图总览</h2>\n'
         + DIAG_A + DIAG_B + DIAG_C +
         "\n<!-- KF-DIAGRAMS:END -->")

html = HTML.read_text()
if "KF-DIAGRAMS:START" in html:
    pre, rest = html.split("<!-- KF-DIAGRAMS:START -->", 1)
    _, post = rest.split("<!-- KF-DIAGRAMS:END -->", 1)
    html = pre + block + post
else:
    anchor = '<div class="lang-en">'
    html = html.replace(anchor, block + "\n\n" + anchor, 1)
HTML.write_text(html)
print("diagrams injected, html", len(html), "bytes")
