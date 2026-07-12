#!/usr/bin/env python3
"""op21 campaign progress plot, faithful to karpathy/autoresearch progress.png form:
x = experiment #, kept improvements = prominent green dots + running-best step line
+ rotated annotations; discarded/falsified = faint gray dots at their measured
(A/B-estimated) y; secondary dtype series thin + direct-labeled.

Data source: op21_gvr_prod/PROGRESS_REPORT.html (iter0.5-14, published 07-08)
+ ITERATIONS.md for the post-report iter15/16 extension.
fp32 P0 gm = nsys pure-kernel cold-L2 geomean vs per-cell best rival, 17 cells.
Falsified-point y values are estimates from their event-A/B ratios, mirroring the
source report's own convention ("y ~ estimate from A/B ratio").
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D

GREEN, DGREEN = "#008300", "#1a5c1a"
GRAY, BLUE, VIOLET, RED = "#9ca3af", "#2a78d6", "#4a3aa7", "#e34948"
INK, MUT = "#0b0b0b", "#52514e"

# (x, y, label, kind)  kind: kept | flat | post
FP32 = [
    (1, 0.830, "iter1 single-CTA v1 (tid0 serial-scan lesson: 45.5->16.3us)", "kept"),
    (2, 1.051, "iter2 row-chunked C-CTA cluster (replicated seeding + DSMEM merge)", "kept"),
    (3, 1.054, "iter3 C8 tier @K2048 hugeN", "kept"),
    (4, 1.054, "iter4 ablation pins P4 = 3.9-7us", "flat"),
    (5, 1.104, "iter5 exact rank-scatter P4 (op8 port)", "kept"),
    (6, 1.125, "iter6 small-bin P4 fast paths (host probe: cnt(b*) max=4)", "kept"),
    (7, 1.249, "iter7 P3 remote-store push -- 17/17 clean sweep", "kept"),
    (8, 1.249, "iter8 16-bit tier", "flat"),
    (9, 1.249, "iter9 native 16-bit ladder", "flat"),
    (10, 1.249, "iter10 B300 HW-invariant (1.268)", "flat"),
    (11, 1.249, "iter11 exactness 0/72 -> 72/72 (perf flat 0.996)", "flat"),
    (12, 1.249, "iter12 PR-1 artifact, upstream 384/384", "flat"),
    (13, 1.249, "iter13 HLS log-falsi fallback (tail axis: K2048-1M 2.105x)", "flat"),
    (14, 1.249, "iter14 distributed msc fallback (1M worst 245->57us)", "flat"),
    (15, 1.276, "iter15 P0 @HEAD (post-report)", "post"),
    (16, 1.298, "iter16 fallback diet, tax<=1% (post-report)", "post"),
]
FALSIFIED = [  # y = estimate from event-A/B ratio (source-report convention)
    (3, 1.020, "dist-P1: +0.6-1.7us/cell worse (L2 makes copies free)"),
    (4, 1.032, "dist-P4: +0.1-1.7us worse (barriers cost)"),
    (5, 1.108, "P1b QBINS=64: flat 1.004"),
    (6, 1.138, "C8@fp32 holes: +0.6-1.3% = noise (flips to WIN @16-bit)"),
]
BF16 = [(8, 1.028), (9, 1.091)]
FP16 = [(8, 1.043), (9, 1.055)]

fig, ax = plt.subplots(figsize=(16, 8))

# faint gray falsified dots (autoresearch "Discarded")
ax.scatter([f[0] for f in FALSIFIED], [f[1] for f in FALSIFIED],
           c=GRAY, s=26, alpha=0.75, zorder=2)
for x, y, lab in FALSIFIED:
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -11),
                fontsize=7.6, color=MUT, rotation=-18, ha="left", va="top")

# running-best step over ALL fp32 points (kept+flat+post)
xs = [p[0] for p in FP32]; ys = [p[1] for p in FP32]
run = [max(ys[:i + 1]) for i in range(len(ys))]
ax.step(xs, run, where="post", color="#27ae60", lw=2, alpha=0.7, zorder=3)

# kept = prominent green; flat = hollow green; post = hollow double-ring
for x, y, lab, kind in FP32:
    if kind == "kept":
        ax.scatter([x], [y], c=GREEN, s=64, zorder=5, edgecolors="black", linewidths=0.6)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 8),
                    fontsize=8.2, color=DGREEN, rotation=30, ha="left", va="bottom")
    elif kind == "flat":
        ax.scatter([x], [y], facecolors="white", edgecolors=GREEN, s=42,
                   linewidths=1.4, zorder=5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 9),
                    fontsize=7.2, color=MUT, rotation=30, ha="left", va="bottom")
    else:
        ax.scatter([x], [y], facecolors="#eaf5ea", edgecolors=GREEN, s=52,
                   linewidths=1.6, zorder=5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 8),
                    fontsize=7.6, color=DGREEN, rotation=30, ha="left", va="bottom")

# exactness-fix marker at iter11
ax.scatter([11], [1.249], facecolors="none", edgecolors=RED, s=150,
           linewidths=1.2, zorder=6)

# 16-bit secondary series, thin + direct labels
for pts, col, lab in [(BF16, BLUE, "bf16 P0 gm"), (FP16, VIOLET, "fp16 P0 gm")]:
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.6,
            marker="o", ms=4.5, zorder=4)
    ax.annotate(f"{lab} {pts[-1][1]:.3f}", pts[-1], textcoords="offset points",
                xytext=(7, -3), fontsize=8, color=col)

ax.axhline(1.0, color=MUT, lw=0.9, ls=":", alpha=0.7)
ax.text(0.62, 1.004, "1.0 = par with the per-cell best rival", fontsize=8,
        color=MUT, va="bottom")

# HLS tail-axis note (P0 flat by design)
ax.annotate("HLS (iter13/14): P0 flat BY DESIGN;\ntail axis: op22 1M worst 245 -> 57 us",
            (13.5, 1.222), fontsize=8.2, color=VIOLET, ha="center", va="top",
            style="italic")

ax.set_xlim(0.4, 17.6); ax.set_ylim(0.78, 1.44)
ax.set_xticks(range(1, 17))
ax.set_xlabel("iteration #  (host prototype iter0.5 gave the GO; 2026-07-05 -> 07-08)", fontsize=11)
ax.set_ylabel("fp32 P0 geomean speedup vs per-cell best rival\n(nsys pure-kernel, cold-L2, B200; higher is better)", fontsize=10)
ax.grid(True, alpha=0.2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("op21 GVR productionization: 16 experiments, 10 kept levers, 6 falsifications, 1 exactness fix"
             "  --  0.830 -> 1.298, executed autonomously in Claude Code (est. cost ~$797)",
             fontsize=12.5, loc="left", color=INK)

handles = [
    Line2D([], [], color="#27ae60", lw=2, label="running best"),
    Line2D([], [], marker="o", ls="", mfc=GREEN, mec="black", ms=8, label="kept improvement"),
    Line2D([], [], marker="o", ls="", mfc="white", mec=GREEN, ms=7, label="fp32 flat (work in other dims)"),
    Line2D([], [], marker="o", ls="", mfc=GRAY, mec=GRAY, ms=5, alpha=0.75,
           label="falsified / discarded (y ~ A/B estimate)"),
    Line2D([], [], marker="o", ls="", mfc="none", mec=RED, ms=10, label="exactness fix (iter11)"),
]
ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=False)

plt.tight_layout()
out = __file__.replace("gen_progress_op21.py", "progress_op21.png")
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print("wrote", out)
