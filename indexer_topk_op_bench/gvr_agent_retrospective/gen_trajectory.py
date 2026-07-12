#!/usr/bin/env python3
"""GVR top-K optimization-history trajectory figure (autoresearch progress-plot style).

Three panels:
  P1  Campaign gantt Jan-Jul 2026, bars colored by outcome (ship/falsified/wash/meta).
  P2  op21->op27 production-candidate trajectory: P0 nsys 17-cell geomean verdict
      vs best rival (consistent metric), kept milestones green + falsification rug.
  P3  Era-2 op-bench commits/day + cumulative shipped vs falsified lever counts.

Data are hand-curated from campaign ledgers (ITERATIONS.md / REPORT.md / git log);
see RETROSPECTIVE.md for sources. Dates approximate to the day.

Palette = pre-validated reference instance from the dataviz skill
(light mode, worst adjacent CVD dE 24.2): blue #2a78d6, green #008300,
yellow #eda100, red #e34948, violet #4a3aa7; gray #9ca3af reserved for
de-emphasized (falsified/discarded) marks.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

C_SHIP, C_META, C_WASH, C_FALS = "#008300", "#2a78d6", "#eda100", "#9ca3af"
C_EVT, C_MATH, INK, MUT = "#e34948", "#4a3aa7", "#0b0b0b", "#52514e"
D = lambda m, d: date(2026, m, d)

# ---------------- P1: campaigns (start, end, outcome, label) ----------------
CAMP = [
    # Era-0
    (D(1,25), D(2,1),  C_META, "onchip_topk / warp-TopL"),
    (D(1,26), D(2,8),  C_SHIP, "preidx GVR embryo (guess-verify)"),
    (D(2,2),  D(2,3),  C_FALS, "gemini sort-based (1 day out)"),
    (D(2,9),  D(2,12), C_FALS, "GVR v1 interp refine (killed by dist)"),
    (D(2,12), D(2,14), C_WASH, "newton root-find branch"),
    (D(2,12), D(2,25), C_WASH, "2-level histogram branch"),
    (D(2,13), D(3,15), C_SHIP, "v2 secant mainline -> V2d-final"),
    (D(3,10), D(3,16), C_SHIP, "TRT-LLM integration + UT"),
    # Era-1
    (D(4,6),  D(4,12), C_META, "Q1 clock64 phase timing"),
    (D(4,20), D(4,25), C_META, "Q3-Q5 deathvalley root cause (NCU)"),
    (D(4,23), D(5,8),  C_SHIP, "Scheme X dispatcher v1.0-1.2"),
    (D(4,25), D(5,8),  C_SHIP, "OPT4-7 + Q1b dtype opts"),
    (D(4,20), D(5,5),  C_FALS, "P1 self-loop v1-v3 (91k cells)"),
    (D(5,8),  D(5,12), C_FALS, "multi-CTA v1/v2 (printf artifact)"),
    (D(5,10), D(5,12), C_SHIP, "CUDA->cuteDSL port (Q16, 2 days)"),
    (D(5,12), D(5,21), C_META, "synth Q18/Q19 temporal + real capture"),
    (D(5,21), D(6,9),  C_META, "V4 sweeps / precision ablation"),
    # Era-2
    (D(6,10), D(6,11), C_META, "op-bench harness (5-op parity)"),
    (D(6,11), D(6,14), C_SHIP, "op7 rank-scatter P4 (PR#15709)"),
    (D(6,11), D(6,13), C_FALS, "op8 turbo 1.5x (infeasible)"),
    (D(6,17), D(6,22), C_FALS, "op9 LB dispatch (always-single wins)"),
    (D(6,20), D(6,21), C_FALS, "op10 GVR-2x (math floor)"),
    (D(6,27), D(6,29), C_WASH, "op12 low-precision (regime dispatch)"),
    (D(6,28), D(7,4),  C_SHIP, "op13 cheaper-P2 log-count"),
    (D(6,28), D(6,29), C_FALS, "op14 1-pass compaction (L2 trap)"),
    (D(6,30), D(7,1),  C_FALS, "op15 smem-resident (warm-L2 veto)"),
    (D(7,1),  D(7,2),  C_FALS, "op16 dual-threshold (2 pivots)"),
    (D(7,1),  D(7,2),  C_SHIP, "op17 cooperative portfolio"),
    (D(7,2),  D(7,3),  C_SHIP, "op18 M-ary ladder + CDF fracs"),
    (D(7,2),  D(7,3),  C_SHIP, "op19 sandwich (720 cells)"),
    (D(7,3),  D(7,4),  C_META, "op20 extreme: wall attribution"),
    (D(7,5),  D(7,8),  C_SHIP, "op21 prod: HLS ship (iter0.5-16)"),
    (D(7,7),  D(7,10), C_META, "op22 12-arm arena (81 batches)"),
    (D(7,7),  D(7,9),  C_META, "op23 UB/LB + op24 favorability"),
    (D(7,8),  D(7,9),  C_SHIP, "op25 HLS win-region expansion"),
    (D(7,9),  D(7,12), C_SHIP, "op26 log-falsi + RS (iter6 live)"),
    (D(7,10), D(7,11), C_SHIP, "op27 K2048 tail ladder"),
]
ERAS = [(D(1,25), D(3,20), "Era-0  birth (manual snapshots, dual-LLM)"),
        (D(4,6),  D(6,9),  "Era-1  ablation (12+2 negative-result ledger)"),
        (D(6,10), D(7,12), "Era-2  op campaigns (git-native, 148 commits)")]
EVENTS = [(D(4,23), "nsys becomes canonical"), (D(5,8), "F006 + printf incident"),
          (D(7,2), "nsys sqlite token leak"), (D(7,7), "HLS math formalization")]

# ------- P2: op21+ P0 verdict trajectory (nsys 17-cell gm vs best rival) -----
KEPT = [  # (seq, gm, label)
    (1, 0.830, "iter1 single-CTA v1"), (2, 1.051, "iter2 row-chunked C-CTA"),
    (4, 1.104, "iter5 rank-scatter P4"), (5, 1.125, "iter6 small-bin P4"),
    (6, 1.249, "iter7 P3 remote-store push"), (8, 1.268, "iter10 B300 HW-invariant"),
    (11, 1.265, "iter13 HLS log-falsi fallback"), (12, 1.276, "iter15 P0 @HEAD"),
    (13, 1.298, "iter16 fallback diet (tax<=1%)"), (14, 1.274, "op25 w3a ladder+slot2+C8"),
]
FALS_RUG = [  # (seq, label) attempts that did not move the line
    (3, "iter3 dist-P1"), (3.5, "iter4 dist-P4"), (5.5, "iter6b C8-at-holes"),
    (7, "iter8 C8 fp32 (flips 16-bit)"), (9, "iter11 P4 path-C 0/72->fixed"),
    (10, "iter12.9 rank-space bridge"), (13.5, "op25 S1b EMA / HLS-MC / kC-diet"),
    (14.5, "op26 secant2 (silicon)"), (15, "op27 M-probe (host replay)"),
]

# ---------------- P3: commits/day + cumulative levers -----------------------
COMMITS = {D(6,27):5, D(6,28):5, D(6,30):5, D(7,1):28, D(7,2):25, D(7,3):13,
           D(7,4):4, D(7,5):5, D(7,6):13, D(7,7):9, D(7,8):15, D(7,9):3,
           D(7,10):14, D(7,12):4}
SHIP_EVT = [D(6,14), D(7,2), D(7,2), D(7,3), D(7,3), D(7,4), D(7,7), D(7,8),
            D(7,8), D(7,9), D(7,10), D(7,12)]      # op7,op17,op13a,op18,op19,op13b,HLS13,HLS14/16,op25,op26/27...
FALS_EVT = [D(6,12), D(6,13), D(6,21), D(6,22), D(6,28), D(6,29), D(6,29),
            D(7,1), D(7,1), D(7,1), D(7,2), D(7,2), D(7,3), D(7,3), D(7,3),
            D(7,4), D(7,5), D(7,5), D(7,6), D(7,6), D(7,7), D(7,8), D(7,8),
            D(7,9), D(7,9), D(7,10), D(7,10), D(7,11)]  # per-campaign falsified levers (ledgers)

fig = plt.figure(figsize=(16, 15.5))
gs = fig.add_gridspec(3, 1, height_ratios=[2.1, 1.15, 0.85], hspace=0.30)

# ---------------- Panel 1 ----------------
ax = fig.add_subplot(gs[0])
for i, (s, e, c, lab) in enumerate(CAMP):
    y = len(CAMP) - i
    ax.barh(y, (e - s).days or 0.7, left=mdates.date2num(s), height=0.62,
            color=c, edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(mdates.date2num(e) + 1.2, y, lab, va="center", fontsize=7.6,
            color=INK if c != C_FALS else MUT, zorder=4)
for s, e, lab in ERAS:
    ax.axvspan(mdates.date2num(s), mdates.date2num(e), color="#2a78d6",
               alpha=0.05, zorder=1)
    ax.text(mdates.date2num(s) + 0.5, len(CAMP) + 1.3, lab, fontsize=8.6,
            color=MUT, style="italic")
for d, lab in EVENTS:
    ax.axvline(mdates.date2num(d), color=C_EVT, lw=0.9, ls=(0, (3, 3)),
               alpha=0.55, zorder=2)
    ax.text(mdates.date2num(d) - 0.9, len(CAMP) - 0.2, lab, rotation=90,
            fontsize=7, color=C_EVT, va="top", ha="right", alpha=0.9)
ax.set_ylim(-0.5, len(CAMP) + 2.6)
ax.set_xlim(mdates.date2num(D(1, 18)), mdates.date2num(D(8, 20)))
ax.set_yticks([])
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.grid(axis="x", alpha=0.15)
for sp in ("left", "top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("GVR threshold top-K: six months of campaigns, Jan-Jul 2026  "
             "(green=shipped, gray=falsified, yellow=wash/partial, blue=infra/meta)",
             fontsize=12, loc="left", color=INK)
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=c, label=l) for c, l in
                   [(C_SHIP, "shipped"), (C_FALS, "falsified"),
                    (C_WASH, "wash / partial"), (C_META, "infra / meta-analysis")]],
          loc="lower left", fontsize=8, frameon=False)

# ---------------- Panel 2 ----------------
ax = fig.add_subplot(gs[1])
xs, ys = [k[0] for k in KEPT], [k[1] for k in KEPT]
run = [max(ys[:i + 1]) for i in range(len(ys))]
ax.step(xs, run, where="post", color=C_SHIP, lw=2, alpha=0.65, zorder=3,
        label="running best (kept)")
ax.scatter(xs, ys, c=C_SHIP, s=52, zorder=4, edgecolors="white", linewidths=0.9,
           label="kept milestone")
for i, (x, y, lab) in enumerate(KEPT):
    dy = 7 if i % 2 == 0 else -13          # stagger to avoid plateau pile-up
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, dy),
                fontsize=7.4, color="#1a5c1a", rotation=15, ha="left",
                va="bottom" if dy > 0 else "top")
rug_y = 0.72
ax.scatter([r[0] for r in FALS_RUG], [rug_y] * len(FALS_RUG), marker="x",
           c=C_FALS, s=34, zorder=4, label="falsified attempt (rug, no y-value)")
for x, lab in FALS_RUG:
    ax.annotate(lab, (x, rug_y), textcoords="offset points", xytext=(2, -9),
                fontsize=6.6, color=MUT, rotation=-15, ha="left", va="top")
ax.axhline(1.0, color=MUT, lw=0.8, ls=":", alpha=0.6)
ax.text(0.55, 1.005, "parity vs best rival", fontsize=7.2, color=MUT, va="bottom")
ax.set_xlim(0.3, 16.6); ax.set_ylim(0.55, 1.47)
ax.set_xlabel("verdict sequence (op21 iter0.5 -> op27, Jul 5-12)", fontsize=10)
ax.set_ylabel("P0 verdict: nsys 17-cell geomean\nvs best rival", fontsize=9.5)
ax.grid(alpha=0.15); ax.set_xticks([])
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("Production trajectory (autoresearch progress-plot form): every kept milestone was "
             "nsys-verdicted; falsified attempts shown as rug ticks (their y is not comparable)",
             fontsize=10.5, loc="left", color=INK)
ax.legend(loc="upper left", fontsize=8, frameon=False)

# ---------------- Panel 3 ----------------
ax = fig.add_subplot(gs[2])
days = sorted(COMMITS)
ax.bar([mdates.date2num(d) for d in days], [COMMITS[d] for d in days],
       width=0.72, color="#2a78d6", alpha=0.55, label="op-bench commits/day", zorder=3)
def cum(evts):
    evts = sorted(evts); xs, ys = [], []
    for i, d in enumerate(evts): xs.append(mdates.date2num(d)); ys.append(i + 1)
    return xs, ys
fx, fy = cum(FALS_EVT); sx, sy = cum(SHIP_EVT)
ax.step(fx, fy, where="post", color=C_FALS, lw=2, label="cumulative falsified levers", zorder=4)
ax.step(sx, sy, where="post", color=C_SHIP, lw=2, label="cumulative shipped levers", zorder=4)
ax.annotate(f"{len(FALS_EVT)} falsified", (fx[-1], fy[-1]), xytext=(6, 2),
            textcoords="offset points", fontsize=8, color=MUT)
ax.annotate(f"{len(SHIP_EVT)} shipped", (sx[-1], sy[-1]), xytext=(6, 2),
            textcoords="offset points", fontsize=8, color="#1a5c1a")
ax.set_xlim(mdates.date2num(D(6, 10)), mdates.date2num(D(7, 16)))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_ylabel("count", fontsize=9.5)
ax.grid(alpha=0.15)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("Era-2 cadence: falsifications outpace ships ~2.3:1 -- the ledger of dead ends "
             "is the main product (counts hand-tallied from campaign ledgers)",
             fontsize=10.5, loc="left", color=INK)
ax.legend(loc="upper left", fontsize=8, frameon=False)

fig.suptitle("GVR (guess-verify-refine) threshold top-K -- human-expert + agent/harness + LLM optimization trajectory",
             fontsize=13.5, y=0.995, color=INK)
out = __file__.replace("gen_trajectory.py", "trajectory.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
print("wrote", out)
