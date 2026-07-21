# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotent injector: Fig. D (ACTUAL executed path of this campaign run)
into KF_PROCESS_LOG.html, appended just before the KF-DIAGRAMS:END marker.

Unlike Figs A-C (the designed loop), Fig. D records what actually happened:
harvest->probe->grid iterations, the contamination incident + anchor
protocol, the engineer grafts, and the ship-bar verdict.
"""
from gen_diagrams import DEFS, G, Y, B, RD, HTML, arrow, box, fig

MARK_END = "<!-- KF-DIAGRAMS:END -->"
TAG = "<!-- KF-FIG-D -->"


def build():
    s = []
    # ---- lane headers ----
    s.append(box(14, 10, 300, 30, ["CLOUD campaign (rounds)"], fill=B, bold_first=True))
    s.append(box(330, 10, 620, 30, ["LOCAL 8×B200 verify arm (nsys cold-L2 paired vs PR head)"], fill=G, bold_first=True))
    s.append(box(966, 10, 320, 30, ["Engineer-in-the-loop"], fill=Y, bold_first=True))

    # ---- Round 1 ----
    s.append(box(14, 60, 300, 74, ["ROUND 1 (01:19–05:29, 6 agents)", "ramp 0.29→1.077 internal",
                                   "a000 41a94aaa 1.041 → a005", "ba1020ce “hybrid_v28” 1.0768"], fill=B, bold_first=True, fs=11))
    s.append(arrow(314, 84, 360, 84, "harvest"))
    s.append(box(360, 60, 260, 46, ["41a94aaa grid r1a: gm 1.316", "865/865 exact · 137 regs"], fs=11))
    s.append(arrow(620, 84, 660, 84))
    s.append(box(660, 60, 290, 46, ["regression map razor-sharp:", "ALL in N∈[16k,65k] crossover band"], fill=RD, fs=11))
    s.append(arrow(950, 84, 1000, 84))
    s.append(box(1000, 60, 286, 46, ["diagnosis: dispatch boundary", "SMALL_N=16384 vs grid npad 16387"], fill=Y, fs=11))

    # ba1020ce verify + sb probes
    s.append(arrow(150, 134, 150, 160))
    s.append(box(14, 160, 300, 40, ["round-1 close 05:29", "winner ba1020ce → round 2 spawns 7"], fill=B, fs=11, bold_first=True))
    s.append(box(360, 160, 260, 46, ["ba1020ce grid r1c: gm 1.3662", "865/865 exact · 85 regs (boundary)"], fs=11))
    s.append(arrow(620, 184, 660, 184))
    s.append(box(660, 160, 290, 46, ["sb17/sb17b probes: 1024thr single-CTA", "heals boundary (0.74→0.89-0.99)"], fill=Y, fs=11))
    s.append(arrow(805, 206, 805, 232, "steering material"))

    # ---- Round 2 ----
    s.append(box(14, 232, 300, 88, ["ROUND 2 (05:29–, 7 new agents)", "cross-pollination via insights DB:",
                                    "a002 0260cee7 1.284 (early-exit+bottom3)", "a002 0197c2a1 1.311 · a003 c74fb3c0 1.339"],
                 fill=B, bold_first=True, fs=11))
    s.append(arrow(314, 262, 360, 262, "harvest ×3"))
    s.append(box(360, 232, 260, 88, ["r2a grid “1.7758” ✗", "r2b grid “1.7101” ✗",
                                     "INVALIDATED: probes launched", "during sharded grids (double driver)"], fill=RD, fs=11, bold_first=True))
    s.append(arrow(620, 276, 660, 276))
    s.append(box(660, 232, 290, 88, ["NEW PROTOCOL:", "serialize ALL GPU work",
                                     "per-rung pr_cold anchor check", "quiet-GPU probe before runs"], fill=Y, fs=11, bold_first=True))

    # clean re-verdicts
    s.append(arrow(490, 320, 490, 346))
    s.append(box(360, 346, 260, 74, ["CLEAN verdicts:", "0260cee7 1.6421 · 4 regs",
                                     "c74fb3c0 1.6713 · 5 regs", "(+2 ext-contaminated rungs re-measured)"], fill=G, fs=11, bold_first=True))
    s.append(arrow(620, 383, 660, 383, "rival join"))
    s.append(box(660, 346, 290, 74, ["compare_rivals.py (PR-normalized):", "vs sglang_v2 1.111 (win 569/865)",
                                     "vs radix_cutedsl 1.611 (864/865)", "first in-tree-family win vs sglang"], fs=11, bold_first=True))
    s.append(arrow(950, 383, 1000, 383))
    s.append(box(1000, 346, 286, 74, ["graft sb<17>@1024thr rung", "onto c74fb3c0 (3 lines;",
                                      "its topk_small already blockDim.x)", "→ c74f_sbx"], fill=Y, fs=11, bold_first=True))

    # ship verdict
    s.append(arrow(1143, 420, 1143, 446))
    s.append(box(660, 446, 626, 52, ["SHIP BAR MET: c74f_sbx — geomean 1.6828 · 865/865 exact · ZERO cold regressions",
                                     "(2 borderline cells adjudicated noise @60 reps: 1.068 / 1.042) · anchors clean"],
                 fill="#e7f6e7", stroke="#070", fs=12, bold_first=True))
    s.append(arrow(660, 472, 490, 446, "", dash=True))
    s.append(box(360, 446, 260, 52, ["campaign continues (plateau 1.3385);", "any later harvest must beat", "the composite to displace it"], fs=11))
    return "".join(s)


def main():
    html = HTML.read_text()
    if TAG in html:
        start = html.index(TAG)
        end = html.index(MARK_END)
        html = html[:start] + html[end:]
    figure = TAG + fig(
        build(),
        "Fig. D — What actually ran: two harvest→verify cycles, the contamination incident that forged the anchor-check "
        "protocol, and the engineer graft that closed the last regressions.",
        "图 D — 本轮实际执行路径:两轮收割→复验循环;污染事故催生锚检协议;工程师嫁接补掉最后的回归,达成 ship 门。",
        (1300, 512))
    html = html.replace(MARK_END, figure + MARK_END)
    HTML.write_text(html)
    print("Fig. D injected")


if __name__ == "__main__":
    main()
