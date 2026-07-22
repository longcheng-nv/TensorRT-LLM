# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R4: build the baseline-source appendix for ws/prompt.md (coldstart).

Platform gap D1 blocks shipping the 248 KB cuteDSL baseline as a
baseline-solution, so the prompt carries a REAL-source digest of the pinned
head (04a0900ff7): full text of the config/dispatch/orchestration layers +
signature & docstring of every phase kernel. No paraphrase — verbatim spans.
"""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "gvrpkg_04a0" / "gvrpkg" / "top_k" / "gvr_topk_decode.py"

FULL = {"GvrParams", "pick_config"}
DOC = ["__init__", "phase1_preidx_stats", "phase1b_hspace_rungs", "block_count_ge",
       "phase2_secant_search", "phase3_collect_candidates",
       "block_fused_snap_iter", "phase4_rank_scatter", "phase4_histogram_snap",
       "_hist_build", "_kth_bin_search", "gvr_topk_kernel", "_run_phases",
       "run_one_row", "launch"]

text = SRC.read_text()
lines = text.splitlines()
tree = ast.parse(text)


def span(node):
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def sig_doc(node):
    body0 = node.body[0]
    end = body0.end_lineno if (isinstance(body0, ast.Expr)
                               and isinstance(body0.value, ast.Constant)) else node.lineno + 1
    dec0 = node.decorator_list[0].lineno - 1 if node.decorator_list else node.lineno - 1
    return "\n".join(lines[dec0:end]) + "\n        ..."


out = ["\n---\n",
       "## APPENDIX — baseline kernel source digest (verbatim spans from the "
       "production CuTe DSL file, PR#16457 pinned head)",
       "",
       "The production kernel is a 5,000-line CuTe DSL (Python) file; it cannot "
       "be shipped whole. Below are VERBATIM spans: full text of the per-K "
       "constants, constructor (every tuning knob + tuned default), launch-time "
       "config dispatch, and the per-row orchestration body; plus signature + "
       "docstring of every phase primitive. Timings in baselines.jsonl were "
       "measured from exactly this code (see Baseline section).",
       "", "```python"]

cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GvrTopKKernel")
gp = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GvrParams")
out.append(span(gp))
out.append("")
out.append("class GvrTopKKernel:")
for node in cls.body:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    if node.name in FULL:
        out.append(span(node))
        out.append("")
    elif node.name in DOC:
        out.append(sig_doc(node))
        out.append("")
out.append("```")
appendix = "\n".join(out)
(HERE / "ws" / "prompt_appendix.md").write_text(appendix)
print(f"appendix {len(appendix)} chars")
