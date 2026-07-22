# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build pr_head_solution.json — R4 coldstart KF baseline solution.

Wraps gvrpkg_04a0/ (verbatim in-tree GVR sources @ PR#16457 pinned head
04a0900ff7c233a03e95dc8c35321c37c256d627) as a cute_dsl SOLBench solution
with a main.py::run DPS entry matching definition indexer_topk_decode_bs1_real:
  run(logits[1,npad] f32, pre_idx[1,k] i32, n_valid scalar, indices[1,k] i32 out)

compress_ratio is derived from k (2048 -> V3.2 cr=1; 512/1024 -> V4 cr=4) and
seq_lens = n_valid * cr, mirroring quick_ab.pr_call — identical production
semantics on every cell.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE / "gvrpkg_04a0"
PIN_SHA = "04a0900ff7c233a03e95dc8c35321c37c256d627"

MAIN_PY = '''import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

_SL_CACHE = {}


def run(logits, pre_idx, n_valid, indices):
    n = int(n_valid)
    k = int(indices.shape[1])
    cr = 1 if k == 2048 else 4
    key = (n, cr, logits.device.index)
    sl = _SL_CACHE.get(key)
    if sl is None:
        sl = torch.full((1,), n * cr, dtype=torch.int32, device=logits.device)
        _SL_CACHE[key] = sl
    GvrTopKKernel.launch(logits, pre_idx, sl, indices, k, compress_ratio=cr)
'''


def main():
    sources = [{"path": "main.py", "content": MAIN_PY}]
    for p in sorted(PKG.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        sources.append({"path": str(p.relative_to(PKG)), "content": p.read_text()})
    sol = {
        "name": "gvr_pr16457_head_04a0900f",
        "definition": "indexer_topk_decode_bs1_real",
        "author": "baseline-pinned-head",
        "spec": {
            "languages": ["cute_dsl"],
            "target_hardware": ["B200"],
            "entry_point": "main.py::run",
            "dependencies": [],
            "destination_passing_style": True,
        },
        "sources": sources,
        "description": (
            "R4 baseline: verbatim TensorRT-LLM PR#16457 pinned-head GVR "
            f"(guess-verify-refine) top-K decode kernel @ {PIN_SHA}. "
            "CuTe DSL, class-level compiled-variant cache, ambient torch stream."
        ),
    }
    out = HERE / "pr_head_solution.json"
    out.write_text(json.dumps(sol, indent=1))
    print(f"wrote {out} ({len(sources)} sources, "
          f"{sum(len(s['content']) for s in sources)} chars)")


if __name__ == "__main__":
    main()
