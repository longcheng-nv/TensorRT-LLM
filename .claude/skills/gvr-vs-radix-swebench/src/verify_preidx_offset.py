#!/usr/bin/env python3
"""Audit the GVR kernel for the V3.2 decode `preIdxOffset = +1` invariant.

Production path: kernel hardcodes `preIdxOffset = (rowIdx % next_n) + 1`.
With `next_n = 1` (decode), this is +1 for every row -- the kernel reads
`preIdx[i]` and scans column `preIdx[i] + 1` of the current-row logits.
The benchmark feeds `preIdx` = exact Top-K of the prev row and relies on
the kernel adding +1 internally.

Standalone path: kernel hardcodes `preIdxOffset = 0`. The benchmark shifts
preIdx by +1 in Python before each call so the effective column shift
matches production.

If the production kernel does NOT bake the +1 shift, GVR is solving the
wrong problem and the comparison against Radix is meaningless. This script
greps both sources and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

PROD_PATTERN = re.compile(
    r"int\s+const\s+preIdxOffset\s*=\s*\(\s*rowIdx\s*%\s*next_n\s*\)\s*\+\s*1"
)
STANDALONE_PATTERN = re.compile(r"/\*\s*preIdxOffset\s*=\s*\*/\s*0")


def audit_production(trtllm_repo: str) -> tuple[bool, list[str]]:
    src = os.path.join(trtllm_repo, "cpp/tensorrt_llm/kernels/heuristicTopKDecode.cu")
    notes: list[str] = ["[production] source: " + src]
    if not os.path.isfile(src):
        notes.append("FATAL: source file not found")
        return False, notes
    with open(src) as f:
        text = f.read()
    matches = PROD_PATTERN.findall(text)
    notes.append(f"matches of '(rowIdx % next_n) + 1': {len(matches)}")
    if len(matches) >= 2:
        notes.append("status: PASS -- kernel applies preIdxOffset=+1 on both paths")
        return True, notes
    notes.append(
        "status: FAIL -- expected >=2 hits (fp32 + half-prec paths). "
        "GVR kernel may be using preIdxOffset=0; production BS=1 numbers "
        "will understate the +1 path by 20-34 % (F006)."
    )
    return False, notes


def audit_standalone(standalone_repo: str) -> tuple[bool, list[str]]:
    """Confirm the standalone JIT kernel still uses preIdxOffset=0.

    The bench shifts preIdx by +1 in Python before each call, so the
    effective column shift matches the TRT-LLM production kernel (+1).
    If the kernel is rebuilt with +1 baked in, the Python shift would
    over-shift and the bench needs adjustment.
    """
    src = os.path.join(standalone_repo, "heuristic_topk.cuh")
    notes: list[str] = ["[standalone] source: " + src]
    if not os.path.isfile(src):
        notes.append("FATAL: source file not found")
        return False, notes
    with open(src) as f:
        text = f.read()
    matches = STANDALONE_PATTERN.findall(text)
    notes.append(f"matches of '/*preIdxOffset=*/0': {len(matches)}")
    if len(matches) >= 1:
        notes.append(
            "status: PASS -- standalone kernel uses preIdxOffset=0; "
            "bench shifts preIdx by +1 in Python so the effective column "
            "offset matches TRT-LLM production (+1)."
        )
        return True, notes
    notes.append(
        "status: FAIL -- expected '/*preIdxOffset=*/0' marker in "
        "heuristic_topk.cuh. Kernel may have been rebuilt with a different "
        "default; the Python +1 shift may over-shift. Inspect the source."
    )
    return False, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trtllm_repo",
        default=os.environ.get(
            "TRTLLM_REPO", "/home/scratch.loncheng_gpu/workspace/perf/TensorRT-LLM"
        ),
    )
    ap.add_argument(
        "--standalone_repo",
        default=os.environ.get(
            "GVR_STANDALONE_REPO",
            "/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1",
        ),
    )
    ap.add_argument(
        "--gvr_backend",
        choices=("production", "standalone", "both"),
        default="production",
        help="Which backend(s) to audit. 'both' runs both and requires both to PASS.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_notes: list[str] = []
    ok = True
    if args.gvr_backend in ("production", "both"):
        ok_p, n_p = audit_production(args.trtllm_repo)
        all_notes.extend(n_p)
        ok = ok and ok_p
    if args.gvr_backend in ("standalone", "both"):
        if all_notes:
            all_notes.append("")
        ok_s, n_s = audit_standalone(args.standalone_repo)
        all_notes.extend(n_s)
        ok = ok and ok_s

    report = "\n".join(all_notes) + "\n"
    sys.stdout.write(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
