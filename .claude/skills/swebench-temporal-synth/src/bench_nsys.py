#!/usr/bin/env python3
"""nsys + L2-flush + NVTX-tagged GVR vs Radix bench for swebench-temporal-synth.

Synthetic data harness; run under:
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi ...
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import nvtx
import torch

LIBTH_COMMON = os.environ.get(
    "LIBTH_COMMON",
    "/home/scratch.loncheng_gpu/workspace/perf/TensorRT-LLM"
    "/cpp/build/tensorrt_llm/thop/libth_common.so",
)
L2_FLUSH_BYTES = 128 * 1024 * 1024  # > B200 L2 (126.5 MiB)


def flush_l2(buf):
    buf.zero_()
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()

    torch.ops.load_library(LIBTH_COMMON)
    op = torch.ops.trtllm.indexer_topk_decode

    flush_buf = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")

    cells = sorted(glob.glob(os.path.join(args.indir, "beta_*_N*_bs*")))
    print(f"Found {len(cells)} cells", file=sys.stderr)

    # Pre-load all bundles into a list (CPU -> GPU) so we don't pay I/O during nsys window
    bundles = []
    for d in cells:
        cfg_n_bs = os.path.basename(d)  # beta_X_NY_bsZ
        logits = torch.load(os.path.join(d, "logits.pt")).cuda()
        preIdx = torch.load(os.path.join(d, "preIdx.pt")).cuda()
        seq_lens = torch.load(os.path.join(d, "seq_lens.pt")).cuda()
        BS, _ = logits.shape
        K = preIdx.shape[1]
        indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
        scratch = torch.empty((BS * K,), dtype=logits.dtype, device="cuda")
        bundles.append(
            {
                "tag": cfg_n_bs,
                "logits": logits,
                "preIdx": preIdx,
                "seq_lens": seq_lens,
                "indices": indices,
                "scratch": scratch,
                "K": K,
                "BS": BS,
            }
        )

    # Warmup ALL cells outside the nsys-tagged region
    for b in bundles:
        for _ in range(args.warmup):
            flush_l2(flush_buf)
            op(b["logits"], b["seq_lens"], b["indices"], 1, b["K"], b["preIdx"], b["scratch"])
            torch.cuda.synchronize()
            flush_l2(flush_buf)
            op(b["logits"], b["seq_lens"], b["indices"], 1, b["K"])
            torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    # Timed region: NVTX-tagged + L2-flushed before each launch
    for b in bundles:
        for rep in range(args.reps):
            flush_l2(flush_buf)
            with nvtx.annotate(f"GVR|{b['tag']}|rep{rep}", color="green"):
                op(b["logits"], b["seq_lens"], b["indices"], 1, b["K"], b["preIdx"], b["scratch"])
                torch.cuda.synchronize()
            flush_l2(flush_buf)
            with nvtx.annotate(f"RADIX|{b['tag']}|rep{rep}", color="red"):
                op(b["logits"], b["seq_lens"], b["indices"], 1, b["K"])
                torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
