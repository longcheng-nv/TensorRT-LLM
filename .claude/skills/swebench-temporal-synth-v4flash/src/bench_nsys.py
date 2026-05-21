#!/usr/bin/env python3
"""nsys + L2-flush + NVTX-tagged GVR vs Radix bench for V4 Flash synth bundles.

Differs from V3.2 sibling:
  - compress_ratio = 4 (read from each bundle's meta.json)
  - preIdx passed directly (V4 caller offset = 0)
  - radix_aux_indices + radix_aux_logits pre-allocated (post-#14297 contract)
  - dtype-aware: heuristic path runs per-dtype; radix baseline always fp32

Run under:
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import nvtx
import torch

LIBTH_COMMON = os.environ.get(
    "LIBTH_COMMON",
    "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM"
    "/cpp/build/tensorrt_llm/thop/libth_common.so",
)
L2_FLUSH_BYTES = 128 * 1024 * 1024
RADIX_AUX_BLOCKS_MAX = 32

DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def flush_l2(buf):
    buf.zero_()
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--dtypes", default="fp32,bf16,fp16",
                   help="comma-separated heuristic-path dtypes. Radix is always fp32.")
    args = p.parse_args()

    torch.ops.load_library(LIBTH_COMMON)
    op = torch.ops.trtllm.indexer_topk_decode

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    flush_buf = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")

    cells = sorted(glob.glob(os.path.join(args.indir, "beta_*_N*_bs*")))
    print(f"Found {len(cells)} cells; dtypes={dtypes}", file=sys.stderr)

    bundles = []
    for d in cells:
        tag = os.path.basename(d)
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        logits_fp32 = torch.load(os.path.join(d, "logits.pt")).cuda().to(torch.float32)
        preIdx = torch.load(os.path.join(d, "preIdx.pt")).cuda()
        seq_lens = torch.load(os.path.join(d, "seq_lens.pt")).cuda()
        BS, Npad = logits_fp32.shape
        K = preIdx.shape[1]
        compress_ratio = meta["compress_ratio"]
        # Pre-cast logits for each dtype
        logits_by_dtype = {}
        for dt in dtypes:
            torch_dt = DTYPE_MAP[dt]
            if torch_dt == torch.float32:
                logits_by_dtype[dt] = logits_fp32
            else:
                logits_by_dtype[dt] = logits_fp32.to(torch_dt)
        scratch_by_dtype = {dt: torch.empty((BS * K,), dtype=DTYPE_MAP[dt], device="cuda")
                            for dt in dtypes}
        indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
        radix_aux_indices = torch.empty(
            (BS * RADIX_AUX_BLOCKS_MAX * K,), dtype=torch.int32, device="cuda")
        radix_aux_logits = torch.empty(
            (BS * RADIX_AUX_BLOCKS_MAX * K,), dtype=torch.float32, device="cuda")
        bundles.append(dict(
            tag=tag,
            preIdx=preIdx, seq_lens=seq_lens, indices=indices,
            logits_by_dtype=logits_by_dtype,
            scratch_by_dtype=scratch_by_dtype,
            radix_aux_indices=radix_aux_indices,
            radix_aux_logits=radix_aux_logits,
            K=K, BS=BS, compress_ratio=compress_ratio,
        ))

    # Warmup outside the nsys-tagged region.
    for b in bundles:
        for dt in dtypes:
            for _ in range(args.warmup):
                flush_l2(flush_buf)
                op(b["logits_by_dtype"][dt], b["seq_lens"], b["indices"], 1, b["K"],
                   pre_idx=b["preIdx"], heuristic_scratch=b["scratch_by_dtype"][dt],
                   compress_ratio=b["compress_ratio"],
                   radix_aux_indices=b["radix_aux_indices"],
                   radix_aux_logits=b["radix_aux_logits"])
                torch.cuda.synchronize()
        for _ in range(args.warmup):
            flush_l2(flush_buf)
            op(b["logits_by_dtype"]["fp32"], b["seq_lens"], b["indices"], 1, b["K"],
               pre_idx=None, heuristic_scratch=None,
               compress_ratio=b["compress_ratio"],
               radix_aux_indices=b["radix_aux_indices"],
               radix_aux_logits=b["radix_aux_logits"])
            torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for b in bundles:
        for rep in range(args.reps):
            for dt in dtypes:
                flush_l2(flush_buf)
                with nvtx.annotate(f"GVR_{dt}|{b['tag']}|rep{rep}", color="green"):
                    op(b["logits_by_dtype"][dt], b["seq_lens"], b["indices"], 1, b["K"],
                       pre_idx=b["preIdx"], heuristic_scratch=b["scratch_by_dtype"][dt],
                       compress_ratio=b["compress_ratio"],
                       radix_aux_indices=b["radix_aux_indices"],
                       radix_aux_logits=b["radix_aux_logits"])
                    torch.cuda.synchronize()
            flush_l2(flush_buf)
            with nvtx.annotate(f"RADIX_fp32|{b['tag']}|rep{rep}", color="red"):
                op(b["logits_by_dtype"]["fp32"], b["seq_lens"], b["indices"], 1, b["K"],
                   pre_idx=None, heuristic_scratch=None,
                   compress_ratio=b["compress_ratio"],
                   radix_aux_indices=b["radix_aux_indices"],
                   radix_aux_logits=b["radix_aux_logits"])
                torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
