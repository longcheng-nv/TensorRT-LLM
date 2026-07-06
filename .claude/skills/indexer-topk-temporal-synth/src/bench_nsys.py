#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""nsys + L2-flush + NVTX-tagged GVR vs Radix bench for unified synth bundles
(v32 / v4flash / v4pro — model contract read from each bundle's meta.json).

Run under:
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
        --capture-range-end=stop -o <rep> python3 bench_nsys.py --indir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import nvtx
import torch

from pathlib import Path

# default: the enclosing repo checkout's build (this file lives at
# <repo>/.claude/skills/<skill>/src/); override with $LIBTH_COMMON
_REPO = Path(__file__).resolve().parents[4]
LIBTH_COMMON = os.environ.get(
    "LIBTH_COMMON",
    str(_REPO / "cpp" / "build" / "tensorrt_llm" / "thop" / "libth_common.so"),
)
L2_FLUSH_BYTES = 128 * 1024 * 1024
RADIX_AUX_BLOCKS_MAX = 32

DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def flush_l2(buf):
    buf.zero_()
    torch.cuda.synchronize()


def load_bundles(indir: str, dtypes):
    bundles = []
    for d in sorted(glob.glob(os.path.join(indir, "*_N*_bs*"))):
        if not os.path.isfile(os.path.join(d, "meta.json")):
            continue
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        logits_fp32 = torch.load(os.path.join(d, "logits.pt")).cuda().to(torch.float32)
        preIdx = torch.load(os.path.join(d, "preIdx.pt")).cuda()
        seq_lens = torch.load(os.path.join(d, "seq_lens.pt")).cuda()
        BS, _ = logits_fp32.shape
        K = preIdx.shape[1]
        logits_by_dtype = {dt: (logits_fp32 if DTYPE_MAP[dt] == torch.float32
                                else logits_fp32.to(DTYPE_MAP[dt]))
                           for dt in dtypes}
        bundles.append(dict(
            tag=os.path.basename(d),
            preIdx=preIdx, seq_lens=seq_lens,
            indices=torch.empty((BS, K), dtype=torch.int32, device="cuda"),
            logits_by_dtype=logits_by_dtype,
            scratch_by_dtype={dt: torch.empty((BS * K,), dtype=DTYPE_MAP[dt],
                                              device="cuda") for dt in dtypes},
            radix_aux_indices=torch.empty((BS * RADIX_AUX_BLOCKS_MAX * K,),
                                          dtype=torch.int32, device="cuda"),
            radix_aux_logits=torch.empty((BS * RADIX_AUX_BLOCKS_MAX * K,),
                                         dtype=torch.float32, device="cuda"),
            K=K, BS=BS, compress_ratio=meta["compress_ratio"],
            model=meta.get("model", "?"),
        ))
    return bundles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--dtypes", default="fp32",
                   help="comma-separated heuristic-path dtypes; radix always fp32")
    args = p.parse_args()

    torch.ops.load_library(LIBTH_COMMON)
    op = torch.ops.trtllm.indexer_topk_decode

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    flush_buf = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    bundles = load_bundles(args.indir, dtypes)
    print(f"Found {len(bundles)} cells; dtypes={dtypes}", file=sys.stderr)

    def gvr(b, dt):
        op(b["logits_by_dtype"][dt], b["seq_lens"], b["indices"], 1, b["K"],
           pre_idx=b["preIdx"], heuristic_scratch=b["scratch_by_dtype"][dt],
           compress_ratio=b["compress_ratio"],
           radix_aux_indices=b["radix_aux_indices"],
           radix_aux_logits=b["radix_aux_logits"])

    def radix(b):
        op(b["logits_by_dtype"]["fp32"], b["seq_lens"], b["indices"], 1, b["K"],
           pre_idx=None, heuristic_scratch=None,
           compress_ratio=b["compress_ratio"],
           radix_aux_indices=b["radix_aux_indices"],
           radix_aux_logits=b["radix_aux_logits"])

    for b in bundles:                      # warmup outside the nsys window
        for dt in dtypes:
            for _ in range(args.warmup):
                flush_l2(flush_buf); gvr(b, dt); torch.cuda.synchronize()
        for _ in range(args.warmup):
            flush_l2(flush_buf); radix(b); torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for b in bundles:
        for rep in range(args.reps):
            for dt in dtypes:
                flush_l2(flush_buf)
                with nvtx.annotate(f"GVR_{dt}|{b['tag']}|rep{rep}", color="green"):
                    gvr(b, dt); torch.cuda.synchronize()
            flush_l2(flush_buf)
            with nvtx.annotate(f"RADIX_fp32|{b['tag']}|rep{rep}", color="red"):
                radix(b); torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
