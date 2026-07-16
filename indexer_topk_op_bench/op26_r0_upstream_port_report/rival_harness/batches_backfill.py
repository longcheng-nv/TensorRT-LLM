# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Ordered batch list for the §8 rival full-ISL BS backfill (2026-07-16).

The 07-15 rival sweep ran the real BS grid at a single representative rung
(ISL=128k); the 07-16 GVR refresh extended the GVR arms to ALL ISL rungs.
This backfill re-runs the EXTERNAL arms (radix_cutedsl / sglang_v2 /
flashinfer_topk) + the op26_r0auto ANCHOR (drift gate only, rows not merged)
over the FULL ISL x BS grid, one nsys batch per (model, dtype, isl) for
8-GPU load balance. 128k is INCLUDED: it gives per-batch drift overlap vs
both the old rival rows (044) and the refresh anchor rows (094).

Line format:  real bs <model> <dtype> <isl>
"""
DTYPES = ["fp32", "fp16", "bf16"]
MODELS = ["flash", "pro", "v32"]
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}

if __name__ == "__main__":
    import sys
    batches = []
    for dt in DTYPES:
        for m in MODELS:
            for isl in REAL_ISLS[m]:
                batches.append(f"real bs {m} {dt} {isl}")
    if "--count" in sys.argv:
        print(f"batches={len(batches)}")
    else:
        print("\n".join(batches))
