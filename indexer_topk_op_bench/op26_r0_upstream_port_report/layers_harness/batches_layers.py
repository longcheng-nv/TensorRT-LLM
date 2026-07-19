#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Batch list for the per-layer backfill: one line per nsys batch.
seqlen batches interleaved across models so 8-way sharding balances;
bs batches (bigger) at the end."""
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
BS3_LAYERS = {"flash": [10, 22, 34], "pro": [14, 30, 46], "v32": [14, 34, 54]}

lines = []
for i in range(9):
    for m in ("flash", "pro", "v32"):
        if i < len(REAL_ISLS[m]):
            lines.append(f"seqlen {m} {REAL_ISLS[m][i]}")
for m in ("flash", "pro", "v32"):
    for L in BS3_LAYERS[m]:
        lines.append(f"bs {m} {L}")
print("\n".join(lines))
