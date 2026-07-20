# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Ordered batch list for the §9 rival sweep. One printed line = one nsys-rep
batch; the driver shards by line index % NW. Trim SCOPE knobs to shrink.

Line formats:
  synth <sweep> <scen> <K> <dtype>
  real  <sweep> <model> <dtype>
"""
SWEEPS = ["seqlen", "bs"]
DTYPES = ["fp32", "fp16", "bf16"]
KS = [512, 1024, 2048]
SCEN = ["best", "worst"]
MODELS = ["flash", "pro", "v32"]

if __name__ == "__main__":
    import sys
    batches = []
    for sw in SWEEPS:
        for dt in DTYPES:
            for K in KS:
                for sc in SCEN:
                    batches.append(f"synth {sw} {sc} {K} {dt}")
            for m in MODELS:
                batches.append(f"real {sw} {m} {dt}")
    if "--count" in sys.argv:
        print(f"batches={len(batches)}")
    else:
        print("\n".join(batches))
