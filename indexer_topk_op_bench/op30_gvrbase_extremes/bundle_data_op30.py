# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op30 — bundle loader for the GVR-base-relative scenario definitions.

get_bundle(scenario, K, dtype, N) with the SAME return contract as
bundle_data_rr.get_bundle (both scenarios live in bundles_op30/).
"""
import json
from pathlib import Path

import torch

from gen_bundles_op30 import SCEN_OP30, bundle_dir  # noqa: F401

HERE = Path(__file__).resolve().parent
_K_MODEL = {512: "v4flash", 1024: "v4pro", 2048: "v32"}
_DT_NAME = {torch.float32: "fp32", torch.bfloat16: "bf16",
            torch.float16: "fp16"}

_mem_cache = {}


def get_bundle(scenario, K, dtype, N, device="cuda"):
    key = (scenario, K, str(dtype), N)
    if key in _mem_cache:
        return _mem_cache[key]
    model = _K_MODEL[K]
    dt = _DT_NAME[dtype] if not isinstance(dtype, str) else dtype
    d = bundle_dir(scenario, model, dt, N)
    meta = json.loads((d / "meta.json").read_text())
    logits = torch.load(d / "logits.pt", map_location=device)
    preIdx = torch.load(d / "preIdx.pt", map_location=device)
    cr = meta["compress_ratio"]
    assert meta["seq_lens_val"] == N * cr
    assert logits.shape[0] == 1 and preIdx.shape == (1, meta["K"])
    bundle = {
        "logits": logits.contiguous(),
        "preIdx": preIdx.to(torch.int32).contiguous(),
        "N": N, "Npad": logits.shape[1], "cr": cr, "K": meta["K"],
        "cfg": f"op30-{scenario}:{meta['cfg']}",
        "kernel_hit_rate": meta["realised_hr_mean"],
        "calibrated_c": None,
        "row_meta": meta["rows"][0],
        "seed": meta["seed"],
    }
    _mem_cache[key] = bundle
    return bundle
