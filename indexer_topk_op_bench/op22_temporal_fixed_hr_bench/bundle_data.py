# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 W2 — disk-bundle data source for the op22 nsys sweep.

get_bundle(scenario, K, dtype, N) loads the pre-generated temporal-synth
bundle (gen_bundles.py output) and returns the SAME dict shape as
harness/synth_data.get_bundle, so harness/sweep._build_inputs and the op
builders work unchanged:
    logits [1, Npad] requested dtype, CUDA
    preIdx [1, K] int32, CUDA (v32 caller -1 offset already applied by skill)
    N, Npad, cr, K, cfg  (+ kernel_hit_rate, row_meta provenance)

seq_lens convention: the harness builds seq_div = N*cr itself (bench
convention, all 5 ops validated on it). The skill's meta seq_lens_val is
N*cr + NEXT_N - 1 with NEXT_N = 1, i.e. IDENTICAL to N*cr — verified per
load below (assert), so there is no convention gap to resolve.
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BUNDLES = HERE / "bundles"

_K_MODEL = {512: "v4flash", 1024: "v4pro", 2048: "v32"}
_DT_NAME = {torch.float32: "fp32", torch.bfloat16: "bf16",
            torch.float16: "fp16"}
_SCEN_CFG = {"best": "beta_deep", "worst": "beta_shallow",
             "real": "aggregate"}

_mem_cache = {}


def bundle_dir(scenario, K, dtype, N):
    model = _K_MODEL[K]
    dt = _DT_NAME[dtype] if not isinstance(dtype, str) else dtype
    cfg = _SCEN_CFG[scenario]
    return BUNDLES / scenario / f"{model}_{dt}_N{N}" / f"{cfg}_N{N}_bs1"


def get_bundle(scenario, K, dtype, N, device="cuda"):
    key = (scenario, K, str(dtype), N)
    if key in _mem_cache:
        return _mem_cache[key]
    d = bundle_dir(scenario, K, dtype, N)
    meta = json.loads((d / "meta.json").read_text())
    logits = torch.load(d / "logits.pt", map_location=device)
    preIdx = torch.load(d / "preIdx.pt", map_location=device)
    cr = meta["compress_ratio"]
    assert meta["seq_lens_val"] == N * cr, (
        f"bundle seq_lens {meta['seq_lens_val']} != bench N*cr {N * cr}")
    assert logits.shape[0] == 1 and preIdx.shape == (1, meta["K"]), (
        logits.shape, preIdx.shape)
    bundle = {
        "logits": logits.contiguous(),          # [1, Npad] bundle dtype
        "preIdx": preIdx.to(torch.int32).contiguous(),  # [1, K]
        "N": N, "Npad": logits.shape[1], "cr": cr, "K": meta["K"],
        "cfg": f"op22-{scenario}:{meta['cfg']}",
        "kernel_hit_rate": meta["realised_hr_mean"],
        "calibrated_c": None,
        "row_meta": meta["rows"][0],
        "seed": meta["seed"],
    }
    _mem_cache[key] = bundle
    return bundle


if __name__ == "__main__":
    for scen in ("best", "worst", "real"):
        for K in (512, 1024, 2048):
            b = get_bundle(scen, K, torch.float32, 65536,
                           device="cuda" if torch.cuda.is_available() else "cpu")
            rm = b["row_meta"]
            print(f"{scen:5s} K={K:4d} cr={b['cr']} Npad={b['Npad']} "
                  f"layer=L{rm['layer']} hr={b['kernel_hit_rate']:.3f} "
                  f"seed={b['seed']}")
    print("bundle_data smoke OK")
