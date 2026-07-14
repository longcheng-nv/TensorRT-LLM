# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 ENV sub-campaign — GVR performance-envelope bundles built by the
LATEST indexer-topk-temporal-synth SKILL, exact user prompt:

  BEST  (最有利于 GVR / op#21) = 逐 K 顺风 cfg + fixed target_hr 0.55
          K512  v4flash  aggregate      hr0.55
          K1024 v4pro     beta_moderate  hr0.55
          K2048 v32       beta_moderate  hr0.55   <-- differs from op22rr best
                                                     (which used beta_shallow)
  WORST (最不利)          = beta_shallow + fixed target_hr 0.05  (K-flat pole)
  slowbase (可选, 基础内核绝对延迟最慢极点) = beta_deep + target_hr 0.90

Protocol knobs (from the prompt): SYNTH_POSITIONAL=1 (positional model ON so
the low-hr preIdx-gather cost is real), --steps 1, seed 42, BS=1 rows (the
harness build_call replicates to BS at bench time; inputs byte-identical
across all 9 arms per cell).

This is a FRESH single-node dataset (bundles_env/) — the generation assets the
skill actually reads (calib_<model>.npz + posz_<model>.npz) are byte-identical
to the GitHub skill, so "latest skill" == deterministic; the new content vs the
existing op22rr best/worst is (a) V3.2-best cfg beta_moderate and (b) a unified
9-arm sweep with NO cross-node anchor transfer.

get_bundle(scenario, K, dtype, N) returns the SAME contract as
bundle_data_rr.get_bundle.
"""
import importlib.util
import json
import os
import sys
import time
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SKILL_SRC = (REPO / ".claude" / "skills" / "indexer-topk-temporal-synth"
             / "src" / "synth_temporal_data.py")
BUNDLES_ENV = HERE / "bundles_env"

MODELS = {"v4flash": 512, "v4pro": 1024, "v32": 2048}
DTYPES = ["fp32"]                      # external arms (sglang_v2/flashinfer) fp32-only
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

# scenario -> model -> (cfg, target_hr)  (user prompt, verbatim)
SCEN_ENV = {
    "best": {"v4flash": ("aggregate", 0.55),
             "v4pro": ("beta_moderate", 0.55),
             "v32": ("beta_moderate", 0.55)},
    "worst": {m: ("beta_shallow", 0.05) for m in MODELS},
    "slowbase": {m: ("beta_deep", 0.90) for m in MODELS},
}

_K_MODEL = {512: "v4flash", 1024: "v4pro", 2048: "v32"}
_DT_NAME = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
_mem_cache = {}
_skill_mod = None


def _skill():
    global _skill_mod
    if _skill_mod is None:
        os.environ.setdefault("SYNTH_POSITIONAL", "1")   # positional model ON
        spec = importlib.util.spec_from_file_location("_tsynth_op22env",
                                                      SKILL_SRC)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_tsynth_op22env"] = m
        spec.loader.exec_module(m)
        _skill_mod = m
    return _skill_mod


def cell_seed(K, N):
    return 42 + zlib.crc32(f"{K}|{N}".encode()) % 1_000_000


def bundle_dir(scenario, model, dtype, N):
    cfg, _ = SCEN_ENV[scenario][model]
    return (BUNDLES_ENV / scenario / f"{model}_{dtype}_N{N}"
            / f"{cfg}_N{N}_bs1")


def get_bundle(scenario, K, dtype, N, device="cuda"):
    dt = _DT_NAME[dtype] if not isinstance(dtype, str) else dtype
    key = (scenario, K, dt, N)
    if key in _mem_cache:
        return _mem_cache[key]
    model = _K_MODEL[K]
    d = bundle_dir(scenario, model, dt, N)
    if not (d / "meta.json").exists():
        _gen_one(scenario, model, dt, N)
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
        "cfg": f"op22env-{scenario}:{meta['cfg']}",
        "kernel_hit_rate": meta["realised_hr_mean"],
        "calibrated_c": None,
        "row_meta": meta["rows"][0],
        "seed": meta["seed"],
    }
    _mem_cache[key] = bundle
    return bundle


def _gen_one(scenario, model, dt, N):
    sk = _skill()
    K = MODELS[model]
    cfg, hr = SCEN_ENV[scenario][model]
    b = sk.synthesize(model, N, 1, cfg, target_hr=hr,
                      seed=cell_seed(K, N), row_mode="independent",
                      sentinel_mode="real", steps=1, dtype=dt)
    b["meta"]["scenario_env"] = scenario
    b["meta"]["objective"] = "GVR perf-envelope (fixed-hr best/worst)"
    b["meta"]["synth_positional"] = os.environ.get("SYNTH_POSITIONAL", "1")
    sk.save(str(bundle_dir(scenario, model, dt, N)), b)


def main():
    os.environ.setdefault("SYNTH_POSITIONAL", "1")
    scens = [s for s in sys.argv[1:] if s in SCEN_ENV] or ["best", "worst"]
    cells = [(scen, model, dt, N)
             for scen in scens
             for model, K in MODELS.items()
             for N in N_GRID if N > 2 * K
             for dt in DTYPES]
    print(f"# gen_bundles_env: {len(cells)} bundles scens={scens} "
          f"SYNTH_POSITIONAL={os.environ['SYNTH_POSITIONAL']}", flush=True)
    t0 = time.time()
    n_done = n_skip = 0
    fails = []
    for i, (scen, model, dt, N) in enumerate(cells):
        d = bundle_dir(scen, model, dt, N)
        if (d / "meta.json").exists():
            n_skip += 1
            continue
        try:
            _gen_one(scen, model, dt, N)
            n_done += 1
        except Exception as e:
            fails.append((scen, model, dt, N, f"{type(e).__name__}: {str(e)[:120]}"))
            print(f"FAIL {d}: {fails[-1][-1]}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cells)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"GEN_ENV DONE: {n_done} new, {n_skip} skipped, {len(fails)} failed "
          f"in {time.time()-t0:.0f}s", flush=True)
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
