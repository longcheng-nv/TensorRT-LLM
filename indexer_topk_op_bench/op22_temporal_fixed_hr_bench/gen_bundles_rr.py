# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr — bundles for the RADIX-RELATIVE scenario definitions (op24 final).

Scenario map (op24 grid-average verdict, RESULTS.md):
    best  : per-K favorable cfg, fixed hr 0.55
            K512  v4flash aggregate      hr0.55
            K1024 v4pro   beta_moderate  hr0.55
            K2048 v32     beta_shallow   hr0.55
    worst : beta_shallow hr0.05 (K-flat adversarial pole)
    real  : NOT generated here — reuses the op22 bundles/real tree verbatim
            (identical definition: aggregate + sampled hr, same cell_seed).

Same grid & seed policy as gen_bundles.py (models x dtypes x N 4K..1M,
cell_seed = 42 + crc32(f"{K}|{N}") % 1e6), output under bundles_rr/.
"""
import importlib.util
import json
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SKILL_SRC = (REPO / ".claude" / "skills" / "indexer-topk-temporal-synth"
             / "src" / "synth_temporal_data.py")
BUNDLES_RR = HERE / "bundles_rr"

MODELS = {"v4flash": 512, "v4pro": 1024, "v32": 2048}
DTYPES = ["fp32", "bf16", "fp16"]
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

# scenario -> model -> (cfg, target_hr)
SCEN_RR = {
    "best": {"v4flash": ("aggregate", 0.55),
             "v4pro": ("beta_moderate", 0.55),
             "v32": ("beta_shallow", 0.55)},
    "worst": {m: ("beta_shallow", 0.05) for m in MODELS},
}


def _skill():
    spec = importlib.util.spec_from_file_location("_tsynth_op22rr", SKILL_SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_tsynth_op22rr"] = m
    spec.loader.exec_module(m)
    return m


def cell_seed(K, N):
    return 42 + zlib.crc32(f"{K}|{N}".encode()) % 1_000_000


def bundle_dir(scenario, model, dtype, N):
    cfg, _ = SCEN_RR[scenario][model]
    return (BUNDLES_RR / scenario / f"{model}_{dtype}_N{N}"
            / f"{cfg}_N{N}_bs1")


def main():
    sk = _skill()
    cells = [(scen, model, dt, N)
             for scen in SCEN_RR
             for model, K in MODELS.items()
             for N in N_GRID if N > 2 * K
             for dt in DTYPES]
    print(f"# gen_bundles_rr: {len(cells)} bundles", flush=True)
    t0 = time.time()
    n_done = n_skip = 0
    fails = []
    for i, (scen, model, dt, N) in enumerate(cells):
        d = bundle_dir(scen, model, dt, N)
        if (d / "meta.json").exists():
            n_skip += 1
            continue
        K = MODELS[model]
        cfg, hr = SCEN_RR[scen][model]
        try:
            b = sk.synthesize(model, N, 1, cfg, target_hr=hr,
                              seed=cell_seed(K, N), row_mode="independent",
                              sentinel_mode="real", dtype=dt)
            b["meta"]["scenario_rr"] = scen
            b["meta"]["objective"] = "radix-relative grid-average (op24)"
            sk.save(str(d), b)
            n_done += 1
        except Exception as e:
            fails.append((scen, model, dt, N,
                          f"{type(e).__name__}: {str(e)[:120]}"))
            print(f"FAIL {d}: {fails[-1][-1]}", flush=True)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(cells)} ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"GEN_RR DONE: {n_done} new, {n_skip} skipped, {len(fails)} failed "
          f"in {time.time() - t0:.0f}s", flush=True)
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
