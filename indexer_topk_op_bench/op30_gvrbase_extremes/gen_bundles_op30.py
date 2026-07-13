# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op30 phase 2 — full bundle trees for the GVR-base-relative scenarios.

Scenario map comes from scen_op30.json (written by pick_scen_op30.py):
    best  : per-K (cfg, hr) minimizing GVR (cuteDSL) base cold-L2 time
    worst : per-K (cfg, hr) maximizing it

Same grid & seed policy as gen_bundles_rr.py (models × dtypes × N 4K..1M,
cell_seed = 42 + crc32("{K}|{N}") % 1e6), output under bundles_op30/.
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
BUNDLES_OP30 = HERE / "bundles_op30"

MODELS = {"v4flash": 512, "v4pro": 1024, "v32": 2048}
DTYPES = ["fp32", "bf16", "fp16"]
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

_scen_path = HERE / "scen_op30.json"
# scenario -> model -> (cfg, target_hr)
SCEN_OP30 = ({s: {m: tuple(v) for m, v in d.items()}
              for s, d in json.loads(_scen_path.read_text()).items()}
             if _scen_path.exists() else None)


def _skill():
    spec = importlib.util.spec_from_file_location("_tsynth_op30g", SKILL_SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_tsynth_op30g"] = m
    spec.loader.exec_module(m)
    return m


def cell_seed(K, N):
    return 42 + zlib.crc32(f"{K}|{N}".encode()) % 1_000_000


def bundle_dir(scenario, model, dtype, N):
    cfg, _ = SCEN_OP30[scenario][model]
    return (BUNDLES_OP30 / scenario / f"{model}_{dtype}_N{N}"
            / f"{cfg}_N{N}_bs1")


def main():
    assert SCEN_OP30 is not None, "scen_op30.json missing — run calibration"
    only = sys.argv[1] if len(sys.argv) > 1 else None   # e.g. "v4pro" shard
    sk = _skill()
    cells = [(scen, model, dt, N)
             for scen in SCEN_OP30
             for model, K in MODELS.items()
             if only in (None, model)
             for N in N_GRID if N > 2 * K
             for dt in DTYPES]
    print(f"# gen_bundles_op30: {len(cells)} bundles (filter={only})",
          flush=True)
    t0 = time.time()
    n_done = n_skip = 0
    fails = []
    for i, (scen, model, dt, N) in enumerate(cells):
        d = bundle_dir(scen, model, dt, N)
        if (d / "meta.json").exists():
            n_skip += 1
            continue
        K = MODELS[model]
        cfg, hr = SCEN_OP30[scen][model]
        try:
            b = sk.synthesize(model, N, 1, cfg, target_hr=hr,
                              seed=cell_seed(K, N), row_mode="independent",
                              sentinel_mode="real", dtype=dt)
            b["meta"]["scenario_op30"] = scen
            b["meta"]["objective"] = "gvr_cutedsl-absolute extremes (op30)"
            sk.save(str(d), b)
            n_done += 1
        except Exception as e:
            fails.append((scen, model, dt, N,
                          f"{type(e).__name__}: {str(e)[:120]}"))
            print(f"FAIL {d}: {fails[-1][-1]}", flush=True)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(cells)} ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"GEN_OP30 DONE: {n_done} new, {n_skip} skipped, "
          f"{len(fails)} failed in {time.time() - t0:.0f}s", flush=True)
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
