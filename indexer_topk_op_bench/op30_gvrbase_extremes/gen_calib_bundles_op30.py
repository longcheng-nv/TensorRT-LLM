# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op30 phase 1 — calibration bundles: cfg × hr × N grid, fp32 BS=1.

Object under test = GVR (cuteDSL) base absolute time, so the grid spans the
full favorability space (op24 showed the GVR-base optimum need not coincide
with either op22rr pole; pooled P2-eval max sits near hr 0.75).

Same seed policy as gen_bundles_rr.py: cell_seed = 42 + crc32("{K}|{N}") % 1e6,
shared across cfg/hr so pairs are matched (aggregate's layer draw included).
"""
import importlib.util
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SKILL_SRC = (REPO / ".claude" / "skills" / "indexer-topk-temporal-synth"
             / "src" / "synth_temporal_data.py")
CALIB_BUNDLES = HERE / "calib_bundles"

MODELS = {"v4flash": 512, "v4pro": 1024, "v32": 2048}
CAL_CFGS = ["aggregate", "beta_shallow", "beta_moderate", "beta_deep"]
CAL_HRS = [0.05, 0.15, 0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90]
CAL_NS = [16384, 65536, 262144]


def _skill():
    spec = importlib.util.spec_from_file_location("_tsynth_op30", SKILL_SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_tsynth_op30"] = m
    spec.loader.exec_module(m)
    return m


def cell_seed(K, N):
    return 42 + zlib.crc32(f"{K}|{N}".encode()) % 1_000_000


def hr_tag(hr):
    return f"{hr:.2f}"


def calib_dir(model, cfg, hr, N):
    return (CALIB_BUNDLES / f"{model}_fp32_N{N}"
            / f"{cfg}_hr{hr_tag(hr)}_N{N}_bs1")


def main():
    only_model = sys.argv[1] if len(sys.argv) > 1 else None
    sk = _skill()
    cells = [(model, cfg, hr, N)
             for model, K in MODELS.items()
             if only_model in (None, model)
             for N in CAL_NS if N > 2 * K
             for cfg in CAL_CFGS
             for hr in CAL_HRS]
    print(f"# gen_calib_bundles_op30: {len(cells)} bundles "
          f"(model filter={only_model})", flush=True)
    t0 = time.time()
    n_done = n_skip = 0
    fails = []
    for i, (model, cfg, hr, N) in enumerate(cells):
        d = calib_dir(model, cfg, hr, N)
        if (d / "meta.json").exists():
            n_skip += 1
            continue
        K = MODELS[model]
        try:
            b = sk.synthesize(model, N, 1, cfg, target_hr=hr,
                              seed=cell_seed(K, N), row_mode="independent",
                              sentinel_mode="real", dtype="fp32")
            b["meta"]["scenario_op30"] = "calib"
            b["meta"]["objective"] = "gvr_cutedsl absolute cold-L2 (op30)"
            sk.save(str(d), b)
            n_done += 1
        except Exception as e:
            fails.append((model, cfg, hr, N,
                          f"{type(e).__name__}: {str(e)[:120]}"))
            print(f"FAIL {d}: {fails[-1][-1]}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cells)} ({time.time() - t0:.0f}s)",
                  flush=True)
    print(f"GEN_CALIB DONE: {n_done} new, {n_skip} skipped, "
          f"{len(fails)} failed in {time.time() - t0:.0f}s", flush=True)
    if fails:
        print("FAILED CELLS (excluded from calibration, noted in report):")
        for f_ in fails:
            print(f"  {f_}")


if __name__ == "__main__":
    main()
