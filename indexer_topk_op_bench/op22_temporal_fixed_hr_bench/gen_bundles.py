# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 W1 — generate all temporal-synth bundles for the fixed-hit-rate bench.

Loops scenario x model x dtype x N and produces one on-disk bundle per cell
via the indexer-topk-temporal-synth skill. Calls synthesize()+save() in-process
(the exact functions the skill CLI wraps) so cuteDSL/torch import cost is paid
once; the byte-identical CLI equivalent is recorded per bundle in meta.json
("gen_cmd") and must reproduce the same tensors (same seed, same code path).

Scenarios (PLAN.md section 2):
  best  : --cfg beta_deep     --target_hr 0.90   (GVR best case)
  worst : --cfg beta_shallow  --target_hr 0.05   (GVR worst case)
  real  : --cfg aggregate     (hr sampled from the real per-step distribution)

Seed policy (PLAN amendment, recorded in the report): synthesize() draws the
row's LAYER as the first rng call from default_rng(seed), so a constant
--seed 42 would give every (K, N) cell the same layer — collapsing the
aggregate/tercile layer mixture (the exact trap synth_data_v2.cell_seed was
built to avoid). We therefore use the SAME derivation as synth_data_v2:
    seed(K, N) = 42 + crc32(f"{K}|{N}") % 1_000_000
shared across dtypes (fp32/bf16/fp16 differ only in precision) AND scenarios
(cross-scenario difference is cfg/target_hr only).

Resumable: a bundle whose meta.json exists and passes verification is skipped.
Verification: realised_hr_mean == target +-0.03 (skill gate G5) for best/worst;
logits [1, Npad] with pad alignment; preIdx [1, K] int32.

Usage:  python3 gen_bundles.py [--only scenario[,scenario]] [--dry]
"""
import argparse
import importlib.util
import json
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # .../TensorRT-LLM
SKILL_SRC = (REPO / ".claude" / "skills" / "indexer-topk-temporal-synth"
             / "src" / "synth_temporal_data.py")
BUNDLES = HERE / "bundles"

BASE_SEED = 42
SCENARIOS = {
    #  name  : (cfg,            target_hr)
    "best":  ("beta_deep",     0.90),
    "worst": ("beta_shallow",  0.05),
    "real":  ("aggregate",     None),
}
MODELS = {"v4flash": 512, "v4pro": 1024, "v32": 2048}
DTYPES = ["fp32", "bf16", "fp16"]
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
HR_TOL = 0.03  # skill gate G5


def cell_seed(K, N, base=BASE_SEED):
    """Deterministic per-(K,N) seed, shared across dtypes AND scenarios
    (identical derivation to harness/synth_data_v2.cell_seed)."""
    return base + zlib.crc32(f"{K}|{N}".encode()) % 1_000_000


def _skill():
    spec = importlib.util.spec_from_file_location("_tsynth_op22", SKILL_SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_tsynth_op22"] = m
    spec.loader.exec_module(m)
    return m


def bundle_dir(scenario, model, dtype, N):
    cfg = SCENARIOS[scenario][0]
    return (BUNDLES / scenario / f"{model}_{dtype}_N{N}"
            / f"{cfg}_N{N}_bs1")


def gen_cmd(scenario, model, dtype, N, seed):
    cfg, hr = SCENARIOS[scenario]
    hr_part = f" --target_hr {hr}" if hr is not None else ""
    outdir = f"op22_temporal_fixed_hr_bench/bundles/{scenario}/{model}_{dtype}_N{N}"
    return (f"python3 .claude/skills/indexer-topk-temporal-synth/src/"
            f"synth_temporal_data.py --model {model} --N {N} --cfg {cfg}"
            f"{hr_part} --bs 1 --dtype {dtype} --seed {seed} "
            f"--outdir {outdir}")


def verify(meta, scenario, model, N):
    K = MODELS[model]
    cfg, hr = SCENARIOS[scenario]
    errs = []
    if meta.get("cfg") != cfg or meta.get("N") != N or meta.get("K") != K:
        errs.append(f"meta mismatch cfg/N/K: {meta.get('cfg')}/{meta.get('N')}/{meta.get('K')}")
    if hr is not None and abs(meta.get("realised_hr_mean", -1) - hr) > HR_TOL:
        errs.append(f"hr {meta.get('realised_hr_mean'):.4f} != {hr} +-{HR_TOL}")
    align = meta.get("logit_alignment", 0)
    if align and meta.get("Npad", 0) % align != 0:
        errs.append(f"Npad {meta.get('Npad')} not aligned to {align}")
    if meta.get("seed") != cell_seed(K, N):
        errs.append(f"seed {meta.get('seed')} != cell_seed {cell_seed(K, N)}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma scenario subset (default: all three)")
    ap.add_argument("--dry", action="store_true", help="list work, no gen")
    args = ap.parse_args()
    scens = ([s.strip() for s in args.only.split(",")] if args.only
             else list(SCENARIOS))
    for s in scens:
        assert s in SCENARIOS, s

    cells = []
    for scen in scens:
        for model, K in MODELS.items():
            for N in N_GRID:
                if N <= 2 * K:
                    continue
                for dt in DTYPES:
                    cells.append((scen, model, dt, N))
    print(f"# gen_bundles: {len(cells)} bundles "
          f"(scenarios={scens})", flush=True)
    if args.dry:
        for c in cells:
            print("  ", c)
        return

    sk = _skill()
    n_done = n_skip = 0
    fails = []
    t0 = time.time()
    for i, (scen, model, dt, N) in enumerate(cells):
        K = MODELS[model]
        cfg, hr = SCENARIOS[scen]
        d = bundle_dir(scen, model, dt, N)
        mj = d / "meta.json"
        if mj.exists():
            meta = json.loads(mj.read_text())
            errs = verify(meta, scen, model, N)
            if not errs:
                n_skip += 1
                continue
            print(f"REGEN {d} (stale: {errs})", flush=True)
        seed = cell_seed(K, N)
        b = sk.synthesize(model, N, 1, cfg, target_hr=hr, seed=seed,
                          row_mode="independent", sentinel_mode="real",
                          dtype=dt)
        b["meta"]["scenario"] = scen
        b["meta"]["gen_cmd"] = gen_cmd(scen, model, dt, N, seed)
        b["meta"]["seed_policy"] = ("cell_seed = 42 + crc32('{K}|{N}') % 1e6, "
                                    "shared across dtypes and scenarios")
        sk.save(str(d), b)
        meta = json.loads(mj.read_text())
        errs = verify(meta, scen, model, N)
        if errs:
            fails.append((scen, model, dt, N, errs))
            print(f"FAIL {d}: {errs}", flush=True)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  [{i+1}/{len(cells)}] generated={n_done} skipped={n_skip} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDONE generated={n_done} skipped={n_skip} fails={len(fails)} "
          f"({time.time()-t0:.0f}s)")
    for f in fails:
        print("  FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
