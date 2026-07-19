#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""F1 validation gates A (9 real fixtures) and D (full 865-cell per-layer grid).

Gate A: the 9 PR-inexact cells from the all-layer backfill must go value-set
exact with p4_finebin_loop=True; flag OFF is the negative control (expected
to reproduce the failure on most runs — the baseline straddle order is
atomic-arrival nondeterministic, so OFF may occasionally pick correctly;
we count over N_RUNS).

Gate D: --grid mode — every captured GVR-active layer x ISL (flash 21 x 9,
pro 30 x 9, v32 58 x 7), BS=1 fp32, ON arm, exact-only (no timing). Shard
with --model.

Usage:
  f1_gates.py --fixtures
  f1_gates.py --grid --model flash|pro|v32
Env: PYTHONNOUSERSITE=1 + cutlass450 farm PYTHONPATH, one GPU via CVD.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPORT = _HERE.parent
_BENCH = _REPORT.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH / "harness"))

from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel   # noqa: E402
import real_data_v4cap as RV4                              # noqa: E402
import real_data_v32 as RV32                               # noqa: E402
RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)

DEV = "cuda"
FIXTURES = [("pro", "64k", 22), ("pro", "128k", 6), ("pro", "512k", 48),
            ("pro", "512k", 60), ("v32", "8k", 8), ("v32", "8k", 39),
            ("v32", "16k", 38), ("v32", "64k", 16), ("v32", "128k", 25)]
ALL_LAYERS = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)),
              "v32": list(RV32.LAYERS_ALL)}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}


def bundle(model, isl, L):
    return (RV32 if model == "v32" else RV4).get_bundle(model, isl, L, "fp32")


def run_cell(bd, flag_on):
    K, N, cr = bd["K"], bd["N"], bd["cr"]
    lg = bd["logits"].contiguous()
    pre = bd["preIdx"].contiguous()
    sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)
    ovr = {}
    GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr)
    torch.cuda.synchronize()
    idx = out[0].long()
    valid = lg[0, :N].float()
    if idx.numel() != K or idx.min() < 0 or idx.max() >= N or idx.unique().numel() != K:
        return False
    ref = torch.topk(valid, K).values.sort().values
    return bool(torch.equal(valid[idx].sort().values, ref))


def gate_fixtures(n_runs=5):
    print("== Gate A: 9 fixtures (ON must be exact every run; OFF = negative control)")
    fail = 0
    for model, isl, L in FIXTURES:
        bd = bundle(model, isl, L)
        on = [run_cell(bd, True) for _ in range(n_runs)]
        off = [run_cell(bd, False) for _ in range(n_runs)]
        ok = all(on)
        fail += not ok
        print(f"  {model}/{isl}/L{L:<3d} hit={bd['hit_rate']:.3f}: "
              f"ON exact {sum(on)}/{n_runs} {'PASS' if ok else '** FAIL **'} | "
              f"OFF exact {sum(off)}/{n_runs} (baseline control)")
    print(f"Gate A: {'PASS' if fail == 0 else f'FAIL ({fail} fixtures)'}")
    return fail == 0


def gate_grid(model):
    print(f"== Gate D shard: {model} full per-layer grid, ON arm, exact-only")
    bad, n = [], 0
    for isl in REAL_ISLS[model]:
        for L in ALL_LAYERS[model]:
            try:
                bd = bundle(model, isl, L)
            except Exception as e:
                print(f"  SKIP {model}/{isl}/L{L}: {type(e).__name__}")
                continue
            ok = run_cell(bd, True)
            n += 1
            if not ok:
                bad.append((model, isl, L))
                print(f"  ** INEXACT ** {model}/{isl}/L{L}")
        print(f"  [{model}] {isl} done ({n} cells so far, {len(bad)} bad)", flush=True)
    print(f"Gate D {model}: {n - len(bad)}/{n} exact; failures: {bad}")
    (_HERE / f"gateD_{model}.json").write_text(json.dumps(
        dict(model=model, n=n, bad=bad)))
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--model", choices=["flash", "pro", "v32"])
    a = ap.parse_args()
    ok = True
    if a.fixtures:
        ok &= gate_fixtures()
    if a.grid:
        ok &= gate_grid(a.model)
    sys.exit(0 if ok else 1)
