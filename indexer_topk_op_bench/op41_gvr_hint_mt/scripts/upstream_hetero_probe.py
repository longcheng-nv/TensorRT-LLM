# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41 option-2 phase A: does the UPSTREAM production kernel (PR#16457 e612
baseline, op40 gvrpkg40b) carry a straggler tax on heterogeneous BS>1
batches? Its qfrac defaults were validated on replicated-row BS grids only.

Batch = all GVR-active layers of one (model, isl) cycled to BS (hetero) vs
p0-proxy = the same batch built from the FASTEST half of layers (we have no
pass counter here; the surrogate bound is hetero vs per-layer-replicated
mean and max). Metrics per (group, BS):
  t_het          hetero-batch time
  sum_rep/BS     mean of per-layer replicated-batch times (amortized ideal)
  t_worst        max per-layer replicated time (all-straggler bound)
Straggler tax estimate = t_het / t_rep_mean. Exactness checked per row.
Event axis + 512MB evict (op40 protocol constants)."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP41 = HERE.parent
BENCH = OP41.parent
OP40 = BENCH / "op40_omni_gvr"
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(OP40 / "scripts"))

import cutlass  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from ab40 import compile_arm, launch_cfg, exact_set  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)
GROUPS = [("v32", "64k"), ("v32", "128k"), ("v32", "256k"),
          ("pro", "256k"), ("pro", "512k"), ("pro", "1024k"),
          ("flash", "256k"), ("flash", "512k")]
BS_LIST = [16, 64, 256]
LAYERS = {"flash": RV4.MODELS["flash"]["layers"],
          "pro": RV4.MODELS["pro"]["layers"],
          "v32": list(RV32.LAYERS_ALL)}


def timeit(fn, reps=15):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        _EVICT.random_()
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return sum(ts[:max(1, len(ts) // 2)]) / max(1, len(ts) // 2)


def main():
    print("group,BS,nlayers,t_het_us,t_rep_mean_us,t_rep_max_us,tax_est")
    for model, isl in GROUPS:
        RD = RV32 if model == "v32" else RV4
        rows = []
        for L in LAYERS[model]:
            try:
                bd = RD.get_bundle(model, isl, L, "fp32")
            except Exception:
                continue
            rows.append((bd["logits"][0].contiguous(),
                         bd["preIdx"][0].contiguous(),
                         bd["N"], bd["K"], bd["cr"]))
            RV4._bundle_cache.clear()
            RV32._bundle_cache.clear()
        if len({r[0].shape[0] for r in rows}) != 1:
            npad0 = rows[0][0].shape[0]
            rows = [r for r in rows if r[0].shape[0] == npad0]
        n = len(rows)
        N, K, cr = rows[0][2], rows[0][3], rows[0][4]
        lg_all = torch.stack([r[0] for r in rows]).cuda()
        pre_all = torch.stack([r[1] for r in rows]).cuda()
        cfg = launch_cfg(lg_all, N)
        fn = compile_arm("base", K, cr, cfg)
        for bs in BS_LIST:
            idx = torch.arange(bs) % n
            lgb = lg_all[idx].contiguous()
            preb = pre_all[idx].contiguous()
            sl = torch.full((bs,), N * cr, dtype=torch.int32, device=DEV)
            oi = torch.full((bs, K), -7, dtype=torch.int32, device=DEV)
            fn(lgb, preb, sl, None, oi, None)
            torch.cuda.synchronize()
            ok = all(exact_set(oi[i:i + 1], lgb[i], K, N) for i in range(bs))
            if not ok:
                print(f"INEXACT {model}_{isl} BS{bs}")
                sys.exit(1)
            for _ in range(8):
                fn(lgb, preb, sl, None, oi, None)
            torch.cuda.synchronize()
            t_het = timeit(lambda: fn(lgb, preb, sl, None, oi, None))
            # per-layer replicated batches
            reps_t = []
            for li in range(n):
                lgr = lg_all[li:li + 1].expand(bs, -1).contiguous()
                prer = pre_all[li:li + 1].expand(bs, -1).contiguous()
                fn(lgr, prer, sl, None, oi, None)
                torch.cuda.synchronize()
                for _ in range(3):
                    fn(lgr, prer, sl, None, oi, None)
                torch.cuda.synchronize()
                reps_t.append(timeit(
                    lambda: fn(lgr, prer, sl, None, oi, None), reps=7))
                del lgr, prer
            m = sum(reps_t) / len(reps_t)
            print(f"{model}_{isl},{bs},{n},{t_het:.1f},{m:.1f},"
                  f"{max(reps_t):.1f},{t_het / m:.3f}", flush=True)
        del lg_all, pre_all
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
