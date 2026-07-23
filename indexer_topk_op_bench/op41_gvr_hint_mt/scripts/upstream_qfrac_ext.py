# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op41 option-2 phase B: qfrac candidates on the upstream kernel's hetero
tax pockets (BS16-64), paired vs the per-K defaults. Candidates translate
op41's v3-winning CCDF rank fractions into r0_qfracs tuples (descending).
Guard rows: BS256 (must not regress) + one replicated bench layer."""
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
from upstream_hetero_probe import timeit, LAYERS  # noqa: E402

DEV = "cuda"
GROUPS = [("flash", "64k"), ("flash", "128k"), ("flash", "512k"),
          ("flash", "1024k"), ("v32", "16k"), ("v32", "32k"),
          ("v32", "256k"), ("pro", "1024k")]
BS_LIST = [16, 64, 256, 1024]
# extended sweep: winning variants only, all remaining groups of their K
CANDS = {
    2048: [("wide", (0.9, 0.6))],
    1024: [("m1lo", (0.70,))],
    512: [("v3port", (0.65, 0.25))],
}


def main():
    print("group,BS,variant,def_us,var_us,x_var")
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
        npad0 = rows[0][0].shape[0]
        rows = [r for r in rows if r[0].shape[0] == npad0]
        n = len(rows)
        N, K, cr = rows[0][2], rows[0][3], rows[0][4]
        lg_all = torch.stack([r[0] for r in rows]).cuda()
        pre_all = torch.stack([r[1] for r in rows]).cuda()
        cfg = launch_cfg(lg_all, N)
        fn_def = compile_arm("base", K, cr, cfg)
        variants = []
        for nm, fr in CANDS[K]:
            c2 = dict(cfg)
            c2["r0_qfracs"] = fr
            variants.append((nm, compile_arm("base", K, cr, c2)))
        for bs in BS_LIST:
            idx = torch.arange(bs) % n
            lgb = lg_all[idx].contiguous()
            preb = pre_all[idx].contiguous()
            sl = torch.full((bs,), N * cr, dtype=torch.int32, device=DEV)
            oi = torch.full((bs, K), -7, dtype=torch.int32, device=DEV)

            def measure(fn):
                fn(lgb, preb, sl, None, oi, None)
                torch.cuda.synchronize()
                ok = all(exact_set(oi[i:i + 1], lgb[i], K, N)
                         for i in range(bs))
                if not ok:
                    return None
                for _ in range(8):
                    fn(lgb, preb, sl, None, oi, None)
                torch.cuda.synchronize()
                return timeit(lambda: fn(lgb, preb, sl, None, oi, None))
            # paired interleaved
            td = measure(fn_def)
            if td is None:
                print(f"INEXACT-DEFAULT {model}_{isl} BS{bs}")
                sys.exit(1)
            for nm, fnv in variants:
                tv = measure(fnv)
                if tv is None:
                    print(f"{model}_{isl},{bs},{nm},{td:.1f},INEXACT,-")
                    continue
                td2 = measure(fn_def)  # re-anchor (paired)
                base = min(td, td2)
                print(f"{model}_{isl},{bs},{nm},{base:.1f},{tv:.1f},"
                      f"{base / tv:.3f}", flush=True)
        del lg_all, pre_all
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
