# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op40 Phase-4 exactness gate — three tracks, tie-aware value-multiset.

Track 1 synthetic (cell-seeded: seed = f(K, N, kind); randn allowed, fp32 only)
Track 2 real captures (representative cells across models/ISL incl. envelope
        corners; the FULL 865-cell exactness check rides along with every A/B)
Track 3 adversarial: plateau (giant tie class), narrow (few coarse bins),
        near-tie 1-2 ULP clusters at the K boundary, forced-hit/forced-miss
        preIdx, short-row degrade (seq fits one CTA slice at cs>1).

  PYTHONNOUSERSITE=1 PYTHONPATH=<cutlass450> python3 gate40.py --arms base
"""
import argparse
import struct
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP40 = HERE.parent
BENCH = OP40.parent
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(HERE))

import cutlass  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from ab40 import compile_arm, exact_set, launch_cfg  # noqa: E402
from arms40 import ARMS as ARM_REG  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
KCRS = [(512, 4), (1024, 4), (2048, 1)]

REAL_CELLS = [
    ("flash", "4k", 2), ("flash", "64k", 2), ("flash", "128k", 42),
    ("flash", "256k", 2), ("flash", "1024k", 2),
    ("pro", "4k", 2), ("pro", "32k", 30), ("pro", "64k", 30),
    ("pro", "256k", 30), ("pro", "1024k", 30),
    ("v32", "4k", 33), ("v32", "8k", 33), ("v32", "32k", 34),
    ("v32", "128k", 14), ("v32", "256k", 34),
]


def ulp_up(x, steps=1):
    return struct.unpack("f", struct.pack("f", x))[0] if steps == 0 else None


def synth_cases():
    cases = []
    for K, cr in KCRS:
        for N in (8192, 65536, 262144):
            g = torch.Generator(device=DEV).manual_seed(hash((K, N)) % (2**31))
            logits = torch.randn(1, N, generator=g, device=DEV)
            cases.append((f"randn_K{K}_N{N}", K, cr, logits, None))
            lp = torch.randn(1, N, generator=g, device=DEV)
            m = torch.rand(1, N, generator=g, device=DEV) < 0.6
            lp[m] = 1.2345
            cases.append((f"plateau_K{K}_N{N}", K, cr, lp, None))
            ln = torch.randn(1, N, generator=g, device=DEV) * 1e-4 + 3.0
            cases.append((f"narrow_K{K}_N{N}", K, cr, ln, None))
            # near-tie: boundary values 1-2 ULP apart around the K-th value
            base = torch.randn(1, N, generator=g, device=DEV)
            v = torch.topk(base[0], K + 64).values
            kth = float(v[K - 1])
            nt = base.clone()
            band = torch.arange(2 * K, device=DEV) % 3
            idx = torch.randperm(N, generator=g, device=DEV)[:2 * K]
            eps = torch.tensor([0.0, 1.0, -1.0], device=DEV)[band] * 1.2e-7 * max(abs(kth), 1e-3)
            nt[0, idx] = kth + eps
            cases.append((f"neartie_K{K}_N{N}", K, cr, nt, None))
            # forced hit / miss preIdx on the randn row
            tk = torch.topk(logits[0], K)
            pre_hit = tk.indices.to(torch.int32).reshape(1, K).contiguous()
            pre_miss = torch.topk(-logits[0], K).indices.to(torch.int32).reshape(1, K).contiguous()
            cases.append((f"hit10_K{K}_N{N}", K, cr, logits, pre_hit))
            cases.append((f"miss_K{K}_N{N}", K, cr, logits, pre_miss))
    return cases


def run_case(arms, name, K, cr, logits, pre, results):
    N = logits.shape[1]
    if pre is None:
        g = torch.Generator(device=DEV).manual_seed(hash((name, 7)) % (2**31))
        noisy = logits[0] + 0.5 * torch.randn(N, generator=g, device=DEV)
        pre = torch.topk(noisy, K).indices.to(torch.int32).reshape(1, K).contiguous()
    sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
    cfg = launch_cfg(logits, N)
    outs = {}
    for arm in arms:
        fn = compile_arm(arm, K, cr, cfg)
        oi = torch.full((1, K), -7, dtype=torch.int32, device=DEV)
        fn(logits, pre, sl, None, oi, None)
        torch.cuda.synchronize()
        ok = exact_set(oi, logits[0], K, N)
        outs[arm] = ok
        results.append((name, arm, cfg["cluster_size"], ok))
    status = "OK" if all(outs.values()) else "FAIL " + str(
        [a for a, v in outs.items() if not v])
    print(f"{name:30s} cs={cfg['cluster_size']:2d} K={K:4d} {status}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base")
    args = ap.parse_args()
    arms = args.arms.split(",")
    for a in arms:
        assert a in ARM_REG, a
    results = []
    for model, isl, layer in REAL_CELLS:
        RD = RV32 if model == "v32" else RV4
        try:
            b = RD.get_bundle(model, isl, layer, "fp32")
        except Exception as e:
            print(f"real_{model}_{isl}_L{layer:02d}  LOAD-SKIP {e!r}", flush=True)
            continue
        run_case(arms, f"real_{model}_{isl}_L{layer:02d}", b["K"], b["cr"],
                 b["logits"].contiguous(), b["preIdx"].contiguous(), results)
        del b
        RV4._bundle_cache.clear()
        RV32._bundle_cache.clear()
    for name, K, cr, logits, pre in synth_cases():
        run_case(arms, name, K, cr, logits, pre, results)
    bad = [(n, a) for n, a, _, ok in results if not ok]
    print(f"\nGATE {'GREEN' if not bad else 'RED'}: {len(results)} checks, "
          f"FAIL {len(bad)}: {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
