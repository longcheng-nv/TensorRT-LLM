# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op37 exactness validation: gvrpkg37 variants vs torch.topk (tie-robust).

Arms: baseline (all flags off; == PR head behavior), d2a, d2b, d1a, all.
Inputs: real decode cells covering cs1/cs4/cs8 x K512/1024/2048, plus
adversarial synthetics that force the non-default paths:
  - plateau: massive ties -> cnt[b*] > 128 -> d2b FALLBACK path + exact tail
  - randn: generic
  - narrow: values squeezed into few bins -> big straddling bin
Run: PYTHONNOUSERSITE=1 PYTHONPATH=<cutlass450> python3 validate_op37.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent            # op37_p4opt
BENCH = HERE.parent                                # indexer_topk_op_bench
REPORT = BENCH / "op26_r0_upstream_port_report"

sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE / "variant"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

from gvrpkg37.top_k.gvr_topk_decode import GvrTopKKernel as K37  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"

ARMS = {
    "base": {},
    "d2a": dict(p4_rs_rw_search=True),
    "d2b": dict(p4_fine_skip=True),
    "d1a": dict(p4_peer_push=True),
    "all": dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True),
}

REAL_CELLS = [
    ("flash", "4k", 2), ("flash", "64k", 2), ("flash", "128k", 42),
    ("flash", "256k", 2), ("flash", "512k", 2),
    ("pro", "4k", 2), ("pro", "64k", 30), ("pro", "256k", 30),
    ("pro", "1024k", 30),
    ("v32", "8k", 33), ("v32", "32k", 34), ("v32", "128k", 14),
    ("v32", "256k", 34),
]

_CACHE = {}


def compile_arm(K, cr, cfg, flags):
    key = (K, cr) + tuple(sorted(cfg.items())) + tuple(sorted(flags.items()))
    c = _CACHE.get(key)
    if c is not None:
        return c
    kobj = K37(dtype=cutlass.Float32, top_k=K, next_n=1, compress_ratio=cr,
               return_output_values=False, **cfg, **flags)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cutlass.Float32, (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, stream=fs,
                     options="--enable-tvm-ffi")
    _CACHE[key] = c
    return c


def launch_cfg(logits, N):
    cfg = K37.pick_config(torch.float32, 1, N)
    if cfg["cluster_size"] > 1:
        try:
            from gvrpkg37.top_k.single_pass_multi_cta_radix_topk_cluster import (
                _query_max_cluster_size,
            )
            cfg["cluster_size"] = min(cfg["cluster_size"], _query_max_cluster_size())
        except ImportError:
            pass
        cfg["cluster_size"] = min(cfg["cluster_size"], 16)
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    return cfg


def exact_set(out, logits_row, K, N):
    lg = logits_row[:N].float()
    idx = out.flatten().to(torch.int64)
    if idx.numel() != K or int(idx.min()) < 0 or int(idx.max()) >= N:
        return False
    if torch.unique(idx).numel() != K:
        return False
    ref = torch.topk(lg, K).values
    return bool(torch.equal(lg[idx].sort(descending=True).values, ref))


def synth_cases():
    g = torch.Generator(device=DEV).manual_seed(1234)
    cases = []
    for K, cr in ((512, 4), (1024, 4), (2048, 1)):
        for N in (8192, 65536, 262144):
            logits = torch.randn(1, N, generator=g, device=DEV)
            cases.append((f"randn_K{K}_N{N}", K, cr, logits))
            # plateau: 60% of mass at one value -> giant tie class,
            # cnt[b*] >> 128 -> forces the d2b fallback + exact tail
            lp = torch.randn(1, N, generator=g, device=DEV)
            m = torch.rand(1, N, generator=g, device=DEV) < 0.6
            lp[m] = 1.2345
            cases.append((f"plateau_K{K}_N{N}", K, cr, lp))
            # narrow: squeeze into few coarse bins
            ln = torch.randn(1, N, generator=g, device=DEV) * 1e-4 + 3.0
            cases.append((f"narrow_K{K}_N{N}", K, cr, ln))
    return cases


def run_case(name, K, cr, logits, pre, results):
    N = logits.shape[1]
    sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
    cfg = launch_cfg(logits, N)
    outs = {}
    for arm, flags in ARMS.items():
        fn = compile_arm(K, cr, cfg, flags)
        oi = torch.full((1, K), -7, dtype=torch.int32, device=DEV)
        fn(logits, pre, sl, None, oi, None)
        torch.cuda.synchronize()
        ok = exact_set(oi, logits[0], K, N)
        outs[arm] = ok
        results.append((name, arm, cfg["cluster_size"], ok))
    flag = "" if all(outs.values()) else "  <-- FAIL " + str(
        [a for a, v in outs.items() if not v])
    print(f"{name:28s} cs={cfg['cluster_size']} K={K} "
          f"{'OK' if all(outs.values()) else 'FAIL'}{flag}", flush=True)


def main():
    results = []
    # real cells
    for model, isl, layer in REAL_CELLS:
        RD = RV32 if model == "v32" else RV4
        b = RD.get_bundle(model, isl, layer, "fp32")
        run_case(f"real_{model}_{isl}_L{layer:02d}", b["K"], b["cr"],
                 b["logits"].contiguous(), b["preIdx"].contiguous(), results)
        del b
        RV4._bundle_cache.clear()
        RV32._bundle_cache.clear()
    # synthetic adversarials (preIdx = random valid hint)
    g = torch.Generator(device=DEV).manual_seed(99)
    for name, K, cr, logits in synth_cases():
        N = logits.shape[1]
        pre = torch.randperm(N * cr, generator=g, device=DEV)[:K].to(
            torch.int32).reshape(1, K).contiguous()
        run_case(name, K, cr, logits, pre, results)

    bad = [(n, a) for n, a, _, ok in results if not ok]
    print(f"\nTOTAL {len(results)} checks, FAIL {len(bad)}: {bad or 'none'}")


if __name__ == "__main__":
    main()
