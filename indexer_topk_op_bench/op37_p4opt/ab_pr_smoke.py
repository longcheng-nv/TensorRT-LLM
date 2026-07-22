# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce-on-current-head smoke A/B for the PR port (gvrpkgpr = worktree
tree @25efb6ec + d2a/d2b/d1a splices + fp32 default-ON).

The 865-grid verdict ran on the campaign snapshot head; the live PR head has
drifted (kc_diet + LB SMEM pinning). This smoke re-verdicts 8 representative
real cells on the PORTED tree: arm base = flags forced False, arm opt =
ctor defaults (fp32 -> ON). Paired same-GPU nsys cold-L2.

Also asserts default resolution: fp32 -> ON/ON/ON, bf16/fp16 -> OFF.

  (under nsys) python3 ab_pr_smoke.py
"""
import csv
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE / "variant"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

from gvrpkgpr.top_k.gvr_topk_decode import GvrTopKKernel as KPR  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from validate_op37 import exact_set  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
WARMUP, REPS = 10, 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

# default resolution sanity (host-only)
for cdt, want in ((cutlass.Float32, True), (cutlass.BFloat16, False),
                  (cutlass.Float16, False)):
    k = KPR(dtype=cdt, top_k=1024, next_n=1, compress_ratio=4,
            return_output_values=False)
    got = (k.p4_rs_rw_search, k.p4_fine_skip, k.p4_peer_push)
    assert got == (want,) * 3, f"default resolution wrong for {cdt}: {got}"
print("[smoke] ctor default resolution OK (fp32 ON, bf16/fp16 OFF)",
      flush=True)

ARMS = [
    ("base", dict(p4_rs_rw_search=False, p4_fine_skip=False,
                  p4_peer_push=False)),
    ("opt", {}),   # ctor defaults -> fp32 ON
]
CELLS = [("flash", "4k", 2), ("flash", "128k", 42), ("flash", "512k", 2),
         ("pro", "8k", 30), ("pro", "128k", 30), ("pro", "1024k", 30),
         ("v32", "16k", 34), ("v32", "256k", 34)]

_CACHE = {}


def compile_arm(K, cr, cfg, flags):
    key = (K, cr) + tuple(sorted(cfg.items())) + tuple(sorted(flags.items()))
    c = _CACHE.get(key)
    if c is not None:
        return c
    kobj = KPR(dtype=cutlass.Float32, top_k=K, next_n=1, compress_ratio=cr,
               return_output_values=False, **cfg, **flags)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cutlass.Float32, (nr, nc),
                                        stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, K),
                                        stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, K),
                                        stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, stream=fs,
                     options="--enable-tvm-ffi")
    _CACHE[key] = c
    return c


def launch_cfg(logits, N):
    cfg = KPR.pick_config(torch.float32, 1, N)
    if cfg["cluster_size"] > 1:
        try:
            from gvrpkgpr.top_k.single_pass_multi_cta_radix_topk_cluster import (
                _query_max_cluster_size,
            )
            cfg["cluster_size"] = min(cfg["cluster_size"],
                                      _query_max_cluster_size())
        except ImportError:
            pass
        cfg["cluster_size"] = min(cfg["cluster_size"], 16)
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    return cfg


def main():
    rows = []
    prof.start()
    for model, isl, layer in CELLS:
        uuid = f"{model}_{isl}_L{layer:02d}"
        RD = RV32 if model == "v32" else RV4
        b = RD.get_bundle(model, isl, layer, "fp32")
        logits, pre = b["logits"].contiguous(), b["preIdx"].contiguous()
        N, K, cr = b["N"], b["K"], b["cr"]
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        cfg = launch_cfg(logits, N)
        for arm, flags in ARMS:
            fn = compile_arm(K, cr, cfg, flags)
            oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
            fn(logits, pre, sl, None, oi, None)
            torch.cuda.synchronize()
            ok = exact_set(oi, logits[0], K, N)
            for _ in range(WARMUP):
                fn(logits, pre, sl, None, oi, None)
            torch.cuda.synchronize()
            for _ in range(REPS):
                _EVICT.random_()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"c|{arm}|{uuid}|fp32")
                fn(logits, pre, sl, None, oi, None)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            rows.append(dict(uuid=uuid, arm=arm, K=K, N=N,
                             cs=cfg["cluster_size"], exact=ok))
            print(f"  {uuid} {arm} exact={ok}", flush=True)
        RV4._bundle_cache.clear()
        RV32._bundle_cache.clear()
    prof.stop()
    with open(HERE / "ship" / "ship_prsmoke.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    bad = [r["uuid"] + "/" + r["arm"] for r in rows if not r["exact"]]
    print(f"[smoke] done; inexact: {bad or 'none'}", flush=True)


if __name__ == "__main__":
    main()
