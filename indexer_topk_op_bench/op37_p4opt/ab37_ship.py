# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op37 FULL 3-axis ship verdict driver: gvrpkg37 {base, all} paired A/B.

Axes (NOTES.md "Remaining before ship" item 1):
  real865  — full real decode grid fp32 (flash 21L x 9 ISL, pro 30L x 9,
             v32 58L x 7 = 865 cells)
  synth    — op26 SS6 scenario grids best/worst x K{512,1024,2048} x
             dtype{fp32,fp16,bf16}, BS=1, N_GRID (N > 2K)
  realdt   — representative real cells (ab_op37.CELLS) in bf16/fp16

One invocation = ONE batch (one nsys rep, <=2 concurrent via drive script).
Per cell: arms back-to-back on the SAME GPU (ratio purity), 10 warmup +
20 cold-L2 NVTX'd launches, tie-robust exactness on every cell x arm.
Idempotent at batch granularity: the per-batch csv is the done marker.

  python3 ab37_ship.py --batch "real865 flash 64k"
  python3 ab37_ship.py --batch "synth worst 512 fp16"
  python3 ab37_ship.py --batch "realdt pro bf16"
  python3 ab37_ship.py --list          # print all batch specs in order
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent

sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(HERE / "variant"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

from gvrpkg37.top_k.gvr_topk_decode import GvrTopKKernel as K37  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
import bundle_data_env as SYNTH  # noqa: E402
from validate_op37 import launch_cfg, exact_set  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
WARMUP, REPS = 10, 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

ARMS = [
    ("base", {}),
    ("all", dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True)),
]

TDT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
CDT = {"fp32": cutlass.Float32, "bf16": cutlass.BFloat16,
       "fp16": cutlass.Float16}

REAL_ISLS = {"flash": RV4.ISLS, "pro": RV4.ISLS,
             "v32": ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
REAL_LAYERS = {"flash": RV4.MODELS["flash"]["layers"],
               "pro": RV4.MODELS["pro"]["layers"],
               "v32": list(RV32.LAYERS_ALL)}
K_OF = {"flash": 512, "pro": 1024, "v32": 2048}
N_GRID = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

# representative dtype-axis cells == ab_op37.CELLS
REALDT_CELLS = {
    "flash": ([("flash", i, 2) for i in RV4.ISLS] + [("flash", "128k", 42)]),
    "pro": [("pro", i, 30) for i in RV4.ISLS],
    "v32": [("v32", i, 34) for i in REAL_ISLS["v32"]],
}

_CACHE = {}


def compile_arm_dt(K, cr, cfg, flags, dt_name):
    key = (K, cr, dt_name) + tuple(sorted(cfg.items())) + tuple(sorted(flags.items()))
    c = _CACHE.get(key)
    if c is not None:
        return c
    cdt = CDT[dt_name]
    kobj = K37(dtype=cdt, top_k=K, next_n=1, compress_ratio=cr,
               return_output_values=False, **cfg, **flags)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cdt, (nr, nc), stride_order=(1, 0),
                                        assumed_align=ia)
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


def launch_cfg_dt(logits, N, dt_name):
    cfg = K37.pick_config(TDT[dt_name], 1, N)
    if cfg["cluster_size"] > 1:
        try:
            from gvrpkg37.top_k.single_pass_multi_cta_radix_topk_cluster import (
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


def all_batches():
    # ENVELOPE RULING (user, 2026-07-22): perf verdict = SS7b real
    # decode-capture only, BS=1, fp32, K={512,1024,2048} (flash/pro/v32),
    # ISL 4k-1M. synth best/worst + realdt bf16/fp16 axes DROPPED from the
    # ship verdict (batch_cells modes kept for ad-hoc probes).
    b = []
    for m in ("flash", "pro", "v32"):
        for isl in REAL_ISLS[m]:
            b.append(f"real865 {m} {isl}")
    return b


def batch_cells(spec):
    """Yield (uuid, dt_name, loader) — loader() -> (logits[1,N], pre, N, K, cr)."""
    parts = spec.split()
    kind = parts[0]
    if kind == "real865":
        m, isl = parts[1], parts[2]
        RD = RV32 if m == "v32" else RV4
        for L in REAL_LAYERS[m]:
            def load(m=m, isl=isl, L=L, RD=RD):
                bd = RD.get_bundle(m, isl, L, "fp32")
                return (bd["logits"].contiguous(), bd["preIdx"].contiguous(),
                        bd["N"], bd["K"], bd["cr"])
            yield f"{m}_{isl}_L{L:02d}", "fp32", load
    elif kind == "synth":
        scen, K, dt = parts[1], int(parts[2]), parts[3]
        for N in N_GRID:
            if N <= 2 * K:
                continue
            def load(scen=scen, K=K, dt=dt, N=N):
                bd = SYNTH.get_bundle(scen, K, torch.float32, N)
                return (bd["logits"].to(TDT[dt]).contiguous(),
                        bd["preIdx"].contiguous(), N, K, bd["cr"])
            yield f"synth_{scen}_K{K}_N{N}", dt, load
    elif kind == "realdt":
        m, dt = parts[1], parts[2]
        RD = RV32 if m == "v32" else RV4
        for _, isl, L in REALDT_CELLS[m]:
            def load(m=m, isl=isl, L=L, dt=dt, RD=RD):
                bd = RD.get_bundle(m, isl, L, dt)
                return (bd["logits"].contiguous(), bd["preIdx"].contiguous(),
                        bd["N"], bd["K"], bd["cr"])
            yield f"{m}_{isl}_L{L:02d}", dt, load
    else:
        raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(all_batches()))
        return
    spec = args.batch
    tag = spec.replace(" ", "_")
    out_csv = HERE / "ship" / f"ship_{tag}.csv"
    out_csv.parent.mkdir(exist_ok=True)

    cells = list(batch_cells(spec))
    print(f"[ship] batch '{spec}': {len(cells)} cells x {len(ARMS)} arms",
          flush=True)
    rows = []
    prof.start()
    for uuid, dt, load in cells:
        try:
            logits, pre, N, K, cr = load()
            sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
            cfg = launch_cfg_dt(logits, N, dt)
            for arm, flags in ARMS:
                fn = compile_arm_dt(K, cr, cfg, flags, dt)
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
                    torch.cuda.nvtx.range_push(f"c|{arm}|{uuid}|{dt}")
                    fn(logits, pre, sl, None, oi, None)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                rows.append(dict(uuid=uuid, dt=dt, arm=arm, K=K, N=N,
                                 cs=cfg["cluster_size"], exact=ok))
                print(f"  {uuid} {dt} {arm} exact={ok}", flush=True)
            del logits, pre
            RV4._bundle_cache.clear()
            RV32._bundle_cache.clear()
            SYNTH._mem_cache.clear()
        except Exception as e:
            rows.append(dict(uuid=uuid, dt=dt, arm="ERROR", K=0, N=0, cs=0,
                             exact=False))
            print(f"[ship] ERROR {uuid}: {e!r}", flush=True)
    prof.stop()

    tmp = out_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.rename(out_csv)   # marker only lands on full-batch completion
    bad = [r["uuid"] + "/" + r["arm"] for r in rows if not r["exact"]]
    print(f"[ship] batch done; inexact: {bad or 'none'}", flush=True)


if __name__ == "__main__":
    main()
