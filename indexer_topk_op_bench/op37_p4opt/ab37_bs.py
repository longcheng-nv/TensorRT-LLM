# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BS-scaling A/B for draft PR #16715 (gvrpkgpr): base (flags OFF =
PR#16457 behavior) vs opt (ctor defaults, fp32 ON), real decode captures
with the SAME row replicated across the batch.

Grid: 3 models x all ISL rungs x BS {1..1024 pow2}; report-matching layers
(flash L22 / pro L30 / v32 L34) so the base arm can be anchor-checked
against REPORT SS7's bs_real.csv `pr` column (fp32 rows).

Per cell: both arms back-to-back on ONE GPU, 10 warmup + 20 cold-L2 NVTX'd
launches; exactness (tie-robust) on row 0 and row BS-1. cluster size follows
production pick_config(fp32, BS, N) (numRows saturation drops cs at high BS).
One invocation = one (model, isl) batch (= one nsys rep); csv is the resume
marker.

  python3 ab37_bs.py --batch "flash 128k"     |  --list
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
sys.path.insert(0, str(HERE / "variant"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

from gvrpkgpr.top_k.gvr_topk_decode import GvrTopKKernel as KPR  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
WARMUP, REPS = 10, 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

ARMS = [
    ("base", dict(p4_rs_rw_search=False, p4_fine_skip=False,
                  p4_peer_push=False)),
    ("opt", {}),   # ctor defaults -> fp32 ON
]
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
REAL_LAYER = {"flash": 22, "pro": 30, "v32": 34}   # == REPORT SS7 layers
ISLS = {"flash": RV4.ISLS, "pro": RV4.ISLS,
        "v32": ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}

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


def launch_cfg(logits, BS, N):
    cfg = KPR.pick_config(torch.float32, BS, N)
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


def exact_rows(oi, row_logits, K, N, BS):
    ref = torch.topk(row_logits[:N].float(), K).values.sort().values
    for r in (0, BS - 1) if BS > 1 else (0,):
        idx = oi[r].long()
        if (idx.numel() != K or idx.min() < 0 or idx.max() >= N
                or idx.unique().numel() != K):
            return False
        if not torch.equal(row_logits[idx].float().sort().values, ref):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    batches = [f"{m} {isl}" for m in ("flash", "pro", "v32")
               for isl in ISLS[m]]
    if args.list:
        print("\n".join(batches))
        return
    model, isl = args.batch.split()
    L = REAL_LAYER[model]
    tag = f"{model}_{isl}"
    out_csv = HERE / "ship" / f"bs_{tag}.csv"

    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, L, "fp32")
    row_logits = b["logits"][0].contiguous()          # [N] source row
    row_pre = b["preIdx"][0].contiguous()             # [K]
    N, K, cr = b["N"], b["K"], b["cr"]
    print(f"[bs] {tag} L{L:02d} N={N} K={K} cr={cr}", flush=True)

    rows = []
    prof.start()
    for BS in BS_GRID:
        uuid = f"{model}_{isl}_L{L:02d}_bs{BS}"
        try:
            logits = row_logits.unsqueeze(0).expand(BS, -1).contiguous()
            pre = row_pre.unsqueeze(0).expand(BS, -1).contiguous()
            sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
            cfg = launch_cfg(logits, BS, N)
            for arm, flags in ARMS:
                fn = compile_arm(K, cr, cfg, flags)
                oi = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                fn(logits, pre, sl, None, oi, None)
                torch.cuda.synchronize()
                ok = exact_rows(oi, row_logits, K, N, BS)
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
                rows.append(dict(uuid=uuid, model=model, isl=isl, BS=BS,
                                 arm=arm, K=K, N=N,
                                 cs=cfg["cluster_size"], exact=ok))
                print(f"  {uuid} cs{cfg['cluster_size']} {arm} exact={ok}",
                      flush=True)
            del logits, pre, oi
            torch.cuda.empty_cache()
        except Exception as e:
            rows.append(dict(uuid=uuid, model=model, isl=isl, BS=BS,
                             arm="ERROR", K=K, N=N, cs=0, exact=False))
            print(f"[bs] ERROR {uuid}: {e!r}", flush=True)
    prof.stop()

    tmp = out_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.rename(out_csv)
    bad = [r["uuid"] + "/" + r["arm"] for r in rows if not r["exact"]]
    print(f"[bs] batch done; inexact: {bad or 'none'}", flush=True)


if __name__ == "__main__":
    main()
