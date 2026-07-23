# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op40 paired A/B driver over the 865-cell real decode grid (envelope:
op26 REPORT SS7b — flash 21L x 9 ISL + pro 30L x 9 + v32 58L x 7, BS=1 fp32).

One invocation = ONE batch (one nsys rep). Per cell: arms back-to-back on the
SAME GPU, 10 warmup + 20 cold-L2 NVTX'd launches, tie-robust exactness per
cell x arm. Idempotent at batch granularity: per-batch csv = done marker.

  python3 ab40.py --arms base --batch "real865 pro 64k"
  python3 ab40.py --arms base,v1 --batch "real865 flash 4k"
  python3 ab40.py --arms base --list
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent          # op40_omni_gvr/scripts
OP40 = HERE.parent
BENCH = OP40.parent                              # indexer_topk_op_bench

sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(HERE))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402
from arms40 import ARMS as ARM_REG, resolve  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"
WARMUP, REPS = 10, 20
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

REAL_ISLS = {"flash": RV4.ISLS, "pro": RV4.ISLS,
             "v32": ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
REAL_LAYERS = {"flash": RV4.MODELS["flash"]["layers"],
               "pro": RV4.MODELS["pro"]["layers"],
               "v32": list(RV32.LAYERS_ALL)}

_CACHE = {}


def compile_arm(arm, K, cr, cfg):
    key = (arm, K, cr) + tuple(sorted(cfg.items()))
    c = _CACHE.get(key)
    if c is not None:
        return c
    kls, flags = resolve(arm)
    flags.update(flags.pop("per_k", {}).get(K, {}))
    kobj = kls(dtype=cutlass.Float32, top_k=K, next_n=1, compress_ratio=cr,
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
    """Config from the BASELINE's pick_config (envelope contract: variants must
    accept the same launch config surface; a variant needing a different config
    space gets its own pick hook in arms40)."""
    import gvrpkg40b.top_k.gvr_topk_decode as B
    cfg = B.GvrTopKKernel.pick_config(torch.float32, 1, N)
    if cfg["cluster_size"] > 1:
        cfg["cluster_size"] = min(cfg["cluster_size"], 16)
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    return cfg


def exact_set(out, logits_row, K, N):
    """Tie-aware value-multiset vs torch.topk + index validity."""
    lg = logits_row[:N].float()
    idx = out.flatten().to(torch.int64)
    if idx.numel() != K or int(idx.min()) < 0 or int(idx.max()) >= N:
        return False
    if torch.unique(idx).numel() != K:
        return False
    ref = torch.topk(lg, K).values
    return bool(torch.equal(lg[idx].sort(descending=True).values, ref))


def all_batches():
    return [f"real865 {m} {isl}" for m in ("flash", "pro", "v32")
            for isl in REAL_ISLS[m]]


def batch_cells(spec):
    parts = spec.split()
    assert parts[0] == "real865", spec
    m, isl = parts[1], parts[2]
    RD = RV32 if m == "v32" else RV4
    for L in REAL_LAYERS[m]:
        def load(m=m, isl=isl, L=L, RD=RD):
            bd = RD.get_bundle(m, isl, L, "fp32")
            return (bd["logits"].contiguous(), bd["preIdx"].contiguous(),
                    bd["N"], bd["K"], bd["cr"])
        yield f"{m}_{isl}_L{L:02d}", load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base")
    ap.add_argument("--batch", default=None)
    ap.add_argument("--tagdir", default="ab", help="results/<tagdir>/ output")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(all_batches()))
        return
    arms = args.arms.split(",")
    for a in arms:
        assert a in ARM_REG, f"unknown arm {a}"
    spec = args.batch
    tag = spec.replace(" ", "_")
    out_csv = OP40 / "results" / args.tagdir / f"{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cells = list(batch_cells(spec))
    print(f"[ab40] batch '{spec}' arms={arms}: {len(cells)} cells", flush=True)
    rows = []
    prof.start()
    for uuid, load in cells:
        try:
            logits, pre, N, K, cr = load()
            sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
            cfg = launch_cfg(logits, N)
            for arm in arms:
                fn = compile_arm(arm, K, cr, cfg)
                oi = torch.full((1, K), -7, dtype=torch.int32, device=DEV)
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
            del logits, pre
            RV4._bundle_cache.clear()
            RV32._bundle_cache.clear()
        except Exception as e:
            rows.append(dict(uuid=uuid, arm="ERROR", K=0, N=0, cs=0,
                             exact=False))
            print(f"[ab40] ERROR {uuid}: {e!r}", flush=True)
    prof.stop()

    tmp = out_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.rename(out_csv)
    bad = [r["uuid"] + "/" + str(r["arm"]) for r in rows if not r["exact"]]
    print(f"[ab40] batch done; inexact: {bad or 'none'}", flush=True)


if __name__ == "__main__":
    main()
