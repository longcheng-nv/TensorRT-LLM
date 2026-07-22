# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D2b probe: measure (cnt[b*], need=K-rank_above) across the 865-cell grid.

One warmup + one read per cell (values are launch-deterministic for fixed
inputs). Decodes phase_ts slot 15; slot==0 => degenerate (pipeline skipped).
Fire classes:
  skip_full : cnt[b*] <= need  (whole straddling bin admitted -> fine level
              + its rescan skippable; by construction cum>=K so == equality)
  tiny_<T>  : cnt[b*] <= T for T in (32, 128)  (direct thread0/p4tt-style
              select over the bin class instead of the fine recursion)
Writes probe_d2b.csv + prints firing-rate summary by rung and K.
"""
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent          # probe_d2b
PP = HERE.parent                                 # p4_pipeline
P4F1 = PP.parent
REPORT = P4F1.parent
BENCH = REPORT.parent

sys.path.insert(0, str(REPORT / "kf_campaign" / "gvrpkg_head"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PP))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel as ProdK  # noqa: E402
from gvrpkgp4probe.top_k.gvr_topk_decode import GvrTopKKernel as ProbeK  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)
DEV = "cuda"

_CACHE = {}


def probe_compile(K, cr, cfg):
    key = (K, cr) + tuple(sorted(cfg.items()))
    c = _CACHE.get(key)
    if c is not None:
        return c
    kobj = ProbeK(dtype=cutlass.Float32, top_k=K, next_n=1, compress_ratio=cr,
                  return_output_values=False, **cfg)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cutlass.Float32, (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    ts_f = crt.make_fake_compact_tensor(cutlass.Int64, (nr, 16), stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, ts_f, stream=fs,
                     options="--enable-tvm-ffi")
    _CACHE[key] = c
    return c


def launch_cfg(logits, N):
    cfg = ProdK.pick_config(torch.float32, 1, N)
    if cfg["cluster_size"] > 1:
        try:
            from gvrpkg.top_k.single_pass_multi_cta_radix_topk_cluster import (
                _query_max_cluster_size,
            )
            cfg["cluster_size"] = min(cfg["cluster_size"], _query_max_cluster_size())
        except ImportError:
            pass
        cfg["cluster_size"] = min(cfg["cluster_size"], 16)
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    return cfg


def main():
    rows = list(csv.DictReader(open(REPORT / "real_3arm_layers_full.csv")))
    out = []
    for k, row in enumerate(rows):
        model, isl, layer = row["model"], row["isl"], int(row["layer"])
        uuid = f"{model}_{isl}_L{layer:02d}"
        try:
            RD = RV32 if model == "v32" else RV4
            b = RD.get_bundle(model, isl, layer, "fp32")
            logits = b["logits"].contiguous()
            pre = b["preIdx"].contiguous()
            N, K, cr = b["N"], b["K"], b["cr"]
            sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
            oi = torch.empty(1, K, dtype=torch.int32, device=DEV)
            ts = torch.zeros(1, 16, dtype=torch.int64, device=DEV)
            cfg = launch_cfg(logits, N)
            fn = probe_compile(K, cr, cfg)
            fn(logits, pre, sl, None, oi, None, ts)
            torch.cuda.synchronize()
            v = int(ts[0, 15].item())
            cntb, need = (v >> 32) & 0xFFFFFFFF, v & 0xFFFFFFFF
            del b
            RV4._bundle_cache.clear()
            RV32._bundle_cache.clear()
            rg = ("cs1-small" if cfg["cluster_size"] == 1 and N <= 8448 else
                  "cs1-mid" if cfg["cluster_size"] == 1 else
                  f"cs{cfg['cluster_size']}")
            out.append(dict(uuid=uuid, model=model, isl=isl, layer=layer,
                            K=K, N=N, cs=cfg["cluster_size"], rung=rg,
                            cntb=cntb, need=need,
                            degenerate=(v == 0),
                            skip_full=(v != 0 and cntb <= need),
                            tiny32=(v != 0 and cntb <= 32),
                            tiny128=(v != 0 and cntb <= 128)))
            if (k + 1) % 100 == 0:
                print(f"[{k+1}/{len(rows)}]", flush=True)
        except Exception as e:
            out.append(dict(uuid=uuid, model=model, isl=isl, layer=layer,
                            error=repr(e)))
            print(f"ERROR {uuid}: {e!r}", flush=True)

    keys = ["uuid", "model", "isl", "layer", "K", "N", "cs", "rung", "cntb",
            "need", "degenerate", "skip_full", "tiny32", "tiny128", "error"]
    with open(HERE / "probe_d2b.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)

    ok = [r for r in out if not r.get("error")]
    live = [r for r in ok if not r["degenerate"]]
    print(f"\ncells {len(ok)} (errors {len(out)-len(ok)}), degenerate "
          f"{len(ok)-len(live)}")
    import statistics as stt
    for grp, sel in ([("ALL", lambda r: True)] +
                     [(f"K{K}", lambda r, K=K: r["K"] == K) for K in (512, 1024, 2048)] +
                     [(rg, lambda r, rg=rg: r["rung"] == rg)
                      for rg in ("cs1-small", "cs1-mid", "cs4", "cs8")]):
        g = [r for r in live if sel(r)]
        if not g:
            continue
        print(f" {grp:9s} n={len(g):3d} cntb med={stt.median([r['cntb'] for r in g]):6.0f} "
              f"p90={sorted(r['cntb'] for r in g)[int(len(g)*.9)]:6d} "
              f"need med={stt.median([r['need'] for r in g]):5.0f} | "
              f"skip_full {100*sum(r['skip_full'] for r in g)/len(g):4.1f}%  "
              f"tiny32 {100*sum(r['tiny32'] for r in g)/len(g):4.1f}%  "
              f"tiny128 {100*sum(r['tiny128'] for r in g)/len(g):4.1f}%")


if __name__ == "__main__":
    main()
