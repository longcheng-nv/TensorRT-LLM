# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-grid (865 real decode cells) sub-P4 pipeline breakdown of the
production GVR top-K kernel at the kf-campaign PR#16457 head. BS=1 fp32,
per (model, ISL, layer) cell. Extends REPORT §9e's [ptime] methodology:
same two-arm design, but the timed twin is gvrpkgp4t_head ([ptime]+[p4sub],
phase_ts int64[1, 16]) which splits P4 select into 8 pipeline stages:

  p4_peer_wait    t5 ->s8   cluster handoff #2 arrive+wait (cs>1 only)
  p4_dsmem_gather s8 ->s9   leader's DSMEM gather of peer candidates
  p4_minmax       s9 ->s10  candidate min/max block reduce (scan #1)
                            (degenerate cand==K / cand<K cells: whole copy
                             path lands here, s11..s14 zero-width)
  p4_coarse_hist  s10->s11  coarse kNumBins histogram zero+build (scan #2)
  p4_coarse_search s11->s12 coarse 3-step high->low bin search
  p4_fine         s12->s13  fine 256-bin re-zero+build+search (scan #3)
  p4_scatter      s13->s14  classify + scatter writeback pass (scan #4)
  p4_tail         s14->t6   output pad + p4_exact_tail / p4tt tie repair

Per cell: 10 warmup + REPS cold-L2 launches per arm (512MB evict outside the
NVTX range), phase_ts read back after every timed launch, per-phase MEDIAN
cycles. Absolute us = fraction x the pristine prod arm's nsys kernel time
(parse with aggregate_p4pipe.py). Validation per cell: (a) both arms exact
vs torch.topk, (b) t0<=..<=t5<=s8<=..<=s14<=t6<=t7 monotone on every launch,
(c) nsys timed-vs-prod overhead gate at aggregation, (d) top-level phase
fractions must reproduce §9e (drift gate on p4_select share).

Run under nsys via drive_p4pipe_shards.sh (8-GPU shard stripe):
  python3 measure_p4pipe_full.py --shard i/m --tag <tag>
Writes p4pipe_<tag>.json(l) next to this script.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent          # p4_pipeline
P4F1 = HERE.parent                               # p4f1_harness
REPORT = P4F1.parent                             # op26_r0_upstream_port_report
BENCH = REPORT.parent                            # indexer_topk_op_bench

sys.path.insert(0, str(REPORT / "kf_campaign" / "gvrpkg_head"))  # gvrpkg (prod)
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(HERE))                                    # gvrpkgp4t_head

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0"), cutlass.__version__

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel as ProdK  # noqa: E402
from gvrpkgp4t_head.top_k.gvr_topk_decode import GvrTopKKernel as TimedK  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)

DEV = "cuda"
WARMUP = 10
NSLOT = 16
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)

# top-level phases (t-slots, §9e-compatible)
PHASES = [
    ("p1_gather_stats", 0, 1),
    ("smem_stage", 1, 2),
    ("p1b_rungs", 2, 3),
    ("p2_count_admission", 3, 4),
    ("p3_collect", 4, 5),
    ("p4_select", 5, 6),
    ("epilogue", 6, 7),
]
# sub-P4 stages (mixed t/s slots); chain t5 -> s8..s14 -> t6
SUBP4 = [
    ("p4_peer_wait", 5, 8),
    ("p4_dsmem_gather", 8, 9),
    ("p4_minmax", 9, 10),
    ("p4_coarse_hist", 10, 11),
    ("p4_coarse_search", 11, 12),
    ("p4_fine", 12, 13),
    ("p4_scatter", 13, 14),
    ("p4_tail", 14, 6),
]
# monotone chain over raw slot indices
CHAIN = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 6, 7]

_TIMED_CACHE = {}


def timed_compile(K, cr, cfg):
    key = (K, cr) + tuple(sorted(cfg.items()))
    c = _TIMED_CACHE.get(key)
    if c is not None:
        return c
    kobj = TimedK(
        dtype=cutlass.Float32,
        top_k=K,
        next_n=1,
        compress_ratio=cr,
        return_output_values=False,
        **cfg,
    )
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cutlass.Float32, (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    ts_f = crt.make_fake_compact_tensor(cutlass.Int64, (nr, NSLOT), stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, ts_f, stream=fs,
                     options="--enable-tvm-ffi")
    _TIMED_CACHE[key] = c
    return c


def launch_cfg(logits, N):
    """Replicate ProdK.launch's cfg post-processing for arm parity."""
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


def exact_set(out, b):
    lg = b["logits"][0, :b["N"]].float()
    idx = out.flatten().to(torch.int64)
    if idx.numel() != b["K"] or int(idx.min()) < 0 or int(idx.max()) >= b["N"]:
        return False
    if torch.unique(idx).numel() != b["K"]:
        return False
    return bool(torch.equal(lg[idx].sort().values,
                            lg[b["ref"].to(torch.int64)].sort().values))


def run_cell(row, reps):
    model, isl, layer = row["model"], row["isl"], int(row["layer"])
    uuid = f"{model}_{isl}_L{layer:02d}"
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, layer, "fp32")
    logits = b["logits"].contiguous()
    pre = b["preIdx"].contiguous()
    N, K, cr = b["N"], b["K"], b["cr"]
    sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
    out_p = torch.empty(1, K, dtype=torch.int32, device=DEV)
    out_t = torch.empty(1, K, dtype=torch.int32, device=DEV)
    ts = torch.zeros(1, NSLOT, dtype=torch.int64, device=DEV)

    cfg = launch_cfg(logits, N)
    timed = timed_compile(K, cr, cfg)

    call_p = lambda: ProdK.launch(logits, pre, sl, out_p, K, compress_ratio=cr)  # noqa: E731
    call_t = lambda: timed(logits, pre, sl, None, out_t, None, ts)  # noqa: E731

    # correctness (pre-timing)
    call_p(); call_t()
    torch.cuda.synchronize()
    ex_p, ex_t = exact_set(out_p, b), exact_set(out_t, b)

    # ---- prod arm: warmup + cold-L2 NVTX reps (nsys absolute anchor) ----
    for _ in range(WARMUP):
        call_p()
    torch.cuda.synchronize()
    for _ in range(reps):
        _EVICT.random_()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"c|prod|{uuid}")
        call_p()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    # ---- timed arm: warmup + cold-L2 NVTX reps + ts readback ----
    for _ in range(WARMUP):
        call_t()
    torch.cuda.synchronize()
    tss = []
    for _ in range(reps):
        _EVICT.random_()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"c|timed|{uuid}")
        call_t()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        tss.append(ts[0].cpu().tolist())

    mono = all(
        all(r[CHAIN[i]] <= r[CHAIN[i + 1]] for i in range(len(CHAIN) - 1))
        for r in tss
    )
    cyc = {n: statistics.median([r[bb] - r[aa] for r in tss])
           for n, aa, bb in PHASES + SUBP4}
    window = statistics.median([r[7] - r[0] for r in tss])
    tot = sum(cyc[n] for n, _, _ in PHASES)
    frac = {k: (v / tot if tot else 0.0) for k, v in cyc.items()}

    del b
    RV4._bundle_cache.clear()
    RV32._bundle_cache.clear()

    return dict(
        uuid=uuid, model=model, isl=isl, layer=layer,
        K=K, N=N, cr=cr, hit=float(row["hit"]),
        cs=cfg["cluster_size"], T=cfg["num_threads"],
        v256=cfg["use_256bit_load"], mbpm=cfg["min_blocks_per_mp"],
        wpr=cfg["enable_warp_parallel_reduce"],
        exact_prod=ex_p, exact_timed=ex_t, mono=mono,
        window_cyc=window, cyc=cyc, frac=frac,
        csv_pr_us=float(row["pr"]) if row.get("pr") else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="i/m stripe")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--cells", default="all", help="uuid,uuid filter")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(REPORT / "real_3arm_layers_full.csv")))
    if args.cells != "all":
        want = set(args.cells.split(","))
        rows = [r for r in rows
                if f"{r['model']}_{r['isl']}_L{int(r['layer']):02d}" in want]
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        rows = rows[i::m]
    print(f"[p4pipe] {len(rows)} cells device={torch.cuda.get_device_name(0)} "
          f"cutlass={cutlass.__version__}", flush=True)

    results = []
    jl = open(HERE / f"p4pipe_{args.tag}.jsonl", "w")
    prof.start()
    for k, row in enumerate(rows):
        try:
            r = run_cell(row, args.reps)
        except Exception as e:  # keep the shard alive; report at aggregation
            r = dict(uuid=f"{row['model']}_{row['isl']}_L{int(row['layer']):02d}",
                     model=row["model"], isl=row["isl"], layer=int(row["layer"]),
                     error=repr(e))
            print(f"[p4pipe] ERROR {r['uuid']}: {e!r}", flush=True)
        results.append(r)
        jl.write(json.dumps(r) + "\n")
        jl.flush()
        if r.get("error") is None:
            p4 = r["frac"]["p4_select"]
            sub = {n: r["frac"][n] for n, _, _ in SUBP4}
            top2 = sorted(sub, key=sub.get, reverse=True)[:2]
            print(f"[{k+1}/{len(rows)}] {r['uuid']:22s} cs={r['cs']} "
                  f"exact={r['exact_prod']}/{r['exact_timed']} mono={r['mono']} "
                  f"p4={100*p4:.0f}% top={top2[0]}:{100*sub[top2[0]]:.0f}% "
                  f"{top2[1]}:{100*sub[top2[1]]:.0f}%", flush=True)
    prof.stop()

    (HERE / f"p4pipe_{args.tag}.json").write_text(json.dumps(results, indent=1))
    bad = [r["uuid"] for r in results
           if r.get("error") or not (r.get("exact_prod") and r.get("exact_timed")
                                     and r.get("mono"))]
    print(f"[p4pipe] done {len(results)} cells; bad={bad or 'none'}", flush=True)


if __name__ == "__main__":
    main()
