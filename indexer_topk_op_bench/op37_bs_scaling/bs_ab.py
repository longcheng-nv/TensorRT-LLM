# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op37 BS-scaling A/B: KF R4 champion r3_v11 (7d8272b7) vs PR#16457 head GVR.

Arms
  champion : torch-extension CUDA kernel, BS=1 contract -> BS rows are
             launched SEQUENTIALLY on the same stream (production shape;
             no batching retrofit allowed per handoff).
  gvr_pr   : gvrpkg_04a0 cuteDSL GVR, native batched launch [BS, Npad].

Data: real rows stacked from different layers of the same (model, ISL)
capture (cycled when BS > n_layers); every row keeps its own preIdx.

Timing: run UNDER nsys (house protocol measure_cell: 10 warmup, warm reps,
cold reps with 512MB evict outside the NVTX range). CUDA-event numbers are
collected in --smoke mode only (coarse screen, never a verdict).

Usage (see run_bs_ab.sh):
  python3 bs_ab.py --cell flash_128k [--bs 1,2,...] [--smoke] [--tag t]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent            # op37_bs_scaling
BENCH = HERE.parent                               # indexer_topk_op_bench
KF = BENCH / "op26_r0_upstream_port_report" / "kf_campaign"
sys.path.insert(0, str(KF / "gvrpkg_04a0"))
sys.path.insert(0, str(BENCH / "harness"))

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from sweep_nsys import measure_cell  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

DEV = "cuda"

# 9-cell matrix from the handoff (K x small/mid/large N, three models).
CELLS = {
    "flash_32k":   dict(model="flash", isl="32k",   K=512,  N=8195),
    "flash_128k":  dict(model="flash", isl="128k",  K=512,  N=32771),
    "flash_1024k": dict(model="flash", isl="1024k", K=512,  N=262127),
    "pro_32k":     dict(model="pro",   isl="32k",   K=1024, N=8195),
    "pro_128k":    dict(model="pro",   isl="128k",  K=1024, N=32771),
    "pro_1024k":   dict(model="pro",   isl="1024k", K=1024, N=262127),
    "v32_16k":     dict(model="v32",   isl="16k",   K=2048, N=16399),
    "v32_64k":     dict(model="v32",   isl="64k",   K=2048, N=65551),
    "v32_256k":    dict(model="v32",   isl="256k",  K=2048, N=163775),
}
BS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def layer_list(model, isl):
    if model == "v32":
        return list(v32.LAYERS_ALL)
    return list(v4.MODELS[model]["layers"])


def build_champion():
    from torch.utils.cpp_extension import load
    cdir = HERE / "champion"
    (cdir / "build_pt").mkdir(exist_ok=True)
    srcs = [str(cdir / "gvr.cu"), str(cdir / "main.cpp")]
    mod = load(name="kf_r3v11", sources=srcs,
               build_directory=str((cdir / "build_pt").resolve()),
               extra_cuda_cflags=["-O3", "-gencode",
                                  "arch=compute_100a,code=sm_100a"],
               verbose=False)
    return mod


class Stack:
    """Real-row stack for one cell at max BS; smaller BS use the prefix."""

    def __init__(self, cell, bs_max):
        mod = v32 if cell["model"] == "v32" else v4
        layers = layer_list(cell["model"], cell["isl"])
        b0 = mod.get_bundle(cell["model"], cell["isl"], layers[0], "fp32")
        self.K, self.N, self.Npad, self.cr = b0["K"], b0["N"], b0["Npad"], b0["cr"]
        assert self.N == cell["N"] and self.K == cell["K"], \
            f"loader N/K mismatch: {self.N}/{self.K} vs cell {cell}"
        self.logits = torch.empty(bs_max, self.Npad, dtype=torch.float32,
                                  device=DEV)
        self.pre = torch.empty(bs_max, self.K, dtype=torch.int32, device=DEV)
        self.refs = []          # per-row ref indices (int64), tie-robust check
        self.rows_layer = []
        for i in range(bs_max):
            L = layers[i % len(layers)]
            b = mod.get_bundle(cell["model"], cell["isl"], L, "fp32")
            self.logits[i] = b["logits"][0]
            self.pre[i] = b["preIdx"][0]
            self.refs.append(b["ref"].to(torch.int64))
            self.rows_layer.append(L)
        # keep GPU bundle caches bounded
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.synchronize()

    def exact(self, out, bs):
        """Tie-robust per-row check (quick_ab.exact vectorized over rows)."""
        bad = []
        for i in range(bs):
            lg = self.logits[i, :self.N]
            idx = out[i].to(torch.int64)
            if int(idx.min()) < 0 or int(idx.max()) >= self.N:
                bad.append((i, "range")); continue
            if torch.unique(idx).numel() != self.K:
                bad.append((i, "dup")); continue
            sel = lg[idx].sort().values
            ref = lg[self.refs[i]].sort().values
            if not torch.equal(sel, ref):
                bad.append((i, f"vdiff={float((sel-ref).abs().max()):.3e}"))
        return bad


def champion_call(stack, cmod, bs):
    lg_rows = [stack.logits[i:i + 1] for i in range(bs)]
    pre_rows = [stack.pre[i:i + 1] for i in range(bs)]
    out = torch.empty(bs, stack.K, dtype=torch.int32, device=DEV)
    out_rows = [out[i:i + 1] for i in range(bs)]
    N = stack.N
    run = cmod.run

    def call():
        for i in range(bs):
            run(lg_rows[i], pre_rows[i], N, out_rows[i])
    return call, out


def head_call(stack, bs):
    lg = stack.logits[:bs]
    pre = stack.pre[:bs]
    sl = torch.full((bs,), stack.N * stack.cr, dtype=torch.int32, device=DEV)
    out = torch.empty(bs, stack.K, dtype=torch.int32, device=DEV)
    K, cr = stack.K, stack.cr

    def call():
        GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr)
    return call, out


def time_events(call, reps=15):
    evict = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    ts = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        evict.random_()
        torch.cuda.synchronize()
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--bs", default=None, help="comma list; default full ladder")
    ap.add_argument("--smoke", action="store_true",
                    help="CUDA-event coarse screen, no nsys ranges")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reps-cold", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=10)
    args = ap.parse_args()

    cell = CELLS[args.cell]
    bss = [int(x) for x in args.bs.split(",")] if args.bs else BS_LIST
    tag = args.tag or args.cell

    cmod = build_champion()
    print(f"[bs_ab] champion built; cell={args.cell} {cell} bs={bss}",
          flush=True)
    stack = Stack(cell, max(bss))
    print(f"[bs_ab] stack ready: Npad={stack.Npad} cr={stack.cr} "
          f"layers cycled={len(set(stack.rows_layer))}", flush=True)

    exact_log, smoke_rows = {}, []

    # pre-warm every head-arm JIT variant BEFORE the profiled region
    for bs in bss:
        call, out = head_call(stack, bs)
        call()
        torch.cuda.synchronize()

    if not args.smoke:
        import torch.cuda.profiler as prof
        prof.start()

    for bs in bss:
        for arm in ("gvr_pr", "champion"):
            if arm == "gvr_pr":
                call, out = head_call(stack, bs)
            else:
                call, out = champion_call(stack, cmod, bs)
            call()
            torch.cuda.synchronize()
            bad = stack.exact(out, bs)
            ok = not bad
            exact_log[f"{args.cell}|BS{bs}|{arm}"] = (
                ok, "" if ok else f"{len(bad)} rows, first={bad[:3]}")
            if not ok:
                print(f"[bs_ab] INEXACT {args.cell} BS{bs} {arm}: "
                      f"{len(bad)} rows, first={bad[:3]}", flush=True)
            if args.smoke:
                us = time_events(call)
                smoke_rows.append((bs, arm, us))
                print(f"[smoke] {args.cell} BS{bs:5d} {arm:9s} "
                      f"{us:10.2f}us exact={ok}", flush=True)
            else:
                measure_cell(call, f"{arm}|{args.cell}|BS{bs}",
                             args.reps_cold, args.reps_warm)
                print(f"[bs_ab] measured {args.cell} BS{bs} {arm} "
                      f"exact={ok}", flush=True)

    if not args.smoke:
        import torch.cuda.profiler as prof
        prof.stop()

    (HERE / f"exact_{tag}.json").write_text(json.dumps(exact_log, indent=1))
    n_bad = sum(1 for v in exact_log.values() if not v[0])
    print(f"[bs_ab] done, inexact {n_bad}/{len(exact_log)}", flush=True)
    if args.smoke and smoke_rows:
        by_bs = {}
        for bs, arm, us in smoke_rows:
            by_bs.setdefault(bs, {})[arm] = us
        for bs, d in sorted(by_bs.items()):
            if "gvr_pr" in d and "champion" in d:
                print(f"[smoke] BS{bs:5d} speedup(champ vs head) "
                      f"{d['gvr_pr'] / d['champion']:.3f}", flush=True)


if __name__ == "__main__":
    main()
