# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op42 A/B: GVR-BSX (base = R4 champion 28dc11f6, src/gvr_bsx.cu) vs
GVR PR#16457 pinned head @04a0900ff7 (gvrpkg_04a0, native batched).

Data spec (user, 2026-07-24): one real decode-capture cell = (model, isl,
layer) row from the §7b capture; BS>1 REPLICATES the identical row across
batch rows (each row its own copy in a [BS, Npad] tensor — distinct
addresses, no aliasing).

Exactness: tie-aware value multiset per row — output indices in-range,
unique, and sorted(logits[idx]) == sorted(torch.topk(logits, K).values).

Timing: --smoke = CUDA-event cold-L2 median (L1 screen only). Default =
house nsys protocol via harness/sweep_nsys.measure_cell (cold-L2 512MB
evict outside NVTX, 10 warmup + warm + cold reps) — L2 ship arbiter.

Cells: any "<model>_<isl>_L<nn>" (e.g. flash_128k_L36, v32_64k_L20).
Usage:
  python3 ab.py --cell flash_128k_L36 --bs 1,8,64,1024 --smoke
  python3 ab.py --cell pro_1024k_L08 --tag shipA        # nsys mode (run under nsys)
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent          # op42/scripts
OP42 = HERE.parent
BENCH = OP42.parent                             # indexer_topk_op_bench
KF = BENCH / "op26_r0_upstream_port_report" / "kf_campaign"
sys.path.insert(0, str(KF / "gvrpkg_04a0"))
sys.path.insert(0, str(BENCH / "harness"))

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
from sweep_nsys import measure_cell  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

DEV = "cuda"
BS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def all_cells():
    cells = []
    for m, d in v4.MODELS.items():
        for isl in v4.ISLS:
            cells += [f"{m}_{isl}_L{L:02d}" for L in d["layers"]]
    for isl in v32.ISLS:
        cells += [f"v32_{isl}_L{L:02d}" for L in v32.LAYERS_ALL]
    return cells


def parse_cell(cell):
    model, isl, lpart = cell.split("_")
    return model, isl, int(lpart[1:])


def build_bsx(src_dir=None, name="op42_gvr_bsx"):
    from torch.utils.cpp_extension import load
    cdir = Path(src_dir) if src_dir else (OP42 / "src")
    bdir = cdir / "build_pt"
    bdir.mkdir(exist_ok=True)
    srcs = [str(cdir / "gvr_bsx.cu"), str(cdir / "main.cpp")]
    return load(name=name, sources=srcs, build_directory=str(bdir.resolve()),
                extra_cuda_cflags=["-O3", "-gencode",
                                   "arch=compute_100a,code=sm_100a"],
                verbose=False)


class Repl:
    """One real cell row replicated to bs_max identical rows."""

    def __init__(self, cell, bs_max):
        model, isl, L = parse_cell(cell)
        mod = v32 if model == "v32" else v4
        b = mod.get_bundle(model, isl, L, "fp32")
        self.K, self.N, self.Npad, self.cr = b["K"], b["N"], b["Npad"], b["cr"]
        self.logits = b["logits"][0:1].repeat(bs_max, 1).contiguous()
        self.pre = b["preIdx"][0:1].repeat(bs_max, 1).contiguous()
        row = self.logits[0, :self.N]
        self.ref_vals = torch.topk(row.float(), self.K).values.sort().values
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.synchronize()

    def exact(self, out, bs):
        bad = []
        for i in range(bs):
            idx = out[i].to(torch.int64)
            if int(idx.min()) < 0 or int(idx.max()) >= self.N:
                bad.append((i, "range")); continue
            if torch.unique(idx).numel() != self.K:
                bad.append((i, "dup")); continue
            sel = self.logits[i, :self.N][idx].sort().values
            if not torch.equal(sel, self.ref_vals):
                bad.append((i, f"vdiff={float((sel - self.ref_vals).abs().max()):.3e}"))
        return bad


def bsx_call(stack, mod, bs):
    out = torch.empty(bs, stack.K, dtype=torch.int32, device=DEV)
    N = stack.N
    if hasattr(mod, "run_batched"):
        lg, pre = stack.logits[:bs], stack.pre[:bs]

        def call():
            mod.run_batched(lg, pre, N, out)
        return call, out
    # fallback: sequential per-row launches (baseline 28dc11f6 shape)
    lg_rows = [stack.logits[i:i + 1] for i in range(bs)]
    pre_rows = [stack.pre[i:i + 1] for i in range(bs)]
    out_rows = [out[i:i + 1] for i in range(bs)]
    run = mod.run

    def call():
        for i in range(bs):
            run(lg_rows[i], pre_rows[i], N, out_rows[i])
    return call, out


def head_call(stack, bs):
    lg, pre = stack.logits[:bs], stack.pre[:bs]
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
    ap.add_argument("--cell", required=True)
    ap.add_argument("--bs", default=None)
    ap.add_argument("--arms", default="gvr_pr,bsx")
    ap.add_argument("--src", default=None, help="alt src dir for bsx arm")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reps-cold", type=int, default=20)
    ap.add_argument("--reps-warm", type=int, default=10)
    args = ap.parse_args()

    bss = [int(x) for x in args.bs.split(",")] if args.bs else BS_LIST
    arms = args.arms.split(",")
    tag = args.tag or args.cell

    mod = build_bsx(args.src) if "bsx" in arms else None
    stack = Repl(args.cell, max(bss))
    print(f"[ab] {args.cell}: K={stack.K} N={stack.N} Npad={stack.Npad} "
          f"cr={stack.cr} bs={bss}", flush=True)

    exact_log, smoke_rows = {}, []
    for bs in bss:  # pre-warm all head JIT variants outside profile window
        if "gvr_pr" in arms:
            call, _ = head_call(stack, bs)
            call()
    torch.cuda.synchronize()

    if not args.smoke:
        import torch.cuda.profiler as prof
        prof.start()

    for bs in bss:
        for arm in arms:
            call, out = head_call(stack, bs) if arm == "gvr_pr" \
                else bsx_call(stack, mod, bs)
            out.fill_(-1)
            call()
            torch.cuda.synchronize()
            bad = stack.exact(out, bs)
            ok = not bad
            exact_log[f"{args.cell}|BS{bs}|{arm}"] = (
                ok, "" if ok else f"{len(bad)} rows, first={bad[:3]}")
            if not ok:
                print(f"[ab] INEXACT {args.cell} BS{bs} {arm}: {len(bad)} "
                      f"rows, first={bad[:3]}", flush=True)
            if args.smoke:
                us = time_events(call)
                smoke_rows.append((bs, arm, us))
                print(f"[smoke] {args.cell} BS{bs:5d} {arm:8s} "
                      f"{us:10.2f}us exact={ok}", flush=True)
            else:
                measure_cell(call, f"{arm}|{args.cell}|BS{bs}",
                             args.reps_cold, args.reps_warm)
                print(f"[ab] measured {args.cell} BS{bs} {arm} exact={ok}",
                      flush=True)

    if not args.smoke:
        import torch.cuda.profiler as prof
        prof.stop()

    (OP42 / "results" / f"exact_{tag}.json").write_text(
        json.dumps(exact_log, indent=1))
    n_bad = sum(1 for v in exact_log.values() if not v[0])
    print(f"[ab] done, inexact {n_bad}/{len(exact_log)}", flush=True)
    if args.smoke and smoke_rows:
        by_bs = {}
        for bs, arm, us in smoke_rows:
            by_bs.setdefault(bs, {})[arm] = us
        for bs, d in sorted(by_bs.items()):
            if "gvr_pr" in d and "bsx" in d:
                print(f"[smoke] BS{bs:5d} speedup(bsx vs head) "
                      f"{d['gvr_pr'] / d['bsx']:.3f}", flush=True)


if __name__ == "__main__":
    main()
