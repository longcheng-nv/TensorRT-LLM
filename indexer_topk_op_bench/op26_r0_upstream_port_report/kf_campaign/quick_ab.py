# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quick local A/B: KernelFactory candidate vs PR head GVR on the 28 campaign cells.

Usage:
  python3 quick_ab.py --cand <dir-with-kernel.cu> [--entry kernel.run] [--cells all]
  python3 quick_ab.py --self-test        # PR arm vs itself (harness sanity)

Candidate contract (CudaGym cuda_cpp): TVM-FFI entry
  fn(TensorView logits[1,npad] f32, TensorView pre_idx[1,k] i32,
     int n_valid (or TensorView scalar), TensorView indices[1,k] i32 out)
built with tvm_ffi.cpp.build; loaded with tvm_ffi.load_module.

Timing: CUDA-event, cold-L2 (512 MB evict buffer between reps), 30 reps,
median; PR arm and candidate arm interleaved back-to-back on the SAME GPU
(pair purity). Exactness: tie-robust value check vs torch.topk.
NOTE: quick loop only — the ship verdict is the nsys full-grid sweep.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BENCH = REPORT.parent
sys.path.insert(0, str(HERE / "gvrpkg_head"))
sys.path.insert(0, str(BENCH / "harness"))

from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402  (gvrpkg_head)
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

DEV = "cuda"
EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)


def flush_l2():
    EVICT.random_()
    torch.cuda.synchronize()


def load_cells():
    return list(csv.DictReader(open(HERE / "cells_meta.csv")))


def bundle_for(row):
    mod = v32 if row["model"] == "v32" else v4
    return mod.get_bundle(row["model"], row["isl"], int(row["layer"]), "fp32")


def pr_call(b):
    K, cr, N = b["K"], b["cr"], b["N"]
    lg = b["logits"].contiguous()
    pre = b["preIdx"].contiguous()
    sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)
    return (lambda: GvrTopKKernel.launch(lg, pre, sl, out, K,
                                         compress_ratio=cr)), out


def build_candidate(cdir, name="kf_cand"):
    """Build a harvested solution dir -> (module, entry).

    Detects torch-extension (PYBIND11 / torch/extension.h) vs TVM-FFI."""
    cdir = Path(cdir)
    srcs = sorted(str(p) for p in cdir.glob("*.cu")) + \
           sorted(str(p) for p in cdir.glob("*.cpp"))
    assert srcs, f"no sources in {cdir}"
    blob = "".join(Path(s).read_text() for s in srcs)
    if "PYBIND11_MODULE" in blob or "torch/extension.h" in blob:
        from torch.utils.cpp_extension import load
        (cdir / "build_pt").mkdir(exist_ok=True)
        mod = load(name=name, sources=srcs,
                   build_directory=str((cdir / "build_pt").resolve()),
                   extra_cuda_cflags=["-O3"], verbose=False)
        fns = [f for f in dir(mod) if not f.startswith("_")]
        return mod, ("run" if "run" in fns else fns[0])
    import tvm_ffi
    lib = tvm_ffi.cpp.build(name=name, sources=srcs,
                            output_dir=str(cdir / "build"))
    mod = tvm_ffi.load_module(lib)
    return mod, [f for f in dir(mod) if not f.startswith("_")][0]


def cand_call(b, mod, entry):
    K, N = b["K"], b["N"]
    lg = b["logits"].float().contiguous()
    pre = b["preIdx"].contiguous()
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)
    fn = getattr(mod, entry)
    return (lambda: fn(lg, pre, N, out)), out


def exact(b, out):
    lg = b["logits"][0, :b["N"]].float()
    idx = out.flatten().to(torch.int64)
    if idx.numel() != b["K"] or int(idx.min()) < 0 or int(idx.max()) >= b["N"]:
        return False, "range/count"
    if torch.unique(idx).numel() != b["K"]:
        return False, "dup"
    sel = lg[idx].sort().values
    ref = lg[b["ref"].to(torch.int64)].sort().values
    ok = bool(torch.equal(sel, ref))
    return ok, "" if ok else f"vdiff={float((sel-ref).abs().max()):.3e}"


def time_call(call, reps=30):
    ts = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        flush_l2()
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)  # us
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", help="candidate solution dir (kernel.cu / sources)")
    ap.add_argument("--entry", default=None, help="entry symbol (default: autodetect)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cmod, entry = None, None
    if not args.self_test:
        import tvm_ffi
        cdir = Path(args.cand)
        srcs = sorted(str(p) for p in cdir.glob("*.cu")) + \
               sorted(str(p) for p in cdir.glob("*.cpp"))
        assert srcs, f"no sources in {cdir}"
        lib = tvm_ffi.cpp.build(name="kf_cand", sources=srcs,
                                output_dir=str(cdir / "build"))
        cmod = tvm_ffi.load_module(lib)
        if args.entry:
            entry = args.entry
        else:
            fns = [f for f in dir(cmod) if not f.startswith("_")]
            assert len(fns) >= 1, fns
            entry = fns[0]
        print(f"candidate entry: {entry}")

    rows = load_cells()
    res, ratios = [], []
    for r in rows:
        b = bundle_for(r)
        pcall, pout = pr_call(b)
        pcall(); torch.cuda.synchronize()          # compile/warm PR
        ok_pr, _ = exact(b, pout)
        if args.self_test:
            ccall, cout = pr_call(b)
        else:
            ccall, cout = cand_call(b, cmod, entry)
        ccall(); torch.cuda.synchronize()          # warm candidate
        ok_c, why = exact(b, cout)
        t_pr = time_call(pcall)
        t_c = time_call(ccall)
        ratio = t_pr / t_c
        ratios.append(ratio)
        res.append(dict(uuid=r["uuid"], pr_us=round(t_pr, 2),
                        cand_us=round(t_c, 2), speedup=round(ratio, 3),
                        pr_exact=ok_pr, cand_exact=ok_c, why=why,
                        csv_pr_us=r["pr_us"]))
        print(f"{r['uuid']:22s} pr={t_pr:7.2f}us cand={t_c:7.2f}us "
              f"x{ratio:5.3f} exact={ok_c} {why}")
    gm = statistics.geometric_mean(ratios)
    n_reg = sum(1 for x in ratios if x < 1.0)
    n_bad = sum(1 for e in res if not e["cand_exact"])
    print(f"\nGEOMEAN {gm:.4f}  regressions(<1.0) {n_reg}/28  inexact {n_bad}/28")
    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(geomean=gm, regressions=n_reg, inexact=n_bad, cells=res),
            indent=1))


if __name__ == "__main__":
    main()
