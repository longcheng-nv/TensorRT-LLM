# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op39 arm v2 gate + L1 screen: full envelope, real preIdx; exactness
(tie-aware multiset) at BS {2,16,256}; event timing at all BS vs report pr.
Also an adversarial mini-track: constant row, near-tie row.

  python3 arm2_gate.py [--perf] [--bs ...]
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent.parent
sys.path.insert(0, str(BENCH / "op38_r3v11_bs"))
sys.path.insert(0, str(BENCH / "harness"))

from probe import bundle, make_batch, exact_rows, timeit  # noqa: E402
from bs38_nsys import all_cells  # noqa: E402
import real_data_v4cap as v4  # noqa: E402
import real_data_v32 as v32  # noqa: E402

CAP = 8192


def build_arm2():
    # Two-step build (iter14): kernel.cu needs -rdc=true for the CDP2 K2
    # tail-launch, and torch's JIT load() does no device-link step — so nvcc
    # builds the pure-CUDA kernel into its own .so (device link happens
    # automatically when nvcc emits a shared lib), and torch only compiles
    # the binding, linking against it.
    import hashlib
    import subprocess
    from torch.utils.cpp_extension import load
    kdir = HERE.parent / "src" / "arm_v2"
    build = kdir / "build_pt"
    build.mkdir(exist_ok=True)
    src = kdir / "kernel.cu"
    lib = build / "libarm39k.so"
    stamp = build / "kernel.hash"
    # ARM39_BUILD_CDP=1 adds -rdc + the CDP2 tail-launch path. Default OFF:
    # -rdc costs ~15-20% on this reg-starved kernel (device runtime reserve).
    import os
    cdp = os.environ.get("ARM39_BUILD_CDP", "0") == "1"
    cmd = ["/usr/local/cuda/bin/nvcc", "-shared", "-Xcompiler", "-fPIC",
           "-O3", "-gencode", "arch=compute_100a,code=sm_100a", str(src)]
    if cdp:
        cmd[5:5] = ["-rdc=true", "-DARM39_CDP"]
        cmd += ["-lcudadevrt"]
    cmd += ["-o", str(lib)]
    h = hashlib.sha256(src.read_bytes() + str(cdp).encode()).hexdigest()
    if not lib.exists() or not stamp.exists() or stamp.read_text() != h:
        subprocess.run(cmd, check=True)
        stamp.write_text(h)
    return load(name="op39_arm_v2",
                sources=[str(kdir / "main.cpp")],
                build_directory=str(build),
                extra_include_paths=["/usr/local/cuda/include"],
                extra_ldflags=[f"-L{build}", "-larm39k",
                               f"-Wl,-rpath,{build}",
                               "-L/usr/local/cuda/lib64", "-lcudart"],
                with_cuda=True,
                verbose=False)


def bufs(bs, K):
    return dict(thr=torch.empty(2 * bs, dtype=torch.float32, device="cuda"),
                cv=torch.empty(bs, CAP, dtype=torch.float32, device="cuda"),
                ci=torch.empty(bs, CAP, dtype=torch.int32, device="cuda"),
                cnt=torch.zeros(bs, dtype=torch.int32, device="cuda"),
                done=torch.zeros(bs, dtype=torch.int32, device="cuda"),
                ovf=torch.zeros(bs, dtype=torch.int32, device="cuda"),
                resc=torch.zeros(bs, dtype=torch.int32, device="cuda"),
                out=torch.empty(bs, K, dtype=torch.int32, device="cuda"))


def adversarial(arm):
    """constant row + near-tie (1-ulp apart) row + degenerate hints."""
    K, npad, bs = 512, 8192, 4
    for name, row in [
            ("const", torch.zeros(npad)),
            ("neartie", torch.nextafter(torch.ones(npad),
                                        torch.tensor(2.0)) *
             (1 + torch.arange(npad) % 3 * 1e-7)),
    ]:
        lg = row.repeat(bs, 1).float().cuda().contiguous()
        pre = torch.randint(0, npad, (bs, K), dtype=torch.int32, device="cuda")
        b = bufs(bs, K)
        arm.run(lg, pre, b["thr"], b["cv"], b["ci"], b["cnt"], b["done"],
                b["ovf"], b["resc"], b["out"], 8)
        torch.cuda.synchronize()
        ref = torch.topk(lg[0], K).values.sort().values
        ok = True
        for r in range(bs):
            idx = b["out"][r].to(torch.int64)
            if int(idx.min()) < 0 or int(idx.max()) >= npad or \
               torch.unique(idx).numel() != K:
                ok = False
                break
            if not torch.equal(lg[0][idx].sort().values, ref):
                ok = False
                break
        print(f"[adv] {name}: {'OK' if ok else 'FAIL'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="2,16,256")
    ap.add_argument("--perf", action="store_true")
    args = ap.parse_args()
    arm = build_arm2()
    adversarial(arm)
    import csv
    pr = {}
    for r in csv.DictReader(open(BENCH / "op26_r0_upstream_port_report" /
                                 "bs_real_layers.csv")):
        pr[(r["model"], r["isl"], int(r["L"]), int(r["BS"]))] = float(r["pr"])
    n_bad = n_tot = 0
    perf_lines = []
    for model, isl, L in all_cells():
        b = bundle(model, isl, L)
        K, N = b["K"], b["N"]
        cname = f"{model}_{isl}_L{L:02d}"
        for bs in (int(x) for x in args.bs.split(",")):
            lg, pre = make_batch(b, bs)
            bb = bufs(bs, K)
            chunks = max(1, 592 // bs)
            arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                    bb["done"], bb["ovf"], bb["resc"], bb["out"], chunks)
            torch.cuda.synchronize()
            bad = exact_rows(b, bb["out"], bs)
            n_tot += 1
            if bad:
                n_bad += 1
                print(f"INEXACT {cname} BS{bs}: {bad}", flush=True)
            if args.perf:
                for _ in range(5):
                    arm.run(lg, pre, bb["thr"], bb["cv"], bb["ci"], bb["cnt"],
                            bb["done"], bb["ovf"], bb["resc"], bb["out"], chunks)
                torch.cuda.synchronize()
                us = timeit(lambda: arm.run(lg, pre, bb["thr"], bb["cv"],
                                            bb["ci"], bb["cnt"], bb["done"],
                                            bb["ovf"], bb["resc"], bb["out"],
                                            chunks), reps=11)
                p = pr.get((model, isl, L, bs))
                perf_lines.append((cname, bs, us, p, (p or 0) / us))
            del lg, pre, bb
        print(f"[gate] {cname} done", flush=True)
        v4._bundle_cache.clear()
        v32._bundle_cache.clear()
        torch.cuda.empty_cache()
    print(f"\n[gate] inexact {n_bad}/{n_tot}")
    if perf_lines:
        import statistics
        sp = [x for *_, x in perf_lines if x]
        print(f"[L1 event-axis vs nsys pr] gm {statistics.geometric_mean(sp):.4f} "
              f"mean {statistics.mean(sp):.4f} min {min(sp):.4f} "
              f"<1.0: {sum(1 for v in sp if v < 1)}/{len(sp)}")
        for cname, bs, us, p, x in sorted(perf_lines, key=lambda r: r[4])[:15]:
            print(f"  worst {cname} BS{bs} arm={us:.2f} pr={p} x={x:.3f}")


if __name__ == "__main__":
    main()
