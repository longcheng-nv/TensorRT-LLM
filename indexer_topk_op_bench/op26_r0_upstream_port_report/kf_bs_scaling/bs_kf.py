# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KF R3 ship kernel (compB) BS-scaling sweep — §7 supplement.

Arms per cell, paired back-to-back on ONE GPU:
  gvr_pr   : REPORT-verbatim PR#16457 arm (gvrpkg_snapshot, ops_rival build:
             cs = 1 if N<64K else 4, seq_len = N*cr, enable_r0) — the local
             anchor that joins this run to rival_bs_layers.csv (b200-027).
  kf_compB : KF R3 ship kernel (harvest/r3_compB, BS=1 single-row contract),
             batched as BS sequential same-stream launches (run_batch loop).

Envelope == rival_bs_layers.csv: real decode captures, fp32, layers
flash {10,22,34} / pro {14,30,46} / v32 {14,34,54}, all ISL rungs,
BS {1..1024 pow2}, SAME ROW replicated across the batch (expand+contiguous).
Timing protocol == harness/sweep_nsys.measure_cell (NVTX c|/w| ranges,
512MB evict OUTSIDE the range, 20 cold + 50 warm). Exactness: tie-robust
top-K value-set check on row 0 and row BS-1.

One invocation = one (model, isl, L) batch = one nsys rep; the jsonl is the
resume marker.   python3 bs_kf.py --batch "flash 4k 22"   |   --list
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent                     # op26_r0_upstream_port_report/
BENCH = REPORT.parent                    # indexer_topk_op_bench/
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(REPORT / "harness"))
sys.path.insert(0, str(REPORT / "gvrpkg_snapshot"))

import cutlass  # noqa: E402

assert cutlass.__version__.startswith("4.5.0"), cutlass.__version__

from sweep_nsys import measure_cell                     # noqa: E402
from exact import compile_kernel                        # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
import real_data_v4cap as RV4                           # noqa: E402
import real_data_v32 as RV32                            # noqa: E402

DEV = "cuda"
REPS_COLD, REPS_WARM = 20, 50
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
LAYERS = {"flash": [10, 22, 34], "pro": [14, 30, 46], "v32": [14, 34, 54]}
ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
        "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
        "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}

_GVR_CACHE = {}


def load_compb():
    from torch.utils.cpp_extension import load
    src = HERE / "compB_src"
    (src / "build_pt").mkdir(exist_ok=True)
    return load(name="kf_compb_bs",
                sources=[str(src / "kernel.cu"), str(src / "main_bs.cpp")],
                build_directory=str(src / "build_pt"),
                extra_cuda_cflags=["-O3"], verbose=False)


def real_bundle(model, isl, L):
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, L, "fp32")
    return b


def gvr_call(b, BS):
    """VERBATIM rival_harness/ops_rival.py gvr_pr build (REPORT §7/§8 arm)."""
    K, cr, N = b["K"], b["cr"], b["N"]
    cs = 1 if N < 65536 else 4
    lg = b["logits"].contiguous().expand(BS, -1).contiguous()
    pre = b["preIdx"].contiguous().expand(BS, -1).contiguous()
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    key = (K, cr, cs)
    f = _GVR_CACHE.get(key)
    if f is None:
        kobj = GvrTopKKernel(dtype=cutlass.Float32, top_k=K, next_n=1,
                             num_threads=1024, compress_ratio=cr,
                             use_256bit_load=True, min_blocks_per_mp=1,
                             cluster_size=cs, return_output_values=False,
                             enable_r0=True)
        f = _GVR_CACHE[key] = compile_kernel(kobj, True)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    call = lambda: f(lg, pre, sl, None, out, None)   # noqa: E731
    call()
    torch.cuda.synchronize()
    return call, [lg, pre, sl, out], {"cs": cs}, out


def compb_call(b, BS, mod):
    """compB: 64-aligned fp32-min padded row (definition.json contract),
    replicated to BS; run_batch = BS sequential same-stream launches."""
    K, N = b["K"], b["N"]
    W = (N + 63) // 64 * 64
    row = torch.full((1, W), torch.finfo(torch.float32).min,
                     dtype=torch.float32, device=DEV)
    row[0, :N] = b["logits"][0, :N].float()
    lg = row.expand(BS, -1).contiguous()
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    call = lambda: mod.run_batch(lg, N, out)         # noqa: E731
    call()
    torch.cuda.synchronize()
    return call, [lg, out], {}, out


def exact_rows(out, b, BS):
    lg = b["logits"][0, :b["N"]].float()
    ref = torch.topk(lg, b["K"]).values.sort().values
    for r in (0, BS - 1) if BS > 1 else (0,):
        idx = out[r].long()
        if (idx.numel() != b["K"] or int(idx.min()) < 0
                or int(idx.max()) >= b["N"]
                or torch.unique(idx).numel() != b["K"]):
            return False
        if not torch.equal(lg[idx].sort().values, ref):
            return False
    return True


def run_batch_cells(model, isl, L):
    tag = f"{model}_{isl}_L{L}"
    out_path = HERE / "results" / f"bs_{tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add((json.loads(line)["op"], json.loads(line)["BS"]))
            except Exception:
                pass
    mod = load_compb()
    b = real_bundle(model, isl, L)
    K, N, cr = b["K"], b["N"], b["cr"]
    hit = b.get("hit_rate")
    f = open(out_path, "a")
    prof.start()
    for BS in BS_GRID:
        for op in ("gvr_pr", "kf_compB"):
            if (op, BS) in done:
                continue
            base = f"{op}|{model}|{isl}|L{L}|{BS}"
            rec = {"model": model, "isl": isl, "L": L, "N": N, "K": K,
                   "cr": cr, "BS": BS, "op": op,
                   "hit": round(float(hit), 4) if hit is not None else None,
                   "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                   "reps_cold": REPS_COLD, "reps_warm": REPS_WARM}
            try:
                if op == "gvr_pr":
                    call, keep, extra, out = gvr_call(b, BS)
                else:
                    call, keep, extra, out = compb_call(b, BS, mod)
                rec.update(extra)
                rec["exact"] = exact_rows(out, b, BS)
                measure_cell(call, base, REPS_COLD, REPS_WARM)
                del call, keep, out
            except Exception as e:  # record, never fake
                rec["error"] = f"{type(e).__name__}: {e}"
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{tag}] {op} BS={BS} "
                  f"{'ERR ' + rec['error'] if 'error' in rec else 'exact=' + str(rec['exact'])}",
                  flush=True)
    prof.stop()
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    batches = [f"{m} {isl} {L}" for m in ("flash", "pro", "v32")
               for isl in ISLS[m] for L in LAYERS[m]]
    if args.list:
        print("\n".join(batches))
        return
    model, isl, L = args.batch.split()
    run_batch_cells(model, isl, int(L))


if __name__ == "__main__":
    main()
