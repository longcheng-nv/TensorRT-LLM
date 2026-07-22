# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""compB BS>1 extension — minimal validation experiment (A + single-wave B).

Per R3_LEDGER "BS>1 extension design analysis" validation order:
  A cells (small/mid-n grid.y batching):
    flash_16k_L22 (N=4096, small<6>), pro_16k_L30 (N=4096, small<6>),
    v32_8k_L34 (N=8192, K=2048, mid<2>), flash_64k_L22 (N=16384, mid<4>)
  B cells (large-n single-wave row teams, N=131072 -> team=64):
    flash_512k_L22 (K=512), pro_512k_L30 (K=1024)

Arms, paired back-to-back on ONE GPU per cell (all local, this node):
  gvr_pr      : REPORT-verbatim PR#16457 arm (local anchor)
  kf_compB    : shipped compB, BS sequential same-stream launches (baseline)
  kf_compB_ext: extension, one run_batch_ext call per batch (candidate)

Protocol == bs_kf.py: NVTX c|/w| ranges, 512MB evict outside range,
20 cold + 50 warm, same real row replicated to BS. Exactness: tie-robust
value-set check on rows {0, BS//2, BS-1}.
  python3 bs_ext.py --batch "flash 512k 22"   |   --list
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BSDIR = HERE.parent                      # kf_bs_scaling/
REPORT = BSDIR.parent                    # op26_r0_upstream_port_report/
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
# (model, isl, L, kind, BS grid)
CELLS = [
    ("flash", "16k", 22, "A", [1, 2, 4, 8, 16, 64, 256, 1024]),
    ("pro", "16k", 30, "A", [1, 2, 4, 8, 16, 64, 256, 1024]),
    ("v32", "8k", 34, "A", [1, 2, 4, 8, 16, 64, 256, 1024]),
    ("flash", "64k", 22, "A", [1, 2, 4, 8, 16, 64, 256, 1024]),
    ("flash", "512k", 22, "B", [1, 2, 4, 8, 16]),
    ("pro", "512k", 30, "B", [1, 2, 4, 8, 16]),
]

_GVR_CACHE = {}


def load_mod(name, sources, build_dir):
    from torch.utils.cpp_extension import load
    Path(build_dir).mkdir(exist_ok=True)
    return load(name=name, sources=[str(s) for s in sources],
                build_directory=str(build_dir),
                extra_cuda_cflags=["-O3"], verbose=False)


def load_compb_seq():
    src = BSDIR / "compB_src"
    return load_mod("kf_compb_bs_039", [src / "kernel.cu", src / "main_bs.cpp"],
                    HERE / "build_pt_seq")


def load_compb_ext():
    return load_mod("kf_compb_ext",
                    [HERE / "kernel_ext.cu", HERE / "main_ext.cpp"],
                    HERE / "build_pt")


def real_bundle(model, isl, L):
    RD = RV32 if model == "v32" else RV4
    return RD.get_bundle(model, isl, L, "fp32")


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


def padded_batch(b, BS):
    N = b["N"]
    W = (N + 63) // 64 * 64
    row = torch.full((1, W), torch.finfo(torch.float32).min,
                     dtype=torch.float32, device=DEV)
    row[0, :N] = b["logits"][0, :N].float()
    return row.expand(BS, -1).contiguous()


def compb_call(b, BS, mod):
    K, N = b["K"], b["N"]
    lg = padded_batch(b, BS)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    call = lambda: mod.run_batch(lg, N, out)         # noqa: E731
    call()
    torch.cuda.synchronize()
    return call, [lg, out], {}, out


def compb_ext_call(b, BS, mod):
    K, N = b["K"], b["N"]
    lg = padded_batch(b, BS)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    path, team, cap, rpw, waves = (int(x) for x in mod.ext_info(N, K, BS))
    call = lambda: mod.run_batch_ext(lg, N, out)     # noqa: E731
    call()
    torch.cuda.synchronize()
    return call, [lg, out], {"path": path, "team": team, "cap": cap,
                             "rpw": rpw, "waves": waves}, out


def exact_rows(out, b, BS):
    lg = b["logits"][0, :b["N"]].float()
    ref = torch.topk(lg, b["K"]).values.sort().values
    rows = sorted({0, BS // 2, BS - 1})
    for r in rows:
        idx = out[r].long()
        if (idx.numel() != b["K"] or int(idx.min()) < 0
                or int(idx.max()) >= b["N"]
                or torch.unique(idx).numel() != b["K"]):
            return False
        if not torch.equal(lg[idx].sort().values, ref):
            return False
    return True


def run_batch_cells(model, isl, L):
    cell = next(c for c in CELLS if c[:3] == (model, isl, L))
    kind, bs_grid = cell[3], cell[4]
    tag = f"{model}_{isl}_L{L}"
    out_path = HERE / "results" / f"ext_{tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add((rec["op"], rec["BS"]))
            except Exception:
                pass
    seq = load_compb_seq()
    ext = load_compb_ext()
    b = real_bundle(model, isl, L)
    K, N, cr = b["K"], b["N"], b["cr"]
    hit = b.get("hit_rate")
    f = open(out_path, "a")
    prof.start()
    for BS in bs_grid:
        for op in ("gvr_pr", "kf_compB", "kf_compB_ext"):
            if (op, BS) in done:
                continue
            base = f"{op}|{model}|{isl}|L{L}|{BS}"
            rec = {"model": model, "isl": isl, "L": L, "N": N, "K": K,
                   "cr": cr, "BS": BS, "op": op, "kind": kind,
                   "hit": round(float(hit), 4) if hit is not None else None,
                   "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                   "reps_cold": REPS_COLD, "reps_warm": REPS_WARM}
            try:
                if op == "gvr_pr":
                    call, keep, extra, out = gvr_call(b, BS)
                elif op == "kf_compB":
                    call, keep, extra, out = compb_call(b, BS, seq)
                else:
                    call, keep, extra, out = compb_ext_call(b, BS, ext)
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
    if args.list:
        print("\n".join(f"{m} {isl} {L}" for m, isl, L, _, _ in CELLS))
        return
    model, isl, L = args.batch.split()
    run_batch_cells(model, isl, int(L))


if __name__ == "__main__":
    main()
