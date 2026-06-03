#!/usr/bin/env python3
r"""
Drive the GVR (Heuristic) Top-K kernel end-to-end on a pre-synthesized
(logits, preIdx, seq_lens) bundle. Pure performance driver for downstream
hardware cycle-level **perfsim**.

Kernel backend selection (`--backend`):
  * `local` (default) — JIT-build the self-contained `gvr_kernel/`
                        torch cpp_extension shipped in this directory.
                        No `tensorrt_llm` install required.
  * `trtllm`          — call `torch.ops.trtllm.indexer_topk_decode`
                        (V4 post-PR-#14297 unified op). Falls back to
                        direct `libth_common.so` load if `import
                        tensorrt_llm` fails.

Both backends drive *the same* GVR Heuristic kernel
(`heuristicTopKMultiRowKernel(Dtype)<K>`); they differ only in how the
launcher is reached.

Target hardware: NVIDIA **Blackwell** (B200 sm_100 / B300 sm_103) with
DSv4 Pro **indexer top-K = 1024** (Blackwell-tuned in `heuristic_topk.cuh`).

Environment knobs read by the kernel (auto-`setdefault`'d):
  TRTLLM_HEURISTIC_NMIN  default 1024 (allow GVR at N≥1024)  [trtllm backend]
  TRTLLM_HEURISTIC_BSMAX default 2048 (allow GVR up to BS=1024)  [trtllm backend]
  TRTLLM_ENABLE_PDL      default 1    (CUDA programmatic stream serialization)
  LIBTH_COMMON           libth_common.so path (auto-detected)  [trtllm backend]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


DEFAULT_LIBTH_CANDIDATES = [
    "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/tensorrt_llm/libs/libth_common.so",
    "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/cpp/build/tensorrt_llm/thop/libth_common.so",
]

RADIX_AUX_BLOCKS_MAX = 32


# ----- Backend: standalone gvr_kernel cpp_extension (no tensorrt_llm dep) -----

def make_local_backend():
    """Import the local `gvr_kernel/` package and return a callable
    `(logits, preIdx, seq_lens, K, compress_ratio) -> indices` that
    matches the trtllm-backend signature used elsewhere in this script.

    Uses `gvr_topk_decode_into` (caller-provided output tensors) so the
    launch contract is identical to TRT-LLM's `indexer_topk_decode`
    (no per-call alloc, no extra copy_()) — gives apples-to-apples
    perfsim-grade timing between the two backends.
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import gvr_kernel  # noqa: E402
    print(f"[info] backend=local  (gvr_kernel JIT, from {here / 'gvr_kernel'})",
          file=sys.stderr)

    def _call(logits, seq_lens, indices_out, next_n, K,
              pre_idx=None, heuristic_scratch=None,
              compress_ratio=4,
              radix_aux_indices=None, radix_aux_logits=None):
        # heuristic_scratch (BS*K, dtype) doubles as the [BS, K] values
        # output tensor — caller already allocated it for the trtllm op
        # contract, so reuse it instead of asking for a separate buffer.
        BS, K_ = indices_out.shape
        values_out = heuristic_scratch.view(BS, K_)
        gvr_kernel.gvr_topk_decode_into(
            logits, pre_idx, seq_lens,
            indices_out, values_out,
            K=int(K), compress_ratio=int(compress_ratio), next_n=int(next_n))
        return indices_out
    return _call


# ----- Backend: tensorrt_llm unified op -----

def make_trtllm_backend():
    """Resolve `torch.ops.trtllm.indexer_topk_decode`. Prefer `import
    tensorrt_llm`; fall back to direct `libth_common.so` load."""
    try:
        import tensorrt_llm  # noqa: F401
        print("[info] backend=trtllm  (via `import tensorrt_llm`)",
              file=sys.stderr)
        op = torch.ops.trtllm.indexer_topk_decode
    except Exception as e:
        print(f"[warn] import tensorrt_llm failed ({type(e).__name__}: {e}); "
              f"falling back to direct libth_common.so load.", file=sys.stderr)
        so = os.environ.get("LIBTH_COMMON", "")
        if not so or not Path(so).exists():
            for cand in DEFAULT_LIBTH_CANDIDATES:
                if Path(cand).exists():
                    so = cand
                    break
        if not so or not Path(so).exists():
            raise FileNotFoundError(
                "libth_common.so not found. Set env LIBTH_COMMON=<path> or "
                "build TRT-LLM first.")
        print(f"[info] backend=trtllm  (libth_common.so {so})", file=sys.stderr)
        torch.ops.load_library(so)
        op = torch.ops.trtllm.indexer_topk_decode

    def _call(logits, seq_lens, indices_out, next_n, K,
              pre_idx=None, heuristic_scratch=None,
              compress_ratio=4,
              radix_aux_indices=None, radix_aux_logits=None):
        return op(logits, seq_lens, indices_out, next_n, K,
                  pre_idx=pre_idx, heuristic_scratch=heuristic_scratch,
                  compress_ratio=compress_ratio,
                  radix_aux_indices=radix_aux_indices,
                  radix_aux_logits=radix_aux_logits)
    return _call


def load_backend(name: str):
    if name == "local":
        return make_local_backend()
    if name == "trtllm":
        return make_trtllm_backend()
    raise ValueError(f"unknown --backend {name}; choose local|trtllm")


def load_case(case_dir: Path, bs_override: int | None):
    meta = json.loads((case_dir / "meta.json").read_text())
    logits = torch.load(case_dir / "logits.pt", map_location="cuda")
    preIdx = torch.load(case_dir / "preIdx.pt", map_location="cuda")
    seq_lens = torch.load(case_dir / "seq_lens.pt", map_location="cuda")

    if bs_override is not None and bs_override != logits.shape[0]:
        if logits.shape[0] != 1:
            raise ValueError(
                f"--bs override requires stored BS=1, got BS={logits.shape[0]}")
        BS = bs_override
        logits = logits.expand(BS, -1).contiguous()
        preIdx = preIdx.expand(BS, -1).contiguous()
        seq_lens_val = int(seq_lens[0].item())
        seq_lens = torch.full((BS,), seq_lens_val, dtype=torch.int32, device="cuda")
        meta["BS"] = BS
        meta["bs_replicated_from"] = 1
    return meta, logits, preIdx, seq_lens


def make_buffers(BS: int, K: int, dtype: torch.dtype, backend_name: str):
    indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
    scratch = torch.empty((BS * K,), dtype=dtype, device="cuda")
    # `radix_aux_*` are required by the trtllm unified dispatcher even for
    # the GVR path. The local backend does not need them.
    if backend_name == "trtllm":
        aux_idx = torch.empty(
            (BS * RADIX_AUX_BLOCKS_MAX * K,), dtype=torch.int32, device="cuda")
        aux_lo = torch.empty(
            (BS * RADIX_AUX_BLOCKS_MAX * K,), dtype=torch.float32, device="cuda")
    else:
        aux_idx = aux_lo = None
    return indices, scratch, aux_idx, aux_lo


def time_one(fn: Callable, warmup: int, reps: int, flush_buf: torch.Tensor):
    for _ in range(warmup):
        flush_buf.zero_(); torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
    walls_us: list[float] = []
    for _ in range(reps):
        flush_buf.zero_(); torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        walls_us.append(start.elapsed_time(end) * 1000.0)
    arr = np.array(walls_us, dtype=np.float64)
    return {
        "median_us": float(np.median(arr)),
        "min_us": float(np.min(arr)),
        "p10_us": float(np.percentile(arr, 10)),
        "p50_us": float(np.percentile(arr, 50)),
        "p90_us": float(np.percentile(arr, 90)),
        "p99_us": float(np.percentile(arr, 99)),
        "n": len(arr),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-dir", required=True, type=Path,
                   help="directory holding logits.pt / preIdx.pt / seq_lens.pt / meta.json")
    p.add_argument("--bs", type=int, default=None,
                   help="override BS via broadcast-replicate of the stored BS=1 row")
    p.add_argument("--backend", choices=["local", "trtllm"], default="local",
                   help="kernel backend: local (cpp_extension, no TRT-LLM) or trtllm")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--l2-flush-mib", type=int, default=128,
                   help="L2 flush buffer size in MiB before every launch (default 128)")
    p.add_argument("--out", type=Path, default=None,
                   help="write JSON summary here (optional)")
    p.add_argument("--label", default=None,
                   help="optional label printed in the summary (e.g. 'case-1a')")
    args = p.parse_args()

    os.environ.setdefault("TRTLLM_HEURISTIC_NMIN", "1024")
    os.environ.setdefault("TRTLLM_HEURISTIC_BSMAX", "2048")

    backend = load_backend(args.backend)

    meta, logits, preIdx, seq_lens = load_case(args.case_dir, args.bs)
    BS = logits.shape[0]
    K = preIdx.shape[1]
    Npad = logits.shape[1]
    N_valid = meta["N"]
    compress_ratio = meta.get("compress_ratio", 4)
    dtype = logits.dtype

    label = args.label or args.case_dir.name
    print(f"\n=== GVR perfsim driver [{label}]  backend={args.backend}  BS={BS}  "
          f"N={N_valid} (Npad={Npad})  K={K}  dtype={dtype}  compress_ratio={compress_ratio} ===")
    print(f"  cfg: {meta.get('cfg_name', '?')}  "
          f"target_hr={meta.get('target_hr', '?')}  "
          f"realised_hr={meta.get('calibration_realised_hr', meta.get('kernel_side_hit_rate', '?')):.4f}  "
          f"noise_c={meta.get('calibrated_noise_c', '?'):.4f}")

    indices, scratch, aux_idx, aux_lo = make_buffers(BS, K, dtype, args.backend)
    flush_bytes = args.l2_flush_mib * 1024 * 1024
    flush_buf = torch.empty(flush_bytes // 4, dtype=torch.float32, device="cuda")

    def gvr_fn():
        backend(logits, seq_lens, indices, 1, K,
                pre_idx=preIdx, heuristic_scratch=scratch,
                compress_ratio=compress_ratio,
                radix_aux_indices=aux_idx, radix_aux_logits=aux_lo)

    t0 = time.time()
    gvr = time_one(gvr_fn, args.warmup, args.reps, flush_buf)
    elapsed = time.time() - t0

    print(f"\n  GVR (Heuristic)   median={gvr['median_us']:8.2f} µs  "
          f"(p10={gvr['p10_us']:7.2f}, p50={gvr['p50_us']:7.2f}, "
          f"p90={gvr['p90_us']:7.2f}, p99={gvr['p99_us']:7.2f}, "
          f"min={gvr['min_us']:7.2f}, n={gvr['n']})")
    print(f"  Bench wall: {elapsed:.1f}s   warmup={args.warmup}  reps={args.reps}  "
          f"l2_flush={args.l2_flush_mib} MiB  backend={args.backend}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "label": label,
            "backend": args.backend,
            "case_dir": str(args.case_dir),
            "meta": meta,
            "BS_used": BS,
            "K": K,
            "N_valid": N_valid,
            "Npad": Npad,
            "compress_ratio": compress_ratio,
            "dtype": str(dtype),
            "warmup": args.warmup,
            "reps": args.reps,
            "l2_flush_mib": args.l2_flush_mib,
            "gvr": gvr,
            "bench_wall_s": elapsed,
        }, indent=2))
        print(f"  → JSON summary: {args.out}")


if __name__ == "__main__":
    main()
