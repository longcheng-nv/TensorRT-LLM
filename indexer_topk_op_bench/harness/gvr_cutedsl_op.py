"""Standalone GVR cuteDSL op driver.

Drives the vendored GvrTopKKernel (cute_vendored/blackwell/top_k/gvr_topk_decode.py)
via cute.compile + TVM-FFI. Imports ONLY cutlass (+ the vendored kernel sources),
no tensorrt_llm runtime. Compile/launch pattern adapted from
dsv4_fused_indexer_gvr_topk/benchmark.py.
"""
import os
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "ops"))  # make cute_vendored importable
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_compiled = {}


def _config(bs, n):
    """Replicate benchmark.py launch-config heuristic for GvrTopKKernel."""
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def compile_gvr_cute(dtype, bs, n, K, cr_val):
    key = (dtype, bs, n, K, cr_val)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrTopKKernel(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
    )
    n_rows, n_cols, n_batch = cute.sym_int(), cute.sym_int(), cute.sym_int()
    in_align = 32 if use256 else 16
    input_fake = cr.make_fake_compact_tensor(_DT[dtype], (n_rows, n_cols), stride_order=(1, 0), assumed_align=in_align)
    pre_idx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_batch, K), stride_order=(1, 0), assumed_align=16)
    seq_lens_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_batch,), stride_order=(0,))
    out_idx_fake = cr.make_fake_compact_tensor(cutlass.Int32, (n_rows, K), stride_order=(1, 0), assumed_align=16)
    fake_stream = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kobj, input_fake, pre_idx_fake, seq_lens_fake, None, out_idx_fake,
                            stream=fake_stream, options="--enable-tvm-ffi")
    _compiled[key] = compiled
    return compiled


def gvr_cutedsl(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None):
    bs, n = logits.shape
    compiled = compile_gvr_cute(logits.dtype, bs, n, index_topk, compress_ratio)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for K, crv, N in ((2048, 1, 32768), (512, 4, 65536), (1024, 4, 32768)):
            logits = torch.randn(1, N, dtype=dt, device="cuda")
            pre_idx = torch.topk(logits[0].float(), K).indices.int().view(1, K).contiguous()
            seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
            out = gvr_cutedsl(logits, pre_idx, seq_lens, K, crv)
            torch.cuda.synchronize()
            # validity + value-equivalence vs torch.topk
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[0].tolist()))
            print(f"  {str(dt):14s} K={K:4d} cr={crv} N={N:6d}: uniq={nuniq}/{K} valdiff_vs_topk={d:.2e}")
    print("GVR cuteDSL smoke OK")
