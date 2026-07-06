#!/usr/bin/env python3
"""iter6 opening ablation: phase cost of the NEW rank-scatter P4 + P3 band
machinery at the remaining hole cells (iter4 no-op subclass pattern)."""
import sys
from pathlib import Path
import torch

_B = Path("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench")
sys.path.insert(0, str(_B / "harness"))
sys.path.insert(0, str(_B / "ops"))
sys.path.insert(0, str(_B / "op18_gvr_1cta_multithresh" / "src"))
sys.path.insert(0, str(_B / "op21_gvr_prod" / "src"))
import synth_data  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as cr  # noqa: E402
import gvr_msc_op as M  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


class NoP4(M.GvrMsClusterKernel):
    @cute.jit
    def phase4_band_rank_scatter(self, smem_keys, smem_vals, smem_hist,
                                 smem_wcnt, s_thr, s_swf, s_iscalars,
                                 output_values_row, output_indices_row, band,
                                 k_rem, m0, tidx, warp_id, lane):
        pass


class NoP34(NoP4):
    # additionally skip the P3 slot walk + emit (from_slots_mc)
    @cute.jit
    def phase3_from_slots_mc(self, smem_slotk, smem_slotv, smem_keys,
                             smem_vals, smem_ptcnt, smem_ptcnt_up,
                             smem_ptcnt_multi, smem_wcnt, s_thr, s_swf,
                             s_iscalars, output_indices_row, d_off, tidx,
                             warp_id, lane):
        pass


def compile_variant(cls, dtype, n, K, cr_val, C, threads=1024):
    use256 = (n >= 16384)
    kobj = cls(dtype=M._DT[dtype], top_k=K, next_n=1, num_threads=threads,
               compress_ratio=cr_val, use_256bit_load=use256,
               enable_unroll_4=True, enable_phase3_unroll=True,
               min_blocks_per_mp=1, return_output_values=False,
               M_thr=4, R_rounds=1, band_accept=64, place_mode=5,
               fuse_collect=True, C_cta=C)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if use256 else 16
    in_f = cr.make_fake_compact_tensor(M._DT[dtype], (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, stream=fs,
                        options="--enable-tvm-ffi")


def cold_us(call, reps=40, warmup=5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    cold = []
    for _ in range(reps):
        _EVICT.uniform_(0, 1); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    cold.sort(); del g
    return cold[len(cold) // 2]


CR = {512: 4, 1024: 4, 2048: 1}
print(f"{'K':>5} {'N':>7} {'C':>2} | {'full':>7} {'noP4':>7} {'noP34':>7} "
      f"| {'P4_us':>6} {'P3_us':>6}")
for K, N, C in ((1024, 262144, 4), (512, 262144, 4), (2048, 262144, 8)):
    crv = CR[K]
    b = synth_data.get_bundle(K, torch.float32, N)
    lg = b["logits"][:1].contiguous()
    pre = b["preIdx"][:1].contiguous()
    sl = torch.full((1,), b["Npad"] * crv, dtype=torch.int32, device=DEV)
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)
    res = {}
    for name, cls in (("full", M.GvrMsClusterKernel), ("noP4", NoP4),
                      ("noP34", NoP34)):
        comp = compile_variant(cls, torch.float32, N, K, crv, C)
        call = lambda: comp(lg, pre, sl, None, out)
        call(); torch.cuda.synchronize()
        res[name] = cold_us(call)
    print(f"{K:>5} {N:>7} {C:>2} | {res['full']:7.2f} {res['noP4']:7.2f} "
          f"{res['noP34']:7.2f} | {res['full']-res['noP4']:6.2f} "
          f"{res['noP4']-res['noP34']:6.2f}")
