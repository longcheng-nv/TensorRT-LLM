#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""iter10 bounded probe: pin the K2048 16-bit BS1 tail (nsys holes
0.88-0.96 both dtypes at 131K/262K, C8 dispatch) — full/noP4/noWG split at
HEAD defaults (p3_push=True, p4 rank-scatter+fast paths ON, native ladder
ON), bf16/fp16, vs a green bf16 K1024 reference cell. Event cold-L2 paired
same-process medians (throttle-immune; screening only — no dispatch or
kernel changes ride on this)."""
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
_DTS = {"bf16": torch.bfloat16, "fp16": torch.float16}


class NoWalkNoGather(M.GvrMsClusterKernel):
    # walk no-op MUST publish deterministic counts (iter7 lesson): the
    # kernel body packs s_iscalars[0] into s_cluster[M]; garbage p_cnt
    # would distort peers' timing. With push ON this also removes the
    # remote-store writes, so downstream P4 sees band=0 (near-free).
    @cute.jit
    def phase3_from_slots_mc(self, smem_slotk, smem_slotv, smem_keys,
                             smem_vals, smem_ptcnt, smem_ptcnt_up,
                             smem_ptcnt_multi, smem_wcnt, s_thr, s_swf,
                             s_iscalars, output_indices_row, d_off, b_off,
                             rank, tidx, warp_id, lane):
        if tidx == 0:
            s_iscalars[4] = cutlass.Int32(0)
            s_iscalars[0] = cutlass.Int32(0)
        cute.arch.barrier()

    @cute.jit
    def _p3_leader_band_gather(self, rank, smem_keys, smem_vals, s_cluster,
                               tidx):
        pass


class NoP4(M.GvrMsClusterKernel):
    @cute.jit
    def phase4_band_rank_scatter(self, smem_keys, smem_vals, smem_hist,
                                 smem_wcnt, s_thr, s_swf, s_iscalars,
                                 output_values_row, output_indices_row, band,
                                 k_rem, m0, tidx, warp_id, lane):
        pass


def compile_variant(cls, dtype, n, K, cr_val, C, threads=1024):
    use256 = (n >= 16384)
    kobj = cls(dtype=M._DT[dtype], top_k=K, next_n=1, num_threads=threads,
               compress_ratio=cr_val, use_256bit_load=use256,
               enable_unroll_4=True, enable_phase3_unroll=True,
               min_blocks_per_mp=1, return_output_values=False,
               M_thr=4, R_rounds=1, band_accept=64, place_mode=5,
               fuse_collect=True, C_cta=C)  # HEAD defaults: push+p4+native ON
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


VARIANTS = (("full", M.GvrMsClusterKernel), ("noP4", NoP4),
            ("noWG", NoWalkNoGather))
CR = {512: 4, 1024: 4, 2048: 1}
CELLS = (  # (K, N, C, dtype) — all BS1; C8 per the 16-bit dispatch rule
    (2048, 131072, 8, "bf16"),
    (2048, 262144, 8, "bf16"),
    (2048, 262144, 8, "fp16"),
    (1024, 262144, 8, "bf16"),  # green reference (nsys 1.035)
)
print(f"{'K':>5} {'N':>7} {'C':>2} {'dt':>4} | " +
      " ".join(f"{n:>7}" for n, _ in VARIANTS) +
      f" | {'P4_us':>6} {'walk_us':>7} {'floor':>6}")
for K, N, C, dtn in CELLS:
    crv = CR[K]
    dt = _DTS[dtn]
    b = synth_data.get_bundle(K, dt, N)
    lg = b["logits"][:1].contiguous().cuda()
    pre = b["preIdx"][:1].contiguous().cuda()
    sl = torch.full((1,), b["Npad"] * crv, dtype=torch.int32, device=DEV)
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)
    res = {}
    for name, cls in VARIANTS:
        comp = compile_variant(cls, dt, N, K, crv, C)
        call = lambda: comp(lg, pre, sl, None, out)
        call(); torch.cuda.synchronize()
        res[name] = cold_us(call)
    print(f"{K:>5} {N:>7} {C:>2} {dtn:>4} | " +
          " ".join(f"{res[n]:7.2f}" for n, _ in VARIANTS) +
          f" | {res['full']-res['noP4']:6.2f} "
          f"{res['noP4']-res['noWG']:7.2f} {res['noWG']:6.2f}")
