# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""op13 iter8: GVR cuteDSL op with LOG-COUNT P2 secant interpolation.

Motivation (iter8a host replay, results/rootfinder_sweep_fp32.{log,json}):
the P2 count-vs-threshold curve (CCDF tail) is ~exponential, so the baseline
LINEAR interpolation  f=(clo-kFT)/(clo-chi)  systematically under-steps and
inflates the eval count at large N / narrow windows. Interpolating in
log-count space  f=log2(clo/kFT)/log2(clo/chi)  is near-exact for an
exponential tail:

  * K512  narrow kc2x(kCC=1024,kFT=614):  evals 3.00 FLAT all N (lin: 3.0->3.75
    at 262K), cand 1.30-1.52xK.
  * K1024 narrow kc2x(kCC=2048,kFT=1024): evals 3.00 flat, cand 1.04-1.11xK
    (lin: 3.92@4K, 3.42/3.83 large-N).
  * K1024 base  (kCC=5120,kFT=1024): large-N free win — evals 3.50->2.75@131K,
    cand 3.49->1.59xK@262K.
  * K2048 base  (kCC=6144,kFT=2048): evals 2.00@8K, 3.00 flat large-N
    (lin: 3.58/3.75), cand 1.13->1.07xK.
  * Illinois == linear everywhere; logillinois == logcount => stateless
    log formula captures ALL the win.

The log interp lives in a GvrTopKKernel subclass that overrides
``phase2_secant_search`` (resolved via MRO at cute.compile trace time) — the
vendored kernel file is NOT edited, mirroring the iter7 GvrP2C ship pattern.
Everything else (compile flags, fake tensors, launch) mirrors
harness/gvr_cutedsl_op.py so local single-op perf == integration perf.

Variant selection is caller-supplied (kCC/kFT/log per call) so the nsys A/B
can probe a portfolio; the ship dispatch table is baked only after iter8c.
"""
import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass.cute import runtime as cr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "ops"))  # make cute_vendored importable
from cute_vendored.blackwell.top_k.gvr_topk_decode import (  # noqa: E402
    GvrTopKKernel,
    _fmin_f32_inline,
)

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

_DT = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
_compiled = {}


class GvrP2CLog(GvrTopKKernel):
    """GvrTopKKernel with log-count-space P2 interpolation + kC/kFT override."""

    def __init__(self, *a, kC_override=None, kFTarget_override=None, **kw):
        super().__init__(*a, **kw)
        if kC_override is not None:
            self.kC = int(kC_override)
        if kFTarget_override is not None:
            self.kFTarget = int(kFTarget_override)

    # Identical to the vendored phase2_secant_search EXCEPT the interpolation
    # block computes f in log-count space (falls back to the linear formula
    # when the log denominator degenerates). Bracket/window/fallback logic,
    # clamps (f in [0.05,0.95], iter0 cap 0.5) and barriers are unchanged, so
    # the exactness guard (done==1 <=> count in [kK,kCC]) is untouched.
    @cute.jit
    def phase2_secant_search(
        self,
        input_row,
        N,
        smem_ptcnt,
        smem_wcnt,
        s_thr,  # [threshold, val_lo, val_hi]
        s_iscalars,  # [cand_count, done, cnt_lo, cnt_hi, out_count]
        tidx,
        warp_id,
        lane,
    ):
        kK = cutlass.const_expr(self.top_k)
        kCC = cutlass.const_expr(self.kC)
        kFTarget = cutlass.const_expr(self.kFTarget)

        # ---- Initial count with the Phase-1 mean as threshold ----
        thr_init = s_thr[0]
        self.block_count_ge(
            input_row,
            N,
            thr_init,
            smem_ptcnt,
            smem_wcnt,
            s_iscalars,
            tidx,
            warp_id,
            lane,
        )

        # tid==0 classifies the initial count.
        if tidx == 0:
            c0 = s_iscalars[0]
            t0 = s_thr[0]
            if c0 >= cutlass.Int32(kK) and c0 <= cutlass.Int32(kCC):
                s_iscalars[1] = cutlass.Int32(1)  # done = 1 (converged)
            elif c0 > cutlass.Int32(kCC):
                s_thr[1] = t0
                s_iscalars[2] = c0
            else:
                s_thr[2] = t0
                s_iscalars[3] = c0
        cute.arch.barrier()

        # ---- Secant refinement loop (LOG-COUNT interpolation) ----
        it = cutlass.Int32(0)
        while it < cutlass.Int32(self.MAX_REFINE_ITERS) and s_iscalars[1] == cutlass.Int32(0):
            if tidx == 0:
                vlo = s_thr[1]
                vhi = s_thr[2]
                clo = s_iscalars[2]
                chi = s_iscalars[3]
                rng = vhi - vlo
                nv = cutlass.Float32(0.0)
                if clo > chi and rng > cutlass.Float32(1e-10):
                    # log-count interpolation: model count(v) ~ exp(a - b*v)
                    # between the brackets; solve count(nv) = kFTarget.
                    clo_f = cutlass.Float32(clo)
                    chi_f = cute.arch.fmax(cutlass.Float32(chi), cutlass.Float32(1.0))
                    den = cmath.log2(clo_f / chi_f, fastmath=True)
                    f = cutlass.Float32(0.0)
                    if den > cutlass.Float32(0.0):
                        f = cmath.log2(clo_f / cutlass.Float32(kFTarget), fastmath=True) / den
                    else:
                        f = cutlass.Float32(clo - cutlass.Int32(kFTarget)) / cutlass.Float32(clo - chi)
                    # clamp f to [0.05, 0.95]
                    f = cute.arch.fmax(cutlass.Float32(0.05), f)
                    f = _fmin_f32_inline(f, cutlass.Float32(0.95))
                    if it == cutlass.Int32(0):
                        f = _fmin_f32_inline(f, cutlass.Float32(0.5))
                    nv = vlo + rng * f
                else:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)

                # clamp nv into (vlo, vhi) range
                if nv <= vlo:
                    nv = vlo + rng * cutlass.Float32(0.05)
                if nv >= vhi:
                    nv = vhi - rng * cutlass.Float32(0.05)

                if nv == vlo or nv == vhi:
                    nv = (vlo + vhi) * cutlass.Float32(0.5)
                    if nv == vlo or nv == vhi:
                        s_thr[0] = vlo
                        s_iscalars[1] = cutlass.Int32(2)  # done = 2 (give up)
                    else:
                        s_thr[0] = nv
                else:
                    s_thr[0] = nv
            cute.arch.barrier()

            if s_iscalars[1] == cutlass.Int32(0):
                new_thr = s_thr[0]
                self.block_count_ge(
                    input_row,
                    N,
                    new_thr,
                    smem_ptcnt,
                    smem_wcnt,
                    s_iscalars,
                    tidx,
                    warp_id,
                    lane,
                )
                if tidx == 0:
                    c_new = s_iscalars[0]
                    t_new = s_thr[0]
                    if c_new >= cutlass.Int32(kK) and c_new <= cutlass.Int32(kCC):
                        s_iscalars[1] = cutlass.Int32(1)
                    elif c_new > cutlass.Int32(kCC):
                        s_thr[1] = t_new
                        s_iscalars[2] = c_new
                    else:
                        s_thr[2] = t_new
                        s_iscalars[3] = c_new
                cute.arch.barrier()
            it = it + cutlass.Int32(1)

        # ---- Post-loop fallback: if still not done, force threshold ----
        if tidx == 0:
            if s_iscalars[1] == cutlass.Int32(0):
                if s_iscalars[2] <= cutlass.Int32(kCC * 2):
                    s_thr[0] = s_thr[1]
                else:
                    s_thr[0] = s_thr[2]
                s_iscalars[1] = cutlass.Int32(2)
        cute.arch.barrier()


def _config(bs, n):
    """Replicate gvr_cutedsl_op.py launch-config heuristic."""
    t = 1024 if (bs <= NUM_SMS and n >= 65536) else 512
    use256 = (n >= 16384)
    min_bpm = 1 if bs <= NUM_SMS else 3
    return t, use256, min_bpm


def compile_gvr_p2clog(dtype, bs, n, K, cr_val, kcc=None, kft=None):
    """Compile the log-interp kernel with optional kCC/kFT override."""
    key = (dtype, bs, n, K, cr_val, kcc, kft)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrP2CLog(
        dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
        use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
        min_blocks_per_mp=min_bpm, return_output_values=False,
        kC_override=kcc, kFTarget_override=kft,
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


def gvr_cutedsl_p2clog(logits, pre_idx, seq_lens, index_topk, compress_ratio=1,
                       kcc=None, kft=None, out=None):
    bs, n = logits.shape
    compiled = compile_gvr_p2clog(logits.dtype, bs, n, index_topk, compress_ratio, kcc, kft)
    if out is None:
        out = torch.empty(bs, index_topk, dtype=torch.int32, device="cuda")
    compiled(logits, pre_idx, seq_lens, None, out)
    return out


# ---------------------------------------------------------------------------
# SHIP dispatch (iter8c, nsys ×3-median 2026-07-04, results/nsys_p2clog_ab_medians.txt)
#
#   K512  fp32: log variants LOSE to the iter7 shipped p2c everywhere (logn
#               4K −8.7% vs p2c −16.6%; large N +7.4/+13.4%) — keep iter7
#               lin-narrow (kCC=1536,kFT=1280) at N<=65536, baseline above.
#   K1024 fp32: logn(2048,1024) ships N-DISPATCHED — wins 8K −32.1%,
#               32K −8.8%, 131K −22.0% (base has reproducible slow spots at
#               8K/131K = the host-replay eval spikes); REAL regressions at
#               65K +13.4% / 262K +9.8% => those N stay baseline. First time
#               K1024 ships (iter7 blanket-narrow was rejected on regressions).
#   K2048 fp32: logn(4096,2048) ships at ALL N>=8192 — worst cell +0.6%
#               (tie band), wins to −12.2% @262K (the −0.75-eval free win).
#   bf16/fp16: host params identical to fp32 but no nsys evidence yet ->
#               baseline (production indexer logits are fp32 anyway).
# ---------------------------------------------------------------------------
def dispatch_p2c_v2(dtype, K, n):
    """-> (use_log, kcc, kft); (False, None, None) = plain baseline."""
    if dtype != torch.float32:
        return False, None, None
    if K == 512:
        if n <= 65536:
            return False, 1536, 1280          # iter7 lin-narrow (GvrP2C)
        return False, None, None
    if K == 1024:
        if n <= 32768 or n == 131072:
            return True, 2048, 1024           # log-narrow
        return False, None, None
    if K == 2048:
        if n >= 8192:
            return True, 4096, 2048           # log-narrow, all measured N
        return False, None, None
    return False, None, None


def gvr_cutedsl_p2c_v2(logits, pre_idx, seq_lens, index_topk, compress_ratio=1, out=None):
    """Final iter8 op: routes per (dtype, K, N) to lin-narrow / log-narrow / baseline."""
    from gvr_p2c_op import gvr_cutedsl_p2c  # lin path (handles K512 narrow + baseline)
    bs, n = logits.shape
    use_log, kcc, kft = dispatch_p2c_v2(logits.dtype, index_topk, n)
    if use_log:
        return gvr_cutedsl_p2clog(logits, pre_idx, seq_lens, index_topk,
                                  compress_ratio, kcc=kcc, kft=kft, out=out)
    # lin path: gvr_p2c_op's own dispatch_params reproduces the K512 table;
    # for all other (dtype,K) it compiles baseline params.
    return gvr_cutedsl_p2c(logits, pre_idx, seq_lens, index_topk, compress_ratio, out=out)


if __name__ == "__main__":
    torch.manual_seed(0)
    for dt in (torch.float32,):
        for K, crv, N, kcc, kft in (
            (512, 4, 16384, 1024, 614),
            (512, 4, 262144, 1024, 614),
            (1024, 4, 32768, 2048, 1024),
            (1024, 4, 262144, None, 1024),   # base window, log interp
            (2048, 1, 32768, None, 2048),
        ):
            logits = torch.randn(1, N, dtype=dt, device="cuda")
            pre_idx = torch.topk(logits[0].float(), K).indices.int().view(1, K).contiguous()
            seq_lens = torch.full((1,), N * crv, dtype=torch.int32, device="cuda")
            out = gvr_cutedsl_p2clog(logits, pre_idx, seq_lens, K, crv, kcc=kcc, kft=kft)
            torch.cuda.synchronize()
            idx = out[0].clamp(min=0).long()
            v = logits[0].float().gather(0, idx).sort(descending=True).values
            ref = torch.topk(logits[0].float(), K).values
            d = (v - ref).abs().max().item()
            nuniq = len(set(out[0].tolist()))
            print(f"  {str(dt):14s} K={K:4d} cr={crv} N={N:6d} kCC={kcc} kFT={kft}: "
                  f"uniq={nuniq}/{K} valdiff={d:.2e}")
    print("GVR p2clog smoke OK")
