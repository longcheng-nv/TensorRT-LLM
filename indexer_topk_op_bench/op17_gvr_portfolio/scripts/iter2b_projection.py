# SPDX-License-Identifier: NVIDIA
# op17 iter2b: assemble the projected portfolio NET from measured terms (K512 fp32).
#
#   portfolio_total ≈ baseline_total − P4_loose_us*(1 − P4_shrink) + sync_us
#
# where (all measured on this B200, cold-L2, report synth data):
#   baseline_total, P4_loose_us  : measure_cute_phases (fraction × trusted prod wall)
#   P4_shrink                    : timed-kernel P4 cycle ratio kC768/kC5120 (iter1b method)
#   sync_us                      : iter2 2-kernel leader-pick proxy (conservative upper bound)
#
# Assumes portfolio holds P2 at ~1 parallel sweep pass (= baseline P2 iters at K512 ≈ 1),
# i.e. NO serial-secant tax. P3 unchanged (still one full-N collect). This is the ceiling
# the real cooperative kernel would approach.
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops", "op13_gvr_p2cand/src"):
    sys.path.insert(0, str(_BENCH / p))
import cutlass, cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as cr  # noqa: E402
from measure_cute_phases import measure, GvrTopKKernelTimed, _config, _DT  # noqa: E402
import synth_data  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)

# iter2 measured leader-sync overhead (us), conservative 2-kernel proxy.
SYNC_US = {4096: 0.0, 8192: 0.0, 16384: 1.95, 65536: 3.42}


class Timed(GvrTopKKernelTimed):
    def __init__(self, *a, kC_override=None, kFTarget_override=None, **kw):
        super().__init__(*a, **kw)
        if kC_override is not None: self.kC = int(kC_override)
        if kFTarget_override is not None: self.kFTarget = int(kFTarget_override)


_compiled = {}


def _compile(dtype, n, K, cr_val, kC, kFT):
    key = (dtype, n, K, cr_val, kC, kFT)
    if key in _compiled: return _compiled[key]
    t, use256, min_bpm = _config(1, n)
    kobj = Timed(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                 use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                 min_blocks_per_mp=min_bpm, return_output_values=False,
                 kC_override=kC, kFTarget_override=kFT)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if use256 else 16
    in_f = cr.make_fake_compact_tensor(_DT[dtype], (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    ts_f = cr.make_fake_compact_tensor(cutlass.Int64, (nr, 6), stride_order=(1, 0), assumed_align=16)
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    c = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, ts_f, stream=fs, options="--enable-tvm-ffi")
    _compiled[key] = c
    return c


def p4_cyc(K, dtype, N, cr_val, kC, reps=30):
    b = synth_data.get_bundle(K, dtype, N)
    logits, pre = b["logits"].to(DEV).contiguous(), b["preIdx"].to(DEV).contiguous()
    Npad = b["Npad"]
    seq_lens = torch.full((1,), Npad * cr_val, dtype=torch.int32, device=DEV)
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)
    ts = torch.zeros(1, 6, dtype=torch.int64, device=DEV)
    c = _compile(dtype, Npad, K, cr_val, kC, 512)
    for _ in range(5): c(logits, pre, seq_lens, None, out, ts)
    torch.cuda.synchronize()
    rows = []
    for _ in range(reps):
        _EVICT.uniform_(); c(logits, pre, seq_lens, None, out, ts); torch.cuda.synchronize()
        rows.append(ts[0].cpu().tolist())
    rows.sort(key=lambda r: r[5] - r[0])
    t = rows[len(rows) // 2]
    return t[4] - t[3]  # P4 cycles


if __name__ == "__main__":
    K, dtype, cr_val = 512, torch.float32, 4
    print(f"K={K} fp32 cr={cr_val} — PROJECTED portfolio net (measured terms)")
    print(f"{'N':>7} | base_us  P4_us  P4shrink  sync_us | proj_us  speedup")
    for N in [4096, 8192, 16384, 65536]:
        r = measure(K, dtype, N, cr_val, reps=25)
        base_us, p4_us = r["total_us"], r["P4_us"]
        p4_loose = p4_cyc(K, dtype, N, cr_val, 5120)
        p4_tight = p4_cyc(K, dtype, N, cr_val, 768)
        shrink = p4_tight / p4_loose
        sync = SYNC_US[N]
        proj = base_us - p4_us * (1 - shrink) + sync
        print(f"{N:>7} | {base_us:6.1f} {p4_us:6.2f}  {shrink:6.2f}   {sync:5.2f}  | "
              f"{proj:6.1f}  {base_us / proj:5.3f}x")
