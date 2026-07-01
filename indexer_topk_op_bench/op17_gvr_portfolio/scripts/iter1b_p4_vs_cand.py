# SPDX-License-Identifier: NVIDIA
# op17 iter1b: isolate P4 cost as a function of cand_count, DECOUPLED from the P2
# tax. Uses the clock64-instrumented timed kernel with a kC (acceptance-ceiling)
# override so tighter kC => tighter cand => (hypothesis) smaller P4. Reports RAW
# P4 cycles (same SM clock across kC at fixed N,K,dtype => directly comparable) so
# the P2-tax confound in iter1's end-to-end wall time is removed.
#
# Interpretation: if P4 cycles fall ~linearly as kC->K, the "P4-shrink via free
# tight threshold" lever is real and the tax-free portfolio would net it at small
# N. If P4 cycles are ~flat, P4 is fixed-cost bound (kNumBins histogram + K
# writeback) and the small-N lever is weak.
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops", "op13_gvr_p2cand/src"):
    sys.path.insert(0, str(_BENCH / p))
import cutlass, cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as cr  # noqa: E402
from measure_cute_phases import GvrTopKKernelTimed, _config, _DT  # noqa: E402
import synth_data  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


class Timed(GvrTopKKernelTimed):
    def __init__(self, *a, kC_override=None, kFTarget_override=None, **kw):
        super().__init__(*a, **kw)
        if kC_override is not None:
            self.kC = int(kC_override)
        if kFTarget_override is not None:
            self.kFTarget = int(kFTarget_override)


_compiled = {}


def compile_timed(dtype, n, K, cr_val, kC, kFT):
    key = (dtype, n, K, cr_val, kC, kFT)
    if key in _compiled:
        return _compiled[key]
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
    compiled = cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, ts_f, stream=fs, options="--enable-tvm-ffi")
    _compiled[key] = compiled
    return compiled


def cyc_split(K, dtype, N, cr_val, kC, kFT, reps=30):
    b = synth_data.get_bundle(K, dtype, N)
    logits, pre = b["logits"].to(DEV).contiguous(), b["preIdx"].to(DEV).contiguous()
    Npad = b["Npad"]
    seq_lens = torch.full((1,), Npad * cr_val, dtype=torch.int32, device=DEV)
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)
    ts = torch.zeros(1, 6, dtype=torch.int64, device=DEV)
    compiled = compile_timed(dtype, Npad, K, cr_val, kC, kFT)
    for _ in range(5):
        compiled(logits, pre, seq_lens, None, out, ts)
    torch.cuda.synchronize()
    # exactness
    idx = out[0].clamp(min=0).long()
    v = logits[0].float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits[0].float(), K).values
    vd = (v - ref).abs().max().item(); uniq = len(set(out[0].tolist()))
    rows = []
    for _ in range(reps):
        _EVICT.uniform_()
        compiled(logits, pre, seq_lens, None, out, ts)
        torch.cuda.synchronize()
        rows.append(ts[0].cpu().tolist())
    rows.sort(key=lambda r: r[5] - r[0])
    t = rows[len(rows) // 2]
    return dict(P1=t[1]-t[0], P2=t[2]-t[1], P3=t[3]-t[2], P4=t[4]-t[3], end=t[5]-t[4],
                tot=t[5]-t[0], vd=vd, uniq=uniq)


if __name__ == "__main__":
    K, dtype, cr_val = 512, torch.float32, 4
    kCs = [5120, 1536, 768]
    print(f"K={K} fp32 cr={cr_val} — RAW clock64 cycles (median), P4 vs acceptance ceiling kC (kFT=512)")
    for N in [4096, 8192, 16384]:
        print(f"--- N={N} ---")
        base = None
        for kC in kCs:
            r = cyc_split(K, dtype, N, cr_val, kC, 512)
            if base is None: base = r
            p4rel = r['P4'] / base['P4']
            totrel = r['tot'] / base['tot']
            ex = "EXACT" if (r['vd'] == 0.0 and r['uniq'] == K) else f"INEXACT vd={r['vd']:.1e} uniq={r['uniq']}"
            print(f"  kC={kC:5d}: P2={r['P2']:8d} P3={r['P3']:8d} P4={r['P4']:8d}cyc "
                  f"(P4 {p4rel:.2f}x, tot {totrel:.2f}x)  {ex}")
