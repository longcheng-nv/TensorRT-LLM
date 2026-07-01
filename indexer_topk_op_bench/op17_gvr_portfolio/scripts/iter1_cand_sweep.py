# SPDX-License-Identifier: NVIDIA
# op17 iter1: does shrinking cand_count (tighter acceptance ceiling kC) keep
# cutting end-to-end GVR time at small N? This bounds the "P4-shrink via free
# tight threshold" lever the user identified.
#
# NOTE: this uses the SERIAL secant (GvrP2C) to reach a tight cand, so it PAYS a
# P2-iter tax the real portfolio would NOT (portfolio gets the tight threshold in
# one parallel sweep). Therefore this A/B is a LOWER BOUND on the portfolio's
# small-N gain: whatever net win survives here, the tax-free portfolio beats it.
#
# Exactness validated every cell (valdiff vs torch.topk == 0, uniq == K). cold-L2
# event median matches harness/sweep.py protocol.
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_BENCH / "ops"))
sys.path.insert(0, str(_BENCH / "op13_gvr_p2cand" / "src"))
import synth_data  # noqa: E402
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
import cutlass, cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as cr  # noqa: E402
from gvr_p2c_op import GvrP2C, _config, _DT  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
_compiled = {}


def compile_variant(dtype, bs, n, K, cr_val, kC, kFT):
    key = (dtype, bs, n, K, cr_val, kC, kFT)
    if key in _compiled:
        return _compiled[key]
    t, use256, min_bpm = _config(bs, n)
    kobj = GvrP2C(dtype=_DT[dtype], top_k=K, next_n=1, num_threads=t, compress_ratio=cr_val,
                  use_256bit_load=use256, enable_unroll_4=True, enable_phase3_unroll=True,
                  min_blocks_per_mp=min_bpm, return_output_values=False,
                  kC_override=kC, kFTarget_override=kFT)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if use256 else 16
    inf = cr.make_fake_compact_tensor(_DT[dtype], (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pf = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sf = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    of = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(kobj, inf, pf, sf, None, of, stream=fs, options="--enable-tvm-ffi")
    _compiled[key] = compiled
    return compiled


def cold_us(call, reps=40, warmup=5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup): call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): call()
    for _ in range(10): g.replay()
    torch.cuda.synchronize()
    cold = []
    for _ in range(reps):
        _EVICT.uniform_(0, 1); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    cold.sort(); del g
    return cold[len(cold) // 2]


def run(K, dtype, N, cr_val, kC, kFT):
    b = synth_data.get_bundle(K, dtype, N)
    logits, pre_idx = b["logits"].to(DEV), b["preIdx"].to(DEV)
    Npad = b["Npad"]
    seq_lens = torch.full((1,), Npad * cr_val, dtype=torch.int32, device=DEV)
    compiled = compile_variant(dtype, 1, Npad, K, cr_val, kC, kFT)
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)

    def call():
        compiled(logits, pre_idx, seq_lens, None, out)

    call(); torch.cuda.synchronize()
    idx = out[0].clamp(min=0).long()
    v = logits[0].float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits[0].float(), K).values
    vd = (v - ref).abs().max().item()
    uniq = len(set(out[0].tolist()))
    us = cold_us(call)
    return us, vd, uniq


if __name__ == "__main__":
    K, dtype, cr_val = 512, torch.float32, 4
    Ns = [4096, 8192, 16384, 65536]
    kCs = [5120, 2560, 1536, 1024, 768]   # acceptance ceiling: 10x..1.5x K ; kFT=512 fixed
    kFT = 512
    print(f"K={K} fp32 cr={cr_val} — cold-L2 median us vs acceptance ceiling kC (kFT={kFT})")
    print(f"{'N':>7} | " + " ".join(f"kC={k:<5d}" for k in kCs) + " | best/base  exact?")
    for N in Ns:
        row, base = [], None
        allok = True
        for kC in kCs:
            us, vd, uniq = run(K, dtype, N, cr_val, kC, kFT)
            ok = (vd == 0.0 and uniq == K)
            allok &= ok
            if base is None: base = us
            row.append((kC, us, ok))
        best = min(r[1] for r in row)
        cells = " ".join(f"{us:6.1f}{'' if ok else '!'}" for _, us, ok in row)
        print(f"{N:>7} | {cells} | {best/base:.3f}x  {'EXACT' if allok else 'INEXACT@!'}")
