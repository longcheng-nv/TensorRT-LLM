# SPDX-License-Identifier: NVIDIA
# op17 iter4: end-to-end A/B — portfolio (single-CTA M-way P2) vs baseline
# gvr_cutedsl. cold-L2 event median (harness/sweep.py protocol), report synth data,
# exactness checked. This is the REAL measurement of the tax-free tight-cand P4 win.
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
from gvr_portfolio_op import gvr_portfolio  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


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


def exact(out, logits, K):
    idx = out[0].clamp(min=0).long()
    v = logits[0].float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits[0].float(), K).values
    return (v - ref).abs().max().item() == 0.0 and len(set(out[0].tolist())) == K


def run(K, dtype, N, cr_val, M_thr=16):
    b = synth_data.get_bundle(K, dtype, N)
    logits, pre = b["logits"].to(DEV), b["preIdx"].to(DEV)
    Npad = b["Npad"]
    seq_lens = torch.full((1,), Npad * cr_val, dtype=torch.int32, device=DEV)
    ob = torch.empty(1, K, dtype=torch.int32, device=DEV)
    op = torch.empty(1, K, dtype=torch.int32, device=DEV)
    cb = lambda: gvr_cutedsl(logits, pre, seq_lens, K, cr_val, out=ob)
    cp = lambda: gvr_portfolio(logits, pre, seq_lens, K, cr_val, out=op, M_thr=M_thr)
    cb(); cp(); torch.cuda.synchronize()
    ok = exact(op, logits, K)
    tb = cold_us(cb); tp = cold_us(cp)
    return tb, tp, ok


if __name__ == "__main__":
    dtype, M = torch.float32, 16
    print(f"portfolio vs baseline gvr_cutedsl — fp32, M={M}, cold-L2 median us")
    print(f"{'K':>5} {'N':>8} | base_us  port_us  speedup  exact")
    for K, cr_val in ((512, 4), (1024, 4), (2048, 1)):
        for N in [4096, 8192, 16384, 32768, 65536, 131072, 262144]:
            if N <= 2 * K:
                continue
            tb, tp, ok = run(K, dtype, N, cr_val, M)
            flag = "OK" if ok else "**FAIL**"
            print(f"{K:>5} {N:>8} | {tb:6.1f}  {tp:6.1f}  {tb/tp:6.3f}x  {flag}")
