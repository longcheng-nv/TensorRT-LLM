#!/usr/bin/env python3
"""op21 iter5 (b) A/B: P1b QBINS=256 vs 64 on the P1 highBS grid (gvr_ms
single-CTA path; rank-scatter P4 on both sides). Cold-L2 CUDA-graph event
medians — SCREENING ONLY. Same-process paired A/B via OP21_QBINS."""
import math
import os
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_BENCH / "ops"))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
from gvr_msc_op import gvr_ms_auto  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
CRMAP = {512: 4, 1024: 4, 2048: 1}


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


def exact_all(out, logits, K):
    lf = logits.float()
    ref = torch.topk(lf, K, dim=1).values
    v = lf.gather(1, out.clamp(min=0).long()).sort(dim=1, descending=True).values
    if (v - ref).abs().max().item() != 0.0:
        return False
    return all(len(set(out[r].tolist())) == K for r in range(out.shape[0]))


CELLS = [
    (512, 4096, 256), (512, 8192, 256), (512, 8192, 1024), (512, 16384, 1024),
    (1024, 4096, 256), (1024, 8192, 256), (1024, 8192, 1024),
    (1024, 16384, 256), (1024, 16384, 1024),
    (2048, 8192, 256), (2048, 16384, 256), (2048, 16384, 1024),
    # BS64 (< NUM_SMS: rule would NOT apply; forced here to bound the tradeoff)
    (1024, 4096, 64), (1024, 16384, 64),
]

print(f"{'K':>5} {'N':>7} {'BS':>5} | {'q256_us':>8} {'q64_us':>8} "
      f"{'q256/q64':>8} exact")
rats = []
for K, N, BS in CELLS:
    cr = CRMAP[K]
    b = synth_data.get_bundle(K, torch.float32, N)
    lg = b["logits"][:1].repeat(BS, 1).contiguous()
    pre = b["preIdx"][:1].repeat(BS, 1).contiguous()
    Npad = b["Npad"]
    sl = torch.full((BS,), Npad * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    res = {}
    exact = {}
    for qb in ("256", "64"):
        os.environ["OP21_QBINS"] = qb
        call = lambda: gvr_ms_auto(lg, pre, sl, K, cr, out=out)
        call(); torch.cuda.synchronize()
        exact[qb] = exact_all(out, lg[:, :N], K)
        res[qb] = cold_us(call)
    r = res["256"] / res["64"]
    rats.append(r)
    ex = "OK" if (exact["256"] and exact["64"]) else "**FAIL**"
    print(f"{K:>5} {N:>7} {BS:>5} | {res['256']:8.2f} {res['64']:8.2f} "
          f"{r:8.3f} {ex}")
print(f"gm q256/q64 = {math.exp(sum(math.log(x) for x in rats)/len(rats)):.3f} "
      f"(>1 = QBINS=64 faster), win {sum(1 for x in rats if x > 1.0)}/{len(rats)}")
