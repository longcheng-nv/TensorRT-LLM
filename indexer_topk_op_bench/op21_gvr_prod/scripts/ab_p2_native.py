#!/usr/bin/env python3
"""op21 iter9 A/B: native 16-bit ladder compares (OP21_P2_NATIVE=1) vs
cvt->fp32 (=0) on gvr_ms_auto, bf16 + fp16. Cold-L2 CUDA-graph event
medians — SCREENING ONLY. Same-process paired A/B via the env-keyed
compile cache."""
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


# P0 grid (cluster cells) + P1 representatives (single-CTA, canary-only)
CELLS = [
    (1024, 65536, 1), (1024, 65536, 16),
    (1024, 131072, 1), (1024, 131072, 16),
    (1024, 262144, 1), (1024, 262144, 4), (1024, 262144, 16),
    (512, 131072, 1), (512, 262144, 1),
    (2048, 131072, 1), (2048, 262144, 1), (2048, 262144, 16),
    (1024, 8192, 256), (2048, 16384, 256),
]

print(f"{'dt':>5} {'K':>5} {'N':>7} {'BS':>5} | {'cvt_us':>8} {'nat_us':>8} "
      f"{'cvt/nat':>9} exact")
rats = []
import itertools
for dtt, (K, N, BS) in itertools.product((torch.bfloat16, torch.float16), CELLS):
    cr = CRMAP[K]
    b = synth_data.get_bundle(K, dtt, N)
    lg = b["logits"][:1].repeat(BS, 1).contiguous()
    pre = b["preIdx"][:1].repeat(BS, 1).contiguous()
    sl = torch.full((BS,), b["Npad"] * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    res = {}
    exact = {}
    for flag in ("0", "1"):
        os.environ["OP21_P2_NATIVE"] = flag
        call = lambda: gvr_ms_auto(lg, pre, sl, K, cr, out=out)
        call(); torch.cuda.synchronize()
        exact[flag] = exact_all(out, lg[:, :N], K)
        res[flag] = cold_us(call)
    r = res["0"] / res["1"]
    rats.append(r)
    ex = "OK" if (exact["0"] and exact["1"]) else "**FAIL**"
    dn = "bf16" if dtt == torch.bfloat16 else "fp16"
    print(f"{dn:>5} {K:>5} {N:>7} {BS:>5} | {res['0']:8.2f} {res['1']:8.2f} "
          f"{r:9.3f} {ex}")
print(f"gm cvt/native = {math.exp(sum(math.log(x) for x in rats)/len(rats)):.3f} "
      f"(>1 = native faster), win {sum(1 for x in rats if x > 1.0)}/{len(rats)}")
