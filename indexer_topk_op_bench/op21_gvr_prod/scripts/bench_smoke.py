#!/usr/bin/env python3
"""op21 iter1 perf screen: gvr_ms vs gvr_x(op20 dispatch) vs radix_cutedsl.
Cold-L2 CUDA-graph event medians — SCREENING ONLY (nsys = the verdict)."""
import sys
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_BENCH / "ops"))
sys.path.insert(0, str(_BENCH / "op20_gvr_extreme" / "src"))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data
from radix_cutedsl_op import radix_cutedsl
from gvr_x_op import gvr_sw_auto as gvr_x_auto
from gvr_ms_op import gvr_ms

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
    (1024, 4096, 16), (1024, 8192, 16), (1024, 8192, 64), (1024, 16384, 64),
    (1024, 65536, 1), (1024, 65536, 16), (1024, 131072, 1), (1024, 262144, 1),
    (1024, 262144, 4),
    (512, 8192, 16), (512, 65536, 1), (512, 131072, 1),
    (2048, 16384, 16), (2048, 262144, 1),
]
print(f"{'K':>5} {'N':>7} {'BS':>4} | {'ms':>7} {'x20':>7} {'radix':>7} | "
      f"{'x20/ms':>7} {'rvl/ms':>7} exact")
import math
g_ms, g_rvl = [], []
for K, N, BS in CELLS:
    cr = CRMAP[K]
    b = synth_data.get_bundle(K, torch.float32, N)
    logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
    pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
    Npad = b["Npad"]
    seq_cr = torch.full((BS,), Npad * cr, dtype=torch.int32, device=DEV)
    seq_nod = torch.full((BS,), Npad, dtype=torch.int32, device=DEV)
    om = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ox = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    orv = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    cm = lambda: gvr_ms(logits, pre, seq_cr, K, compress_ratio=cr, out=om)
    cx = lambda: gvr_x_auto(logits, pre, seq_cr, K, cr, out=ox)
    cv = lambda: radix_cutedsl(logits, seq_nod, K, out=orv)
    cm(); cx(); cv(); torch.cuda.synchronize()
    ex = exact_all(om, logits[:, :Npad], K)
    t_m, t_x, t_v = cold_us(cm), cold_us(cx), cold_us(cv)
    g_ms.append(t_x / t_m); g_rvl.append(t_v / t_m)
    print(f"{K:>5} {N:>7} {BS:>4} | {t_m:7.1f} {t_x:7.1f} {t_v:7.1f} | "
          f"{t_x/t_m:7.3f} {t_v/t_m:7.3f} {'OK' if ex else 'FAIL'}")
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
print(f"\ngeomean x20/ms={gm(g_ms):.3f}  rvl/ms={gm(g_rvl):.3f}  (>1 = ms faster)")
