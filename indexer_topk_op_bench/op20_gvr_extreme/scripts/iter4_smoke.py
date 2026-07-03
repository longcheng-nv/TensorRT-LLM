# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op20 iter4 smoke: fused P2+P3 (fuse=True) vs classic (fuse=False) A/B at the
# small-N wall cells + a mid-N sanity cell. Exactness on 3 perturbed inputs,
# cold-L2 graph-median timing, in-run radix rival reference.
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
from radix_cutedsl_op import radix_cutedsl  # noqa: E402
from gvr_x_op import gvr_sw  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


def cold_us(call, reps=30, warmup=5):
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


def exact(out, logits, K):
    lf = logits.float()
    ref = torch.topk(lf, K, dim=1).values
    idx = out.clamp(min=0).long()
    v = lf.gather(1, idx).sort(dim=1, descending=True).values
    if (v - ref).abs().max().item() != 0.0:
        return False
    return all(len(set(out[r].tolist())) == K for r in range(out.shape[0]))


CELLS = [(512, 4096, 1), (512, 4096, 4), (512, 4096, 16), (512, 4096, 64),
         (512, 8192, 1), (512, 8192, 4), (512, 8192, 16), (512, 8192, 64),
         (1024, 4096, 1), (1024, 4096, 4), (1024, 4096, 16), (1024, 4096, 64),
         (1024, 8192, 1), (1024, 8192, 4), (1024, 8192, 16), (1024, 8192, 64),
         (512, 65536, 4), (1024, 65536, 4)]  # mid-N sanity (fuse must not regress)
CFGS = (("M2R1p4", 2), ("M4R1p4", 4), ("M6R1p4", 6))

if __name__ == "__main__":
    torch.manual_seed(0)
    n_bad = 0
    for K, N, BS in CELLS:
        cr = 4
        b = synth_data.get_bundle(K, torch.float32, N)
        logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
        pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
        seq_cr = torch.full((BS,), b["Npad"] * cr, dtype=torch.int32, device=DEV)
        seq_nod = torch.full((BS,), b["Npad"], dtype=torch.int32, device=DEV)
        o = torch.empty(BS, K, dtype=torch.int32, device=DEV)
        line = f"K{K} N{N:>6} BS{BS:>3} | radix={cold_us(lambda: radix_cutedsl(logits, seq_nod, K, out=o)):6.1f}"
        for nm, M in CFGS:
            for fuse in (False, True):
                tag = nm + ("f" if fuse else "")
                try:
                    call = (lambda M=M, fuse=fuse: gvr_sw(
                        logits, pre, seq_cr, K, cr, out=o, M=M, R=1,
                        place_mode=4, fuse=fuse))
                    # exactness: bundle + 2 perturbed copies (fresh noise seeds)
                    ok = True
                    for sd in range(3):
                        if sd == 0:
                            li = logits
                        else:
                            g0 = torch.Generator(device=DEV).manual_seed(1000 + sd)
                            li = logits + 1e-3 * torch.randn(
                                logits.shape, generator=g0, device=DEV, dtype=logits.dtype)
                        call2 = (lambda li=li, M=M, fuse=fuse: gvr_sw(
                            li, pre, seq_cr, K, cr, out=o, M=M, R=1,
                            place_mode=4, fuse=fuse))
                        call2(); torch.cuda.synchronize()
                        if not exact(o, li, K):
                            ok = False
                    t = cold_us(call)
                    line += f" {tag}={t:6.1f}{'' if ok else '!EX'}"
                    if not ok:
                        n_bad += 1
                except Exception:
                    line += f" {tag}=ERR"
                    n_bad += 1
        print(line, flush=True)
    print(f"BAD: {n_bad}", flush=True)
