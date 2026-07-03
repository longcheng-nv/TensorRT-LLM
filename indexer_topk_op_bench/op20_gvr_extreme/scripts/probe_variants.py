# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op20 iter2 attribution probe: at the losing tier1 cells, time every available
# GVR variant knob (single-CTA sandwich configs, cluster G, baseline) vs the
# in-run radix_cutedsl rival. Answers: is the hole the threshold-parallel
# cluster's O(N)-per-CTA pass structure, or config choice?
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
from radix_cutedsl_op import radix_cutedsl  # noqa: E402
from gvr_x_op import gvr_sw  # noqa: E402
from gvr_swc_op import gvr_swc  # noqa: E402

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


CELLS = [(512, 4096, 1), (512, 4096, 16), (512, 8192, 4), (1024, 8192, 4),
         (512, 131072, 1), (512, 262144, 1), (1024, 262144, 4)]

if __name__ == "__main__":
    torch.manual_seed(0)
    for K, N, BS in CELLS:
        cr = 4
        b = synth_data.get_bundle(K, torch.float32, N)
        logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
        pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
        seq_cr = torch.full((BS,), b["Npad"] * cr, dtype=torch.int32, device=DEV)
        seq_nod = torch.full((BS,), b["Npad"], dtype=torch.int32, device=DEV)
        o = torch.empty(BS, K, dtype=torch.int32, device=DEV)
        res = []
        variants = [("radix", lambda: radix_cutedsl(logits, seq_nod, K, out=o)),
                    ("base", lambda: gvr_cutedsl(logits, pre, seq_cr, K, cr, out=o))]
        for G in (4, 8, 16):
            variants.append((f"swcG{G}", lambda G=G: gvr_swc(logits, pre, seq_cr, K, cr, out=o, G=G)))
        for cfg in (("M2R1p4", 2, 1, 4), ("M4R1p4", 4, 1, 4), ("M6R1p4", 6, 1, 4),
                    ("M4R2p4", 4, 2, 4)):
            nm, M, R, pm = cfg
            variants.append((nm, lambda M=M, R=R, pm=pm: gvr_sw(
                logits, pre, seq_cr, K, cr, out=o, M=M, R=R, place_mode=pm)))
        line = f"K{K} N{N:>6} BS{BS:>2} |"
        for nm, call in variants:
            try:
                call(); torch.cuda.synchronize()
                ok = exact(o, logits, K) if nm != "radix" else True
                t = cold_us(call)
                line += f" {nm}={t:6.1f}{'' if ok else '!EX'}"
            except Exception as e:
                line += f" {nm}=ERR"
        print(line, flush=True)
