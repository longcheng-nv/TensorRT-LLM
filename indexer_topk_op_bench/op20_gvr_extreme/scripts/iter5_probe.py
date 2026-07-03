# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op20 iter5 probe: op17-v2 P4T4 fusion vs the current mc/mcC8 routing at the
# 131K/262K low-BS keys (the K1024-262K residual hole + K512 parity cells).
# Per-BS-bucket probing (protocol red line); exactness on every variant.
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_BENCH / "op17_gvr_portfolio" / "v2"))
import synth_data  # noqa: E402
from radix_cutedsl_op import radix_cutedsl  # noqa: E402
from gvr_multicta_cutedsl_op import gvr_multicta_cutedsl  # noqa: E402
from gvr_portfolio_fusion_op import gvr_portfolio_fusion  # noqa: E402

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


CELLS = [(K, N, BS) for K in (512, 1024) for N in (131072, 262144)
         for BS in (1, 4, 16)]

if __name__ == "__main__":
    tbl = json.load(open(_HERE.parent / "results" / "dispatch_table_fp32.json"))
    for K, N, BS in CELLS:
        cr = 4
        key = f"{K}_{N}_{BS}"
        cur = tbl.get(key, {}).get("cfg", "?")
        b = synth_data.get_bundle(K, torch.float32, N)
        logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
        pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
        seq = torch.full((BS,), b["Npad"] * cr, dtype=torch.int32, device=DEV)
        o = torch.empty(BS, K, dtype=torch.int32, device=DEV)
        variants = [("radix", lambda: radix_cutedsl(
            logits, torch.full((BS,), b["Npad"], dtype=torch.int32, device=DEV), K, out=o), False)]
        variants.append(("mc-auto", lambda: gvr_multicta_cutedsl(
            logits, pre, seq, K, cr, out=o), True))
        variants.append(("mcC8", lambda: gvr_multicta_cutedsl(
            logits, pre, seq, K, cr, out=o, cluster_size=8), True))
        for P, T in ((4, 4), (8, 4)):
            variants.append((f"fusP{P}T{T}", lambda P=P, T=T: gvr_portfolio_fusion(
                logits, pre, seq, K, cr, out=o, P=P, T=T), True))
        line = f"{key:>16} cur={cur:>7} |"
        for nm, call, chk in variants:
            try:
                call(); torch.cuda.synchronize()
                ok = exact(o, logits, K) if chk else True
                line += f" {nm}={cold_us(call):6.1f}{'' if ok else '!EX'}"
            except Exception as e:
                line += f" {nm}=ERR({type(e).__name__})"
        print(line, flush=True)
