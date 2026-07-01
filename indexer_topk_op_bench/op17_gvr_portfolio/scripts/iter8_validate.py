# SPDX-License-Identifier: NVIDIA
# op17 iter8: ×3-median cold-L2 A/B of the cooperative-cluster portfolio (G=16)
# vs BOTH baselines — single-CTA gvr_cutedsl AND the existing PR#15198 multicta
# cluster — across fp32/bf16/fp16. Exactness checked. Reports median-of-3 speedup.
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
from gvr_multicta_cutedsl_op import gvr_multicta_cutedsl  # noqa: E402
from gvr_portfolio_cluster_op import gvr_portfolio_cluster  # noqa: E402

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
    c = []
    for _ in range(reps):
        _EVICT.uniform_(0, 1); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        c.append(e0.elapsed_time(e1) * 1e3)
    c.sort(); del g
    return c[len(c) // 2]


def med3(call):
    return sorted(cold_us(call) for _ in range(3))[1]


def exact(out, logits, K):
    idx = out[0].clamp(min=0).long()
    v = logits[0].float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits[0].float(), K).values
    return (v - ref).abs().max().item() == 0.0 and len(set(out[0].tolist())) == K


if __name__ == "__main__":
    G = 16
    Ns = [4096, 16384, 65536, 262144]
    print(f"op17 cooperative-cluster G={G}, ×3-median cold-L2. port/base = vs single-CTA gvr_cutedsl; port/mc = vs PR#15198 cluster")
    print(f"{'dt':>5} {'K':>5} {'N':>8} | base  mc   port | port/base  port/mc  exact")
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        for K, crv in ((512, 4), (1024, 4), (2048, 1)):
            for N in Ns:
                if N <= 2 * K:
                    continue
                b = synth_data.get_bundle(K, dt, N)
                lo, pr = b["logits"].to(DEV), b["preIdx"].to(DEV)
                Npad = b["Npad"]
                sl = torch.full((1,), Npad * crv, dtype=torch.int32, device=DEV)
                ob = torch.empty(1, K, dtype=torch.int32, device=DEV)
                om = torch.empty(1, K, dtype=torch.int32, device=DEV)
                op = torch.empty(1, K, dtype=torch.int32, device=DEV)
                cb = lambda: gvr_cutedsl(lo, pr, sl, K, crv, out=ob)
                cm = lambda: gvr_multicta_cutedsl(lo, pr, sl, K, crv, out=om)
                cp = lambda: gvr_portfolio_cluster(lo, pr, sl, K, crv, out=op, G=G)
                cb(); cm(); cp(); torch.cuda.synchronize()
                ok = exact(op, lo, K)
                tb = med3(cb); tm = med3(cm); tp = med3(cp)
                dtn = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[dt]
                flag = "OK" if ok else "**FAIL**"
                print(f"{dtn:>5} {K:>5} {N:>8} | {tb:5.1f} {tm:5.1f} {tp:5.1f} | "
                      f"{tb/tp:8.3f}  {tm/tp:7.3f}  {flag}")
