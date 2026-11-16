"""Per-kernel GPU throughput of the fused decode path (fixed-expert mode): capture every custom
kernel launch of one token (mk.* kernels) and every remaining MLX op group, replay each kernel's
launches N times back to back with one eval, report us/launch, total ms/token and GB/s.
usage: mk_kprof.py [--n 20]"""
from __future__ import annotations

import argparse
import collections
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

CAP = collections.defaultdict(list)
FACT = ["_moe_up_kernel", "_moe_down_kernel", "_hc_partial_kernel", "_hc_finalize_kernel", "_hc_fused_kernel", "_gemv_mxfp8_kernel",
        "_rs_up_kernel", "_rs_down_kernel", "_attn_aux_kernel", "_sparse_attn_kernel"]


def wrap_factory(name):
    f = getattr(mk, name)
    k = f()

    class W:
        def __call__(self, *a, **kw):
            out = k(*a, **kw)
            tmpl = dict(kw.get("template", []))
            ins = kw.get("inputs", [])
            nbytes = sum(int(x.nbytes) for x in ins if isinstance(x, mx.array) and x.nbytes < (1 << 31))
            if name in ("_moe_up_kernel", "_moe_down_kernel"):  # slab inputs: count only the touched slots
                nbytes = -1
            tag = name.strip("_").replace("_kernel", "")
            if name == "_gemv_mxfp8_kernel":
                tag += f"[K={tmpl.get('K')},N={kw['output_shapes'][0][0]}]"
            CAP[tag].append((k, a, kw, nbytes))
            return out

    w = W()
    setattr(mk, name, lambda: w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    M.MoE.SPEC_K = 0
    M.MoE.FIXED_EXPERTS = list(range(6))
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=16)
    ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    for _ in range(3):
        tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
    for nm in FACT:
        wrap_factory(nm)
    t0 = time.perf_counter()
    mx.eval(llm.model.forward(st, [tok]))
    print(f"one token (capturing): {(time.perf_counter() - t0) * 1e3:.1f} ms")
    rows, total = [], 0.0
    for tag, calls in CAP.items():
        def run():
            outs = []
            for k, a_, kw, _ in calls:
                outs += k(*a_, **kw)
            return outs
        mx.eval(run())
        t = time.perf_counter()
        outs = []
        for _ in range(a.n):
            outs += run()
        mx.eval(outs)
        dt = (time.perf_counter() - t) / a.n
        total += dt
        nb = sum(c[3] for c in calls)
        rows.append((tag, dt, len(calls), nb))
    for tag, dt, n, nb in sorted(rows, key=lambda r: -r[1]):
        bw = f"{nb / dt / 1e9:6.1f} GB/s" if nb > 0 else "   (slab)"
        print(f"  {tag:44s} {dt * 1e3:7.2f} ms/token  {dt / n * 1e6:7.1f} us x {n:3d}  {bw}")
    print(f"  {'sum of fused kernels':44s} {total * 1e3:7.2f} ms/token")


if __name__ == "__main__":
    main()
