"""Pure GPU throughput of the per-layer decode kernels, measured by replaying captured arguments
50x back-to-back with one eval at the end (no per-call sync): isolates GPU execution time from
dispatch latency / pipeline bubbles / SSD waits. Run from the DeepInfer-Mac repo root.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402
from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

M.MoE.SPEC_K = 4
llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=18.0)
ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
st, logits, _, _ = llm._prefill(ids, False)
tok = int(np.argmax(np.array(logits[-1])))
cap = {}


def capture(obj, name, tag):
    f = getattr(obj, name)

    def g(*a, **k):
        cap.setdefault(tag, []).append((f, a, k))
        return f(*a, **k)
    setattr(obj, name, g)


for b in llm.model.layers:
    at = b.attn; at._compiled()
    capture(at, "_proj_c", "attn.proj"); capture(at, "_core_c", "attn.core")
    if at.is_index_source: capture(at, "_indexer", "attn.indexer")
    if not hasattr(b.ffn, "_rs_c"): b.ffn._rs_c = mx.compile(b.ffn._route_shared)
    capture(b.ffn, "_rs_c", "moe.route+shared")
    capture(b.ffn, "_experts_into", "moe.experts_into(slab gather_qmm; slots already resident on replay)")
lg = llm.model.forward(st, [tok]); mx.eval(lg)
# the capture above recorded one decode step (40 layers). Replay each stage's calls N times.
N = 20
print("captured per token: " + ", ".join(f"{k}: {len(v)} calls" for k, v in cap.items()))
total = 0.0
for tag, calls in cap.items():
    # warm
    outs = [f(*a, **k) for f, a, k in calls]; mx.eval([o for o in outs if isinstance(o, mx.array)] or [o[0] for o in outs])
    t0 = time.perf_counter()
    flat = []
    for _ in range(N):
        outs = [f(*a, **k) for f, a, k in calls]
        for o in outs:
            if isinstance(o, mx.array): flat.append(o)
            elif isinstance(o, (tuple, list)): flat += [v for v in o if isinstance(v, mx.array)]
    mx.eval(flat)  # every iteration's graph actually runs
    dt = (time.perf_counter() - t0) / N
    total += dt
    print(f"  {tag:70s} {dt*1000:7.1f} ms/token GPU-throughput ({dt/len(calls)*1000:.2f} ms/call)")
print(f"  sum of stages: {total*1000:.0f} ms/token  (vs ~270 ms/token measured end-to-end)")
