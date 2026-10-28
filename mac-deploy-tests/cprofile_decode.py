"""cProfile of the host side of decode: where does CPU time go per token (Python, numpy, MLX graph
building, host syncs)? Run from the DeepInfer-Mac repo root with SPEC_K=3.
"""
import cProfile
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402
from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

M.MoE.SPEC_K = 3
llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=float(sys.argv[1]) if len(sys.argv) > 1 else 18.0)
ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
st, logits, _, _ = llm._prefill(ids, False)
tok = int(np.argmax(np.array(logits[-1])))
for _ in range(6):
    lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))
N = 10
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
for _ in range(N):
    lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))
pr.disable()
wall = (time.perf_counter() - t0) / N
io = llm.weights.experts.stats
print(f"wall {wall*1000:.0f} ms/token (profiled; cProfile adds overhead), cumulative SSD wait so far {io['wait_s']:.1f}s")
ps = pstats.Stats(pr)
ps.sort_stats("tottime")
print("\n=== top 30 by own time (per token = total/10) ===")
ps.print_stats(30)
