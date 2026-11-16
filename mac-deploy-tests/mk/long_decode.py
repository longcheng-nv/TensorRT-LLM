import sys, time, numpy as np
sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx
from deepinfer_mac import LLM
from deepinfer_mac.model import model as M
import os
M.MoE.SPEC_K = int(sys.argv[2]); M.MoE.SUB_RANK = None if sys.argv[1] == "none" else int(sys.argv[1])
M.MoE.SUB_GAP = float(os.environ.get("SUB_GAP", "inf")); M.MoE.SUB_FREE_RANK = int(os.environ.get("SUB_FREE", "99"))
llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=16)
ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
st, logits, _, _ = llm._prefill(ids, False)
tok = int(np.argmax(np.array(logits[-1])))
ts = []
for i in range(200):
    t = time.perf_counter(); tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1]))); ts.append(time.perf_counter() - t)
ts = np.array(ts) * 1e3
print(f"sub={sys.argv[1]} gap={M.MoE.SUB_GAP} free={M.MoE.SUB_FREE_RANK} spec={sys.argv[2]} subs/tok={M.MoE.sub_stats['subs']/len(ts):.1f}: " + " | ".join(f"tok {a}-{a+39}: med {np.median(ts[a:a+40]):.0f} ms ({1000/np.median(ts[a:a+40]):.1f}/s)" for a in range(0, 200, 40)))
print(f"   overall tokens 20-200: median {np.median(ts[20:]):.1f} ms = {1000/np.median(ts[20:]):.2f} tok/s, mean {np.mean(ts[20:]):.1f} ms = {1000/np.mean(ts[20:]):.2f} tok/s")
