import sys, time, os, json, numpy as np
sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx
from deepinfer_mac import LLM
from deepinfer_mac.model import model as M
sub = None if sys.argv[1] == "none" else int(sys.argv[1]); spec = int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
M.MoE.SPEC_K = spec; M.MoE.SUB_RANK = sub
M.MoE.SUB_TARGET_MS = float(os.environ.get("TARGET", "0")); M.MoE.SUB_RANK_MILD = None if os.environ.get("MILD", "none") == "none" else int(os.environ["MILD"])
llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=16)
prompts = ["Write a long paragraph about the history of tea.",
           "Write a Python function that parses a CSV file and computes per-column statistics (mean, median, std), with comments and a usage example.",
           "用三段话解释量子计算的基本原理，并举一个实际应用的例子。",
           "Solve step by step: a train travels 120 km at 60 km/h and then 180 km at 90 km/h. What is its average speed for the whole trip?"]
allts = []
for p in prompts:
    llm.reset()
    ids, _ = llm.chat_prompt([{"role": "user", "content": p}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1]))); ts = []; out = [tok]
    for i in range(n):
        t = time.perf_counter(); tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1]))); ts.append(time.perf_counter() - t); out.append(tok)
        if tok == 1: break
    ts = np.array(ts[20:]) * 1e3
    allts += ts.tolist()
    print(f"  {p[:28]!r:32s} {len(ts)+20:3d} tok: median {np.median(ts):6.1f} ms ({1000/np.median(ts):5.2f} tok/s), mean {np.mean(ts):6.1f} ms ({1000/np.mean(ts):5.2f} tok/s) | {llm.tokenizer.decode(out[:16])!r}", flush=True)
a = np.array(allts)
s_ = M.MoE.sub_stats; print(f"   subs/token {s_['subs']/max(1,len(allts)+80):.1f}, mild layers {s_.get('mild_layers',0)/max(1,len(allts)+80):.1f}/token")
print(f"ALL sub={sub} target={M.MoE.SUB_TARGET_MS} mild={M.MoE.SUB_RANK_MILD} spec={spec}: median {np.median(a):.1f} ms = {1000/np.median(a):.2f} tok/s, mean {np.mean(a):.1f} ms = {1000/np.mean(a):.2f} tok/s (tokens 20+ of each prompt)")
