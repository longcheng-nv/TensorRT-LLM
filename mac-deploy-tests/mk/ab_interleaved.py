"""Per-token interleaved A/B of decode-loop switches in ONE process.

Separate benchmark processes differ by +-3-5% run to run (cache history, background load, GPU clock),
which hides 1-3% effects. Here the arms take turns token by token (round robin, offset per prompt), so
slow drifts hit every arm alike; the paired difference of neighbouring tokens gives a tight estimate.

usage: ab_interleaved.py fast|lossless N_TOKENS ARM [ARM ...]
  ARM = comma list of io, engram, defer, tick, split ("base" = none); e.g. base io,engram,tick,defer
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

mode, N, ARMS = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
M.MoE.SUB_RANK, M.MoE.SPEC_K = (2, 2) if mode == "fast" else (None, 4)
SKIP = 20
PROMPTS = ["Write a long paragraph about the history of tea.",
           "Write a Python function that parses a CSV file and computes per-column statistics (mean, median, std), with comments and a usage example.",
           "用三段话解释量子计算的基本原理，并举一个实际应用的例子。",
           "Solve step by step: a train travels 120 km at 60 km/h and then 180 km at 90 km/h. What is its average speed for the whole trip?",
           "Explain how a hash map works internally, including collision handling and resizing, for a junior engineer.",
           "写一篇关于秋天的短文，大约三百字。"]


def setting(s: str):
    on = set(s.split(",")) - {"base"}
    assert on <= {"io", "engram", "defer", "tick", "split"}, s
    M.MoE.IO_FIRST = "io" in on
    M.Transformer.ENGRAM_EARLY = "engram" in on
    M.Transformer.DEFER_MISS = "defer" in on
    mk.HOT_TICK_US = 60 if "tick" in on else 0
    M.MoE.SPLIT_SHARED = "split" in on


llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=16)
rec = []  # (prompt, step, arm, ms)
t_start = time.time()
for pi, p in enumerate(PROMPTS):
    setting(ARMS[0])
    llm.reset()
    ids, _ = llm.chat_prompt([{"role": "user", "content": p}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    for i in range(N):
        arm = (i + pi) % len(ARMS)
        setting(ARMS[arm])
        t = time.perf_counter()
        tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
        rec.append((pi, i, arm, (time.perf_counter() - t) * 1e3))
        if tok == 1:
            break
    r = [x for x in rec if x[0] == pi and x[1] >= SKIP]
    print(f"  prompt {pi} ({len([x for x in rec if x[0] == pi])} tok): " + "  ".join(
        f"{ARMS[a]} {np.mean([x[3] for x in r if x[2] == a]):6.1f}" for a in range(len(ARMS))), flush=True)

R = np.array([x for x in rec if x[1] >= SKIP])
print(f"{mode}: {len(R)} tokens after the first {SKIP} of each prompt, {time.time() - t_start:.0f} s")
for a in range(len(ARMS)):
    v = R[R[:, 2] == a, 3]
    print(f"  {ARMS[a]:24s} n {len(v):4d}  mean {v.mean():6.1f} ms ({1000 / v.mean():5.2f} tok/s)  median {np.median(v):6.1f} ms")
# paired: each token of arm a vs the mean of the base-arm tokens within +-len(ARMS) steps of the same prompt
rng = np.random.default_rng(0)
by = {(int(pi), int(i)): (int(a), ms) for pi, i, a, ms in R}
for a in range(1, len(ARMS)):
    d = []
    for (pi, i), (arm, ms) in by.items():
        if arm != a:
            continue
        nb = [by[(pi, j)][1] for j in range(i - len(ARMS), i + len(ARMS) + 1) if (pi, j) in by and by[(pi, j)][0] == 0]
        if nb:
            d.append(ms - np.mean(nb))
    d = np.array(d)
    boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    print(f"  {ARMS[a]:24s} - {ARMS[0]}: {d.mean():+6.2f} ms/token (95% CI {np.percentile(boot, 2.5):+.2f} .. {np.percentile(boot, 97.5):+.2f}), n {len(d)}")
