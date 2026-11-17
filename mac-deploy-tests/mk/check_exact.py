"""Lossless decode must not change when the decode loop is only reordered: greedy-decode the same
prompts under two switch settings in one process and compare the tokens and the logits bit for bit
(exact routing: no substitution). Settings are comma lists of: io (MoE.IO_FIRST), engram
(Transformer.ENGRAM_EARLY), defer (Transformer.DEFER_MISS), tick (mk.HOT_TICK_US=60); "off" = none.

usage: check_exact.py N_TOKENS SETTING_A SETTING_B      e.g. check_exact.py 40 off io,engram,tick,defer
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
A = sys.argv[2] if len(sys.argv) > 2 else "off"
B = sys.argv[3] if len(sys.argv) > 3 else "io,engram,tick,defer"
M.MoE.SUB_RANK, M.MoE.SPEC_K = None, 4  # --lossless
import os  # noqa: E402

llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=float(os.environ.get("CACHE_GB", "16")))
PROMPTS = ["Write a long paragraph about the history of tea.", "用三段话解释量子计算的基本原理，并举一个实际应用的例子。"]


def setting(s: str):
    on = set(s.split(",")) - {"off", "base"}
    assert on <= {"io", "engram", "defer", "tick", "split"}, s
    M.MoE.IO_FIRST = "io" in on
    M.Transformer.ENGRAM_EARLY = "engram" in on
    M.Transformer.DEFER_MISS = "defer" in on
    mk.HOT_TICK_US = 60 if "tick" in on else 0
    M.MoE.SPLIT_SHARED = "split" in on


def run(s: str, prompt: str):
    setting(s)
    llm.reset()
    ids, _ = llm.chat_prompt([{"role": "user", "content": prompt}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    toks, lg, ts = [tok], [], []
    for _ in range(N):
        t = time.perf_counter()
        out = np.array(llm.model.forward(st, [tok])[-1], dtype=np.float32)
        ts.append(time.perf_counter() - t)
        lg.append(out)
        tok = int(np.argmax(out))
        toks.append(tok)
    return toks, np.stack(lg), np.median(ts[5:]) * 1e3


ok = True
for p in PROMPTS:
    a_t, a_l, a_ms = run(A, p)
    b_t, b_l, b_ms = run(B, p)
    same_tok = a_t == b_t
    diff = a_l != b_l
    first = int(np.argmax(diff.any(axis=1))) if diff.any() else -1
    rel = float(np.abs(a_l - b_l).max() / np.abs(a_l).max())
    ok &= same_tok and not diff.any()
    print(f"{p[:24]!r:28s} [{A}] vs [{B}]: tokens equal {same_tok}, differing logits {int(diff.sum())} / {a_l.size}, "
          f"first differing step {first}, max |diff| / max |logit| {rel:.2e}; median ms/token {a_ms:.1f} vs {b_ms:.1f}", flush=True)
print("EXACT" if ok else "MISMATCH")
