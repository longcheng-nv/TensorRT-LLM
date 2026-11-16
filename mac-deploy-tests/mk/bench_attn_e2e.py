"""End-to-end effect of the split sparse-attention kernel on BS=1 decode, compute path only.

Fixed-expert mode (routing overridden to experts 0-5, all resident: no SSD reads), so the per-token time
is GPU work + host syncs. On the same decode state, alternates blocks of decode steps with
mk.SPLIT_ATTN on (two-dispatch kernel) and off (previous single-dispatch kernel), for a short prompt and
a ~4K-token prompt (top-512 compressed keys active). Numerics are meaningless in fixed mode.

usage: bench_attn_e2e.py [--ctx 4096] [--steps 24] [--rounds 3]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    M.MoE.SPEC_K = 0
    M.MoE.FIXED_EXPERTS = list(range(6))
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=16)
    short_ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
    root = "/Users/loncheng/Models/DeepInfer-Mac"
    text = "".join(open(f"{root}/{p}").read() for p in ("README.md", "deepinfer_mac/model/model.py", "deepinfer_mac/engine.py"))
    body = llm.tokenizer.decode(llm.tokenizer.encode(text, add_special_tokens=False).ids[: a.ctx])
    long_ids, _ = llm.chat_prompt([{"role": "user", "content": "Summarize the following code and notes:\n\n" + body}])
    for name, ids in (("short", short_ids), ("long", long_ids)):
        llm.reset()
        st, logits, _, _ = llm._prefill(ids, False)
        tok = int(np.argmax(np.array(logits[-1])))
        for _ in range(4):
            tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
        res = {True: [], False: []}
        for _ in range(a.rounds):
            for split in (True, False):
                mk.SPLIT_ATTN = split
                for i in range(a.steps):
                    t = time.perf_counter()
                    tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
                    if i >= 2:  # first steps after a switch absorb kernel-cache effects
                        res[split].append(time.perf_counter() - t)
        mk.SPLIT_ATTN = True
        new, old = np.median(res[True]) * 1e3, np.median(res[False]) * 1e3
        print(f"[{name}] prompt {len(ids)} tok | fixed-mode decode: split kernel {new:.1f} ms/token, old kernel {old:.1f} ms/token "
              f"-> {old - new:+.1f} ms/token saved ({len(res[True])} + {len(res[False])} steps)", flush=True)


if __name__ == "__main__":
    main()
