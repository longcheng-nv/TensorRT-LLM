"""MoE.PREFILL_LOOKAHEAD (prefill cross-layer lookahead) moves reads only: the prefill's logits must be bit-identical
with it on and off. Runs the prompt of a recorded agent request (agent_capture.py) as _prefill chunks it -- a non-final
chunk (the decoder skipped) then a final one (bounded replay) -- alternating off / on, compares every logit bit for
bit and prints the times and bytes read.

usage: test_prefill_lookahead.py [CAPTURE.jsonl] [--tokens 600] [--cache-gb 8] [--k 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
MODEL = "/Users/loncheng/Models/dsv41-mac"
CAPTURE = "/Users/loncheng/Models/logs/mk/agent/capture-20261127.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", default=CAPTURE)
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()

    from deepinfer_mac import LLM
    from deepinfer_mac.model import model as M

    ids = [int(t) for t in json.loads(open(a.capture).readline())["prompt"]][: a.tokens]
    llm = LLM(MODEL, expert_cache_gb=a.cache_gb)
    keep, store = llm.keep_tokens, llm.weights.experts
    cut = len(ids) // 2
    assert len(ids) - cut > keep, "the final chunk must hold more than keep_tokens"

    llm.model.prefilling = True  # as LLM._prefill sets it around its chunks

    def run(k):
        M.MoE.PREFILL_LOOKAHEAD = k
        st = llm.model.new_state()
        b0, t = store.stats["bytes"], time.perf_counter()
        llm.model.forward(st, ids[:cut], replay_from=cut)  # non-final chunk: the decoder replays nothing
        lg = llm.model.forward(st, ids[cut:], replay_from=len(ids) - cut - keep, all_logits=True)
        out = np.array(lg)
        return out, time.perf_counter() - t, (store.stats["bytes"] - b0) / 1e9

    outs, ok = {}, True
    for k in (0, a.k, 0, a.k):
        out, dt, gb = run(k)
        print(f"lookahead {k:>3}: {dt:6.2f} s, {gb:6.1f} GB read, logits {out.shape}", flush=True)
        if k in outs:
            same = np.array_equal(outs[k], out)
            print(f"   repeat of {k}: bit-identical {same}")
            ok &= same
        else:
            outs[k] = out
    same = np.array_equal(outs[0], outs[a.k])
    diff = int(np.count_nonzero(outs[0] != outs[a.k]))
    print(f"off vs on: bit-identical {same} ({diff} differing of {outs[0].size})")
    ok &= same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
