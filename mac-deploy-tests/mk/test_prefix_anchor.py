"""Shared-prefix anchors (LLM._from_anchor) on the real model, with the recorded dsh system prompt.

Session A (the recorded first request) and a second session A2 start with the same system prompt: the second
occurrence makes the anchor; a new session B -- same system prompt and tools, another user message -- then restores it. B is also prefilled cold (anchors off) and exactly (bounded replay off: every layer
sees every token, the reference). Prints the times, how close each path's logits are to the exact ones, and the
greedy tokens of each path (lossless decode).

usage: test_prefix_anchor.py [CAPTURE.jsonl] [--cache-gb 8] [--gen 24]
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
QUESTIONS = ["17 乘以 23 等于多少？只回答数字。", "当前目录下有哪些文件？"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", default=CAPTURE)
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--gen", type=int, default=24)
    a = ap.parse_args()

    from deepinfer_mac import LLM
    from deepinfer_mac.__main__ import build_parser, decode_mode

    print(decode_mode(build_parser().parse_args(["bench", "--cache-gb", str(a.cache_gb), "--lossless"])), flush=True)
    llm = LLM(MODEL, expert_cache_gb=a.cache_gb)
    tk, A = llm.tokenizer, [int(t) for t in json.loads(open(a.capture).readline())["prompt"]]
    end = A.index(llm._user_id) + 1
    tail = tk.decode(A[end:], skip_special_tokens=False)
    assert tk.encode(tail, add_special_tokens=False).ids == A[end:], "the tail does not round-trip"
    user_a = tail.split("\n\n<system-reminder>", 1)[0]

    def prompt_for(q):
        return A[:end] + tk.encode(tail.replace(user_a, q, 1), add_special_tokens=False).ids

    def greedy(st, logits, n):
        out, lg = [], logits
        for _ in range(n):
            t = int(np.argmax(np.array(lg[-1])))
            out.append(t)
            lg = llm.model.forward(st, [t])
        return out

    def run(prompt, label):
        t = time.perf_counter()
        st, logits, _, reused = llm._prefill(prompt, False)
        lg = np.array(logits[-1], dtype=np.float64)
        dt = time.perf_counter() - t
        toks = greedy(st, logits, a.gen)
        llm._state = st
        print(f"  {label:28s} {dt:6.2f} s  reused {reused:5d}  {tk.decode(toks)!r}", flush=True)
        return lg, toks, dt

    ok = True
    print(f"anchor = {end} tokens (system prompt + tool definitions + <｜User｜>); A has {len(A)} tokens", flush=True)
    for q in QUESTIONS:
        B = prompt_for(q)
        print(f"B: {q!r} ({len(B)} tokens)")
        llm.reset()
        llm.PREFIX_ANCHOR, llm.bounded_replay = False, False
        exact = run(B, "exact (no bounded replay)")
        llm.reset()
        llm.bounded_replay = True
        cold = run(B, "cold, no anchor")
        llm.reset()
        llm.PREFIX_ANCHOR = True
        run(A, "A: first time seen")
        made = run(prompt_for("你好"), "A2: makes the anchor")
        anch = run(B, "B: restores the anchor")

        def cos(x, y):
            return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))

        for name, r in (("cold", cold), ("anchor", anch)):
            print(f"    {name:6s} vs exact: cos {cos(r[0], exact[0]):.6f}, max |diff| {np.abs(r[0] - exact[0]).max():.3f}, "
                  f"top-1 same {int(r[0].argmax()) == int(exact[0].argmax())}, greedy tokens same {r[1] == exact[1]}")
        print(f"    time: cold {cold[2]:.1f} s -> with the anchor {anch[2]:.1f} s ({(anch[2] / cold[2] - 1) * 100:+.0f}%); "
              f"the session that made it: {made[2]:.1f} s")
        ok &= int(anch[0].argmax()) == int(exact[0].argmax())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
