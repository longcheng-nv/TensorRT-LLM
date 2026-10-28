"""Expert-routing coverage on coding prompts: how many of the 384 routed experts per layer carry
99.5% / 99% / 95% of the routing-weight mass, and how many a held-out prompt would miss if the
rest were pruned. Records (layer, token, expert ids, weights) during prefill + short greedy decode.

Run from the DeepInfer-Mac repo root:
  env -u HTTP_PROXY -u HTTPS_PROXY .venv/bin/python /Users/loncheng/Models/dsv41-mac-tests/routing_coverage.py [--cache-gb 9] [--decode 16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402
from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

CODE_PROMPTS = [
    "Implement an LRU cache in Python with O(1) get and put using a doubly linked list and a dict. Include type hints and docstrings.",
    "Write a C++17 function that parses a CSV line with quoted fields and escaped quotes into std::vector<std::string>. Handle edge cases.",
    "Write a CUDA kernel that computes a row-wise softmax for a [M, N] fp32 matrix using warp shuffles, and the host launch code.",
    "In Rust, implement a thread-safe bounded MPSC queue using Mutex and Condvar, with blocking push and pop. Show a small test.",
    "Write a Python function that validates an IPv6 address string according to RFC 4291, including :: compression and embedded IPv4.",
    "Given this buggy JavaScript, find and fix the bug:\n```js\nfunction debounce(fn, ms){ let t; return (...a)=>{ clearTimeout(t); t=setTimeout(fn(...a), ms);} }\n```",
    "Write a SQL query (PostgreSQL) returning each customer's second most recent order date, using window functions, and explain the plan.",
    "Implement Dijkstra's shortest path in Go with a container/heap priority queue over an adjacency list; return distances and predecessors.",
    "Write a PyTorch module implementing rotary position embeddings (RoPE) applied to q and k of shape [B, H, T, D], with caching of cos/sin.",
    "Write a bash script that finds the 10 largest files under a directory, excluding .git, printing sizes in human-readable form; handle spaces in names.",
    "Explain and fix the data race in this Java code:\n```java\nclass Counter { int n; void inc(){ n++; } }\n```\nProvide the corrected class.",
    "Write a Python asyncio HTTP client with bounded concurrency (semaphore), retries with exponential backoff, and per-request timeouts.",
]
HELD_OUT = [
    "Implement a trie in TypeScript with insert, search and startsWith, plus a method returning all words with a given prefix sorted.",
    "Write a NumPy function that computes the 2D convolution of an image with a kernel using stride tricks (no loops), with 'same' padding.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/loncheng/Models/dsv41-mac")
    ap.add_argument("--cache-gb", type=float, default=9.0)
    ap.add_argument("--decode", type=int, default=16)
    a = ap.parse_args()
    llm = LLM(a.model, expert_cache_gb=a.cache_gb)
    n_layers, n_exp = llm.args.n_layers, llm.args.n_routed_experts
    mass = {s: np.zeros((n_layers, n_exp)) for s in ("train", "test")}
    count = {s: np.zeros((n_layers, n_exp)) for s in ("train", "test")}
    tokens = {"train": 0, "test": 0}
    split = {"s": "train"}

    orig = M.MoE.__call__

    def hooked(self, x, image_mask=None):
        w, idx = self.route(x, image_mask)
        wn, ids = np.array(w.astype(mx.float32)), np.array(idx).astype(np.int64)
        layer = int(self.key[1:]) if self.key.startswith("L") else None
        if layer is not None and self.key.startswith("L"):
            s = split["s"]
            share = wn / np.maximum(wn.sum(-1, keepdims=True), 1e-20)
            np.add.at(mass[s][layer], ids.reshape(-1), share.reshape(-1))
            np.add.at(count[s][layer], ids.reshape(-1), 1.0)
        return orig(self, x, image_mask)

    M.MoE.__call__ = hooked
    t0 = time.time()
    for s, prompts in (("train", CODE_PROMPTS), ("test", HELD_OUT)):
        split["s"] = s
        for p in prompts:
            llm.reset()
            ids, _ = llm.chat_prompt([{"role": "user", "content": p}])
            tokens[s] += len(ids) + a.decode
            st, logits, _, _ = llm._prefill(ids, False)
            tok = int(np.argmax(np.array(logits[-1])))
            for _ in range(a.decode):
                lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))
            print(f"  [{s}] {len(ids)} prompt tokens done ({time.time() - t0:.0f}s)", flush=True)

    out = {"tokens": tokens, "layers": []}
    print(f"\nrouting mass over {tokens['train']} train tokens (12 coding prompts), {tokens['test']} held-out tokens")
    print("layer | experts used | K for 95% | 99% | 99.5% | 99.9% | held-out mass outside top-48 | outside top-96 | outside top-192")
    for layer in range(n_layers):
        m = mass["train"][layer]; tot = m.sum()
        if tot == 0:
            continue
        order = np.argsort(-m); cum = np.cumsum(m[order]) / tot
        ks = {q: int(np.searchsorted(cum, q) + 1) for q in (0.95, 0.99, 0.995, 0.999)}
        used = int((m > 0).sum())
        mt = mass["test"][layer]; tt = max(mt.sum(), 1e-20)
        outside = {K: float(mt[order[K:]].sum() / tt) for K in (48, 96, 192)}
        print(f"L{layer:02d}  | {used:4d} | {ks[0.95]:4d} | {ks[0.99]:4d} | {ks[0.995]:4d} | {ks[0.999]:4d} | {outside[48]:.4f} | {outside[96]:.4f} | {outside[192]:.4f}")
        out["layers"].append({"layer": layer, "used": used, "K": {str(k): v for k, v in ks.items()}, "heldout_outside": {str(k): v for k, v in outside.items()}})
    tr = mass["train"].sum(1) > 0
    agg = {q: float(np.mean([l["K"][str(q)] for l in out["layers"]])) for q in (0.95, 0.99, 0.995, 0.999)}
    ho = {K: float(np.mean([l["heldout_outside"][str(K)] for l in out["layers"]])) for K in (48, 96, 192)}
    print(f"\nmean experts/layer for 95/99/99.5/99.9% mass: {agg[0.95]:.0f} / {agg[0.99]:.0f} / {agg[0.995]:.0f} / {agg[0.999]:.0f}  (of {n_exp}; MoE layers {int(tr.sum())})")
    print(f"held-out routing mass lost if pruned to top-48 / 96 / 192 per layer: {ho[48]:.3%} / {ho[96]:.3%} / {ho[192]:.3%}")
    out["summary"] = {"mean_K": {str(k): v for k, v in agg.items()}, "heldout_lost": {str(k): v for k, v in ho.items()}}
    json.dump(out, open("/Users/loncheng/Models/logs/prune/routing_coverage.json", "w"), indent=1)
    np.save("/Users/loncheng/Models/logs/prune/routing_mass_train.npy", mass["train"])
    np.save("/Users/loncheng/Models/logs/prune/routing_mass_test.npy", mass["test"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
