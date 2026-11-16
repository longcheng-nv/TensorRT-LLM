"""Where does real-mode decode wall time go? Per MoE layer, host timestamps at:
  enter -> synced (routing ids on host: includes GPU running everything queued since the last sync)
  -> routed (substitution, cancel, prefetch/speculation issue) -> hits_submitted (async_eval)
  -> misses_ready (blocking wait for SSD) -> exit; and exit -> next enter (Python building the next
  layer's graph while the GPU sits idle). Reports ms/token per phase, median over tokens.
usage: mk_timeline.py [--cache-gb 16] [--sub-rank R] [--spec-k K] [--tokens 24]"""
from __future__ import annotations

import argparse
import collections
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

T = []


def main():
    import os
    if os.environ.get("SWITCH"):
        sys.setswitchinterval(float(os.environ["SWITCH"]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-gb", type=float, default=16.0)
    ap.add_argument("--sub-rank", type=int, default=None)
    ap.add_argument("--spec-k2", type=int, default=0)
    ap.add_argument("--spec-k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--fixed", action="store_true")
    a = ap.parse_args()
    M.MoE.SPEC_K = a.spec_k
    M.MoE.SPEC_K2 = a.spec_k2
    M.MoE.SUB_RANK = a.sub_rank
    if a.fixed:
        M.MoE.SPEC_K = 0
        M.MoE.FIXED_EXPERTS = list(range(6))
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=a.cache_gb)
    store = llm.weights.experts
    ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))

    # instrumentation
    orig_call, orig_slab, orig_into = M.MoE.__call__, M.MoE._slab_moe_mk, M.MoE._experts_into
    orig_get = store.get_slots
    cur = {}

    def rec(tag):
        cur.setdefault(tag, []).append(time.perf_counter())

    def call(self, x, image_mask=None):
        if x.shape[0] == 1:
            rec("enter")
        out = orig_call(self, x, image_mask)
        if x.shape[0] == 1:
            rec("exit")
        return out

    def slab(self, x, xq, w, idx_np, y, lim):
        rec("synced_routed")  # routing ids are on host and substitution is done by now
        return orig_slab(self, x, xq, w, idx_np, y, lim)

    calls = {"n": 0}

    def into(self, y, x, xq, w, idx_np, group):
        rec("into_enter")
        out = orig_into(self, y, x, xq, w, idx_np, group)
        rec("into_exit")
        return out

    def get_slots(key, ids_):
        rec("get_enter")
        r = orig_get(key, ids_)
        rec("get_exit")
        return r

    M.MoE.__call__, M.MoE._slab_moe_mk, M.MoE._experts_into = call, slab, into
    store.get_slots = get_slots
    # also time the routing sync itself
    orig_eval = mx.eval

    for _ in range(4):
        lg = llm.model.forward(st, [tok])
        tok = int(np.argmax(np.array(lg[-1])))
    per_tok = collections.defaultdict(list)
    tok_ms = []
    for _ in range(a.tokens):
        cur.clear()
        t0 = time.perf_counter()
        lg = llm.model.forward(st, [tok])
        t1 = time.perf_counter()
        tok = int(np.argmax(np.array(lg[-1])))
        t2 = time.perf_counter()
        tok_ms.append((t2 - t0) * 1e3)
        en, ex, sr = cur["enter"], cur["exit"], cur["synced_routed"]
        ge, gx = cur.get("get_enter", []), cur.get("get_exit", [])
        ie, ix = cur.get("into_enter", []), cur.get("into_exit", [])
        n = len(en)
        # phases
        per_tok["a: enter->routed (GPU queue drain + routing sync + substitution/spec issue)"].append(sum(sr[i] - en[i] for i in range(n)) * 1e3)
        per_tok["b: routed->exit (slots, hits submit, SSD wait, misses submit)"].append(sum(ex[i] - sr[i] for i in range(n)) * 1e3)
        per_tok["   of which get_slots (incl. SSD wait)"].append(sum(gx[i] - ge[i] for i in range(len(ge))) * 1e3)

        per_tok["c: exit->next enter (python builds hc+attention graph, lazily)"].append(sum(en[i + 1] - ex[i] for i in range(n - 1)) * 1e3)
        per_tok["d: forward start->first enter + last exit->return"].append(((en[0] - t0) + (t1 - ex[-1])) * 1e3)
        per_tok["e: logits sync (GPU tail: last misses + head)"].append((t2 - t1) * 1e3)
    print(f"cache {a.cache_gb} sub {a.sub_rank} spec {a.spec_k} fixed {a.fixed}: median {np.median(tok_ms):.1f} ms/token")
    for k, v in per_tok.items():
        print(f"  {k:70s} {np.median(v):7.1f} ms")
    s = store.stats
    print("  store stats:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
