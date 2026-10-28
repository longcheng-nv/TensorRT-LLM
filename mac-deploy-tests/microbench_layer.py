"""Per-stage GPU cost of ONE decode step, with the SSD out of the picture: runs a real prompt once
(so caches/state exist), then re-times each stage of every layer's decode path on the live state
with mx.eval after each stage (dispatch + GPU time per stage), and separately times the whole
40-layer forward for one token with experts already resident (hit rate ~1) by replaying the same
token sequence. Prints ms/token per stage and the resident-expert forward time.

Run from the DeepInfer-Mac repo root:
  SPEC_K=3 env -u HTTP_PROXY -u HTTPS_PROXY .venv/bin/python /Users/loncheng/Models/dsv41-mac-tests/microbench_layer.py [--cache-gb 18]
"""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-gb", type=float, default=18.0)
    ap.add_argument("--model", default="/Users/loncheng/Models/dsv41-mac")
    a = ap.parse_args()
    M.MoE.SPEC_K = 3
    llm = LLM(a.model, expert_cache_gb=a.cache_gb)
    ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))

    # 1) warm: decode 8 tokens normally
    toks = []
    for _ in range(8):
        lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1]))); toks.append(tok)

    # 2) stage timing with a sync after every stage (upper bound: includes dispatch + GPU, no overlap)
    T = collections.defaultdict(float)

    def wrap(obj, name, tag):
        f = getattr(obj, name)

        def g(*args, **kw):
            t = time.perf_counter(); out = f(*args, **kw)
            if out is not None:
                mx.eval(out if isinstance(out, (mx.array, list)) else [v for v in out if isinstance(v, mx.array)])
            T[tag] += time.perf_counter() - t
            return out
        setattr(obj, name, g)

    for b in llm.model.layers:
        at = b.attn; at._compiled()
        wrap(at, "_proj_c", "attn.proj (wq_a,wq_b,wkv,rope,act_quant)")
        wrap(at, "_core_c", "attn.core (scores,softmax,wo_a,wo_b)")
        if at.is_kv_source: wrap(at, "_compress", "attn.compressor")
        if at.is_index_source: wrap(at, "_indexer", "attn.indexer")
        wrap(b.ffn, "_slab_moe", "moe.experts (get+gather_qmm, incl SSD wait)")
        if not hasattr(b.ffn, "_rs_c"): b.ffn._rs_c = mx.compile(b.ffn._route_shared)
        wrap(b.ffn, "_rs_c", "moe.route+shared")
        if b.engram is not None: wrap(b.engram, "__call__", "engram")
    n = 6
    io0 = dict(llm.weights.experts.stats); t0 = time.perf_counter()
    for _ in range(n):
        lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))
    total = (time.perf_counter() - t0) / n
    io = llm.weights.experts.stats
    print(f"\n[staged, sync after each stage] total {total*1000:.0f} ms/token; SSD wait {(io['wait_s']-io0['wait_s'])/n*1000:.0f} ms/token; {(io['bytes']-io0['bytes'])/n/1e9:.2f} GB/token")
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"   {k:48s} {v/n*1000:7.1f} ms/token")
    other = total - sum(T.values()) / n
    print(f"   {'other (Block fused chains, hc_post, embed/head, Python)':48s} {other*1000:7.1f} ms/token")

    # 3) resident-expert forward: replay the same tokens so every expert is already in the slab cache
    for b in llm.model.layers:  # unwrap
        at = b.attn
        for name in ("_proj_c", "_core_c", "_compress", "_indexer"):
            if name in at.__dict__: del at.__dict__[name]
        for name in ("_slab_moe", "_rs_c"):
            if name in b.ffn.__dict__: del b.ffn.__dict__[name]
        if b.engram is not None and "__call__" in b.engram.__dict__: del b.engram.__dict__["__call__"]
    st2 = llm.model.new_state()
    llm.model.forward(st2, ids)  # prefill into fresh state (experts for ids mostly cached now)
    seq = list(toks[:6])
    for t in seq: llm.model.forward(st2, [t])      # pass 1: fills cache for these tokens
    st3 = llm.model.new_state(); llm.model.forward(st3, ids)
    io0 = dict(llm.weights.experts.stats); t0 = time.perf_counter()
    for t in seq:
        mx.eval(llm.model.forward(st3, [t]))
    dt = (time.perf_counter() - t0) / len(seq); io = llm.weights.experts.stats
    hit = (io['hits']-io0['hits']) / max(1, (io['hits']-io0['hits']) + (io['misses']-io0['misses']))
    print(f"\n[resident replay] {dt*1000:.0f} ms/token with expert hit rate {hit:.2f}, SSD wait {(io['wait_s']-io0['wait_s'])/len(seq)*1000:.0f} ms/token  <- compute-side floor of the current code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
