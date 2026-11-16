"""Megakernel-effort decode benchmark (BS=1). Two modes:

  real   : normal routing, real SSD streaming (end-to-end tok/s)
  fixed  : every layer's router output is overridden to a fixed expert set (all resident after the
           first token, SPEC_K forced 0) -> zero SSD traffic. Measures the compute path alone:
           GPU work + host-sync bubbles + Python overhead. Numerics are meaningless in this mode.

usage: mk_bench.py [--mode real|fixed] [--cache-gb 16] [--tokens 24] [--warm 4] [--prompt TEXT]
Prints one JSON line (also appended to --out if given).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.environ.get("DEEPINFER_REPO", "/Users/loncheng/Models/DeepInfer-Mac")
sys.path.insert(0, REPO)
import mlx.core as mx  # noqa: E402

from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="real", choices=["real", "fixed"])
    ap.add_argument("--cache-gb", type=float, default=16.0)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--spec-k2", type=int, default=0)
    ap.add_argument("--spec-k", type=int, default=None)
    ap.add_argument("--prompt", default="Write a long paragraph about the history of tea.")
    ap.add_argument("--model", default="/Users/loncheng/Models/dsv41-mac")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--sub-rank", type=int, default=None)
    ap.add_argument("--sub-gap", type=float, default=float("inf"))
    ap.add_argument("--sub-free", type=int, default=99)
    ap.add_argument("--mk", type=int, default=1)
    a = ap.parse_args()

    if a.spec_k is not None:
        M.MoE.SPEC_K = a.spec_k
    M.MoE.SPEC_K2 = a.spec_k2
    M.MoE.SUB_RANK = a.sub_rank
    M.MoE.SUB_GAP = a.sub_gap
    M.MoE.SUB_FREE_RANK = a.sub_free
    M.Transformer.MK = bool(a.mk)
    if a.mode == "fixed":
        M.MoE.SPEC_K = 0
        M.MoE.FIXED_EXPERTS = list(range(6))
    t_load = time.perf_counter()
    llm = LLM(a.model, expert_cache_gb=a.cache_gb)
    t_load = time.perf_counter() - t_load
    ids, _ = llm.chat_prompt([{"role": "user", "content": a.prompt}])
    t0 = time.perf_counter()
    st, logits, _, _ = llm._prefill(ids, False)
    mx.eval(logits)
    ttft = time.perf_counter() - t0
    tok = int(np.argmax(np.array(logits[-1])))
    store = llm.weights.experts
    ts, toks = [], [tok]
    s0 = None
    for i in range(a.warm + a.tokens):
        if i == a.warm:
            s0 = dict(store.stats)
            sub0 = M.MoE.sub_stats["subs"]
        t = time.perf_counter()
        lg = llm.model.forward(st, [tok])
        tok = int(np.argmax(np.array(lg[-1])))
        dt = time.perf_counter() - t
        toks.append(tok)
        if i >= a.warm:
            ts.append(dt)
    s = store.stats
    n = a.tokens
    subs = M.MoE.sub_stats["subs"] - sub0
    acc = (s["hits"] - s0["hits"]) + (s["misses"] - s0["misses"]) + (s["prefetch_hits"] - s0["prefetch_hits"])
    rec = {
        "label": a.label, "mode": a.mode, "cache_gb": a.cache_gb, "spec_k": M.MoE.SPEC_K, "spec_k2": M.MoE.SPEC_K2,
        "prompt_tokens": len(ids), "ttft_s": round(ttft, 2), "load_s": round(t_load, 1),
        "ms_median": round(float(np.median(ts)) * 1e3, 1), "ms_mean": round(float(np.mean(ts)) * 1e3, 1),
        "ms_p10": round(float(np.percentile(ts, 10)) * 1e3, 1), "ms_p90": round(float(np.percentile(ts, 90)) * 1e3, 1),
        "tok_s_median": round(1.0 / float(np.median(ts)), 2),
        "hit_rate": round(((s["hits"] - s0["hits"]) + (s["prefetch_hits"] - s0["prefetch_hits"])) / max(1, acc), 3),
        "GB_per_token": round((s["bytes"] - s0["bytes"]) / n / 1e9, 3),
        "spec_GB_per_token": round((s.get("spec_bytes", 0) - s0.get("spec_bytes", 0)) / n / 1e9, 3),
        "ssd_wait_ms": round((s["wait_s"] - s0["wait_s"]) / n * 1e3, 1),
        "sub_rank": a.sub_rank, "sub_gap": a.sub_gap, "sub_free": a.sub_free, "subs_per_tok": round(subs / n, 1),
        "text": llm.tokenizer.decode(toks[: min(len(toks), 24)]) if a.mode == "real" else "",
    }
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
