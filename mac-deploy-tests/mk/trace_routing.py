"""Record real routing traces (BS=1 decode, greedy) for offline cache / prefetch / substitution
simulation. For every decode token and layer: the router's raw scores for all 384 experts (fp16),
and the next layer's router applied to this layer's MoE input (what SPEC_K speculation uses).
Prefill: the selected top-k per prompt row and layer (to seed the simulated cache).

usage: trace_routing.py OUT.npz [--tokens 160] [--cache-gb 16]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

PROMPTS = [
    "Write a long paragraph about the history of tea.",
    "用三段话解释量子计算的基本原理，并举一个实际应用的例子。",
    "Write a Python function that parses a CSV file and computes per-column statistics (mean, median, std), with comments and a small usage example.",
    "Solve step by step: a train travels 120 km at 60 km/h and then 180 km at 90 km/h. What is its average speed for the whole trip?",
    "Give me a detailed 3-day travel plan for Kyoto in autumn, including food recommendations.",
    "LONGDOC",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--cache-gb", type=float, default=16.0)
    a = ap.parse_args()
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=a.cache_gb)
    layers = llm.model.layers
    nL = len(layers)
    rec = {"phase": None}
    buf_scores, buf_pred, buf_pref = [], [], []

    def make_wrapper(li, ffn):
        orig = ffn.__call__.__func__ if hasattr(ffn.__call__, "__func__") else None

        def call(x, image_mask=None):
            xf = x.astype(mx.float32)
            sc = mx.sqrt(mx.logaddexp(xf @ ffn.gate_w.T, 0.0))
            if x.shape[0] == 1 and rec.get("tok", -1) >= 0:
                nxt = ffn.next_moe
                pr = mx.sqrt(mx.logaddexp(xf @ nxt.gate_w.T, 0.0)) if nxt is not None else mx.zeros_like(sc)
                mx.eval(sc, pr)
                buf_scores.append((rec["tok"], li, np.array(sc[0]).astype(np.float16)))
                buf_pred.append((rec["tok"], li, np.array(pr[0]).astype(np.float16)))
            else:
                b = sc + ffn.bias
                top = mx.argpartition(-b, kth=ffn.topk - 1, axis=-1)[:, : ffn.topk]
                mx.eval(top)
                buf_pref.append((rec["prompt"], li, np.array(top).astype(np.int16)))
            return M.MoE.__call__(ffn, x, image_mask)

        return call

    for li, b in enumerate(layers):
        b.ffn.__call__ = make_wrapper(li, b.ffn)  # instance attribute shadows the method for direct calls
        # Block calls self.ffn(h, image_mask) -> python looks up type(self.ffn).__call__, so patch via a subclass
    for li, b in enumerate(layers):
        f = b.ffn
        cls = type("TracedMoE", (M.MoE,), {"__call__": lambda self, x, image_mask=None, _w=f.__call__: _w(x, image_mask)})
        f.__class__ = cls

    biases = np.stack([np.array(b.ffn.bias) for b in layers]).astype(np.float32)
    all_tokens, meta = [], []
    tok_counter = 0
    longdoc = open("/Users/loncheng/Models/DeepInfer-Mac/README.md").read()[:6000]
    for pi, p in enumerate(PROMPTS):
        text = p if p != "LONGDOC" else "Summarize the following document in 5 bullet points:\n\n" + longdoc
        llm.reset()
        ids, _ = llm.chat_prompt([{"role": "user", "content": text}])
        rec["prompt"] = pi
        rec["tok"] = -1
        t0 = time.perf_counter()
        st, logits, _, _ = llm._prefill(ids, False)
        tok = int(np.argmax(np.array(logits[-1])))
        gen = [tok]
        t1 = time.perf_counter()
        for i in range(a.tokens):
            rec["tok"] = tok_counter
            meta.append((pi, i))
            tok_counter += 1
            lg = llm.model.forward(st, [tok])
            tok = int(np.argmax(np.array(lg[-1])))
            gen.append(tok)
        t2 = time.perf_counter()
        rec["tok"] = -1
        all_tokens.append((ids, gen))
        print(f"prompt {pi}: {len(ids)} prompt tok, prefill {t1-t0:.1f}s, decode {a.tokens} tok {t2-t1:.1f}s "
              f"({a.tokens/(t2-t1):.2f} tok/s) | {llm.tokenizer.decode(gen[:40])!r}", flush=True)

    T = tok_counter
    S = np.zeros((T, nL, 384), dtype=np.float16)
    P = np.zeros((T, nL, 384), dtype=np.float16)
    for t, li, v in buf_scores:
        S[t, li] = v
    for t, li, v in buf_pred:
        P[t, li] = v
    pref = {}
    for pi, li, top in buf_pref:
        pref.setdefault((pi, li), []).append(top)
    pref_arrays = {f"pref_{pi}_{li}": np.concatenate(v) for (pi, li), v in pref.items()}
    np.savez_compressed(
        a.out, scores=S, pred=P, biases=biases, meta=np.array(meta, dtype=np.int32),
        topk=np.int32(6), n_prompts=np.int32(len(PROMPTS)),
        tokens=np.array([np.array(g, dtype=np.int32) for _, g in all_tokens], dtype=object),
        prompts=np.array([np.array(i, dtype=np.int32) for i, _ in all_tokens], dtype=object),
        **pref_arrays,
    )
    print("saved", a.out, S.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
