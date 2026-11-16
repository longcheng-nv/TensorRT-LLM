"""forward_mk (fused decode path) vs the reference forward on real decode steps: per step, run the
reference from a state snapshot, restore, run the fused path, compare logits; continue greedily
from the fused path. usage: test_mk_forward.py [--steps 12] [--cache-gb 16] [--fixed]"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cache-gb", type=float, default=16.0)
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--prompt", default="Write a long paragraph about the history of tea.")
    a = ap.parse_args()
    M.MoE.SPEC_K = 0
    if a.fixed:
        M.MoE.FIXED_EXPERTS = list(range(6))
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=a.cache_gb)
    ids, _ = llm.chat_prompt([{"role": "user", "content": a.prompt}])
    M.Transformer.MK = False
    M.MoE.MK = False
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    agree, cosines, maxrel = 0, [], []
    out = [tok]
    for step in range(a.steps):
        snap = st.snapshot()
        M.Transformer.MK = False
        M.MoE.MK = False
        lr = np.array(llm.model.forward(st, [tok])[-1])
        st.restore(snap)
        M.Transformer.MK = True
        M.MoE.MK = True
        lm = np.array(llm.model.forward(st, [tok])[-1])
        cos = float(np.dot(lr, lm) / (np.linalg.norm(lr) * np.linalg.norm(lm)))
        rel = float(np.abs(lr - lm).max() / np.abs(lr).max())
        same = int(np.argmax(lr)) == int(np.argmax(lm))
        agree += same
        cosines.append(cos)
        maxrel.append(rel)
        p_r = np.exp(lr - lr.max()); p_r /= p_r.sum()
        p_m = np.exp(lm - lm.max()); p_m /= p_m.sum()
        kl = float(np.sum(p_r * (np.log(p_r + 1e-30) - np.log(p_m + 1e-30))))
        print(f"step {step}: logits cos {cos:.7f} max rel {rel:.2e} KL {kl:.2e} argmax {'same' if same else 'DIFF'}", flush=True)
        tok = int(np.argmax(lm))
        out.append(tok)
    print(f"argmax agreement {agree}/{a.steps}; min cos {min(cosines):.7f}")
    if not a.fixed:
        print("text:", llm.tokenizer.decode(out))


if __name__ == "__main__":
    main()
