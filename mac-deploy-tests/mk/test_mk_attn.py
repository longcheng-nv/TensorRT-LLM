"""Attention._decode_mk vs Attention.__call__ (reference) on real decode states: per layer, run the
reference attention from a state snapshot, restore, run the fused one, compare outputs.
usage: test_mk_attn.py [--long]  (--long: a ~1.7K-token prompt so the indexer's top-k is exercised)"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

M.MoE.SPEC_K = 0
llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=8)
if "--long" in sys.argv:
    doc = open("/Users/loncheng/Models/DeepInfer-Mac/README.md").read()[:6000]
    ids, _ = llm.chat_prompt([{"role": "user", "content": "Summarize:\n\n" + doc}])
else:
    ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
M.Attention.MK = False
st, logits, _, _ = llm._prefill(ids, False)
tok = int(np.argmax(np.array(logits[-1])))
for _ in range(3):  # a few plain decode steps so the ring / compressed state is realistic
    tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
# capture each layer's attention input for one decode step
cap = []
orig = M.Attention.__call__


def call(self, x, positions, st_, ring, sh, start, floor=0):
    snap = (ring.copy(), st_.snapshot(), (sh.comp, sh.index_k, sh.topk, sh.candidates))
    M.Attention.MK = False
    out_r = orig(self, x, positions, st_, ring, sh, start, floor)
    mx.eval(out_r)
    ring_r = (np.array(ring.kv.astype(mx.float32)), ring.pos.copy())
    # restore and run the fused path on the same inputs
    r0, s0, sh0 = snap
    ring.kv, ring.pos = r0.kv, r0.pos
    st_.restore(s0)
    sh.comp, sh.index_k, sh.topk, sh.candidates = sh0
    M.Attention.MK = True
    out_m = orig(self, x, positions, st_, ring, sh, start, floor)
    mx.eval(out_m)
    M.Attention.MK = False
    ring_m = (np.array(ring.kv.astype(mx.float32)), ring.pos.copy())
    a, b = np.array(out_r.astype(mx.float32)).ravel(), np.array(out_m.astype(mx.float32)).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-30))
    cap.append((self.layer_id, cos, rel, float((a == b).mean()), bool((ring_r[0] == ring_m[0]).all() and (ring_r[1] == ring_m[1]).all())))
    return out_r


M.Attention.__call__ = call
llm.model.forward(st, [tok])
M.Attention.__call__ = orig
worst = min(c[1] for c in cap)
for lid, cos, rel, ex, ringok in cap:
    if cos < 0.9999999 or not ringok or ex < 0.999:
        print(f"layer {lid:2d}: out cos {cos:.7f} max rel {rel:.1e} exact {ex:.3f} ring {'same' if ringok else 'DIFF'}")
print(f"prompt {len(ids)} tok | worst cos {worst:.7f} | rings identical: {all(c[4] for c in cap)} -> {'OK' if worst > 0.9999 and all(c[4] for c in cap) else 'CHECK'}")
# component check on one layer: qr / kv / q of the fused projections vs the reference _proj
at = llm.model.layers[8].attn
x = mx.random.normal((1, 5120), key=mx.random.key(7)).astype(mx.bfloat16)
positions = mx.array([st.pos])
cos_, sin_ = at.rope.cos_sin(positions)
qr_r, q_r, kv_r = at._proj(x, cos_, sin_)
from deepinfer_mac.model import mk  # noqa: E402
cs = mx.concatenate([cos_[0], sin_[0]])
eps = mx.array(at.a.norm_eps, dtype=mx.float32)
qakv = mk.gemv_mxfp8(x, at.wq_a_kv.w, at.wq_a_kv.scales, pro=1, rs=8, sgs=1, split=4)
qa_r, kvr_r = at.wq_a_kv(x)
qr_m, kv_m = mk.attn_aux(qakv, at.q_norm, at.kv_norm, eps, cs)
q_m = mk.gemv_mxfp8(qr_m, at.wq_b.w, at.wq_b.scales, pro=1, rs=8, sgs=2, rope_cs=cs)
for name, r, m_ in (("qa|kvr", mx.concatenate([qa_r[0], kvr_r[0]]), qakv), ("qr", qr_r[0], qr_m), ("kv", kv_r[0], kv_m), ("q", q_r.reshape(-1), q_m)):
    a_, b_ = np.array(r.astype(mx.float32)).ravel(), np.array(m_.astype(mx.float32)).ravel()
    print(f"  {name:6s}: exact {(a_ == b_).mean():.4f} max rel {np.abs(a_ - b_).max() / np.abs(a_).max():.1e}")
