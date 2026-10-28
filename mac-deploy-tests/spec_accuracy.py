"""How well can layer l+1's routed experts be predicted before its attention runs?
(a) current SPEC_K: next layer's router applied to layer l's MoE INPUT (post-attention hidden of l)
(b) proposed: next layer's router applied to layer l+1's BLOCK INPUT (= layer l's output, before attention l+1)
Reports coverage of the true top-6 by the top-K prediction for K in 3..12, averaged over layers/tokens.
Run from the DeepInfer-Mac repo root with the real model.
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402
from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402

M.MoE.SPEC_K = 0
llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=12.0)
rec = {"moe_in": {}, "blk_in": {}, "actual": {}}
orig_moe = M.MoE.__call__
orig_blk = M.Block.__call__

def moe_hook(self, x, image_mask=None):
    if x.shape[0] == 1 and self.key.startswith("L"):
        l = int(self.key[1:])
        w, idx = self.route(x, image_mask)
        rec["actual"].setdefault(l, []).append(np.array(idx)[0].tolist())
        rec["moe_in"].setdefault(l, []).append(np.array(x[-1:].astype(mx.float32)))
    return orig_moe(self, x, image_mask)

def blk_hook(self, x, positions, pre_mix, st, ring, sh, image_mask=None, start=0, floor=0):
    if x.shape[0] == 1:
        # the MoE router input is rms_norm(hc_pre(x_after_attn, attn_pre)); before attention we only
        # have x and pre_mix -> use rms-normed hc_pre(x, pre_mix) as the proxy (what the router would see if attention were identity)
        h = M._rms_f32(M._hc_pre_c(x, pre_mix), self.ffn_norm, mx.array(self.a.norm_eps, dtype=mx.float32))
        rec["blk_in"].setdefault(self.layer_id, []).append(np.array(h.astype(mx.float32)))
    return orig_blk(self, x, positions, pre_mix, st, ring, sh, image_mask, start, floor)

M.MoE.__call__ = moe_hook
M.Block.__call__ = blk_hook
prompts = ["Write a long paragraph about the history of tea.", "Implement an LRU cache in Python with O(1) operations.", "用三段话解释量子计算的基本原理。"]
for p in prompts:
    llm.reset(); ids, _ = llm.chat_prompt([{"role": "user", "content": p}])
    st, logits, _, _ = llm._prefill(ids, False); tok = int(np.argmax(np.array(logits[-1])))
    for _ in range(12):
        lg = llm.model.forward(st, [tok]); tok = int(np.argmax(np.array(lg[-1])))

layers = llm.model.layers
Ks = [3, 4, 6, 8, 12]
cov = {"moe_in": {k: [] for k in Ks}, "blk_in": {k: [] for k in Ks}}
for l in range(len(layers) - 1):
    nxt = layers[l + 1].ffn
    gw, gb = nxt.gate_w, nxt.bias
    for src in ("moe_in", "blk_in"):
        xs = rec[src][l] if src == "moe_in" else rec["blk_in"][l + 1]
        for t, x in enumerate(xs):
            actual = set(rec["actual"][l + 1][t])
            sc = np.array(mx.sqrt(mx.logaddexp(mx.array(x).astype(mx.float32) @ gw.T, 0.0)) + gb)[0]
            order = np.argsort(-sc)
            for k in Ks:
                cov[src][k].append(len(actual & set(order[:k].tolist())) / 6.0)
print("coverage of the true top-6 experts of layer l+1 by top-K predictions (mean over 39 layer pairs x 36 tokens):")
print("K     | (a) from layer-l MoE input (current SPEC_K) | (b) from layer-(l+1) block input (proposed)")
for k in Ks:
    print(f"{k:<5} | {np.mean(cov['moe_in'][k]):.3f}                                        | {np.mean(cov['blk_in'][k]):.3f}")

# --- confidence-adaptive rule: predict experts within margin d of the 6th-best predicted score (cap 8)
print("\nadaptive rule: predicted set = top-8 candidates with score >= s(6) - d   (s(6) = 6th best predicted score)")
print("d      | mean #predicted | precision | recall (of true top-6) | expected wasted reads/layer")
for d in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
    npred, prec, recl = [], [], []
    for l in range(len(layers) - 1):
        nxt = layers[l + 1].ffn; gw, gb = nxt.gate_w, nxt.bias
        for t, x in enumerate(rec["moe_in"][l]):
            actual = set(rec["actual"][l + 1][t])
            sc = np.array(mx.sqrt(mx.logaddexp(mx.array(x).astype(mx.float32) @ gw.T, 0.0)) + gb)[0]
            order = np.argsort(-sc); s6 = sc[order[5]]
            pred = [int(i) for i in order[:8] if sc[i] >= s6 - d]
            hit = len(actual & set(pred))
            npred.append(len(pred)); prec.append(hit / max(1, len(pred))); recl.append(hit / 6.0)
    print(f"{d:<6} | {np.mean(npred):15.2f} | {np.mean(prec):9.3f} | {np.mean(recl):22.3f} | {np.mean(npred) - np.mean(recl)*6:.2f}")
print("\nfixed-K reference: K=3 precision", f"{np.mean(cov['moe_in'][3])*6/3:.3f}", "recall", f"{np.mean(cov['moe_in'][3]):.3f}", "| K=4 precision", f"{np.mean(cov['moe_in'][4])*6/4:.3f}", "recall", f"{np.mean(cov['moe_in'][4]):.3f}")

# --- two layers ahead: layer l+2's router applied to layer l's MoE input
print("\ntwo layers ahead (l+2) coverage of true top-6 by top-K from layer-l MoE input; precision = coverage*6/K")
for k in (1, 2, 3, 4):
    c = []
    for l in range(len(layers) - 2):
        nxt = layers[l + 2].ffn; gw, gb = nxt.gate_w, nxt.bias
        for t, x in enumerate(rec["moe_in"][l]):
            actual = set(rec["actual"][l + 2][t])
            sc = np.array(mx.sqrt(mx.logaddexp(mx.array(x).astype(mx.float32) @ gw.T, 0.0)) + gb)[0]
            c.append(len(actual & set(np.argsort(-sc)[:k].tolist())) / 6.0)
    print(f"K={k}: coverage {np.mean(c):.3f}  precision {np.mean(c)*6/k:.3f}")

# --- confidence filter for the fixed top-4: keep the top-3 always; keep the 4th only if its predicted
#     score exceeds the predicted 7th-best by at least delta (margin in router-score units)
print("\nconfidence filter on rank 4 (keep top-3 always): delta | mean #pred | precision | recall | wasted/layer")
for d in (None, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
    npred, prec, recl = [], [], []
    for l in range(len(layers) - 1):
        nxt = layers[l + 1].ffn; gw, gb = nxt.gate_w, nxt.bias
        for t, x in enumerate(rec["moe_in"][l]):
            actual = set(rec["actual"][l + 1][t])
            sc = np.array(mx.sqrt(mx.logaddexp(mx.array(x).astype(mx.float32) @ gw.T, 0.0)) + gb)[0]
            order = np.argsort(-sc)
            pred = order[:3].tolist()
            if d is None or sc[order[3]] - sc[order[6]] >= d:
                pred.append(int(order[3]))
            hit = len(actual & set(pred)); npred.append(len(pred)); prec.append(hit / len(pred)); recl.append(hit / 6.0)
    print(f"{str(d):5} | {np.mean(npred):10.2f} | {np.mean(prec):9.3f} | {np.mean(recl):6.3f} | {np.mean(npred) - np.mean(recl)*6:.2f}")
print("per-rank precision of predictions (rank1..6):")
pr = np.zeros(6); cnt = 0
for l in range(len(layers) - 1):
    nxt = layers[l + 1].ffn; gw, gb = nxt.gate_w, nxt.bias
    for t, x in enumerate(rec["moe_in"][l]):
        actual = set(rec["actual"][l + 1][t])
        sc = np.array(mx.sqrt(mx.logaddexp(mx.array(x).astype(mx.float32) @ gw.T, 0.0)) + gb)[0]
        order = np.argsort(-sc)
        for r in range(6): pr[r] += int(order[r]) in actual
        cnt += 1
print("  ", np.round(pr / cnt, 3).tolist())
