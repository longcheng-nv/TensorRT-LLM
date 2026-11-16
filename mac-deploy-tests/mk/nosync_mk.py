import sys, time, numpy as np
sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx
from deepinfer_mac import LLM
from deepinfer_mac.model import model as M, mk
M.MoE.SPEC_K = 0
M.MoE.FIXED_EXPERTS = list(range(6))
llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=16)
ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
st, logits, _, _ = llm._prefill(ids, False)
tok = int(np.argmax(np.array(logits[-1])))
for _ in range(3):
    tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
def timed(n=10):
    global tok
    tb, te = [], []
    for _ in range(n):
        t = time.perf_counter(); lg = llm.model.forward(st, [tok]); t1 = time.perf_counter()
        tok = int(np.argmax(np.array(lg[-1]))); t2 = time.perf_counter()
        tb.append(t1 - t); te.append(t2 - t)
    return np.median(te) * 1e3, np.median(tb) * 1e3
print("fixed, per-layer sync: %.1f ms/token (python before final eval %.1f ms)" % timed())
# no host sync: fixed slots, router kernels still run, nothing read back per layer
store = llm.weights.experts
slots = {}
for b in llm.model.layers:
    slab, sl = store.get_slots(b.ffn.key, list(range(6)))
    slots[b.ffn.key] = (slab, mx.array(np.array(sl, dtype=np.uint32)))
def decode_nosync(self, x, bias):
    rsc, lim = self._mk_consts
    sw13, sw2 = self.shared_w13, self.shared_w2
    idx, w, scores, pred, _, xq, y = mk.router_shared(x, self.gate_bf16, sw13.w, sw13.scales, sw2.w, sw2.scales, bias, rsc, lim, topk=self.topk)
    slab, sl = slots[self.key]
    y = mk.moe_experts(xq, sl, w, lim, slab.proj("w1"), slab.proj("w3"), slab.proj("w2"), y)
    return y[None].astype(x.dtype)
M.MoE._decode_mk = decode_nosync
print("fixed, NO per-layer sync (one lazy graph per token): %.1f ms/token (python graph build %.1f ms)" % timed())
# same, but async_eval after every layer's MoE so the GPU starts early
orig = M.MoE._decode_mk
def decode_nosync_async(self, x, bias):
    y = orig(self, x, bias); mx.async_eval(y); return y
M.MoE._decode_mk = decode_nosync_async
print("fixed, no sync + async_eval per layer: %.1f ms/token (python %.1f ms)" % timed())
