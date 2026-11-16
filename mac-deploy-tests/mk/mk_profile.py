"""GPU-throughput profile of one BS=1 decode token, every stage, in fixed-expert mode (no SSD).

Captures every compiled call of one decode step (Block chains, attention projections / indexer /
compressor / core, router+shared, routed experts, head), then replays each stage's calls N times
back to back with ONE eval at the end: per-stage GPU execution time without host syncs.
Also times (a) the normal forward (per-layer host sync) and (b) a sync-free forward (fixed slots,
whole token as one lazy graph) to split host-sync bubbles from GPU time.

usage: mk_profile.py [--cache-gb 16] [--n 20]
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
    ap.add_argument("--cache-gb", type=float, default=16.0)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    M.MoE.SPEC_K = 0
    M.MoE.FIXED_EXPERTS = list(range(6))
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=a.cache_gb)
    ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    for _ in range(4):
        lg = llm.model.forward(st, [tok])
        tok = int(np.argmax(np.array(lg[-1])))

    # (a) normal forward
    ts = []
    for _ in range(8):
        t = time.perf_counter()
        lg = llm.model.forward(st, [tok])
        tok = int(np.argmax(np.array(lg[-1])))
        ts.append(time.perf_counter() - t)
    print(f"(a) normal fixed-mode forward: median {np.median(ts)*1e3:.1f} ms/token", flush=True)

    # (b) sync-free forward: fixed slots resolved once, no np.array(idx), no store calls
    store = llm.weights.experts
    slots = {}
    for b in llm.model.layers:
        slab, sl = store.get_slots(b.ffn.key, list(range(6)))
        slots[b.ffn.key] = (slab, mx.array(np.array(sl, dtype=np.uint32))[:, None])
    orig_call = M.MoE.__call__

    def nosync_call(self, x, image_mask=None):
        if x.shape[0] != 1:
            return orig_call(self, x, image_mask)
        if not hasattr(self, "_rs_c"):
            self._rs_c = mx.compile(self._route_shared)
        w, idx, xq, y, _ = self._rs_c(x, self._bias(x, image_mask))
        slab, rhs = slots[self.key]
        q1, q3, q2 = slab.proj("w1"), slab.proj("w3"), slab.proj("w2")
        key = (q1[3], q3[3], q2[3])
        f = M._ffn_fns.get(key)
        if f is None:
            f = M._ffn_fns[key] = mx.compile(lambda xr, rhs, wk, lim, a1, b1, c1, a3, b3, c3, a2, b2, c2: M._slab_ffn(
                xr, rhs, wk, lim, (a1, b1, c1, key[0]), (a3, b3, c3, key[1]), (a2, b2, c2, key[2])))
        xr = mx.broadcast_to(xq[:, None, None, :], (6, 1, 1, xq.shape[-1]))
        wk = w.reshape(6, 1, 1, 1).astype(mx.float32)
        o = f(xr, rhs, wk, mx.array(self.a.swiglu_limit, dtype=mx.float32), *q1[:3], *q3[:3], *q2[:3])
        return (y + o.sum(axis=0, keepdims=True)).astype(x.dtype)

    M.MoE.__call__ = nosync_call
    tb, te = [], []
    for _ in range(8):
        t = time.perf_counter()
        lg = llm.model.forward(st, [tok])
        t1 = time.perf_counter()
        tok = int(np.argmax(np.array(lg[-1])))
        t2 = time.perf_counter()
        tb.append(t1 - t)
        te.append(t2 - t)
    print(f"(b) sync-free forward: median {np.median(te)*1e3:.1f} ms/token  (python graph build {np.median(tb)*1e3:.1f} ms, "
          f"then eval {np.median(np.array(te)-np.array(tb))*1e3:.1f} ms)", flush=True)

    # (c) per-stage GPU throughput by replay
    cap = collections.defaultdict(list)

    def capture(obj, name, tag):
        f = getattr(obj, name)

        def g(*args, **kw):
            out = f(*args, **kw)
            cap[tag].append((f, args, kw))
            return out
        setattr(obj, name, g)
        return f

    restore = []
    for b in llm.model.layers:
        at = b.attn
        at._compiled()
        restore.append((at, "_proj_c", capture(at, "_proj_c", "attn.proj")))
        restore.append((at, "_core_c", capture(at, "_core_c", "attn.core")))
        if at.is_index_source:
            restore.append((at, "_indexer", capture(at, "_indexer", "attn.indexer")))
        if at.is_kv_source:
            restore.append((at, "_compress", capture(at, "_compress", "attn.compressor")))
        restore.append((b.ffn, "_rs_c", capture(b.ffn, "_rs_c", "moe.route+shared")))
    for name, tag in (("_block_pre_attn_c", "block.pre_attn (hc mixes+sinkhorn+pre+rms)"),
                      ("_block_post_attn_pre_ffn_c", "block.post_attn_pre_ffn (hc_post+mixes+sinkhorn+pre+rms)"),
                      ("_hc_post_c", "block.hc_post (after ffn)")):
        f0 = getattr(M, name)

        def mk(f0=f0, tag=tag):
            def g(*args, **kw):
                out = f0(*args, **kw)
                cap[tag].append((f0, args, kw))
                return out
            return g
        setattr(M, name, mk())
    # routed experts: capture the compiled slab ffn
    for key, f0 in list(M._ffn_fns.items()):
        def mk2(f0=f0):
            def g(*args, **kw):
                out = f0(*args, **kw)
                cap["moe.routed experts (3x gather_qmm + swiglu)"].append((f0, args, kw))
                return out
            return g
        M._ffn_fns[key] = mk2()
    head = llm.model.head

    class HeadCap:
        def __call__(self, x):
            cap["head (8-bit 129280x5120)"].append((Linear_call, (x,), {}))
            return Linear_call(x)
    Linear_call = head.__call__
    llm.model.head = HeadCap()
    lg = llm.model.forward(st, [tok])
    mx.eval(lg)
    llm.model.head = head
    print("captured per token: " + ", ".join(f"{k}: {len(v)}" for k, v in cap.items()), flush=True)
    total = 0.0
    rows = []
    for tag, calls in cap.items():
        def run_all():
            flat = []
            for f, args, kw in calls:
                o = f(*args, **kw)
                if isinstance(o, mx.array):
                    flat.append(o)
                elif isinstance(o, (tuple, list)):
                    flat += [v for v in o if isinstance(v, mx.array)]
            return flat
        mx.eval(run_all())
        t0 = time.perf_counter()
        flat = []
        for _ in range(a.n):
            flat += run_all()
        mx.eval(flat)
        dt = (time.perf_counter() - t0) / a.n
        total += dt
        rows.append((tag, dt, len(calls)))
    for tag, dt, n in sorted(rows, key=lambda r: -r[1]):
        print(f"  {tag:62s} {dt*1e3:7.2f} ms/token  ({dt/n*1e6:7.1f} us/call x {n})")
    print(f"  {'sum of stages':62s} {total*1e3:7.2f} ms/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
