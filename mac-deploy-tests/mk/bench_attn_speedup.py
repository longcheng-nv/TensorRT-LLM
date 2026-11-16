"""Sparse-attention megakernel (mk.sparse_attn) vs the same computation as separate MLX ops, BS=1 decode.

For every layer, the inputs of mk.sparse_attn are captured during one fused decode step on a real state
(a short prompt, and a long prompt where the indexer's top-512 selection is active). The reference
rebuilds what the reference decode path computes for that step, minus the wo_a / wo_b projections:
window + compressed-KV gathers, QK, sink softmax, PV with bf16 probabilities, inverse RoPE, fp8
act-quant and the ring update (Attention.__call__ / Attention._core). Both are first checked to agree,
then timed for one whole token (all 40 layers) back to back with one eval; each call's ring output
feeds the next call, so the calls run in sequence as in decode. Reference variants:
  ops  = every op dispatched on its own (pure single-op composition)
  prod = the core compiled with mx.compile, exactly as the reference decode path runs it
Expert routing is fixed to experts 0-5 (no SSD reads); attention timing does not depend on it.

usage: bench_attn_speedup.py [--ctx 4096] [--n 60] [--reps 7] [--check-only]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402
from deepinfer_mac.model.ops import act_quant, rope_tail  # noqa: E402

F32, BF16 = mx.float32, mx.bfloat16


def make_core(rd: int, o_groups: int):
    def core(q, g, valid, sink, scale, cs):
        H = q.size // 512
        qh = q.reshape(1, H, 512)
        s = (qh.astype(F32) @ g.astype(F32).swapaxes(-1, -2)) * scale
        s = mx.where(valid[:, None, :], s, -mx.inf)
        m = mx.maximum(s.max(-1, keepdims=True), -1e30)
        p = mx.exp(s - m)
        denom = p.sum(-1, keepdims=True) + mx.exp(sink[None, :, None] - m)
        o = ((p.astype(BF16).astype(F32) @ g.astype(F32)) / denom).astype(qh.dtype)
        half = cs.size // 2
        o = rope_tail(o, (cs[:half][None], cs[half:][None]), rd, inverse=True)
        return act_quant(o.reshape(1, o_groups, -1)).reshape(-1)

    return core


def ref_step(c, ring, core_fn):
    kv2 = c["kv"].reshape(1, -1)
    wsrc = mx.concatenate([ring, kv2])
    wi = c["widx"][None]
    g, valid = wsrc[mx.maximum(wi, 0)], wi >= 0
    if c["comp"] is not None:
        ci = c["cidx"].reshape(1, -1)
        g = mx.concatenate([g, c["comp"][mx.maximum(ci, 0)]], axis=1)
        valid = mx.concatenate([valid, ci >= 0], axis=1)
    o = core_fn(c["q"], g, valid, c["sink"], c["scale"], c["cs"])
    has = mx.arange(ring.shape[0]) == c["rslot"]
    return o, mx.where(has[:, None], kv2, ring)


def mk_step(c, ring):
    return mk.sparse_attn(c["q"], ring, c["kv"], c["widx"], c["comp"], c["cidx"], c["sink"], c["scale"], c["cs"], c["rslot"],
                          rope_dim=c["rd"])


def mk_old_step(c, ring):
    """The previous single-dispatch kernel (mk_sparse_attn, 16 threadgroups)."""
    split, mk.SPLIT_ATTN = mk.SPLIT_ATTN, False
    try:
        return mk_step(c, ring)
    finally:
        mk.SPLIT_ATTN = split


def capture(llm, ids):
    llm.reset()
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    for _ in range(3):  # a few plain decode steps so the ring / compressed state is realistic
        tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))
    caps, orig = [], mk.sparse_attn

    def cap(q, ring, kvnew, widx, comp, cidx, sink, scale, rope_cs, rslot, heads_per_tg=4, rope_dim=64):
        out = orig(q, ring, kvnew, widx, comp, cidx, sink, scale, rope_cs, rslot, heads_per_tg=heads_per_tg, rope_dim=rope_dim)
        caps.append(dict(q=q, ring=ring, kv=kvnew, widx=widx, comp=comp, cidx=cidx, sink=sink, scale=scale, cs=rope_cs,
                         rslot=rslot, rd=rope_dim, out=out))
        return out

    mk.sparse_attn = cap
    try:
        mx.eval(llm.model.forward(st, [tok]))
    finally:
        mk.sparse_attn = orig
    mx.eval([c["out"] for c in caps])
    return caps, st.pos if hasattr(st, "pos") else None


def time_token(step, caps, n, reps):
    """Seconds per token (all captured layers once), median over reps; one eval per rep."""
    def build():
        ring, outs = caps[0]["ring"], []
        for _ in range(n):
            for c in caps:
                o, ring = step(c, ring)
                outs.append(o)
        outs.append(ring)
        return outs

    mx.eval(build())  # warm-up (compiles, keeps the GPU clock up)
    ts = []
    for _ in range(reps):
        outs = build()
        t = time.perf_counter()
        mx.eval(outs)
        ts.append((time.perf_counter() - t) / n)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096, help="long-context prompt length in tokens")
    ap.add_argument("--n", type=int, default=60, help="tokens per timed batch")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--check-only", action="store_true", help="exactness check only, no timing")
    a = ap.parse_args()
    M.MoE.SPEC_K = 0
    M.MoE.FIXED_EXPERTS = list(range(6))
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=8)
    args = llm.args
    core_ops = make_core(args.rope_head_dim, args.o_groups)
    core_prod = mx.compile(make_core(args.rope_head_dim, args.o_groups))
    short_ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
    root = "/Users/loncheng/Models/DeepInfer-Mac"
    text = "".join(open(f"{root}/{p}").read() for p in ("README.md", "deepinfer_mac/model/model.py", "deepinfer_mac/engine.py"))
    body = llm.tokenizer.decode(llm.tokenizer.encode(text, add_special_tokens=False).ids[: a.ctx])
    long_ids, _ = llm.chat_prompt([{"role": "user", "content": "Summarize the following code and notes:\n\n" + body}])
    print(f"model: {args.n_layers} layers, {args.n_heads} heads x {args.head_dim}, window {args.window_size}, top-k {args.index_topk}; "
          f"n={a.n} tokens per batch, median of {a.reps}", flush=True)
    for name, ids in (("short", short_ids), ("long", long_ids)):
        t0 = time.perf_counter()
        caps, _ = capture(llm, ids)
        prep = time.perf_counter() - t0
        # correctness: exact element counts, new split kernel vs the reference chain and vs the old kernel
        n_el = int(caps[0]["q"].size)
        stats = {"new vs ref": [], "new vs old": [], "old vs ref": []}
        worst_cos, rings_ok, max_rel = 1.0, True, 0.0
        for c in caps:
            o_m, r_m = mk_step(c, c["ring"])
            o_o, r_o = mk_old_step(c, c["ring"])
            o_r, r_r = ref_step(c, c["ring"], core_ops)
            mx.eval(o_m, r_m, o_o, r_o, o_r, r_r)
            x, y, z = (np.array(t.astype(F32)).ravel() for t in (o_r, o_m, o_o))
            worst_cos = min(worst_cos, float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)))
            max_rel = max(max_rel, float(np.abs(x - y).max() / (np.abs(x).max() + 1e-30)))
            stats["new vs ref"].append(int((x != y).sum()))
            stats["new vs old"].append(int((z != y).sum()))
            stats["old vs ref"].append(int((x != z).sum()))
            rm = np.array(r_m.astype(F32))
            rings_ok &= bool(np.array_equal(rm, np.array(r_r.astype(F32))) and np.array_equal(rm, np.array(r_o.astype(F32))))
        diff_txt = " | ".join(f"{k}: {sum(1 for d in v if d)} of {len(v)} layers differ, max {max(v)} / {n_el} elements per layer"
                              for k, v in stats.items())
        nkeys = [int((np.array(c["cidx"]) >= 0).sum()) if c["comp"] is not None else 0 for c in caps]
        groups = {
            "window only": [c for c in caps if c["comp"] is None],
            "window + compressed": [c for c in caps if c["comp"] is not None],
        }
        print(f"\n[{name}] prompt {len(ids)} tok (prefill+3 steps {prep:.1f} s) | {len(caps)} layers | compressed keys attended "
              f"per layer: min {min(nkeys)} max {max(nkeys)} | new vs reference: worst cos {worst_cos:.7f}, max rel diff {max_rel:.1e} | "
              f"rings identical {rings_ok}\n  exact compare: {diff_txt}", flush=True)
        if a.check_only:
            continue
        variants = (("megakernel (split)", mk_step), ("old megakernel", mk_old_step),
                    ("ref prod", lambda c, r: ref_step(c, r, core_prod)), ("ref ops", lambda c, r: ref_step(c, r, core_ops)))
        for gname, gc in (("all layers", caps), *groups.items()):
            if not gc:
                continue
            ts = {vn: time_token(fn, gc, a.n, a.reps) / len(gc) * 1e6 for vn, fn in variants}
            base = ts["megakernel (split)"]
            print(f"  {gname:20s} ({len(gc):2d} layers) " + " | ".join(
                f"{vn} {t:6.1f} us/layer" + ("" if vn.startswith("megakernel") else f" ({t / base:4.2f}x)") for vn, t in ts.items()),
                flush=True)


if __name__ == "__main__":
    main()
