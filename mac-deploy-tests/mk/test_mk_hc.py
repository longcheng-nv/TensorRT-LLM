"""Check mk.hc_stage against the engine's compiled mHC chains on real activations of a decode step
(captured from the real model), and time both. usage: test_mk_hc.py"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac import LLM  # noqa: E402
from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402


def cmp(name, a, b):
    a, b = np.array(a.astype(mx.float32)).ravel(), np.array(b.astype(mx.float32)).ravel()
    d = np.abs(a - b)
    rel = d.max() / (np.abs(b).max() + 1e-12)
    exact = (d == 0).mean()
    return f"{name}: max|d|/max {rel:.1e} exact {exact:.3f}", rel


def main():
    M.MoE.SPEC_K = 0
    M.MoE.FIXED_EXPERTS = list(range(6))
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=4)
    ids, _ = llm.chat_prompt([{"role": "user", "content": "Write a long paragraph about the history of tea."}])
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    cap = {"pre": [], "post": []}
    f_pre, f_post = M._block_pre_attn_c, M._block_post_attn_pre_ffn_c

    def g_pre(*a):
        out = f_pre(*a)
        cap["pre"].append((a, out))
        return out

    def g_post(*a):
        out = f_post(*a)
        cap["post"].append((a, out))
        return out

    M._block_pre_attn_c, M._block_post_attn_pre_ffn_c = g_pre, g_post
    M.Transformer.MK = False  # the reference Block path is what gets captured
    lg = llm.model.forward(st, [tok])
    M.Transformer.MK = True
    mx.eval(lg, [o for _, o in cap["pre"]], [o for _, o in cap["post"]])
    M._block_pre_attn_c, M._block_post_attn_pre_ffn_c = f_pre, f_post
    worst = 0.0
    for li, ((x, pre_mix, fn, sc, base, norm_w, eps), (h_r, pre_r, post_r, comb_r)) in enumerate(cap["pre"]):
        X, h, pre, post, comb = mk.hc_stage(x[0], pre_mix[0], fn, sc, base, norm_w, eps)
        mx.eval(X, h, pre, post, comb)
        rows = [cmp("h", h, h_r[0]), cmp("pre", pre, pre_r[0]), cmp("post", post, post_r[0]), cmp("comb", comb, comb_r[0])]
        worst = max(worst, max(r for _, r in rows))
        if li in (0, 1, 20, 39):
            print(f"pre_attn L{li}: " + " | ".join(s for s, _ in rows))
    for li, ((h_in, res, a_post, a_comb, a_pre, fn, sc, base, norm_w, eps), (x_r, hn_r, pre_r, post_r, comb_r)) in enumerate(cap["post"]):
        X, h, pre, post, comb = mk.hc_stage(res[0], a_pre[0], fn, sc, base, norm_w, eps, hin=h_in[0], post=a_post[0], comb=a_comb[0])
        mx.eval(X, h, pre, post, comb)
        rows = [cmp("X", X, x_r[0]), cmp("h", h, hn_r[0]), cmp("pre", pre, pre_r[0]), cmp("post", post, post_r[0]), cmp("comb", comb, comb_r[0])]
        worst = max(worst, max(r for _, r in rows))
        if li in (0, 1, 20, 39):
            print(f"post_attn L{li}: " + " | ".join(s for s, _ in rows))
    print(f"worst relative error {worst:.2e} -> {'OK' if worst < 2e-2 else 'MISMATCH'}")
    # timing: the 40 captured pre_attn calls, reference vs fused, back to back
    for name, fn_ in (("mlx compiled chain", lambda a: f_pre(*a)),
                      ("mk hc_stage", lambda a: mk.hc_stage(a[0][0], a[1][0], *a[2:]))):
        for rep in range(3):
            t = time.perf_counter()
            outs = [fn_(a) for a, _ in cap["pre"]]
            mx.eval(outs)
            dt = (time.perf_counter() - t) / len(cap["pre"])
        print(f"  pre_attn {name:20s}: {dt*1e6:7.1f} us/call")
    for name, fn_ in (("mlx compiled chain", lambda a: f_post(*a)),
                      ("mk hc_stage", lambda a: mk.hc_stage(a[1][0], a[4][0], *a[5:], hin=a[0][0], post=a[2][0], comb=a[3][0]))):
        for rep in range(3):
            t = time.perf_counter()
            outs = [fn_(a) for a, _ in cap["post"]]
            mx.eval(outs)
            dt = (time.perf_counter() - t) / len(cap["post"])
        print(f"  post_attn {name:19s}: {dt*1e6:7.1f} us/call")


if __name__ == "__main__":
    main()
