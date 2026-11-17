"""mk.router_shared(split=True) (routing and shared expert as separate dispatches) must equal the
two-dispatch kernel bit for bit: every output element is computed by the same code. usage: test_mk_router_split.py"""
import sys

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model.model import Args, MoE  # noqa: E402

root = "/Users/loncheng/Models/dsv41-mac"
a = Args()
ok = True
for la in (0, 5, 21, 38):
    d = mx.load(f"{root}/dense/L{la:02d}.safetensors")
    dn = mx.load(f"{root}/dense/L{la + 1:02d}.safetensors")
    moe, nxt = MoE(a, la, d, None, f"L{la:02d}"), MoE(a, la + 1, dn, None, f"L{la + 1:02d}")
    rsc = mx.array(a.route_scale, dtype=mx.float32)
    lim = mx.array(a.swiglu_limit, dtype=mx.float32)
    for trial in range(3):
        x = (mx.random.normal((1, 5120), key=mx.random.key(10 * la + trial)) * (0.3 + 0.5 * trial)).astype(mx.bfloat16)
        for speck in (0, 2, 4):
            kw = dict(topk=6, gate_next=nxt.gate_bf16 if speck else None, bias_next=nxt.bias if speck else None, spec_k=speck)
            r0 = mk.router_shared(x, moe.gate_bf16, moe.shared_w13.w, moe.shared_w13.scales, moe.shared_w2.w, moe.shared_w2.scales,
                                  moe.bias, rsc, lim, **kw)
            r1 = mk.router_shared(x, moe.gate_bf16, moe.shared_w13.w, moe.shared_w13.scales, moe.shared_w2.w, moe.shared_w2.scales,
                                  moe.bias, rsc, lim, split=True, **kw)
            mx.eval(*r0, *r1)
            names = ["idx", "w", "scores", "pred", "pred2", "xq", "y"]
            for n, u, v in zip(names, r0, r1):
                if n == "pred" and not speck or n == "pred2":
                    continue
                same = np.array_equal(np.array(u.astype(mx.float32) if u.dtype == mx.bfloat16 else u),
                                      np.array(v.astype(mx.float32) if v.dtype == mx.bfloat16 else v))
                if not same:
                    ok = False
                    print(f"L{la} trial {trial} spec_k {speck}: {n} differs")
    print(f"L{la:02d}: checked", flush=True)
print("PASS (bit-identical)" if ok else "FAIL")
