"""Check mk.moe_experts against the engine's MLX chain (_slab_ffn) on real expert weights from the
converted checkpoint, for every expert layout (3/3/3, 2/2/3, 4/4/4), and time both.
usage: test_mk_moe.py [--model DIR]"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402
from deepinfer_mac.model.ops import act_quant  # noqa: E402
from deepinfer_mac.store import ExpertStore  # noqa: E402


def ref(x, slots, w, lim, p1, p3, p2, shared):
    xq = act_quant(x[None])  # [1, dim]
    n = slots.shape[0]
    xr = mx.broadcast_to(xq[:, None, None, :], (n, 1, 1, xq.shape[-1]))
    rhs = slots[:, None]
    wk = w.reshape(n, 1, 1, 1).astype(mx.float32)
    o = M._slab_ffn(xr, rhs, wk, lim, p1, p3, p2)
    return shared + o.sum(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/loncheng/Models/dsv41-mac")
    a = ap.parse_args()
    store = ExpertStore(a.model, 3 << 30)
    rng = np.random.default_rng(0)
    lim = mx.array(10.0, dtype=mx.float32)
    ok = True
    for key in ("L00", "L05", "L39"):
        ids = sorted(rng.choice(384, 6, replace=False).tolist())
        slab, sl = store.get_slots(key, ids)
        slots = mx.array(np.array(sl, dtype=np.uint32))
        p1, p3, p2 = slab.proj("w1"), slab.proj("w3"), slab.proj("w2")
        for trial in range(3):
            x = (mx.random.normal((5120,), key=mx.random.key(trial)) * (0.5 + trial)).astype(mx.bfloat16)
            w = mx.array(rng.dirichlet(np.ones(6)).astype(np.float32) * 1.5)
            shared = mx.random.normal((5120,), key=mx.random.key(100 + trial)).astype(mx.float32) * 0.1
            y_ref = ref(x, slots, w, lim, p1, p3, p2, shared)
            y = mk.moe_experts(act_quant(x[None])[0], slots, w, lim, p1, p3, p2, shared)
            mx.eval(y_ref, y)
            d = np.abs(np.array(y) - np.array(y_ref))
            rel = float(d.max() / (np.abs(np.array(y_ref)).max() + 1e-9))
            cos = float(np.dot(np.array(y).ravel(), np.array(y_ref).ravel()) /
                        (np.linalg.norm(np.array(y)) * np.linalg.norm(np.array(y_ref)) + 1e-12))
            exact = float((d == 0).mean())
            flag = "OK" if cos > 0.99999 and rel < 2e-2 else "MISMATCH"
            ok &= flag == "OK"
            print(f"{key} bits={p1[3]}/{p3[3]}/{p2[3]} trial {trial}: cos {cos:.7f} max|d|/max|y| {rel:.2e} exact {exact:.3f} {flag}")
        # timing: 40 back-to-back calls each (one layer's worth per call), one eval
        xs = [(mx.random.normal((5120,), key=mx.random.key(1000 + i))).astype(mx.bfloat16) for i in range(40)]
        mx.eval(xs)
        for name, fn in (("mlx chain", ref), ("mk fused", lambda xi, *r: mk.moe_experts(act_quant(xi[None])[0], *r))):
            for rep in range(2):
                t = time.perf_counter()
                outs = [fn(xi, slots, w, lim, p1, p3, p2, shared) for xi in xs]
                mx.eval(outs)
                dt = (time.perf_counter() - t) / 40
            nbytes = 6 * store.layouts[key].nbytes
            print(f"   {name:10s}: {dt*1e6:7.1f} us/layer-call  ({nbytes/dt/1e9:6.1f} GB/s effective)")
    print("ALL OK" if ok else "SOME MISMATCH")


if __name__ == "__main__":
    main()
