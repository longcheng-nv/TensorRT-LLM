"""mk.router_shared vs the engine's MoE._route_shared (+ next-layer speculation) on real layer
weights and realistic inputs. usage: test_mk_router.py"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac.model import mk  # noqa: E402
from deepinfer_mac.model.model import Args, MoE  # noqa: E402

root = "/Users/loncheng/Models/dsv41-mac"
a = Args()
d = mx.load(f"{root}/dense/L05.safetensors")
dn = mx.load(f"{root}/dense/L06.safetensors")
moe = MoE(a, 5, d, None, "L05")
nxt = MoE(a, 6, dn, None, "L06")
gw, gwn = d["ffn.gate.weight"], dn["ffn.gate.weight"]  # bf16 [384, 5120] each
w13, w13s = moe.shared_w13.w, moe.shared_w13.scales
w2, w2s = moe.shared_w2.w, moe.shared_w2.scales
rsc = mx.array(a.route_scale, dtype=mx.float32)
lim = mx.array(a.swiglu_limit, dtype=mx.float32)
ok = True
for trial in range(4):
    x = (mx.random.normal((1, 5120), key=mx.random.key(trial)) * (0.3 + 0.5 * trial)).astype(mx.bfloat16)
    w_r, idx_r, xq_r, y_r, sc_r = moe._route_shared(x, moe.bias)
    spec = mx.sqrt(mx.logaddexp(x.astype(mx.float32) @ nxt.gate_bf16.astype(mx.float32).T, 0.0)) + nxt.bias
    pred_r = mx.argsort(-spec, axis=-1)[0, :4]
    idx, w, sc, pred, _, xq, y = mk.router_shared(x, gw, w13, w13s, w2, w2s, moe.bias, rsc, lim, topk=6,
                                               gate_next=gwn, bias_next=nxt.bias, spec_k=4)
    mx.eval(w_r, idx_r, xq_r, y_r, sc_r, pred_r, idx, w, sc, pred, xq, y)
    i_r, i_m = np.array(idx_r[0]), np.array(idx)
    same_set = set(i_r.tolist()) == set(i_m.tolist())
    wr = dict(zip(i_r.tolist(), np.array(w_r[0]).tolist()))
    wm = dict(zip(i_m.tolist(), np.array(w).tolist()))
    wd = max(abs(wr[e] - wm.get(e, 0)) for e in wr)
    scd = float(np.abs(np.array(sc_r[0]) - np.array(sc)).max())
    xqx = float((np.array(xq_r[0].astype(mx.float32)) == np.array(xq.astype(mx.float32))).mean())
    yd = np.abs(np.array(y_r[0]) - np.array(y))
    yrel = float(yd.max() / np.abs(np.array(y_r[0])).max())
    pr = set(np.array(pred_r).tolist()) == set(np.array(pred).tolist())
    good = same_set and wd < 1e-5 and scd < 1e-4 and xqx == 1.0 and yrel < 2e-2 and pr
    ok &= good
    print(f"trial {trial}: idx set {'same' if same_set else 'DIFF'} | max|dw| {wd:.1e} | max|dscores| {scd:.1e} | xq exact {xqx:.3f} | "
          f"shared y max rel {yrel:.1e} exact {(yd == 0).mean():.3f} | pred {'same' if pr else 'DIFF'} -> {'OK' if good else 'MISMATCH'}")
print("ALL OK" if ok else "SOME MISMATCH")
# timing: 40 calls back to back
xs = [(mx.random.normal((1, 5120), key=mx.random.key(100 + i))).astype(mx.bfloat16) for i in range(40)]
mx.eval(xs)
rs_c = mx.compile(moe._route_shared)


def ref_call(x):
    o = rs_c(x, moe.bias)
    sc = mx.sqrt(mx.logaddexp(x.astype(mx.float32) @ nxt.gate_bf16.astype(mx.float32).T, 0.0)) + nxt.bias
    return list(o) + [mx.argpartition(-sc, kth=3, axis=-1)[:, :4]]


for name, fn in (("mlx compiled route+shared+spec", ref_call),
                 ("mk router_shared", lambda x: mk.router_shared(x, gw, w13, w13s, w2, w2s, moe.bias, rsc, lim, 6, gwn, nxt.bias, 4))):
    for rep in range(3):
        t = time.perf_counter()
        outs = [fn(x) for x in xs]
        mx.eval(outs)
        dt = (time.perf_counter() - t) / 40
    print(f"  {name:32s}: {dt*1e6:7.1f} us/call")
