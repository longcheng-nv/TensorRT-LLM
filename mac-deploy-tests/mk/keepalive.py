import sys, time, threading, numpy as np
sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx
from deepinfer_mac.model import mk
root = "/Users/loncheng/Models/dsv41-mac"
W = []
for l in range(40):
    d = mx.load(f"{root}/dense/L{l:02d}.safetensors")
    W.append((d["attn.wo_b.weight"], d["attn.wo_b.scales"])); mx.eval(W[-1])
x = mx.random.normal((8192,)).astype(mx.bfloat16); mx.eval(x)
spin_src = """
  uint t = thread_position_in_grid.x;
  float a = float(t) * 1e-3f;
  for (int i = 0; i < ITERS; i++) a = metal::fma(a, 0.9999f, 1e-4f);
  if (a == 12345.0f) out[t] = a;   // keep the loop
"""
spin_k = mx.fast.metal_kernel(name="spin", input_names=["dummy"], output_names=["out"], source=spin_src)
dummy = mx.zeros((1,)); mx.eval(dummy)
s2 = mx.new_stream(mx.gpu)
def spin(iters, tgs, stream=None):
    return spin_k(inputs=[dummy], template=[("ITERS", iters)], grid=(32 * tgs, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(32 * tgs,)], output_dtypes=[mx.float32], stream=stream)[0]
# calibrate spin duration
for iters, tgs in ((20000, 1), (20000, 20), (5000, 20), (2000, 40)):
    o = spin(iters, tgs); mx.eval(o)
    t = time.perf_counter(); o = spin(iters, tgs); mx.eval(o); print(f"spin iters={iters} tgs={tgs}: {(time.perf_counter()-t)*1e6:.0f} us")
def one(l):
    t = time.perf_counter(); y = mk.gemv_mxfp8(x, W[l][0], W[l][1], pro=1, rs=8, sgs=1, split=2); mx.eval(y); return time.perf_counter() - t
GAP = 4e-3
def run(mode, iters=0, tgs=0):
    ts = []
    for rep in range(3):
        for l in range(40):
            t_end = time.perf_counter() + GAP
            if mode == "none":
                while time.perf_counter() < t_end: pass
            elif mode == "same-stream-chain":   # small kernels back to back on the default stream until the gap ends
                while time.perf_counter() < t_end:
                    o = spin(iters, tgs); mx.eval(o)
            elif mode == "other-stream":        # long-ish spin on a second stream, real work does not wait for it
                o = spin(iters, tgs, stream=s2); mx.async_eval(o)
                while time.perf_counter() < t_end: pass
            ts.append(one(l))
    mx.synchronize()
    print(f"gap 4 ms, keep-alive {mode:18s} iters={iters:6d} tgs={tgs:3d}: wo_b median {np.median(ts)*1e6:6.1f} us")
run("none")
run("same-stream-chain", 2000, 1)
run("same-stream-chain", 2000, 20)
run("other-stream", 20000, 1)
run("other-stream", 20000, 20)
run("other-stream", 60000, 4)
