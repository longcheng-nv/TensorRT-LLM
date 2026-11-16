import sys, time, numpy as np
sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx
from deepinfer_mac.model import mk
root = "/Users/loncheng/Models/dsv41-mac"
W = []
for l in range(40):
    d = mx.load(f"{root}/dense/L{l:02d}.safetensors")
    W.append((d["attn.wo_b.weight"], d["attn.wo_b.scales"])); mx.eval(W[-1])
x = mx.random.normal((8192,)).astype(mx.bfloat16); mx.eval(x)
def one(l):
    t = time.perf_counter(); y = mk.gemv_mxfp8(x, W[l][0], W[l][1], pro=1, rs=8, sgs=1, split=2); mx.eval(y); return time.perf_counter() - t
for gap_ms in (0, 0.5, 1, 2, 4, 8):
    ts = []
    for rep in range(3):
        for l in range(40):
            if gap_ms:
                t_end = time.perf_counter() + gap_ms / 1e3
                while time.perf_counter() < t_end: pass   # CPU busy-wait: GPU idle
            ts.append(one(l))
    print(f"GPU idle gap {gap_ms:4.1f} ms before each call: wo_b GEMV median {np.median(ts)*1e6:6.1f} us (incl. submit+sync)")
