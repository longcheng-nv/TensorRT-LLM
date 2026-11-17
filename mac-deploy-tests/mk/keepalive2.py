"""Does a keep-alive strategy hold the GPU clock during an SSD wait? Emulate a wait of GAP ms with each
strategy, then time one chained wo_b GEMV (as decode runs it right after a wait). A clocked-down GPU
shows up as a slower GEMV (see dvfs.py: 1.2 ms instead of ~0.19 ms after a 4 ms idle gap).

  old       : mk.gpu_tick() back to back (queue a tiny kernel, wait for it) -- the shipped hot_wait
  async<N>  : queue a tiny kernel without waiting, sleep N us (threading.Event.wait, GIL released), repeat
  idle      : no keep-alive (reference)
"""
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import mlx.core as mx  # noqa: E402

from deepinfer_mac.model import mk  # noqa: E402

root = "/Users/loncheng/Models/dsv41-mac"
W = []
for l in range(40):
    d = mx.load(f"{root}/dense/L{l:02d}.safetensors")
    W.append((d["attn.wo_b.weight"], d["attn.wo_b.scales"]))
    mx.eval(W[-1])
x = mx.random.normal((8192,)).astype(mx.bfloat16)
mx.eval(x)
ev = threading.Event()


def gemv(l):
    t = time.perf_counter()
    y = mk.gemv_mxfp8(x, W[l][0], W[l][1], pro=1, rs=8, sgs=1, split=2)
    mx.eval(y)
    return time.perf_counter() - t


def wait(mode, gap):
    end = time.perf_counter() + gap
    n = 0
    if mode == "idle":
        while time.perf_counter() < end:
            pass
    elif mode == "old":
        while time.perf_counter() < end:
            mk.gpu_tick()
            n += 1
    else:
        us = int(mode[5:])
        while time.perf_counter() < end:
            mx.async_eval(mk._spin())
            ev.wait(us * 1e-6)
            n += 1
    return n


for gap_ms in (1.0, 2.5, 5.0):
    for mode in ("idle", "old", "async100", "async50", "async200", "old"):
        ts, ns = [], []
        for rep in range(3):
            for l in range(40):
                ns.append(wait(mode, gap_ms / 1e3))
                ts.append(gemv(l))
        print(f"wait {gap_ms:3.1f} ms, keep-alive {mode:9s} ({np.mean(ns):5.1f} ticks): wo_b after it median {np.median(ts) * 1e6:6.1f} us, "
              f"p90 {np.percentile(ts, 90) * 1e6:6.1f} us", flush=True)
