# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op37 mechanism probe: champion r3_v11 sequential 1-stream vs S-stream
round-robin per-row launches (no kernel change). CUDA-event coarse timing,
cold-L2 evict per rep, exact check per config. Quantifies how much of the
BS>1 linear decay is pure same-stream serialization of ~1-cluster kernels."""
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bs_ab import CELLS, Stack, build_champion, head_call  # noqa: E402

DEV = "cuda"


def seq_call(stack, cmod, bs, streams):
    lg = [stack.logits[i:i + 1] for i in range(bs)]
    pre = [stack.pre[i:i + 1] for i in range(bs)]
    out = torch.empty(bs, stack.K, dtype=torch.int32, device=DEV)
    orow = [out[i:i + 1] for i in range(bs)]
    N, run = stack.N, cmod.run
    if streams == 1:
        def call():
            for i in range(bs):
                run(lg[i], pre[i], N, orow[i])
        return call, out
    ss = [torch.cuda.Stream() for _ in range(streams)]
    main = torch.cuda.current_stream()

    def call():
        ev = torch.cuda.Event()
        ev.record(main)
        for j, s in enumerate(ss[:min(streams, bs)]):
            s.wait_event(ev)
        for i in range(bs):
            with torch.cuda.stream(ss[i % streams]):
                run(lg[i], pre[i], N, orow[i])
        for s in ss[:min(streams, bs)]:
            main.wait_stream(s)
    return call, out


def timeit(call, reps=15):
    evict = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        evict.random_()
        torch.cuda.synchronize()
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def main():
    cmod = build_champion()
    for cname in ("flash_128k", "pro_1024k", "flash_32k"):
        cell = CELLS[cname]
        stack = Stack(cell, 64)
        for bs in (8, 64):
            base = None
            for streams in (1, 2, 4, 8, 16):
                if streams > bs:
                    continue
                call, out = seq_call(stack, cmod, bs, streams)
                call(); torch.cuda.synchronize()
                bad = stack.exact(out, bs)
                us = timeit(call)
                if streams == 1:
                    base = us
                # head reference once per (cell, bs)
                print(f"{cname:12s} BS{bs:3d} S{streams:2d} {us:9.2f}us "
                      f"vs1str x{base / us:5.2f} exact={not bad}", flush=True)
            hcall, hout = head_call(stack, bs)
            hcall(); torch.cuda.synchronize()
            hus = timeit(hcall)
            print(f"{cname:12s} BS{bs:3d} head {hus:9.2f}us", flush=True)
        del stack
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
