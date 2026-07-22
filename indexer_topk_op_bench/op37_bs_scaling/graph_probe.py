# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op37 mechanism probe 2: CUDA-graph replay removes host issue cost.

Captures the BS-row champion launch DAG as (a) 1-stream chain and
(b) S-stream fork-join, replays each as a single graph. If (b) collapses
toward single-row latency, rows are hardware-concurrent and the BS>1 decay
is host-issue + same-stream serialization, not a kernel-side conflict."""
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bs_ab import CELLS, Stack, build_champion  # noqa: E402

DEV = "cuda"


def make_graph(stack, cmod, bs, streams):
    lg = [stack.logits[i:i + 1] for i in range(bs)]
    pre = [stack.pre[i:i + 1] for i in range(bs)]
    out = torch.empty(bs, stack.K, dtype=torch.int32, device=DEV)
    orow = [out[i:i + 1] for i in range(bs)]
    N, run = stack.N, cmod.run
    ss = [torch.cuda.Stream() for _ in range(streams)]
    g = torch.cuda.CUDAGraph()
    # warm on side stream (torch.cuda.graph does this internally too)
    with torch.cuda.graph(g):
        cap = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cap)
        for s in ss[:min(streams, bs)]:
            s.wait_event(ev)
        for i in range(bs):
            with torch.cuda.stream(ss[i % streams]):
                run(lg[i], pre[i], N, orow[i])
        for s in ss[:min(streams, bs)]:
            cap.wait_stream(s)
    return g, out


def timeit(fn, reps=30):
    evict = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(reps):
        evict.random_()
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def main():
    cmod = build_champion()
    for cname in ("flash_128k", "pro_1024k", "flash_32k"):
        stack = Stack(CELLS[cname], 64)
        # single-row anchor
        g1, o1 = make_graph(stack, cmod, 1, 1)
        g1.replay(); torch.cuda.synchronize()
        assert not stack.exact(o1, 1)
        t1 = timeit(g1.replay)
        print(f"{cname:12s} BS  1 graph        {t1:9.2f}us", flush=True)
        for bs in (8, 64):
            for streams in (1, 8, 16, 32):
                if streams > bs:
                    continue
                g, out = make_graph(stack, cmod, bs, streams)
                g.replay(); torch.cuda.synchronize()
                bad = stack.exact(out, bs)
                us = timeit(g.replay)
                print(f"{cname:12s} BS{bs:3d} graph S{streams:2d} {us:9.2f}us "
                      f"us/row={us / bs:6.2f} exact={not bad}", flush=True)
        del stack
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
