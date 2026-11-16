#!/usr/bin/env python3
"""Write a realistic *synthetic* DATA.json for testing trace_viewer_template.html.

usage: python3 make_synthetic.py [DATA.json] [--seed N] [--tokens N]

Two runs in the viewer schema: "fast" (--fast, ~95-110 ms/token, a few demand reads) and
"lossless" (--lossless, ~210-240 ms/token, ~70-80 demand reads of 15.5 MB per token).
The CPU timeline is simulated per layer (build -> route_build -> sync -> post -> launch_hit ->
io_issue -> wait -> launch_miss, then a per-token tail); GPU blocks are derived the way the real
tracer infers them (attn ends at the sync end, MoE durations are estimates, hot-wait fills the
waits); SSD reads go through a small discrete-event model of the native IO pool: two priorities
(demand first, speculative after), non-preemptive ~2 MB chunks on N worker threads, with
speculative requests promoted or cancelled at the next layer's io_issue. Every io chunk belongs
to a req, and each req's t_done is the end of its last chunk.
"""
from __future__ import annotations

import argparse
import copy
import json
import random

LAYERS = 40
N_EXPERTS = 384
TOPK = 6
EXPERT_KB = 15872          # one routed expert (w1|w3|w2) = 15.5 MB
CHUNK_KB = 2048
IO_THREADS = 8
SOURCE_LAYERS = {2, 8, 14, 20, 24, 28, 32, 36}   # compressor / indexer layers: slightly longer attention


class IOSim:
    """Native IO pool model: demand chunks before speculative ones, FIFO within a priority,
    a chunk runs to completion once started (cancel / promote only affect chunks not yet started)."""

    def __init__(self, rng: random.Random, n_threads: int, chunk_ms: tuple[float, float], spec_conc: int):
        self.rng = rng
        self.free = [0.0] * n_threads
        self.chunk_ms = chunk_ms
        self.spec_conc = spec_conc          # at most this many speculative chunks in flight
        self.spec_running: list[float] = []  # end times of speculative chunks that may still run
        self.pending: list[dict] = []
        self.reqs: list[dict] = []
        self.chunks: list[dict] = []
        self.next_id = 1001

    def submit(self, t: float, layer: int, expert: int, spec: bool) -> dict:
        sizes, left = [], EXPERT_KB
        while left > 0:
            sizes.append(min(CHUNK_KB, left))
            left -= sizes[-1]
        r = {"req": self.next_id, "l": layer, "e": expert, "spec": 1 if spec else 0,
             "t_sub": t, "t_done": None, "t_promote": None, "t_cancel": None, "kb": EXPERT_KB,
             "_prio": 1 if spec else 0, "_ready": t, "_left": sizes, "_ends": []}
        self.next_id += 1
        self.reqs.append(r)
        self.pending.append(r)
        return r

    def _plan(self):
        """Next chunk to start: (thread, start time, req) or None when nothing is queued."""
        live = [r for r in self.pending if r["_left"]]
        if not live:
            return None
        th = min(range(len(self.free)), key=lambda i: self.free[i])
        start = max(self.free[th], min(r["_ready"] for r in live))
        floor = min(self.free)
        self.spec_running = [e for e in self.spec_running if e > floor]
        while True:
            cands = [r for r in live if r["_ready"] <= start]
            r = min(cands, key=lambda q: (q["_prio"], q["_ready"], q["req"]))
            if r["_prio"] == 0:
                return th, start, r
            ends = sorted(e for e in self.spec_running if e > start)
            if len(ends) < self.spec_conc:
                return th, start, r
            # wait for a speculative slot, unless a queued demand request shows up first
            t_slot = ends[len(ends) - self.spec_conc]
            t_dem = min((q["_ready"] for q in live if q["_prio"] == 0 and q["_ready"] > start), default=t_slot)
            start = min(t_slot, t_dem)

    def step(self) -> bool:
        plan = self._plan()
        if plan is None:
            return False
        th, start, r = plan
        kb = r["_left"].pop(0)
        lo, hi = self.chunk_ms
        t1 = start + self.rng.uniform(lo, hi) * kb / CHUNK_KB
        self.free[th] = t1
        if r["_prio"]:
            self.spec_running.append(t1)
        self.chunks.append({"th": th, "t0": start, "t1": t1, "spec": r["_prio"], "req": r["req"],
                            "l": r["l"], "e": r["e"], "kb": kb})
        r["_ends"].append(t1)
        if not r["_left"]:
            r["t_done"] = max(r["_ends"])
            self.pending.remove(r)
        return True

    def advance(self, t: float) -> None:
        """Start every chunk whose start time is <= t (nothing new is submitted before t)."""
        while True:
            plan = self._plan()
            if plan is None or plan[1] > t:
                return
            self.step()

    def finish(self, reqs: list[dict]) -> float:
        """Run the pool until every chunk of `reqs` has started; return when the last one ends."""
        while any(r["_left"] for r in reqs):
            if not self.step():
                break
        return max(r["t_done"] for r in reqs)

    def promote(self, r: dict, t: float) -> None:
        # chunks not started by now continue as demand reads, from the promotion time on
        r["t_promote"] = t
        r["_prio"] = 0
        r["_ready"] = max(r["_ready"], t)

    def cancel(self, r: dict, t: float) -> None:
        r["t_cancel"] = t
        r["_left"] = []
        r["t_done"] = max(r["_ends"]) if r["_ends"] else None
        if r in self.pending:
            self.pending.remove(r)

    def is_done_by(self, r: dict, t: float) -> bool:
        return not r["_left"] and r["t_done"] is not None and r["t_done"] <= t


def pick(rng: random.Random, weights: dict[int, float]) -> int:
    ks = list(weights)
    return rng.choices(ks, weights=[weights[k] for k in ks])[0]


def r4(x):
    return None if x is None else round(x, 4)


def ev(lst, k, l, t0, t1, n=None, **kw):
    e = {"k": k, "l": l, "t0": t0, "t1": t1}
    e.update(kw)
    if n:
        e["n"] = n
    lst.append(e)
    return t1


def gen_token(rng: random.Random, cfg: dict, st: dict, cpu: list, gpu: list, ti: int) -> dict:
    """Simulate one decode token from st["t"]; appends to cpu / gpu / st["io"] and returns the token."""
    fast, io, spec_for = cfg["fast"], st["io"], st["spec_for"]
    t, gpu_free = st["t"], st["gpu_free"]
    tok0 = t
    t = ev(cpu, "other", -1, t, t + rng.uniform(0.15, 0.3), "token 开始：embedding 与 Engram 行预取下发")
    if cfg["gc_token"] == ti:
        # an occasional interpreter pause; deliberately an unknown kind (the viewer shows it as "other")
        t = ev(cpu, "gc", -1, t, t + rng.uniform(0.9, 1.3), "Python 垃圾回收暂停")
    for L in range(LAYERS):
        t = ev(cpu, "build", L, t, t + rng.uniform(0.20, 0.32))
        t = ev(cpu, "route_build", L, t, t + rng.uniform(0.05, 0.09))
        # sync: the graph is submitted at eval; the GPU starts once it is free; the host wakes at the end
        s0 = t
        a0 = max(s0, gpu_free) + rng.uniform(0.01, 0.03)
        a1 = a0 + rng.uniform(1.05, 1.22) + (rng.uniform(0.10, 0.20) if L in SOURCE_LAYERS else 0.0)
        ev(gpu, "attn", L, a0, a1, est=False)
        t = ev(cpu, "sync", L, s0, a1)
        gpu_free = a1

        # routing outcome for this layer
        specs = spec_for.pop(L, [])
        needed_spec = [r for r in specs if rng.random() < cfg["p_need"]]
        n_dem = pick(rng, cfg["demand"])
        n_sub = pick(rng, {0: 0.45, 1: 0.35, 2: 0.15, 3: 0.05}) if fast else 0
        post_n = f"替换 {n_sub} 个未缓存专家（第 3–6 名）" if n_sub else None
        t = ev(cpu, "post", L, t, t + rng.uniform(0.06, 0.11) + (0.03 if fast else 0.0), post_n)

        # launch_hit, then io_issue decides promote / cancel for the speculative reads of this layer
        lh0 = t
        lh1 = lh0 + rng.uniform(0.05, 0.09)
        io.advance(lh1)
        promoted, cancelled = [], []
        for r in specs:
            if r in needed_spec:
                # routed to: promote if still in flight and it must be exact; otherwise it is a hit
                # (already read) or, in --fast, substituted and left to finish at low priority
                if not io.is_done_by(r, lh1) and rng.random() < cfg["p_exact"]:
                    promoted.append(r)
            elif r["_left"]:
                cancelled.append(r)
        n_need = min(TOPK, n_dem + len(promoted))
        n_hit = TOPK - n_need
        t = ev(cpu, "launch_hit", L, lh0, lh1, f"命中 {n_hit} 个专家")
        m0 = max(t, gpu_free) + 0.01
        m1 = m0 + 0.055 * n_hit + rng.uniform(0.08, 0.12)
        ev(gpu, "moe", L, m0, m1, est=True, n=f"命中专家 {n_hit} 个（时长按专家数估计）")
        gpu_free = m1

        io0 = t
        for r in cancelled:
            io.cancel(r, io0)
        for r in promoted:
            io.promote(r, io0)
        used = {r["e"] for r in specs}
        demand = []
        for _ in range(n_dem):
            e = rng.randrange(N_EXPERTS)
            while e in used:
                e = rng.randrange(N_EXPERTS)
            used.add(e)
            demand.append(io.submit(io0 + 0.01, L, e, spec=False))
        new_spec = []
        if L + 1 < LAYERS:
            taken = set()
            for _ in range(pick(rng, cfg["spec"])):
                e = rng.randrange(N_EXPERTS)
                while e in taken:
                    e = rng.randrange(N_EXPERTS)
                taken.add(e)
                new_spec.append(io.submit(io0 + 0.02, L + 1, e, spec=True))
            spec_for[L + 1] = new_spec
        parts = []
        if cancelled:
            parts.append(f"取消 {len(cancelled)}")
        if promoted:
            parts.append(f"提升 {len(promoted)}")
        parts.append(f"需求读 {len(demand)}")
        parts.append(f"推测 {len(new_spec)}")
        t = ev(cpu, "io_issue", L, io0,
               io0 + 0.03 + 0.012 * (len(demand) + len(new_spec) + len(cancelled) + len(promoted)),
               " · ".join(parts))

        # wait for this layer's demand + promoted reads (hot-wait keeps the GPU clocked meanwhile)
        w0 = t
        need = demand + promoted
        if need:
            done = io.finish(need)
            w1 = max(w0 + 0.01, done + rng.uniform(0.02, 0.05))
            wn = f"等 {len(need)} 个读完" + (f"（含 {len(promoted)} 个提升）" if promoted else "")
        else:
            w1 = w0 + rng.uniform(0.004, 0.012)
            wn = None
        ev(cpu, "wait", L, w0, w1, wn)
        if w1 - w0 > 0.25:
            h0 = max(w0, gpu_free) + 0.02
            if w1 - h0 > 0.05:
                ev(gpu, "hot", L, h0, w1, est=False, n=f"保温 kernel ×{max(1, round((w1 - h0) / 0.06))}")
                gpu_free = w1
        t = w1

        lm0 = t
        if n_need:
            t = ev(cpu, "launch_miss", L, lm0, lm0 + rng.uniform(0.04, 0.07), f"未命中专家 {n_need} 个")
            m0 = max(t, gpu_free) + 0.01
            m1 = m0 + 0.055 * n_need + rng.uniform(0.04, 0.06)
            ev(gpu, "moe", L, m0, m1, est=True, n=f"未命中专家 {n_need} 个（时长按专家数估计）")
            gpu_free = m1
        else:
            t = ev(cpu, "launch_miss", L, lm0, lm0 + rng.uniform(0.006, 0.015))

    # token tail: head / logits on the GPU, then sampling on the host
    tail0 = t
    h0 = max(tail0, gpu_free) + 0.01
    h1 = h0 + rng.uniform(2.5, 2.9)
    ev(gpu, "head", -1, h0, h1, est=False, n="head / logits（0.7 GB）")
    t = h1 + rng.uniform(0.25, 0.5)
    ev(cpu, "tail", -1, tail0, t, "head 与采样同步")
    st["t"], st["gpu_free"] = t, h1
    return {"i": ti, "t0": tok0, "t1": t}


def gen_run(rng: random.Random, mode: str, n_tokens: int) -> dict:
    fast = mode == "fast"
    cfg = {
        "fast": fast,
        # demand reads per layer (top-2 misses only in --fast; every miss in --lossless)
        "demand": {0: 0.84, 1: 0.15, 2: 0.01} if fast else {0: 0.06, 1: 0.32, 2: 0.42, 3: 0.17, 4: 0.03},
        # speculative reads issued for the next layer (predictions that are not cached yet)
        "spec": {0: 0.40, 1: 0.50, 2: 0.10} if fast else {0: 0.45, 1: 0.40, 2: 0.15},
        "p_need": 0.35 if fast else 0.40,       # a speculative read turns out to be routed to
        "p_exact": 0.4 if fast else 1.0,        # ... and must be exact (--fast: top-2 only, else substituted)
        "chunk_ms": (1.85, 2.2) if not fast else (2.05, 2.45),   # one 2 MB chunk, 8 threads busy
        "spec_conc": 4,                         # speculative chunks in flight at most
        "target_ms": (96.0, 109.0) if fast else (212.0, 238.0),
        "gc_token": None if fast else 2,
    }
    st = {"io": IOSim(rng, IO_THREADS, cfg["chunk_ms"], cfg["spec_conc"]), "spec_for": {},
          "t": 0.0, "gpu_free": 0.0}
    cpu: list[dict] = []
    gpu: list[dict] = []
    tokens: list[dict] = []
    lo, hi = cfg["target_ms"]
    for ti in range(n_tokens):
        # snapshot so a token that lands outside the target band can be re-drawn with the IO state intact
        snap = copy.deepcopy((st["io"], st["spec_for"], st["t"], st["gpu_free"]))
        n_cpu, n_gpu = len(cpu), len(gpu)
        for attempt in range(80):
            if attempt:
                io_c, spec_c, t_c, g_c = copy.deepcopy(snap)
                io_c.rng = rng
                st.update(io=io_c, spec_for=spec_c, t=t_c, gpu_free=g_c)
                del cpu[n_cpu:]
                del gpu[n_gpu:]
            tok = gen_token(rng, cfg, st, cpu, gpu, ti)
            if lo <= tok["t1"] - tok["t0"] <= hi:
                break
        tokens.append(tok)
    io = st["io"]

    # reads still queued when tracing stopped are dropped (chunks already started still finish)
    io.advance(st["t"])
    for r in list(io.pending):
        io.cancel(r, st["t"])

    for e in cpu + gpu:
        e["t0"], e["t1"] = r4(e["t0"]), r4(e["t1"])
    chunks = sorted(io.chunks, key=lambda c: (c["t0"], c["th"]))
    for c in chunks:
        c["t0"], c["t1"] = r4(c["t0"]), r4(c["t1"])
    reqs = [{k: (r4(v) if k.startswith("t_") else v) for k, v in r.items() if not k.startswith("_")}
            for r in io.reqs]
    for tk in tokens:
        tk["t0"], tk["t1"] = r4(tk["t0"]), r4(tk["t1"])
    ms = [round(tk["t1"] - tk["t0"], 1) for tk in tokens]

    meta = {
        "cache_gb": 16, "sub_rank": 2 if fast else None, "spec_k": 2 if fast else 4,
        "prompt_tokens": 14, "warm_tokens": 8, "traced_tokens": n_tokens,
        "io_threads": IO_THREADS, "chunk_kb": CHUNK_KB, "power": "AC", "ms_per_token": ms,
        "notes": [
            "合成数据：只用于测试查看器，数量级仿照实测，不是实测结果。",
            "GPU 区间由主机同步点推断：注意力块的结束取路由同步返回的时刻，MoE 块的时长按专家数估计。",
            ("--fast（有损）：第 3–6 名未缓存专家换成缓存内专家，需求读只剩前 2 名的缺失；推测预取 2 个。"
             if fast else
             "--lossless（无损）：6 个专家全部精确，缺失即需求读；推测预取 4 个，命中时提升、落空时取消。"),
        ] + ([] if fast else ["token 3 开头有一次 Python 垃圾回收暂停（类型 gc，查看器按「其他」显示）。"]),
    }
    return {
        "id": mode,
        "label": "默认 --fast（有损）" if fast else "--lossless（无损）",
        "meta": meta,
        "tokens": tokens,
        "cpu": cpu,
        "gpu": gpu,
        "io": chunks,
        "reqs": reqs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", default="DATA.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tokens", type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    data = {"generated": "2026-11-16 23:00",
            "runs": [gen_run(rng, "fast", a.tokens), gen_run(rng, "lossless", a.tokens)]}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    for run in data["runs"]:
        n = len(run["tokens"]) or 1
        dem = sum(1 for r in run["reqs"] if not r["spec"])
        spc = sum(1 for r in run["reqs"] if r["spec"])
        prom = sum(1 for r in run["reqs"] if r["t_promote"] is not None)
        canc = sum(1 for r in run["reqs"] if r["t_cancel"] is not None)
        print(f"{run['id']:9s} ms/token {run['meta']['ms_per_token']}  per token: demand {dem / n:.1f}, "
              f"spec {spc / n:.1f}, promoted {prom / n:.1f}, cancelled {canc / n:.1f}; "
              f"events cpu {len(run['cpu'])} gpu {len(run['gpu'])} io {len(run['io'])} reqs {len(run['reqs'])}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
