"""Pipeline trace of a recorded agent session (prefill + decode) on the real engine.

Replays a session recorded by agent_capture.py request by request, as the server ran it: each prompt goes
through LLM._prefill (prefix reuse, chunking and bounded replay as served), then the recorded output is
fed back token by token (teacher forcing: the same decode work and the same prefix for the next request
as in the live run, whatever --fast substitutes).

Tracks (trace_decode.py's schema, so trace/embed.py's viewer shows them):
  CPU  main thread. Prefill, per layer: build (lazy graph: hc + attention; Engram lookups) -> sync (routing
       read-back: the GPU runs the previous layer's last expert batch, this layer's attention and routing)
       -> post -> per batch of MoE.EXPERT_BATCH experts: io_issue (lookahead read of the next batch) -> wait
       (this batch's reads) -> launch (graph) -> sync (eval: the GPU computes the batch; not after the last
       batch). The prompt's last token runs the decode path and is traced like a decode token; decode
       tokens get trace_decode.py's per-layer phases.
  SSD  native IO trace: one event per 2 MB chunk; demand reads vs lookahead (low priority).
  GPU  inferred from host syncs (no GPU timestamps without Xcode): a block ends when its sync returns.

usage: trace_agent.py CAPTURE.jsonl OUT.json [--cache-gb 8] [--lossless] [--viewer-decode 4] [--label L]
  OUT.json            viewer data (runs[0]: tokens = named segments, cpu / gpu / io / reqs) + "summary"
  OUT.perfetto.json   every event, for ui.perfetto.dev
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
import trace_decode as TD  # noqa: E402

MODEL = "/Users/loncheng/Models/dsv41-mac"
now = time.perf_counter_ns
BUILD_NOTE = "Python 构建 hc + 注意力计算图（惰性，GPU 尚未执行）"


# ------------------------------------------------------------------------------ capture

def capture(a):
    import mlx.core as mx

    from deepinfer_mac import LLM
    from deepinfer_mac.__main__ import build_parser, decode_mode
    from deepinfer_mac.model import mk
    from deepinfer_mac.model import model as M

    recs = [json.loads(line) for line in open(a.capture) if line.strip()]
    mode = decode_mode(build_parser().parse_args(["bench", "--cache-gb", str(a.cache_gb)] + (["--lossless"] if a.lossless else [])))
    print(mode, flush=True)
    t = time.perf_counter()
    llm = LLM(MODEL, expert_cache_gb=a.cache_gb)
    print(f"model loaded in {time.perf_counter() - t:.1f} s", flush=True)
    store = llm.weights.experts
    io = store.prio
    assert hasattr(io, "trace_start"), "native IO with tracing is required"
    off = [io.lib.mkio_now() - (now() + now()) // 2 for _ in range(5)]
    assert max(abs(o) for o in off) < 50_000, f"clocks differ: {off}"

    raw, S, mark, uninstall_decode = TD.install_hooks(M, mk, mx, store)
    P = {"in_prefill": False, "in_pf": False, "in_slab": False, "layer": -1, "last": None}

    def pm(tag, layer=None, extra=None):
        raw.append((tag, P["layer"] if layer is None else layer, now(), extra))

    o = SimpleNamespace(fwd=M.Transformer.forward, call=M.MoE.__call__, prune=M.MoE._prune, slab=M.MoE._slab_moe,
                        into=M.MoE._experts_into, kv=M.Engram.kv, pf=store.prefetch, gs=store.get_slots, ev=mx.eval)

    def fwd(self, st, tokens, *args, **kw):
        if len(tokens) > 1:
            P["in_pf"] = True
            pm("pf_fwd_enter", -1, {"n": len(tokens), "replay_from": kw.get("replay_from")})
            try:
                return o.fwd(self, st, tokens, *args, **kw)
            finally:
                P["in_pf"] = False
                pm("pf_fwd_exit", -1)
        if P["in_prefill"]:  # the prompt's last token: the decode path, traced as a decode token
            S["layer"] = -1
            mark("tok_start")
            t0 = raw[-1][2]
            r = o.fwd(self, st, tokens, *args, **kw)
            S["layer"] = -1
            mark("fwd_ret")
            P["last"] = (t0, raw[-1][2])
            return r
        return o.fwd(self, st, tokens, *args, **kw)

    def call(self, x, image_mask=None):
        if P["in_pf"] and x.shape[0] > 1:
            layer = int(self.key[1:])
            P["layer"] = layer
            pm("pf_moe_enter", layer)
            try:
                return o.call(self, x, image_mask)
            finally:
                pm("pf_moe_exit", layer)
        return o.call(self, x, image_mask)

    def prune(self, x, w, idx_np):
        if P["in_pf"] and x.shape[0] > 1:
            ids = [e for e in np.unique(idx_np).tolist() if e >= 0]
            pm("pf_sync_ret", None, {"ids": len(ids), "resident": sum(1 for e in ids if store.resident(self.key, e)),
                                     "rows": int(x.shape[0])})
        return o.prune(self, x, w, idx_np)

    def slab(self, x, xq, w, idx_np, y):
        if P["in_pf"] and x.shape[0] > 1:
            P["in_slab"] = True
            pm("pf_slab_enter")
            try:
                return o.slab(self, x, xq, w, idx_np, y)
            finally:
                P["in_slab"] = False
                pm("pf_slab_exit")
        return o.slab(self, x, xq, w, idx_np, y)

    def into(self, y, x, xq, w, idx_np, group):
        if not P["in_slab"]:
            return o.into(self, y, x, xq, w, idx_np, group)
        pm("pf_into_enter", None, {"n": len(group)})
        try:
            return o.into(self, y, x, xq, w, idx_np, group)
        finally:
            pm("pf_into_exit")

    def pf(key, ids, optional=False, cold=False):
        if not P["in_slab"]:
            return o.pf(key, ids, optional=optional, cold=cold)
        ids = [int(e) for e in ids]
        need = [e for e in ids if (key, e) not in store.slabs[key].lru]
        pm("pf_prefetch_enter", None, {"optional": bool(optional)})
        try:
            return o.pf(key, ids, optional=optional, cold=cold)
        finally:
            issued = sum(1 for e in need if (key, e) in store.inflight)
            pm("pf_prefetch_exit", None, {"optional": bool(optional), "n": len(ids), "need": len(need), "issued": issued})

    def gs(key, ids):
        if not P["in_slab"]:
            return o.gs(key, ids)
        ids_l = [int(e) for e in ids]
        miss = sum(1 for e in ids_l if not store.resident(key, e))
        flying = sum(1 for e in ids_l if (key, e) in store.inflight and not store.inflight[(key, e)].done())
        pm("pf_gs_enter", None, {"n": len(ids_l), "miss": miss, "inflight": flying})
        try:
            return o.gs(key, ids)
        finally:
            pm("pf_gs_exit")

    def ev(*args, **kw):
        if P["in_slab"] and not S["in_hot"]:
            pm("pf_eval_call")
            try:
                return o.ev(*args, **kw)
            finally:
                pm("pf_eval_ret")
        return o.ev(*args, **kw)

    def kv(self, hash_ids):
        if not P["in_pf"]:
            return o.kv(self, hash_ids)
        pm("pf_engram_enter", self.layer_id)
        try:
            return o.kv(self, hash_ids)
        finally:
            pm("pf_engram_exit", self.layer_id)

    M.Transformer.forward, M.MoE.__call__, M.MoE._prune, M.MoE._slab_moe = fwd, call, prune, slab
    M.MoE._experts_into, M.Engram.kv = into, kv
    store.prefetch, store.get_slots = pf, gs
    mx.eval = ev

    per_req, base = [], None
    try:
        for i, rec in enumerate(recs):
            if rec.get("images"):
                print(f"request {i + 1}: has images, skipped (the replay covers text prompts)", flush=True)
                continue
            prompt, gen = [int(x) for x in rec["prompt"]], [int(x) for x in rec["generated"]]
            feed = gen[:-1]  # the server never feeds its last sampled token
            s0 = dict(store.stats)
            io.trace_start(1 << 18)
            r0 = len(raw)
            pm("req_start", -1, {"i": i})
            t0 = raw[-1][2]
            base = t0 if base is None else base
            P["in_prefill"], P["last"] = True, None
            try:
                st, logits, _, reused = llm._prefill(prompt, False)
            finally:
                P["in_prefill"] = False
            np.array(logits[-1])  # the logits sync before the first output token is sampled
            S["layer"] = -1
            mark("tok_end")
            t_first = raw[-1][2]
            toks = [(P["last"][0], P["last"][1], t_first)]
            for tok in feed:
                S["layer"] = -1
                mark("tok_start")
                ta = raw[-1][2]
                lg = llm.model.forward(st, [tok])
                S["layer"] = -1
                mark("fwd_ret")
                tb = raw[-1][2]
                np.array(lg[-1])
                mark("tok_end")
                toks.append((ta, tb, raw[-1][2]))
            llm._state = st
            chunks, ioreqs = io.trace_stop()
            s1 = dict(store.stats)
            live = rec.get("stats", {})
            per_req.append({
                "i": i, "prompt": len(prompt), "reused": reused, "new": len(prompt) - reused, "generated": len(gen),
                "live": {"reused": live.get("reused_tokens"), "ttft_s": rec["t_tokens"][0] if rec["t_tokens"] else None,
                         "prefill_s": live.get("prefill_s"), "decode_tok_s": live.get("decode_tok_s"),
                         "GB_read": live.get("expert_GB_read"), "total_s": rec.get("t_end")},
                "t0": t0, "t_first": t_first, "toks": toks, "raw": (r0, len(raw)), "chunks": chunks, "ioreqs": ioreqs,
                "store": {k: s1[k] - s0.get(k, 0) for k in s1 if isinstance(s1[k], (int, float))},
            })
            dec = [(c - a_) / 1e6 for a_, _, c in toks[1:]]
            print(f"request {i + 1}: prompt {len(prompt)} reused {reused} (live {live.get('reused_tokens')}) | "
                  f"first token {(t_first - t0) / 1e9:.2f} s (live {per_req[-1]['live']['ttft_s']}) | "
                  f"decode {len(dec)} x {statistics.mean(dec) if dec else 0:.1f} ms | "
                  f"read {(s1['bytes'] - s0['bytes']) / 1e9:.1f} GB | chunk events {len(chunks)}", flush=True)
    finally:
        M.Transformer.forward, M.MoE.__call__, M.MoE._prune, M.MoE._slab_moe = o.fwd, o.call, o.prune, o.slab
        M.MoE._experts_into, M.Engram.kv = o.into, o.kv
        store.prefetch, store.get_slots = o.pf, o.gs
        mx.eval = o.ev
        uninstall_decode()
    return mode, M, store, raw, per_req, base


# ------------------------------------------------------------------------------ tracks

def build_prefill(markers, ms):
    """CPU and inferred GPU events of the multi-token forwards, plus one record per MoE layer."""
    cpu, gpu, layers = [], [], []
    free = [None]

    def C(kind, layer, t0, t1, note=""):
        if t1 > t0:
            cpu.append({"k": kind, "l": layer, "t0": ms(t0), "t1": ms(t1), **({"n": note} if note else {})})

    def G(kind, layer, t_submit, t_end, note=""):
        start = t_submit if free[0] is None else max(t_submit, free[0])
        if t_end <= start:
            start = max(t_submit, t_end - 1)
        gpu.append({"k": kind, "l": layer, "t0": ms(start), "t1": ms(t_end), "est": False, **({"n": note} if note else {})})
        free[0] = t_end if free[0] is None else max(free[0], t_end)

    cur, L, batch, hot0, chunk = None, None, {}, None, None
    for tag, mlayer, t, ex in markers:
        cl = L["l"] if L is not None else -1
        if tag == "req_start":
            cur = ("other", -1, t)
        elif tag == "pf_fwd_enter":
            if cur is not None:
                C(cur[0], cur[1], cur[2], t, "请求开始：前缀匹配 / 快照恢复" if cur[1] == -1 and cur[0] == "other" else "")
            cur, chunk = ("build", 0, t), ex
        elif tag == "pf_engram_enter":
            C(cur[0], mlayer, cur[2], t, BUILD_NOTE if cur[0] == "build" else "")
            cur = ("other", mlayer, t)
        elif tag == "pf_engram_exit":
            C("other", mlayer, cur[2], t, "Engram：主机查 n-gram 表行 + 构建值投影图")
            cur = ("build", mlayer, t)
        elif tag == "pf_moe_enter":
            C(cur[0], mlayer, cur[2], t, BUILD_NOTE if cur[0] == "build" else "")
            L = {"l": mlayer, "rows": None, "ids": 0, "resident": 0, "batches": 0, "wait_ms": 0.0, "eval_ms": 0.0,
                 "sync_ms": 0.0, "need_look": 0, "issued_look": 0, "demand": 0, "waited_batches": 0,
                 "chunk_n": chunk["n"] if chunk else None, "t_enter": t}
            cur = ("sync", mlayer, t)
        elif tag == "pf_sync_ret":
            C("sync", cl, cur[2], t, "路由同步：GPU 执行上一层最后一批专家 + 本层注意力与路由")
            G("attn", cl, cur[2], t, "上一层最后一批专家 + 本层 hc / 注意力 / 路由（prefill）；结束时刻实测")
            L.update(sync_ms=(t - cur[2]) / 1e6, ids=ex["ids"], resident=ex["resident"], rows=ex["rows"], t_sync=t)
            cur = ("post", cl, t)
        elif tag == "pf_slab_enter":
            C(cur[0], cl, cur[2], t, "读回路由结果、整理专家")
            cur = ("io_issue", cl, t)
        elif tag == "pf_prefetch_enter":
            if cur[0] != "wait":
                C(cur[0], cl, cur[2], t)
                cur = ("io_issue", cl, t)
        elif tag == "pf_prefetch_exit":
            if ex["optional"]:
                C("io_issue", cl, cur[2], t, f"预读下一批：{ex['issued']}/{ex['need']} 个未驻留专家下发读盘（低优先级）")
                L["need_look"] += ex["need"]
                L["issued_look"] += ex["issued"]
                cur = ("other", cl, t)
            else:
                L["demand"] += ex["issued"]
        elif tag == "pf_into_enter":
            C(cur[0], cl, cur[2], t)
            cur, batch = ("other", cl, t), dict(ex)
        elif tag == "pf_gs_enter":
            C(cur[0], cl, cur[2], t)
            batch.update(ex)
            cur = ("wait", cl, t)
        elif tag == "hot_enter":
            hot0 = t
        elif tag == "hot_exit":
            if hot0 is not None:
                G("hot", cl, hot0, t, "hot-wait 保温小 kernel")
            hot0 = None
        elif tag == "pf_gs_exit":
            d = (t - cur[2]) / 1e6
            C("wait", cl, cur[2], t, f"等这一批 {batch.get('n')} 个专家读完（进入时 {batch.get('miss')} 个未就绪，"
                                      f"其中 {batch.get('inflight')} 个已在读）")
            L["batches"] += 1
            L["wait_ms"] += d
            L["waited_batches"] += d > 0.2
            cur = ("launch_miss" if batch.get("miss") else "launch_hit", cl, t)
        elif tag == "pf_into_exit":
            C(cur[0], cl, cur[2], t, f"构建这一批 {batch.get('n')} 个专家的计算图")
            cur = ("other", cl, t)
        elif tag == "pf_eval_call":
            C(cur[0], cl, cur[2], t)
            cur = ("sync", cl, t)
        elif tag == "pf_eval_ret":
            C("sync", cl, cur[2], t, "等 GPU 算完这一批专家（prefill）")
            G("moe", cl, cur[2], t, f"专家批 ×{batch.get('n')}（prefill）；结束时刻实测")
            L["eval_ms"] += (t - cur[2]) / 1e6
            cur = ("io_issue", cl, t)
        elif tag == "pf_slab_exit":
            C(cur[0], cl, cur[2], t)
            cur = ("other", cl, t)
        elif tag == "pf_moe_exit":
            C(cur[0], cl, cur[2], t)
            L["t_exit"] = t
            layers.append(L)
            cur, L = ("build", mlayer + 1, t), None
        elif tag == "pf_fwd_exit":
            if cur is not None:
                C(cur[0], cur[1], cur[2], t, BUILD_NOTE if cur[0] == "build" else "")
            cur = ("other", -1, t)
    return cpu, gpu, layers


def io_events(chunks, ioreqs, keyname, ms, io_ev, rq):
    for tq, t0, t1, gen, tagv, thr, lo, a_, b_ in chunks.tolist():
        if tagv < 0:
            continue
        key = keyname.get(tagv >> 20, "?")
        layer = int(key[1:]) if key[1:].isdigit() else -1
        e = (tagv >> 1) & 0x7FFFF
        io_ev.append({"th": thr, "t0": ms(t0), "t1": ms(t1), "spec": lo, "req": gen, "l": layer, "e": e, "kb": round((b_ - a_) / 1024)})
        r = rq.setdefault(gen, {"req": gen, "l": layer, "e": e, "spec": tagv & 1, "t_sub": ms(tq), "t_done": None,
                                "t_promote": None, "t_cancel": None, "kb": None})
        r["t_done"] = max(r["t_done"] or 0, ms(t1))
    for t, gen, tagv, kind, high, total in ioreqs.tolist():
        if tagv < 0:
            continue
        key = keyname.get(tagv >> 20, "?")
        r = rq.setdefault(gen, {"req": gen, "l": int(key[1:]) if key[1:].isdigit() else -1, "e": (tagv >> 1) & 0x7FFFF,
                                "spec": tagv & 1, "t_sub": ms(t), "t_done": None, "t_promote": None, "t_cancel": None, "kb": None})
        r["kb"] = round(total / 1024)
        if kind == 0:
            r["t_sub"] = ms(t)
        elif kind == 1:
            r["t_promote"] = ms(t)
        elif kind == 2:
            r["t_cancel"] = ms(t)


# ------------------------------------------------------------------------------ analysis

def union(iv):
    out = []
    for a, b in sorted((a, b) for a, b in iv if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def clip(iv, t0, t1):
    return [(max(a, t0), min(b, t1)) for a, b in iv if b > t0 and a < t1]


def length(iv):
    return sum(b - a for a, b in iv)


def inter(x, y):
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if b > a:
            out.append((a, b))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def complement(busy, t0, t1):
    out, t = [], t0
    for a, b in busy:
        if a > t:
            out.append((t, min(a, t1)))
        t = max(t, b)
        if t >= t1:
            break
    if t < t1:
        out.append((t, t1))
    return out


def by_phase(idle, cpu_sorted):
    """Length of `idle` covered by each CPU phase (the main thread's events are sequential)."""
    out, i, j = {}, 0, 0
    while i < len(idle) and j < len(cpu_sorted):
        a, b = max(idle[i][0], cpu_sorted[j]["t0"]), min(idle[i][1], cpu_sorted[j]["t1"])
        if b > a:
            out[cpu_sorted[j]["k"]] = out.get(cpu_sorted[j]["k"], 0) + b - a
        if idle[i][1] < cpu_sorted[j]["t1"]:
            i += 1
        else:
            j += 1
    return {k: round(v, 1) for k, v in sorted(out.items(), key=lambda x: -x[1])}


class Index:
    """Events sorted by start, sliced by time window with bisect (windows are many, events ~10^5)."""

    def __init__(self, events):
        self.ev = events
        self.st = [e["t0"] for e in events]
        self.maxd = max((e["t1"] - e["t0"] for e in events), default=0.0)

    def over(self, t0, t1):
        i, j = bisect.bisect_left(self.st, t0 - self.maxd), bisect.bisect_left(self.st, t1)
        return [e for e in self.ev[i:j] if e["t1"] > t0]


def uclip(u, ust, t0, t1):
    """Clip a sorted disjoint interval list (with its starts `ust`) to [t0, t1]."""
    i, j = max(0, bisect.bisect_right(ust, t0) - 1), bisect.bisect_left(ust, t1)
    return [(max(a, t0), min(b, t1)) for a, b in u[i:j] if b > t0 and a < t1]


def window(work, ssd, cpu_ix, io_ix, t0, t1):
    g, s = uclip(work[0], work[1], t0, t1), uclip(ssd[0], ssd[1], t0, t1)
    both = length(inter(g, s))
    G, Sb, T = length(g), length(s), t1 - t0
    io_w = io_ix.over(t0, t1)
    chunk_ms = sum(min(e["t1"], t1) - max(e["t0"], t0) for e in io_w)
    kb = sum(e["kb"] for e in io_w if e["t0"] >= t0)
    cpu_w = cpu_ix.over(t0, t1)
    kinds = {}
    for e in cpu_w:
        kinds[e["k"]] = kinds.get(e["k"], 0) + min(e["t1"], t1) - max(e["t0"], t0)
    return {
        "T": round(T, 1), "gpu": round(G, 1), "ssd": round(Sb, 1), "both": round(both, 1), "gpu_only": round(G - both, 1),
        "ssd_only": round(Sb - both, 1), "idle": round(T - G - Sb + both, 1), "GB": round(kb * 1024 / 1e9, 2),
        "GBps_busy": round(kb * 1024 / 1e9 / (Sb / 1e3), 2) if Sb else None, "GBps_wall": round(kb * 1024 / 1e9 / (T / 1e3), 2) if T else None,
        "io_concurrency": round(chunk_ms / Sb, 2) if Sb else None,
        "ssd_idle_by_cpu": by_phase(complement(s, t0, t1), cpu_w),
        "gpu_idle_by_cpu": by_phase(complement(g, t0, t1), cpu_w),
        "cpu_by_kind": {k: round(v, 1) for k, v in sorted(kinds.items(), key=lambda x: -x[1])},
    }


def build(a, mode, M, store, raw, per_req, base):
    ms = lambda t: round((t - base) / 1e6, 4)  # noqa: E731
    keyname = {v: k for k, v in store.trace_keys.items()}
    cpu, gpu, io_ev, rq, segs, reqsum = [], [], [], {}, [], []
    empty_c, empty_r = np.zeros((0, 9), dtype=np.int64), np.zeros((0, 6), dtype=np.int64)
    for R in per_req:
        mk_ = raw[R["raw"][0]:R["raw"][1]]
        t_last0 = R["toks"][0][0]
        pcpu, pgpu, layers = build_prefill([m for m in mk_ if m[2] < t_last0], ms)
        cpu += pcpu
        gpu += pgpu
        segs.append({"name": f"请求 {R['i'] + 1} · prefill（新 {R['new']} token）", "t0": ms(R["t0"]), "t1": ms(t_last0),
                     "kind": "prefill", "req": R["i"] + 1})
        starts = [w[0] for w in R["toks"]]
        ends = [w[2] for w in R["toks"]]

        def inside(t):
            j = bisect.bisect_right(starts, t) - 1
            return j >= 0 and t <= ends[j]

        sub = SimpleNamespace(lossless=a.lossless, cache_gb=a.cache_gb, warm=0, tokens=len(R["toks"]))
        run = TD.build_run(sub, mode, M, [0] * R["prompt"], R["toks"], [m for m in mk_ if inside(m[2])], empty_c, empty_r, store, {})
        shift = (R["toks"][0][0] - base) / 1e6
        for e in run["cpu"] + run["gpu"]:
            e["t0"], e["t1"] = round(e["t0"] + shift, 4), round(e["t1"] + shift, 4)
        cpu += run["cpu"]
        gpu += run["gpu"]
        for j, tk in enumerate(run["tokens"]):
            segs.append({"name": f"请求 {R['i'] + 1} · " + ("首个输出 token（prompt 末 token，解码路径）" if j == 0 else f"解码 {j}"),
                         "t0": round(tk["t0"] + shift, 4), "t1": round(tk["t1"] + shift, 4),
                         "kind": "first" if j == 0 else "decode", "req": R["i"] + 1})
        io_events(R["chunks"], R["ioreqs"], keyname, ms, io_ev, rq)
        R["layers"] = layers
    cpu.sort(key=lambda e: e["t0"])
    gpu.sort(key=lambda e: e["t0"])
    io_ev.sort(key=lambda e: e["t0"])
    w_u = union((e["t0"], e["t1"]) for e in gpu if e["k"] != "hot")
    s_u = union((e["t0"], e["t1"]) for e in io_ev)
    work, ssd = (w_u, [a_ for a_, _ in w_u]), (s_u, [a_ for a_, _ in s_u])
    cpu_ix, io_ix = Index(cpu), Index(io_ev)
    for R in per_req:
        pf = window(work, ssd, cpu_ix, io_ix, ms(R["t0"]), ms(R["toks"][0][0]))
        first = window(work, ssd, cpu_ix, io_ix, ms(R["toks"][0][0]), ms(R["t_first"]))
        dec_ms = [(c - a_) / 1e6 for a_, _, c in R["toks"][1:]]
        dec = None
        if dec_ms:
            parts = [window(work, ssd, cpu_ix, io_ix, ms(a_), ms(c)) for a_, _, c in R["toks"][1:]]
            dec = {k: round(sum(p[k] for p in parts) / len(parts), 2) for k in ("T", "gpu", "ssd", "both", "gpu_only", "ssd_only", "idle", "GB")}
            dec.update(n=len(dec_ms), median_ms=round(statistics.median(dec_ms), 1), first5_ms=round(statistics.mean(dec_ms[:5]), 1),
                       rest_ms=round(statistics.mean(dec_ms[5:]), 1) if len(dec_ms) > 5 else None,
                       tok_s=round(1e3 / statistics.mean(dec_ms), 2))
        st = R["store"]
        reqsum.append({
            "req": R["i"] + 1, "prompt": R["prompt"], "reused": R["reused"], "new": R["new"], "generated": R["generated"],
            "live": R["live"], "first_token_s": round((R["t_first"] - R["t0"]) / 1e9, 2),
            "prefill_chunks_s": round((R["toks"][0][0] - R["t0"]) / 1e9, 2), "last_token_ms": round((R["t_first"] - R["toks"][0][0]) / 1e6, 1),
            "store": {"hits": st.get("hits"), "misses": st.get("misses"), "prefetch_hits": st.get("prefetch_hits"),
                      "GB_read": round(st.get("bytes", 0) / 1e9, 2), "spec_GB": round(st.get("spec_bytes", 0) / 1e9, 2),
                      "wait_s": round(st.get("wait_s", 0), 2)},
            "prefill": pf, "first": first, "decode": dec,
            "layers": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in L.items() if not k.startswith("t_")} for L in R["layers"]],
        })
    run = {
        "id": f"agent-{a.label}", "label": f"agent 会话回放 · {a.label}",
        "meta": {"cache_gb": a.cache_gb, "sub_rank": M.MoE.SUB_RANK, "spec_k": M.MoE.SPEC_K, "expert_batch": M.MoE.EXPERT_BATCH,
                 "prompt_tokens": per_req[0]["prompt"] if per_req else 0, "io_threads": 8, "chunk_kb": 2048,
                 "power": TD.power_source(), "mode": mode, "requests": len(per_req),
                 "notes": [
                     "回放 agent_capture.py 录下的真实 dsh 会话：每个请求按服务端的方式 prefill（前缀复用、分块、Bounded Replay），"
                     "再把当时生成的 token 逐个喂回（teacher forcing），解码工作量和下一个请求的前缀与实际运行一致。",
                     "GPU 轨为推断：本机拿不到 Metal GPU 时间戳。prefill 的注意力块与专家批都以主机同步返回的时刻为结束；"
                     "解码同 trace_decode.py（MoE 时长为估计）。",
                     "SSD 轨来自原生 IO 的追踪：每个事件是一个 2 MB 块；prefill 的「推测」是预读下一批专家（低优先级）。",
                 ]},
        "tokens": [{"i": k, **s} for k, s in enumerate(segs)], "cpu": cpu, "gpu": gpu, "io": io_ev,
        "reqs": sorted(rq.values(), key=lambda r: r["t_sub"] or 0),
    }
    return run, reqsum


def merge_io(events, gap=0.02):
    """One bar per IO thread for each run of back-to-back chunks of the same layer and kind: a prefill holds
    ~10^5 chunk events and the viewer needs each thread's occupancy (and its gaps), not every 2 MB chunk."""
    out, last = [], {}
    for e in sorted(events, key=lambda e: (e["th"], e["t0"])):
        m = last.get(e["th"])
        if m is not None and m["l"] == e["l"] and m["spec"] == e["spec"] and e["t0"] - m["t1"] <= gap:
            m["t1"], m["kb"], m["n"] = max(m["t1"], e["t1"]), m["kb"] + e["kb"], m["n"] + 1
            m["e"], m["req"] = "", ""
            continue
        m = dict(e, n=1)
        last[e["th"]] = m
        out.append(m)
    return out


def slim(run, keep_decode):
    """Viewer copy: every prefill segment, the first-token segments and the first `keep_decode` decode tokens of each request."""
    keep, n = [], {}
    for s in run["tokens"]:
        if s["kind"] == "decode":
            n[s["req"]] = n.get(s["req"], 0) + 1
            if n[s["req"]] > keep_decode:
                continue
        keep.append(s)
    wins = sorted((s["t0"], s["t1"]) for s in keep)
    starts = [w[0] for w in wins]

    def inside(t):
        j = bisect.bisect_right(starts, t) - 1
        return j >= 0 and t <= wins[j][1]

    pf = sorted((s["t0"], s["t1"]) for s in keep if s["kind"] == "prefill")
    pf_starts = [w[0] for w in pf]

    def in_prefill(t):
        j = bisect.bisect_right(pf_starts, t) - 1
        return j >= 0 and t <= pf[j][1]

    io = [e for e in run["io"] if inside(e["t0"])]
    out = dict(run)
    out["tokens"] = [{**s, "i": k} for k, s in enumerate(keep)]
    out["cpu"] = [e for e in run["cpu"] if inside(e["t0"])]
    out["gpu"] = [e for e in run["gpu"] if inside(e["t0"])]
    out["io"] = sorted([e for e in io if not in_prefill(e["t0"])] + merge_io([e for e in io if in_prefill(e["t0"])]),
                       key=lambda e: e["t0"])
    out["reqs"] = [r for r in run["reqs"] if r["t_sub"] is not None and inside(r["t_sub"])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--lossless", action="store_true")
    ap.add_argument("--viewer-decode", type=int, default=4, help="decode tokens per request kept in the viewer data")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    a.label = a.label or f"{'--lossless' if a.lossless else '--fast'} · 缓存 {a.cache_gb:g} GiB"
    mode, M, store, raw, per_req, base = capture(a)
    run, reqsum = build(a, mode, M, store, raw, per_req, base)
    json.dump(TD.chrome_trace(run), open(os.path.splitext(a.out)[0] + ".perfetto.json", "w"), ensure_ascii=False)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": [slim(run, a.viewer_decode)], "summary": reqsum}
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    for r in reqsum:
        p, d = r["prefill"], r["decode"] or {}
        print(json.dumps({"req": r["req"], "new": r["new"], "first_token_s": r["first_token_s"], "live_ttft_s": r["live"]["ttft_s"],
                          "prefill": {k: p[k] for k in ("T", "gpu_only", "both", "ssd_only", "idle", "GB", "GBps_busy", "io_concurrency")},
                          "ssd_idle_by_cpu": p["ssd_idle_by_cpu"], "decode_ms": d.get("T"), "decode_n": d.get("n")}, ensure_ascii=False))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB) and the Perfetto JSON")


if __name__ == "__main__":
    main()
