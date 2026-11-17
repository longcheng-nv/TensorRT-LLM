"""nsys-style event trace of BS=1 decode on the real engine, in the CLI's default mode (--fast) or --lossless.

Tracks
  CPU  main (Python) thread, per layer and in order: build (hc + attention graph, lazy) -> route_build ->
       sync (routing eval: the GPU runs everything queued) -> post (routing read-back, substitution) ->
       launch_hit (MoE of resident experts, async) -> io_issue (cancel stale guesses, demand reads,
       speculation) -> wait (SSD, with hot-wait) -> launch_miss (MoE of the misses, async); then the token
       tail (head graph + logits sync).
  SSD  the native IO trace (deepinfer_mac/native/mkio.c): one event per 2 MB chunk with its worker thread
       and whether it ran at speculative priority; request submit / promote / cancel events.
  GPU  inferred: MLX only executes at eval / async_eval and this Mac has no Xcode (no Metal System Trace),
       so GPU work is placed as an in-order queue: an "attn" block starts no earlier than the routing eval
       call (or the previous block) and ends exactly when that eval returns; MoE blocks start at their
       async_eval and use a duration estimate (EST_US_PER_EXPERT, from the fixed-expert kernel profile);
       "hot" covers hot-wait keep-alive ticks; "head" ends when the logits sync returns.
Clocks: time.perf_counter_ns() and mkio's CLOCK_UPTIME_RAW are the same clock on macOS (checked at start).

Since mk-7 (2026-11-17) also: the shared expert as its own GPU block after the routing sync (MoE.SPLIT_SHARED),
the Engram value projections launched into an SSD wait (fill), and misses whose MoE graph is built before the
read lands and launched by the decode loop after it has built the next layer's attention (MoE.pending).

usage: trace_decode.py OUT.json [--lossless] [--tokens 5] [--warm 8] [--cache-gb 16]
       trace_decode.py --combine FAST.json LOSSLESS.json OUT.json      (viewer data with both runs)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")

EST_US_PER_EXPERT = (234.2 + 134.8) / 6  # moe_up + moe_down per layer of 6 experts (mk_kprof, fixed-expert replay)
EST_US_SHARED = 35.4e6 / 225e9 * 1e6  # split shared expert: 35.4 MB of mxfp8 weights at the measured ~225 GB/s
EST_US_ENGRAM = 162e6 / 225e9 * 1e6  # one Engram value projection (engram.wkv, 162 MB)
now = time.perf_counter_ns


def power_source() -> str:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        return "AC" if "AC Power" in out else "battery"
    except Exception:  # noqa: BLE001
        return "unknown"


def capture(a):
    import mlx.core as mx

    from deepinfer_mac import LLM
    from deepinfer_mac.__main__ import build_parser, decode_mode
    from deepinfer_mac.model import mk
    from deepinfer_mac.model import model as M

    mode = decode_mode(build_parser().parse_args(["bench", "--cache-gb", str(a.cache_gb)] + (["--lossless"] if a.lossless else [])))
    print(mode, flush=True)
    llm = LLM("/Users/loncheng/Models/dsv41-mac", expert_cache_gb=a.cache_gb)
    store = llm.weights.experts
    io = store.prio
    assert hasattr(io, "trace_start"), "native IO with tracing is required"
    off = [io.lib.mkio_now() - (now() + now()) // 2 for _ in range(5)]
    assert max(abs(o) for o in off) < 50_000, f"clocks differ: {off}"

    ids, _ = llm.chat_prompt([{"role": "user", "content": a.prompt}], thinking=False)
    st, logits, _, _ = llm._prefill(ids, False)
    tok = int(np.argmax(np.array(logits[-1])))
    for _ in range(a.warm):
        tok = int(np.argmax(np.array(llm.model.forward(st, [tok])[-1])))

    raw = []  # (tag, layer, t_ns, extra)
    S = {"layer": -1, "in_dmk": False, "in_slab": False, "in_hot": False, "in_gs": False, "synced": False}

    def mark(tag, extra=None):
        raw.append((tag, S["layer"], now(), extra))

    o_dmk, o_slab = M.MoE._decode_mk, M.MoE._slab_moe_mk
    o_eval, o_aeval = mx.eval, mx.async_eval
    o_gs, o_cu, o_pf, o_sp = store.get_slots, store.cancel_unneeded, store.prefetch, store.speculate
    o_hw = mk.hot_wait
    o_rs = store.reserve_slots
    o_res = M.Transformer._resolve_pending
    o_kv = M.Engram.kv

    def dmk(self, x, bias):
        S.update(layer=int(self.key[1:]), in_dmk=True, synced=False)
        mark("dmk_enter")
        try:
            return o_dmk(self, x, bias)
        finally:
            mark("dmk_exit")
            S["in_dmk"] = False

    def slab(self, *args):
        mark("slab_enter")
        S["in_slab"] = True
        try:
            return o_slab(self, *args)
        finally:
            S["in_slab"] = False
            mark("slab_exit")

    def ev(*args, **kw):
        if S["in_dmk"] and not S["synced"] and not S["in_hot"]:
            mark("sync_call")
            r = o_eval(*args, **kw)
            mark("sync_ret")
            S["synced"] = True
            return r
        return o_eval(*args, **kw)

    def aev(*args, **kw):
        if S["in_slab"] and not S["in_hot"]:
            mark("aeval_call")
            r = o_aeval(*args, **kw)
            mark("aeval_ret")
            return r
        if not S["in_dmk"] and not S["in_hot"]:  # the decode loop launches attention (ASYNC_ATTN)
            mark("aeval_loop")
        return o_aeval(*args, **kw)

    def rs(key, ids_):
        if S["in_slab"] and not S["in_gs"]:  # deferred misses: graph built now, launched after the read lands
            mark("reserve", {"n": len(list(ids_))})
        return o_rs(key, ids_)

    def res(launch=True):
        if M.MoE.pending is None:
            return o_res(launch)
        mark("resolve_enter")
        try:
            return o_res(launch)
        finally:
            mark("resolve_exit")

    def kv(self, hash_ids):
        if S["in_slab"]:
            mark("fill_enter")
        return o_kv(self, hash_ids)

    def gs(key, ids_):
        if S["in_slab"]:
            miss = [int(e) for e in ids_ if not store.resident(key, int(e))]
            mark("gs_enter", {"n": len(list(ids_)), "miss": miss})
            S["in_gs"] = True
            try:
                r = o_gs(key, ids_)
            finally:
                S["in_gs"] = False
            mark("gs_exit")
            return r
        return o_gs(key, ids_)

    def hw(fut):
        S["in_hot"] = True
        mark("hot_enter", getattr(fut, "gen", -1))
        try:
            return o_hw(fut)
        finally:
            mark("hot_exit")
            S["in_hot"] = False

    def cu(key, ids_):
        if S["in_slab"] and not S["in_gs"]:
            mark("cancel_enter")
        return o_cu(key, ids_)

    def pf(key, ids_, optional=False, cold=False):
        if S["in_slab"] and not S["in_gs"]:
            mark("prefetch_enter", {"spec": bool(optional)})
        return o_pf(key, ids_, optional=optional, cold=cold)

    def sp(key, ids_, after):
        if S["in_slab"] and not S["in_gs"]:
            mark("speculate_enter")
        return o_sp(key, ids_, after)

    M.MoE._decode_mk, M.MoE._slab_moe_mk = dmk, slab
    mx.eval, mx.async_eval = ev, aev
    store.get_slots, store.cancel_unneeded, store.prefetch, store.speculate = gs, cu, pf, sp
    mk.hot_wait = hw
    store.reserve_slots = rs
    M.Transformer._resolve_pending = staticmethod(res)
    M.Engram.kv = kv
    stats0 = dict(store.stats)
    toks = []
    io.trace_start()
    try:
        for _ in range(a.tokens):
            S["layer"] = -1
            t0 = now()
            mark("tok_start")
            lg = llm.model.forward(st, [tok])
            t1 = now()
            S["layer"] = -1
            mark("fwd_ret")
            tok = int(np.argmax(np.array(lg[-1])))
            t2 = now()
            mark("tok_end")
            toks.append((t0, t1, t2))
    finally:
        chunks, reqs = io.trace_stop()
        M.MoE._decode_mk, M.MoE._slab_moe_mk = o_dmk, o_slab
        mx.eval, mx.async_eval = o_eval, o_aeval
        store.get_slots, store.cancel_unneeded, store.prefetch, store.speculate = o_gs, o_cu, o_pf, o_sp
        mk.hot_wait = o_hw
        store.reserve_slots = o_rs
        M.Transformer._resolve_pending = staticmethod(o_res)
        M.Engram.kv = o_kv
    stats = {k: (store.stats[k] - stats0.get(k, 0)) if isinstance(store.stats[k], (int, float)) else store.stats[k] for k in store.stats}
    return build_run(a, mode, M, ids, toks, raw, chunks, reqs, store, stats)


def build_run(a, mode, M, prompt_ids, toks, raw, chunks, reqs, store, stats):
    base = toks[0][0]
    ms = lambda t: round((t - base) / 1e6, 4)  # noqa: E731
    cpu, gpu = [], []
    keyname = {v: k for k, v in store.trace_keys.items()}

    def C(kind, l, t0, t1, note=""):
        if t1 > t0:
            cpu.append({"k": kind, "l": l, "t0": ms(t0), "t1": ms(t1), **({"n": note} if note else {})})

    gpu_free = [base]

    def G(kind, l, t_submit, t_end=None, dur_ns=None, note=""):
        start = max(t_submit, gpu_free[0])
        est = t_end is None
        end = t_end if not est else start + dur_ns
        if not est and end <= start:  # the observed end bounds everything queued before it
            start = max(t_submit, end - 1)
        gpu.append({"k": kind, "l": l, "t0": ms(start), "t1": ms(end), "est": est, **({"n": note} if note else {})})
        gpu_free[0] = max(gpu_free[0], end)

    # walk the host markers in order; per layer the phases are contiguous
    BUILD = "Python 构建 hc + 注意力计算图（惰性，GPU 尚未执行）"
    split = bool(getattr(M.MoE, "SPLIT_SHARED", False))
    cur = None  # (phase, layer, t_start)
    hits_n = 0
    attn_submit = None  # the decode loop's async launch of the next attention (ASYNC_ATTN)
    deferred = None  # (layer, misses) whose MoE waits in MoE.pending
    n_fill = 0
    for tag, l, t, extra in raw:
        if tag == "tok_start":
            cur = ("build", 0, t)
            gpu_free[0] = max(gpu_free[0], t)
            attn_submit = None
        elif tag == "aeval_loop":
            attn_submit = t
        elif tag == "dmk_enter":
            C(cur[0], cur[1] if cur[0] == "build" else l, cur[2], t, BUILD if cur[0] == "build" else "")
            cur = ("route_build", l, t)
        elif tag == "sync_call":
            C("route_build", l, cur[2], t)
            sync_call = t
            cur = ("sync", l, t)
        elif tag == "sync_ret":
            C("sync", l, cur[2], t, "等 GPU 执行注意力与路由" + ("（共享专家已拆到同步之后）" if split else ""))
            # the attention block cannot start before queued MoE of the previous layer ends; trim estimates
            if gpu and gpu[-1]["est"] and gpu[-1]["t1"] > ms(t):
                gpu[-1]["t1"] = ms(max(sync_call, t - 1))
                gpu_free[0] = min(gpu_free[0], max(sync_call, t - 1))
            sub = sync_call if attn_submit is None else min(attn_submit, sync_call)
            G("attn", l, sub, t_end=t, note="hc + 注意力 + 路由；结束时刻实测，开始时刻不早于提交（异步提交或 eval 调用）")
            attn_submit = None
            if split:
                G("shared", l, t, dur_ns=int(EST_US_SHARED * 1e3), note="共享专家（拆分后排在路由同步之后执行），时长按带宽估计")
            cur = ("post", l, t)
        elif tag == "slab_enter":
            C("post", l, cur[2], t, "读回路由结果" + ("、专家替换" if M.MoE.SUB_RANK is not None else ""))
            cur = ("launch_hit", l, t)
            hits_n = 0
        elif tag == "gs_enter":
            if extra["miss"]:  # the miss group: waits for SSD inside
                C(cur[0], l, cur[2], t)
                cur = ("launch_miss", l, t)
                miss_n = extra["n"]
            else:
                hits_n = extra["n"]
        elif tag in ("cancel_enter", "prefetch_enter", "speculate_enter"):
            if cur[0] == "launch_hit":  # no resident expert was launched: straight to the IO bookkeeping
                C("post", l, cur[2], t, "分出命中 / 未命中")
                cur = ("io_issue", l, t)
        elif tag == "hot_enter":
            C(cur[0], l, cur[2], t)
            cur = ("wait", l, t)
            hot_gen = extra
        elif tag == "hot_exit":
            C("wait", l, cur[2], t, f"等 SSD 读完（请求 {hot_gen}），期间 hot-wait 提交保温小 kernel")
            G("hot", l, cur[2], t_end=t, note="hot-wait 保温小 kernel")
            cur = ("launch_miss", l, t)
        elif tag == "fill_enter":
            if cur[0] != "fill":
                C(cur[0], l, cur[2], t)
                cur = ("fill", l, t)
                n_fill = 0
            n_fill += 1
        elif tag == "reserve":
            C(cur[0], l, cur[2], t)
            miss_n = extra["n"]
            deferred = (l, miss_n)
            cur = ("launch_miss", l, t)
        elif tag == "resolve_enter":
            C(cur[0], cur[1], cur[2], t, BUILD + "；上一层的读盘仍在进行" if cur[0] == "build" else "")
            nxt_build = cur[1] if cur[0] == "build" else min(l + 1, 39)
            cur = ("wait", l, t)
        elif tag == "resolve_exit":
            C(cur[0], l, cur[2], t, "读盘完成：未命中专家的 MoE 与下一层注意力一起提交" if cur[0] == "launch_miss" else "")
            if deferred is not None:
                dl, dn = deferred
                G("moe", dl, t, dur_ns=int(EST_US_PER_EXPERT * 1e3 * max(1, dn)),
                  note=f"未命中专家 ×{dn}（图在等盘前已构建，读完后提交），时长估计")
                deferred = None
            cur = ("build", nxt_build, t)
        elif tag == "aeval_ret":
            if cur[0] == "fill":
                C("other", l, cur[2], t, f"Engram 值投影 ×{n_fill}：取表行、构建并提交（填进等盘时的 GPU 空档）")
                G("engram", l, t, dur_ns=int(EST_US_ENGRAM * 1e3 * n_fill), note=f"Engram 值投影 ×{n_fill}（engram.wkv GEMV），时长按带宽估计")
                cur = ("other", l, t)
            elif cur[0] == "launch_hit":
                C("launch_hit", l, cur[2], t, f"命中 {hits_n} 个专家")
                G("moe", l, t, dur_ns=int(EST_US_PER_EXPERT * 1e3 * max(1, hits_n)), note=f"命中专家 ×{hits_n}，时长估计")
                cur = ("io_issue", l, t)
            elif cur[0] == "launch_miss":
                C("launch_miss", l, cur[2], t, f"未命中 {miss_n} 个专家")
                G("moe", l, t, dur_ns=int(EST_US_PER_EXPERT * 1e3 * max(1, miss_n)), note=f"未命中专家 ×{miss_n}，时长估计")
                cur = ("other", l, t)
        elif tag == "slab_exit":
            C(cur[0], l, cur[2], t)
            cur = ("other", l, t)
        elif tag == "dmk_exit":
            C(cur[0], l, cur[2], t)
            cur = ("build", min(l + 1, 39), t)
        elif tag == "fwd_ret":
            C("tail", -1, cur[2], t, "构建 head 计算图")
            fwd_ret = t
            cur = ("tail", -1, t)
        elif tag == "tok_end":
            C("tail", -1, cur[2], t, "logits 同步（GPU 执行最后的 MoE 与 head）")
            G("head", -1, fwd_ret, t_end=t, note="最后一层 MoE 尾部 + head；结束时刻实测")
            cur = None

    io_ev, rq = [], {}
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
    for t, gen, tagv, kind, high, total in reqs.tolist():
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
    tokens = [{"i": i, "t0": ms(t0), "t1": ms(t2)} for i, (t0, t1, t2) in enumerate(toks)]
    msk = [round((t2 - t0) / 1e6, 2) for t0, _, t2 in toks]
    pw = power_source()
    notes = [
        "GPU 轨为推断：本机没有 Xcode，拿不到 Metal GPU 时间戳。MLX 只在 eval / async_eval 时提交，注意力块的结束时刻取路由同步返回的实测值；MoE 块从提交时刻起按专家全命中模式的实测吞吐估计时长（斜线），hot-wait 期间 GPU 在跑保温小 kernel。",
        "SSD 轨来自原生 IO 的追踪：每个事件是一个 2 MB 块，所在行是执行它的 IO 线程；推测读在开始时若仍是低优先级则标为推测。",
        f"CPU 轨是主线程的主机时间戳；插桩本身每层约增加十几微秒。采集时电源：{'接电源' if pw == 'AC' else ('电池' if pw == 'battery' else '未知')}。",
        f"预热 {a.warm} 个 token 后连续记录 {a.tokens} 个 token；提示为 {len(prompt_ids)} token 的短提示。",
    ]
    run = {
        "id": "lossless" if a.lossless else "fast",
        "label": "--lossless（无损）" if a.lossless else "默认 --fast（有损）",
        "meta": {"cache_gb": a.cache_gb, "sub_rank": M.MoE.SUB_RANK, "spec_k": M.MoE.SPEC_K, "prompt_tokens": len(prompt_ids),
                 "warm_tokens": a.warm, "traced_tokens": a.tokens, "io_threads": max(8, int(max([c[5] for c in chunks.tolist()] or [7]) + 1)),
                 "chunk_kb": 2048, "power": pw, "ms_per_token": msk, "mode": mode,
                 "store": {k: v for k, v in stats.items() if k in ("hits", "misses", "prefetch_hits", "spec_cancelled")},
                 "notes": notes},
        "tokens": tokens, "cpu": cpu, "gpu": gpu, "io": io_ev, "reqs": sorted(rq.values(), key=lambda r: r["t_sub"] or 0),
    }
    return run


def chrome_trace(run):
    """Chrome / Perfetto trace-event JSON (open in ui.perfetto.dev or chrome://tracing)."""
    ev = [{"ph": "M", "name": "process_name", "pid": 1, "args": {"name": f"DSV4.1-Flash decode · {run['label']}"}}]
    tid_names = {1: "CPU 主线程", 2: "GPU（推断）"}
    for th in range(run["meta"]["io_threads"]):
        tid_names[10 + th] = f"SSD IO {th}"
    for tid, name in tid_names.items():
        ev.append({"ph": "M", "name": "thread_name", "pid": 1, "tid": tid, "args": {"name": name}})
        ev.append({"ph": "M", "name": "thread_sort_index", "pid": 1, "tid": tid, "args": {"sort_index": tid}})
    for c in run["cpu"]:
        ev.append({"ph": "X", "pid": 1, "tid": 1, "name": c["k"], "ts": c["t0"] * 1e3, "dur": (c["t1"] - c["t0"]) * 1e3,
                   "args": {"layer": c["l"], **({"note": c["n"]} if "n" in c else {})}})
    for g in run["gpu"]:
        ev.append({"ph": "X", "pid": 1, "tid": 2, "name": g["k"] + (" (est)" if g["est"] else ""), "ts": g["t0"] * 1e3,
                   "dur": (g["t1"] - g["t0"]) * 1e3, "args": {"layer": g["l"], **({"note": g["n"]} if "n" in g else {})}})
    for x in run["io"]:
        ev.append({"ph": "X", "pid": 1, "tid": 10 + x["th"], "name": ("spec " if x["spec"] else "demand ") + f"L{x['l']:02d} e{x['e']}",
                   "ts": x["t0"] * 1e3, "dur": (x["t1"] - x["t0"]) * 1e3, "args": {"req": x["req"], "kb": x["kb"]}})
    for t in run["tokens"]:
        ev.append({"ph": "i", "s": "g", "pid": 1, "tid": 1, "name": f"token {t['i'] + 1}", "ts": t["t0"] * 1e3})
    return {"traceEvents": ev, "displayTimeUnit": "ms"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--lossless", action="store_true")
    ap.add_argument("--tokens", type=int, default=5)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--cache-gb", type=float, default=16.0)
    ap.add_argument("--prompt", default="Write a long paragraph about the history of tea.")
    ap.add_argument("--combine", nargs=3, metavar=("FAST", "LOSSLESS", "OUT"))
    a = ap.parse_args()
    if a.combine:
        runs = [json.load(open(p)) for p in a.combine[:2]]
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": runs}, open(a.combine[2], "w"), ensure_ascii=False)
        return
    run = capture(a)
    json.dump(run, open(a.out, "w"), ensure_ascii=False)
    json.dump(chrome_trace(run), open(a.out.replace(".json", ".chrome.json"), "w"), ensure_ascii=False)
    m = run["meta"]
    print(f"{run['label']}: {len(run['cpu'])} cpu / {len(run['gpu'])} gpu / {len(run['io'])} io chunk events, "
          f"{len(run['reqs'])} requests | ms/token {m['ms_per_token']} | power {m['power']} | store {m['store']}", flush=True)


if __name__ == "__main__":
    main()
