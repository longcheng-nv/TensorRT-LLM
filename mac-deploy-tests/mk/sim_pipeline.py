"""Time model of BS=1 decode with SSD expert streaming, replayed on recorded routing.

Per layer: GPU pre-routing work (hc + attention + router) G_PRE ms -> routing known -> needed
experts (rank < R; ranks >= R are used if resident, else substituted) must be resident -> MoE
G_MOE ms (+ SYNC host overhead) -> next layer. One SSD server, BW GB/s, in CHUNK-ms chunks:
demand reads always before speculative ones (a demand waits at most the chunk in progress); a
speculative read that becomes needed is promoted. Speculation: at layer l's routing, the next
layer's top-K prediction (gate_{l+1} on layer l's MoE input) is queued for non-resident experts,
inserted cold into the LRU. Prints ms/token, SSD busy fraction, reads/token, substitutions/token.
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from sim_cache import slab_plan


class Item:
    __slots__ = ("key", "rem")

    def __init__(self, key, rem):
        self.key, self.rem = key, rem


class SSD:
    def __init__(self, chunk_ms, ready_at):
        self.t, self.chunk, self.busy = 0.0, chunk_ms, 0.0
        self.dq, self.sq = collections.deque(), collections.deque()
        self.ready_at = ready_at
        self.where = {}  # key -> Item (queued or in progress)

    def _step(self, q):
        it = q[0]
        step = min(self.chunk, it.rem)
        self.t += step
        self.busy += step
        it.rem -= step
        if it.rem <= 1e-12:
            q.popleft()
            self.ready_at[it.key] = self.t
            self.where.pop(it.key, None)

    def advance(self, until):
        while self.t < until:
            if self.dq:
                self._step(self.dq)
            elif self.sq:
                self._step(self.sq)
            else:
                self.t = until
                return

    def demand(self, key, dur, now):
        self.advance(now)
        it = Item(key, dur)
        self.dq.append(it)
        self.where[key] = it
        self.ready_at[key] = float("inf")

    def promote(self, key):
        it = self.where.get(key)
        if it is None or it not in self.sq:
            return
        self.sq.remove(it)
        self.dq.append(it)

    def spec(self, key, dur):
        it = Item(key, dur)
        self.sq.append(it)
        self.where[key] = it
        self.ready_at[key] = float("inf")

    def drain_demands(self):
        while self.dq:
            self._step(self.dq)
        return self.t

    def drop(self, key):
        it = self.where.pop(key, None)
        if it is not None and it in self.sq and it is not (self.sq[0] if self.sq else None):
            self.sq.remove(it)


def simulate(tr, budget=16.0, R=6, K=0, g_pre=1.15, g_moe=0.40, sync=0.30, bw=6.05, chunk_mb=2.0, sub=True,
             cancel=True, tokens=None):
    S, P, B, meta = tr["scores"].astype(np.float32), tr["pred"].astype(np.float32), tr["biases"], tr["meta"]
    T, nL, nE = S.shape
    slab_of, cap, stride = slab_plan(budget)
    lru = {lay: collections.OrderedDict() for lay in cap}
    ready_at = {}
    ssd = SSD(chunk_mb / bw, ready_at)  # ms per chunk
    stats = collections.Counter()
    order_all = np.argsort(-(S + B[None]), axis=-1)[..., :6]
    pred_all = np.argsort(-(P + B[np.r_[1:nL, nL - 1]][None]), axis=-1)[..., : max(K, 1)] if K else None
    t = 0.0
    cur_prompt = -1

    def ms_of(l):
        return stride[l] / 1e6 / bw

    def ins(l, e, cold=False):
        od = lru[slab_of[l]]
        k = (l, e)
        if k in od:
            od.move_to_end(k)
            return
        while len(od) >= cap[slab_of[l]]:
            # evict the oldest entry that is not being read
            for v in od:
                if ready_at.get(v, 0.0) != float("inf"):
                    break
            del od[v]
            ready_at.pop(v, None)
        od[k] = 1
        if cold:
            od.move_to_end(k, last=False)

    n_tok = 0
    for ti in range(T):
        if tokens is not None and n_tok >= tokens:
            break
        pi = int(meta[ti][0])
        if pi != cur_prompt:
            cur_prompt = pi
            ssd.dq.clear(); ssd.sq.clear(); ssd.where.clear()
            for k in [k for k, v in ready_at.items() if v == float("inf")]:
                ready_at.pop(k)
                lru[slab_of[k[0]]].pop(k, None)
            for l in range(nL):
                key = f"pref_{pi}_{l}"
                if key in tr:
                    for e in np.unique(tr[key]).tolist():
                        ins(l, e)
                        ready_at[(l, e)] = 0.0
            ssd.t = t
        t0 = t
        for l in range(nL):
            t += g_pre
            ssd.advance(t)
            od = lru[slab_of[l]]
            ranked = order_all[ti, l].tolist()
            used = []
            for r, e in enumerate(ranked):
                k = (l, e)
                if k in od:
                    od.move_to_end(k)
                    used.append(k)
                    if ready_at.get(k, 0.0) <= t:
                        stats["hit"] += 1
                    else:
                        stats["inflight"] += 1
                        ssd.promote(k)
                    continue
                if sub and r >= R:
                    stats["sub"] += 1
                    continue
                ins(l, e)
                ssd.demand(k, ms_of(l), t)
                used.append(k)
                stats["demand"] += 1
            if cancel:  # queued (not started) guesses for this layer that were not selected are dropped
                for k in [k for k in list(ssd.where) if k[0] == l and k not in used]:
                    it = ssd.where[k]
                    if it in ssd.sq and it.rem >= ms_of(l) - 1e-12:
                        ssd.sq.remove(it)
                        ssd.where.pop(k, None)
                        ready_at.pop(k, None)
                        lru[slab_of[l]].pop(k, None)
                        stats["cancel"] += 1
            if any(ready_at.get(k, 0.0) > t for k in used):
                ssd.drain_demands()
                t = max(t, max(ready_at.get(k, 0.0) for k in used))
            if K and l + 1 < nL:
                for e in pred_all[ti, l, :K].tolist():
                    k = (l + 1, e)
                    if k not in lru[slab_of[l + 1]]:
                        ins(l + 1, e, cold=True)
                        ssd.spec(k, ms_of(l + 1))
                        stats["spec"] += 1
            t += sync + g_moe
        n_tok += 1
        stats["tokens"] += 1
        stats["time"] += t - t0
    n = max(1, stats["tokens"])
    return {"budget": budget, "R": R, "K": K, "ms_per_tok": round(stats["time"] / n, 1), "tok_s": round(1000 * n / stats["time"], 2),
            "demand/tok": round(stats["demand"] / n, 1), "spec/tok": round(stats["spec"] / n, 1),
            "cancel/tok": round(stats["cancel"] / n, 1), "inflight/tok": round(stats["inflight"] / n, 1),
            "sub/tok": round(stats["sub"] / n, 1), "ssd_busy": round(ssd.busy / max(1e-9, ssd.t), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budget", type=float, nargs="+", default=[16.0])
    ap.add_argument("--R", type=int, nargs="+", default=[6, 4, 3])
    ap.add_argument("--K", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--g-pre", type=float, default=1.15)
    ap.add_argument("--g-moe", type=float, default=0.40)
    ap.add_argument("--sync", type=float, default=0.30)
    ap.add_argument("--tokens", type=int, default=None)
    a = ap.parse_args()
    tr = dict(np.load(a.trace, allow_pickle=True))
    for b in a.budget:
        for R in a.R:
            for K in a.K:
                print(json.dumps(simulate(tr, b, R, K, g_pre=a.g_pre, g_moe=a.g_moe, sync=a.sync, tokens=a.tokens)), flush=True)
