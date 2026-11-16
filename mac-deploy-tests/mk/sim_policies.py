"""Replacement policies for the expert slab cache, replayed on recorded routing (prefill accesses as
warm-up per prompt, decode accesses counted). Same slab capacities as ExpertStore at the budget.
Policies: lru, 2q (probation FIFO of size f*cap; a hit promotes to the main LRU), lru2 (evict the
largest backward K=2 distance), dlfu (exponentially decayed frequency, evict min), and Belady.
Optional --rank-limit R: only accesses at rank < R count / need residency (ranks >= R are
substituted when missing; a resident one is still used and refreshed)."""
from __future__ import annotations

import argparse
import collections
import heapq
import math

import numpy as np

from sim_cache import slab_plan


def accesses(tr, topk=6):
    S, B, meta = tr["scores"].astype(np.float32), tr["biases"], tr["meta"]
    T, nL, _ = S.shape
    out = []  # (decode?, layer, expert, rank)
    cur = -1
    for t in range(T):
        pi = int(meta[t][0])
        if pi != cur:
            cur = pi
            for l in range(nL):
                key = f"pref_{pi}_{l}"
                if key in tr:
                    for e in np.unique(tr[key]).tolist():
                        out.append((False, l, e, 0))
        order = np.argsort(-(S[t] + B), axis=-1)[:, :topk]
        for l in range(nL):
            for r, e in enumerate(order[l].tolist()):
                out.append((True, l, e, r))
    return out


class LRU:
    def __init__(self, cap):
        self.cap, self.od = cap, collections.OrderedDict()

    def access(self, k, insert=True):
        if k in self.od:
            self.od.move_to_end(k)
            return True
        if insert:
            if len(self.od) >= self.cap:
                self.od.popitem(last=False)
            self.od[k] = 1
        return False

    def has(self, k):
        return k in self.od


class TwoQ:
    """S3-FIFO-like: new keys enter a small probation FIFO; a hit there promotes to main LRU.
    Eviction: from probation while it exceeds its share, else main LRU tail."""

    def __init__(self, cap, frac=0.1):
        self.cap, self.pcap = cap, max(1, int(cap * frac))
        self.prob, self.main = collections.OrderedDict(), collections.OrderedDict()

    def has(self, k):
        return k in self.prob or k in self.main

    def access(self, k, insert=True):
        if k in self.main:
            self.main.move_to_end(k)
            return True
        if k in self.prob:
            del self.prob[k]
            self.main[k] = 1
            self._fit()
            return True
        if insert:
            self.prob[k] = 1
            self._fit()
        return False

    def _fit(self):
        while len(self.prob) + len(self.main) > self.cap:
            if len(self.prob) > self.pcap or not self.main:
                self.prob.popitem(last=False)
            else:
                self.main.popitem(last=False)


class DLFU:
    def __init__(self, cap, half_life=200.0):
        self.cap, self.lam = cap, math.log(2) / half_life
        self.score, self.heap, self.t = {}, [], 0

    def has(self, k):
        return k in self.score

    def access(self, k, insert=True):
        self.t += 1
        hit = k in self.score
        if not hit and not insert:
            return False
        # stored as log-domain priority: s * exp(lam * t) -> add exp(lam * t)
        v = self.score.get(k, 0.0) + math.exp(self.lam * self.t)
        self.score[k] = v
        heapq.heappush(self.heap, (v, k))
        while len(self.score) > self.cap:
            v0, k0 = heapq.heappop(self.heap)
            if self.score.get(k0) == v0:
                del self.score[k0]
        if self.t % 20000 == 0:  # renormalize to avoid overflow
            f = math.exp(-self.lam * self.t)
            self.score = {kk: vv * f for kk, vv in self.score.items()}
            self.lam_t0 = self.t
            self.heap = [(vv, kk) for kk, vv in self.score.items()]
            heapq.heapify(self.heap)
            self.t = 0
        return hit


def run(acc, budget, policy, rank_limit=6, **kw):
    slab_of, cap, stride = slab_plan(budget)
    caches = {}
    for lay, c in cap.items():
        caches[lay] = {"lru": LRU, "2q": TwoQ, "dlfu": DLFU}[policy](c, **kw)
    reads = hits = need = 0
    for dec, l, e, r in acc:
        c = caches[slab_of[l]]
        k = (l, e)
        if not dec:
            c.access(k)
            continue
        if r >= rank_limit:  # substitutable: use if resident (refresh), never read
            if c.has(k):
                c.access(k, insert=False)
            continue
        need += 1
        if c.access(k):
            hits += 1
        else:
            reads += 1
    return hits / max(1, need), reads


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budget", type=float, default=16.0)
    ap.add_argument("--rank-limit", type=int, nargs="+", default=[6, 3])
    a = ap.parse_args()
    tr = dict(np.load(a.trace, allow_pickle=True))
    acc = accesses(tr)
    T = len(tr["meta"])
    for R in a.rank_limit:
        for name, pol, kw in (("lru", "lru", {}), ("2q f=0.05", "2q", {"frac": 0.05}), ("2q f=0.1", "2q", {"frac": 0.1}),
                              ("2q f=0.25", "2q", {"frac": 0.25}), ("dlfu hl=100", "dlfu", {"half_life": 100.0}),
                              ("dlfu hl=400", "dlfu", {"half_life": 400.0}), ("dlfu hl=1600", "dlfu", {"half_life": 1600.0})):
            hr, reads = run(acc, a.budget, pol, R, **kw)
            print(f"budget {a.budget} rank<{R} {name:12s}: hit {hr:.3f}  reads/token {reads / T:5.1f}", flush=True)
