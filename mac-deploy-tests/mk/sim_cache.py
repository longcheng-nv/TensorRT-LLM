"""Offline expert-cache simulator over recorded routing traces (trace_routing.py output).

Replays prefill accesses + decode routing per prompt against the engine's slab structure (one LRU
per expert layout, capacity from the cache budget like ExpertStore), with optional next-layer
speculation (SPEC_K) and lossy miss policies. Reports, per decode token: demand reads, speculative
reads (useful / wasted), substitutions, and the bytes read from SSD.

Policies (applied at decode only):
  none                  baseline (every miss is read)
  sub:R[:G]             a miss at rank >= R (0-based) that is not already resident/in flight is
                        replaced by the best-scoring resident expert of that layer not already
                        selected, if its biased score is within G of the missed one (G=inf default)
  drop:R                a miss at rank >= R is dropped (weights renormalized)
  topk:K                global top-K instead of 6
"""
from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np

LAYOUT = json.load(open("/Users/loncheng/Models/dsv41-mac/layout.json"))["experts"]


def slab_plan(budget_gb: float):
    groups = collections.defaultdict(list)
    for k, v in LAYOUT.items():
        if k.startswith("L"):
            groups[(v["bits_w1"], v["bits_w3"], v["bits_w2"], v["stride"])].append(int(k[1:]))
    total = sum(len(ls) * lay[3] for lay, ls in groups.items())
    budget = budget_gb * (1 << 30)
    slab_of, cap, stride = {}, {}, {}
    for lay, ls in groups.items():
        slots = max(96, int(budget * len(ls) * lay[3] / total) // lay[3])
        for l in ls:
            slab_of[l] = lay
            stride[l] = LAYOUT[f"L{l:02d}"]["nbytes"]
        cap[lay] = slots
    return slab_of, cap, stride


class Slab:
    def __init__(self, cap):
        self.cap = cap
        self.od = collections.OrderedDict()

    def touch(self, k):
        self.od.move_to_end(k)

    def has(self, k):
        return k in self.od

    def insert(self, k, cold=False, protect=()):
        if k in self.od:
            return
        while len(self.od) >= self.cap:
            for victim in self.od:
                if victim not in protect:
                    break
            del self.od[victim]
        self.od[k] = 1
        if cold:
            self.od.move_to_end(k, last=False)


def run(tr, budget_gb=16.0, spec_k=4, policy="none", topk=6, prompts=None, verbose=False, touch=0, touch_mode="lru"):
    S, P, B, meta = tr["scores"].astype(np.float32), tr["pred"].astype(np.float32), tr["biases"], tr["meta"]
    T, nL, nE = S.shape
    slab_of, cap, stride = slab_plan(budget_gb)
    slabs = {lay: Slab(c) for lay, c in cap.items()}
    pol, arg = (policy.split(":", 1) + [""])[:2]
    R = G = None
    if pol in ("sub", "drop"):
        parts = arg.split(":")
        R = int(parts[0])
        G = float(parts[1]) if len(parts) > 1 else math.inf
    if pol == "topk":
        topk = int(arg)
    stats = collections.Counter()
    gaps = []
    rank_miss = collections.Counter()
    by_prompt = collections.defaultdict(collections.Counter)
    cur_prompt = -1
    spec_pending = set()  # (layer, e) prefetched speculatively and not yet used
    for t in range(T):
        pi, step = int(meta[t][0]), int(meta[t][1])
        if prompts is not None and pi not in prompts:
            continue
        if pi != cur_prompt:  # new prompt: replay its prefill accesses (expert-major, ascending ids)
            cur_prompt = pi
            for l in range(nL):
                key = f"pref_{pi}_{l}"
                if key in tr:
                    ids = np.unique(tr[key])
                    sl = slabs[slab_of[l]]
                    for e in ids.tolist():
                        k = (l, e)
                        if sl.has(k):
                            sl.touch(k)
                        else:
                            sl.insert(k)
            spec_pending.clear()
        for l in range(nL):
            sl = slabs[slab_of[l]]
            b = S[t, l] + B[l]
            order = np.argsort(-b, kind="stable")
            sel = order[:topk].tolist()
            used = []
            for r, e in enumerate(sel):
                k = (l, e)
                if sl.has(k):
                    if k in spec_pending:
                        stats["spec_hit"] += 1
                        spec_pending.discard(k)
                    else:
                        stats["hit"] += 1
                    sl.touch(k)
                    used.append(e)
                    continue
                rank_miss[r] += 1
                if pol == "sub" and r >= R:
                    # best resident not already selected/used
                    cand = None
                    for e2 in order[topk:].tolist():
                        if sl.has((l, e2)) and e2 not in used and e2 not in sel:
                            cand = e2
                            break
                    if cand is not None and b[e] - b[cand] <= G:
                        stats["sub"] += 1
                        gaps.append(float(b[e] - b[cand]))
                        sl.touch((l, cand))
                        used.append(cand)
                        continue
                if pol == "drop" and r >= R:
                    stats["drop"] += 1
                    continue
                stats["demand"] += 1
                stats["bytes"] += stride[l]
                sl.insert(k, protect={(l, x) for x in used})
                used.append(e)
            stats["act"] += len(sel)
            if touch:  # score-aware recency: refresh resident experts ranked just below the selection
                for e2 in order[topk : topk + touch].tolist()[::-1]:
                    if sl.has((l, e2)) and e2 not in used:
                        sl.touch((l, e2))
                for e in used[::-1]:
                    sl.touch((l, e))
            # speculation for layer l+1 from this layer's input
            if spec_k and l + 1 < nL:
                l2 = l + 1
                sl2 = slabs[slab_of[l2]]
                pb = P[t, l] + B[l2]
                pred = np.argsort(-pb, kind="stable")[:spec_k].tolist()
                for e in pred:
                    k = (l2, e)
                    if not sl2.has(k):
                        stats["spec_read"] += 1
                        stats["bytes"] += stride[l2]
                        sl2.insert(k, cold=True)
                        spec_pending.add(k)
        # mispredictions are just cold entries; count wasted at token end (rough: still pending)
        stats["tokens"] += 1
        by_prompt[pi]["tokens"] += 1
    n = max(1, stats["tokens"])
    out = {
        "budget_gb": budget_gb, "spec_k": spec_k, "policy": policy, "topk": topk,
        "hit_rate_cache": round(stats["hit"] / max(1, stats["act"]), 3),
        "hit_rate_incl_spec": round((stats["hit"] + stats["spec_hit"]) / max(1, stats["act"]), 3),
        "demand_reads_per_tok": round(stats["demand"] / n, 1),
        "spec_reads_per_tok": round(stats["spec_read"] / n, 1),
        "spec_useful_per_tok": round(stats["spec_hit"] / n, 1),
        "sub_per_tok": round(stats["sub"] / n, 1), "drop_per_tok": round(stats["drop"] / n, 1),
        "GB_per_tok": round(stats["bytes"] / n / 1e9, 3),
        "miss_by_rank": [rank_miss[r] for r in range(topk)],
    }
    if gaps:
        g = np.array(gaps)
        out["gap_p50_p90"] = [round(float(np.percentile(g, 50)), 4), round(float(np.percentile(g, 90)), 4)]
    return out


def belady(tr, budget_gb=16.0, topk=6):
    """Upper bound: Belady/OPT replacement per slab on the decode accesses (prefill as warmup, LRU)."""
    S, B, meta = tr["scores"].astype(np.float32), tr["biases"], tr["meta"]
    T, nL, _ = S.shape
    slab_of, cap, stride = slab_plan(budget_gb)
    seq = collections.defaultdict(list)  # slab -> list of (layer, e)
    for t in range(T):
        for l in range(nL):
            sel = np.argsort(-(S[t, l] + B[l]), kind="stable")[:topk].tolist()
            for e in sel:
                seq[slab_of[l]].append((l, e))
    hits = acc = 0
    for lay, s in seq.items():
        nxt = {}
        nextuse = [0] * len(s)
        for i in range(len(s) - 1, -1, -1):
            nextuse[i] = nxt.get(s[i], math.inf)
            nxt[s[i]] = i
        cache = {}
        import heapq
        heap = []
        for i, k in enumerate(s):
            acc += 1
            if k in cache:
                hits += 1
            elif len(cache) >= cap[lay]:
                while True:
                    nu, kk = heapq.heappop(heap)
                    if cache.get(kk) == -nu:
                        break
                del cache[kk]
            cache[k] = nextuse[i]
            heapq.heappush(heap, (-nextuse[i], k))
    return round(hits / acc, 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budget", type=float, nargs="+", default=[16.0])
    ap.add_argument("--spec-k", type=int, nargs="+", default=[4])
    ap.add_argument("--policy", nargs="+", default=["none"])
    ap.add_argument("--belady", action="store_true")
    ap.add_argument("--touch", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    tr = dict(np.load(a.trace, allow_pickle=True))
    for bgt in a.budget:
        if a.belady:
            print(json.dumps({"budget_gb": bgt, "belady_hit_rate": belady(tr, bgt)}), flush=True)
        for k in a.spec_k:
            for p in a.policy:
                for tch in a.touch:
                    r = run(tr, bgt, k, p, touch=tch)
                    r["touch"] = tch
                    print(json.dumps(r), flush=True)
