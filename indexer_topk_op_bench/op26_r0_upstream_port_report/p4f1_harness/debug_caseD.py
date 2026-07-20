# [p4tt] targeted debug for battery_p4tt caseD failures (-FLT_MAX straddle).
# Runs fast / slow / pristine arms on both caseD rows and dumps per-arm
# diagnostics: which check fails, invalid-index counts, value mismatches.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as KMOD
from gvrpkgprod2_pristine.top_k.gvr_topk_decode import GvrTopKKernel as KPRI

DEV = "cuda"
CR = 4
NEGMAX = torch.finfo(torch.float32).min

def run(kcls, logits, pre, k, **ov):
    nrows, n = logits.shape
    sl = torch.full((nrows,), n * CR, dtype=torch.int32, device=DEV)
    out = torch.empty((nrows, k), dtype=torch.int32, device=DEV)
    kcls.launch(logits, pre, sl, out, k, compress_ratio=CR, **ov)
    torch.cuda.synchronize()
    return out

def diag(name, logits, out, k):
    idx = out[0].long()
    neg = int((idx < 0).sum())
    uniq = int(idx.unique().numel())
    ref = torch.topk(logits[0].float(), k).values
    ok_shape = neg == 0 and uniq == k
    if ok_shape:
        vals = logits[0, idx].float().sort(descending=True).values
        vok = bool(torch.equal(vals, ref))
        nbad = int((vals != ref).sum())
    else:
        vok, nbad = False, -1
    print(f"  {name}: neg={neg} uniq={uniq}/{k} value_exact={vok} nbad={nbad}")
    if not ok_shape:
        # where are the bad slots?
        bad = torch.nonzero(idx < 0).flatten().tolist()[:10]
        print(f"    first neg-idx slots: {bad}")
        # duplicate analysis
        v, c = idx.unique(return_counts=True)
        dups = v[c > 1]
        print(f"    n_dup_indices={int(dups.numel())} sample={dups[:10].tolist()}")
        # slot values around the tail
        print(f"    tail idx[1014:1024]={idx[1014:1024].tolist()}")
        vals_raw = torch.where(idx >= 0, logits[0][idx.clamp(min=0)],
                               torch.tensor(float('nan'), device=DEV))
        print(f"    tail vals[1014:1024]={vals_raw[1014:1024].tolist()}")
    return vok

k = 1024
rng = np.random.default_rng(77)
for n, label in ((1088, "cnt=69 fast"), (1216, "cnt=197 radix")):
    cnt = n - (k - 5)
    row = np.full(n, NEGMAX, dtype=np.float32)
    pos = rng.choice(n, size=k - 5, replace=False)
    row[pos] = (0.9 + 0.0999 * rng.random(k - 5)).astype(np.float32)
    logits = torch.from_numpy(row).view(1, n).to(DEV)
    pre = torch.arange(k, dtype=torch.int32, device=DEV).view(1, k).contiguous()
    print(f"=== caseD {label} N={n} cnt={cnt} need=5 ===")
    of = run(KMOD, logits, pre, k, p4_tail_fast=True)
    os_ = run(KMOD, logits, pre, k, p4_tail_fast=False)
    op = run(KPRI, logits, pre, k)
    diag("fast    ", logits, of, k)
    diag("slow    ", logits, os_, k)
    diag("pristine", logits, op, k)
