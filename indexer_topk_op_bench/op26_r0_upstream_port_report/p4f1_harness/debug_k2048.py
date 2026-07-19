# [p4f1] debug: reproduce the failing case-3 K=2048 row + bit-identity evidence
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "gvrpkg_snapshot"))

import numpy as np
import torch

from battery_f1 import (
    CR, DEV, ParamsF1, host_levels, kbins_of, make_preidx, run_kernel,
    valueset_exact_rows,
)
from gvrpkgf1.top_k.gvr_topk_decode import GvrTopKKernel as KF1
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel as KBase

n = 65536

# ---------- part A: find + diagnose the failing K=2048 planted row ----------
rng = np.random.default_rng(42)
for k in (512, 1024, 2048):
    kb = kbins_of(k)
    kc = ParamsF1.get("float32", k, CR).kC
    for fi, frac in enumerate((0.45, 0.55, 0.6499, 0.65, 0.6501, 0.75)):
        for jit in (-0.25, 0.0, 0.25, 0.5):
            row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
            row[0], row[1] = np.float32(0.0), np.float32(0.9999)
            pos = rng.choice(np.arange(2, n), size=k + 1, replace=False)
            row[pos[: k - 2]] = (0.9 + 0.0999 * rng.random(k - 2)).astype(np.float32)
            cand_min_est = float(np.partition(row, n - kc)[n - kc])
            candrange = 0.9999 - cand_min_est
            coarse_w = candrange / kb
            fine_w = candrange / (kb * 255.99)
            gap = fine_w / 2.0 if fi % 2 == 0 else fine_w / 16.0
            v1 = np.float32(frac + jit * coarse_w)
            v2 = np.float32(v1 + gap)
            row[pos[k - 1]] = v2
            row[pos[k]] = v1
            if k != 2048:
                continue  # keep rng stream aligned with battery, only test K2048
            logits = torch.from_numpy(row).view(1, n).to(DEV)
            pre_idx = make_preidx(logits, k)
            out_on = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
            ok = valueset_exact_rows(logits, out_on, k)[0]
            if not ok:
                print(f"FAILING ROW: K={k} frac={frac} jit={jit} gap={gap:.3e} "
                      f"v1={float(v1):.9f} v2={float(v2):.9f}")
                idx = out_on[0].long()
                neg = int((idx < 0).sum())
                uniq = idx.unique().numel()
                ref = torch.topk(logits[0], k).values
                vals = logits[0, idx.clamp(min=0)].sort(descending=True).values
                mism = (vals != ref).nonzero().flatten()
                print(f"  neg={neg} uniq={uniq}/{k} mismatches={mism.numel()}")
                if mism.numel():
                    mm = mism[:4]
                    print(f"  at {mm.tolist()}: got {vals[mm].tolist()}")
                    print(f"             want {ref[mm].tolist()}")
                # rerun same row 3x — deterministic failure?
                for t in range(3):
                    o2 = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
                    print(f"  rerun{t}: exact={valueset_exact_rows(logits, o2, k)[0]}")
                # flag OFF on the same row (defect expected: may fail)
                o_off = run_kernel(KF1, logits, pre_idx, k)
                print(f"  flag OFF exact={valueset_exact_rows(logits, o_off, k)[0]}")
                print(f"  host_levels={host_levels(row, k, kb, kc)}")
                np.save(os.path.join(_HERE, "fail_row_k2048.npy"), row)
                np.save(os.path.join(_HERE, "fail_row_k2048_preidx.npy"),
                        pre_idx.cpu().numpy())

# ---------- part B: bit-identity worlds evidence (K=1024 for speed) ----------
print("\n--- bit-identity evidence ---")
k = 1024
torch.manual_seed(99)
logits = torch.rand((4, n), dtype=torch.float32, device=DEV)
pre_idx = make_preidx(logits, k)
b1 = run_kernel(KBase, logits, pre_idx, k)
b2 = run_kernel(KBase, logits, pre_idx, k)
f1 = run_kernel(KF1, logits, pre_idx, k)
print(f"random rows (cand>K): base-run1 == base-run2 bitwise: {torch.equal(b1, b2)}")
print(f"random rows: base == F1-OFF bitwise: {torch.equal(b1, f1)}")
print(f"random rows: base == F1-OFF sorted-index-set: "
      f"{torch.equal(b1.sort(dim=1).values, f1.sort(dim=1).values)}")

# deterministic rows (cand_count == kK early path): bitwise must match
rng2 = np.random.default_rng(11)
row = (rng2.random(n, dtype=np.float32) * 0.35).astype(np.float32)
posd = rng2.choice(n, size=k, replace=False)
row[posd] = (10.0 + rng2.random(k)).astype(np.float32)
logits_d = torch.from_numpy(row).view(1, n).to(DEV)
pre_d = torch.from_numpy(posd.astype(np.int32)).view(1, k).to(DEV).contiguous()
db = run_kernel(KBase, logits_d, pre_d, k)
db2 = run_kernel(KBase, logits_d, pre_d, k)
df = run_kernel(KF1, logits_d, pre_d, k)
dn = run_kernel(KF1, logits_d, pre_d, k, p4_finebin_loop=True)
print(f"deterministic row: base-run1 == base-run2 bitwise: {torch.equal(db, db2)}")
print(f"deterministic row: base == F1-OFF bitwise: {torch.equal(db, df)}")
print(f"deterministic row: base == F1-ON  bitwise: {torch.equal(db, dn)}")
