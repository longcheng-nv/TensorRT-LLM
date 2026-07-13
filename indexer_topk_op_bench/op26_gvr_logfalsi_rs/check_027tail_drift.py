#!/usr/bin/env python3
# NVIDIA Copyright 2026
# Targeted anchor-drift check for the 027-tail fin2 batches (§4 step 3).
# The last 9 markers before the external job landed (13:49-13:55 on 027) were:
#   real/bs K512, real/bs_hugeN K512, best/seqlen K512  (all 3 dtypes).
# Verify gvr_cutedsl (BASE, the cross-node invariant anchor) per-cell cold-us
# ratio fin2/ORIG stays <=1.05 on those batches; anything higher => marker was
# taken under contention and must be deleted & re-run.
import importlib.util as iu, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIB = HERE.parents[0] / "op22_temporal_fixed_hr_bench"
sys.path.insert(0, str(SIB))
spec = iu.spec_from_file_location("u5", SIB / "update_report_op26_iter5.py")
u5 = iu.module_from_spec(spec); spec.loader.exec_module(u5)
u27 = u5.u27
BASE = u27.BASE

FIN = SIB.parents[0] / "indexer_topk_op_bench" / "results_b200_op26_fin2"
FIN = (SIB.parent / "results_b200_op26_fin2")

def key(r): return (r["s"], r["w"], r["K"], r["d"], r["N"], r["B"])

fin = {key(r): r for r in u27.load(FIN, {BASE})}
orig = {key(r): r for r in u27.load(u27.ORIG_ROOT, {BASE})}

# 027-tail batch selectors: (scenario, sweep-w, K)
TAIL = {("real", "bs", 512), ("real", "bs_hugeN", 512), ("best", "seqlen", 512)}

rows = []
for k, fr in fin.items():
    s, w, K, d, N, B = k
    if (s, w, K) not in TAIL:
        continue
    o = orig.get(k)
    if not o:
        continue
    ratio = fr["c"] / o["c"]
    rows.append((ratio, k))

rows.sort(reverse=True)
over = [r for r in rows if r[0] > 1.05]
import statistics as st
ratios = [r[0] for r in rows]
print(f"027-tail cells matched: {len(rows)}")
print(f"  median={st.median(ratios):.4f}  max={max(ratios):.4f}  min={min(ratios):.4f}")
print(f"  cells >1.05: {len(over)}")
for ratio, k in over[:20]:
    print(f"    DRIFT {ratio:.4f}  {k}")
if not over:
    print("  PASS: no 027-tail cell exceeds 1.05x -- markers clean.")
