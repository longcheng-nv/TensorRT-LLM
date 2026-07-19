# [p4f1] Gate-C attribution: host-replay the level-0 need_more predicate on
# the 25 real bench cells (flash L22 x 9 ISL, pro L30 x 9 ISL, v32 L34 x 7 ISL).
import os
import sys

sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/"
                   "TensorRT-LLM/indexer_topk_op_bench/harness")
import numpy as np
import torch
import real_data_v4cap as RD4
import real_data_v32 as RDV


def level0_replay(row_np, top_k, kbins, kc):
    """Replay coarse + level-0 fine; return (need_more, ra_fine, cnt_str, need,
    width, ulp)."""
    v = np.asarray(row_np, dtype=np.float32)
    cand = np.sort(v)[::-1][: min(kc, v.size)].astype(np.float32)
    if cand.size <= top_k:
        return dict(fires=False, note="cand<=K early path")
    bmin, bmax = np.float32(cand.min()), np.float32(cand.max())
    if bmax <= bmin:
        bmax = np.float32(bmin + np.float32(1e-6))
    inv1 = np.float32((np.float32(kbins - 1) + np.float32(0.99)) / (bmax - bmin))
    cb = np.clip(((cand - bmin) * inv1).astype(np.int32), 0, kbins - 1)
    cum = 0
    b_star, rank_above = kbins - 1, 0
    for b in range(kbins - 1, -1, -1):
        c = int((cb == b).sum())
        if cum + c >= top_k:
            b_star, rank_above = b, cum
            break
        cum += c
    f_lo = np.float32(bmin + np.float32(b_star) / inv1)
    finv = np.float32(np.float32(255.99) * inv1)
    chain = cand[cb == b_star]
    sb = np.clip(((chain - f_lo) * finv).astype(np.int32), 0, 255)
    cum = rank_above
    sb_star, ra = 255, rank_above
    for b in range(255, -1, -1):
        c = int((sb == b).sum())
        if cum + c >= top_k:
            sb_star, ra = b, cum
            break
        cum += c
    cnt = int((sb == sb_star).sum())
    width = float(np.float32(1.0) / finv)
    ulp = float(max(abs(float(f_lo)), 1e-30) * 1.1920928955078125e-07)
    fires = (ra + cnt > top_k) and (width > ulp)
    return dict(fires=fires, ra_fine=ra, cnt_str=cnt, need=top_k - ra,
                width=width, ulp=ulp)


CELLS = (
    [("flash", 22, isl, RD4, 512) for isl in RD4.ISLS]
    + [("pro", 30, isl, RD4, 1024) for isl in RD4.ISLS]
    + [("v32", 34, isl, RDV, 2048) for isl in RDV.ISLS]
)
KBINS = {512: 1024, 1024: 1024, 2048: 2048}   # fp32 GvrParams kNumBins

print(f"{'cell':28s} {'N':>8s} fires  ra_fine cnt_str need  cnt<=128")
n_fire = 0
for model, L, isl, RD, K in CELLS:
    try:
        b = RD.get_bundle(model, isl, L, "fp32")  # v32 loader ignores model
    except Exception as e:  # cell missing on disk
        print(f"{model}:{isl}:L{L:02d}".ljust(28), "LOAD-FAIL", type(e).__name__)
        continue
    N = b["N"]
    row = b["logits"][0, :N].float().cpu().numpy()
    # kC mirrors ctor: K512 cs=1 (N<64K) -> 3072 diet, else 5120; K1024 5120;
    # K2048 cr=1 6144
    if K == 512:
        kc = 3072 if N < 65536 else 5120
    elif K == 1024:
        kc = 5120
    else:
        kc = 6144
    r = level0_replay(row, K, KBINS[K], kc)
    tag = f"{model}:{isl}:L{L:02d}"
    if "note" in r:
        print(f"{tag:28s} {N:8d} {r['note']}")
        continue
    n_fire += int(r["fires"])
    print(f"{tag:28s} {N:8d} {str(r['fires']):5s}  {r['ra_fine']:7d} "
          f"{r['cnt_str']:7d} {r['need']:4d}  {r['cnt_str'] <= 128}")
print(f"\nfired: {n_fire}/{len(CELLS)} cells")
