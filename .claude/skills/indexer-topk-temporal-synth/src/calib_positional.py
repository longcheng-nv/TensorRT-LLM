#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
"""Part-3 · Stage 1 — per-(model, layer) POSITIONAL calibration from real captures.

Part 2 validated the value MARGINAL (empirical inv-CDF + GPD tail) but assigns
values to positions IID -> the preIdx spatial gather pattern GVR actually reads
is uniform-random, whereas real preIdx (prev-step top-K positions) is heavily
CLUSTERED (measured: median gap = 4-14% of uniform, 50-67% of consecutive pairs
adjacent) and layer-dependently sink/recency-concentrated.

This stage measures, per (model, layer):
  * mu_norm[32]    winner-position density (normalised 0..1), mean 1.0
  * frac_adj       fraction of consecutive sorted-preIdx gaps <= 2  (clustering)
  * gap_ratio      median(preIdx gap) / uniform gap                 (clustering)
  * recency, sink  winner fraction in last 10% / first 1% of positions
  * sink_W, rec_W  ABSOLUTE-token band widths (for N-generalisation)
  * unique_frac    fraction of distinct logit values (tie granularity, C3)
  * N_calib        the sequence length these were calibrated at

Sources (per REAL_DATA_INVENTORY):
  V4-Flash / V4-Pro : q9j_preidx.in.pt + q9j_topk.out.pt + q9j_logits.in.pt
                      (preIdx = captured GVR pre_idx, positions in compressed space)
  V3.2              : Layer_<L>_pd.npy consecutive rows (preIdx = prev-row top-K,
                      DERIVED; no captured preidx stream exists)
"""
from __future__ import annotations
import argparse, collections, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"

V4FLASH_CAP = ("/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1/"
               "ablation_study/gvr_phase_timing/09_precision_ablation/"
               "11_dsv4_trtllm_indexer_data_capture/data/"
               "capture_20260520T082958_alllayers_swe64k")
V4PRO_CAP = ("/home/scratch.loncheng_gpu/workspace/CUDAProgram/auto_optimization_v1/"
             "ablation_study/gvr_phase_timing/09_precision_ablation/"
             "12_dsv4_pro_indexer_data_capture/data/"
             "capture_20260520T164146Z_v4pro_K1024_64k")
V32_DIR = ("/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/"
           "data_distri/deepseek-v3.2-logging/notebooks/SWE_Bench_64K_decode_logits")

NBIN = 32            # normalised winner-density bins
MODEL_K = {"v4flash": 512, "v4pro": 1024, "v32": 2048}


def _pos_stats(pre_pos, win_pos, N, K):
    """preIdx gap/clustering + winner-position density for one (row/step)."""
    pre_pos = pre_pos[(pre_pos >= 0) & (pre_pos < N)]
    win_pos = win_pos[(win_pos >= 0) & (win_pos < N)]
    if pre_pos.size < 8 or win_pos.size < 8:
        return None
    ps = np.sort(pre_pos)
    gaps = np.diff(ps)
    uni = N / max(pre_pos.size, 1)
    pn = win_pos / max(N - 1, 1)
    hist, _ = np.histogram(pn, bins=NBIN, range=(0, 1))
    return dict(
        hist=hist.astype(np.float64),
        frac_adj=float((gaps <= 2).mean()) if gaps.size else 0.0,
        gap_ratio=float(np.median(gaps) / uni) if gaps.size else 1.0,
        recency=float((pn > 0.90).mean()),
        sink=float((pn < 0.01).mean()),
        hr=float(np.isin(pre_pos, win_pos).mean()),
        N=int(N),
    )


def _miss_depths(x, kernel_read_pos, topk_pos, K):
    """Value-depth (thr - value)/sigma of the KERNEL-READ preIdx entries that
    are NOT current winners — the 'bad bracket' structure GVR's search sees.
    Purely a data measurement; never tuned to any kernel timing."""
    x = np.asarray(x, np.float64)
    if x.size <= K:
        return np.empty(0)
    thr = np.partition(x, -K)[-K]                    # K-th largest = threshold
    sigma = x.std() or 1.0
    kr = kernel_read_pos[(kernel_read_pos >= 0) & (kernel_read_pos < x.size)]
    miss = kr[~np.isin(kr, topk_pos)]
    if miss.size == 0:
        return np.empty(0)
    return (thr - x[miss]) / sigma                   # >0 = below threshold


def _pool_depths(mdepths, cap=4000, seed=0):
    """Pool per-step miss-depth samples, subsample to cap. Fallback = [1.0]."""
    if not mdepths:
        return np.array([1.0])
    d = np.concatenate(mdepths)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.array([1.0])
    if d.size > cap:
        d = d[np.random.default_rng(seed).choice(d.size, cap, replace=False)]
    return d.astype(np.float32)


def _agg(records):
    if not records:
        return None
    hist = np.sum([r["hist"] for r in records], axis=0)
    mu = hist / max(hist.sum(), 1) * NBIN            # density, mean 1.0
    scal = lambda k: float(np.mean([r[k] for r in records]))
    Nc = int(np.median([r["N"] for r in records]))
    # absolute band widths: contiguous end/head bins whose density > 1.3x uniform
    binw = Nc / NBIN
    rec_bins = 0
    for b in range(NBIN - 1, -1, -1):
        if mu[b] > 1.3:
            rec_bins += 1
        else:
            break
    sink_bins = 0
    for b in range(NBIN):
        if mu[b] > 1.3:
            sink_bins += 1
        else:
            break
    return dict(mu_norm=mu.astype(np.float32),
                frac_adj=scal("frac_adj"), gap_ratio=scal("gap_ratio"),
                recency=scal("recency"), sink=scal("sink"), hr=scal("hr"),
                N_calib=Nc,
                rec_W=int(max(rec_bins, 1) * binw),
                sink_W=int(max(sink_bins, 0) * binw))


# ------------------------- per-model real loaders -------------------------
def calib_v4(model, cap_dir, n_steps=40):
    import torch
    K = MODEL_K[model]
    pre = torch.load(f"{cap_dir}/q9j_preidx.in.pt", map_location="cpu")
    top = torch.load(f"{cap_dir}/q9j_topk.out.pt", map_location="cpu")
    logi = torch.load(f"{cap_dir}/q9j_logits.in.pt", map_location="cpu")
    byl = collections.defaultdict(list)
    for k in pre:
        byl[k[0]].append(k)
    out = {}
    for L in sorted(byl):
        ks = sorted(byl[L], key=lambda x: x[1])
        ks = ks[len(ks) // 10:]                       # drop early warmup steps
        pick = ks[:: max(1, len(ks) // n_steps)][:n_steps]
        recs, ufracs, mdepths = [], [], []
        for k in pick:
            p = pre[k].flatten().numpy()
            t = top[k].flatten().numpy()
            tv = t[t >= 0]
            if tv.size < 8:
                continue
            N = int(max(tv.max(), p[p >= 0].max())) + 1
            s = _pos_stats(p, t, N, K)
            if s:
                recs.append(s)
            if k in logi:
                lv = logi[k].flatten().numpy()[:N]
                ufracs.append(len(np.unique(lv)) / max(lv.size, 1))
                pv = p[(p >= 0) & (p < N)]             # V4: kernel-read = preIdx (offset 0)
                md = _miss_depths(lv, pv, tv, K)
                if md.size:
                    mdepths.append(md)
        rec = _agg(recs)
        if rec:
            rec["unique_frac"] = float(np.mean(ufracs)) if ufracs else 1.0
            rec["miss_depth"] = _pool_depths(mdepths)
            out[int(L)] = rec
    return out


def calib_v32(n_rows=40):
    K = MODEL_K["v32"]
    layers = [0, 1, 20, 21, 22, 40, 41, 42, 60]
    out = {}
    for L in layers:
        mm = np.load(f"{V32_DIR}/Layer_{L}_pd.npy", mmap_mode="r")
        R = mm.shape[0]
        rows = np.linspace(int(R * 0.4), R - 2, n_rows).astype(int)
        recs, ufracs, mdepths = [], [], []
        for r in rows:
            row0 = np.asarray(mm[r - 1], dtype=np.float32)
            row1 = np.asarray(mm[r], dtype=np.float32)
            N0 = 70690 - int((row0[::-1] != 0).argmax())
            N1 = 70690 - int((row1[::-1] != 0).argmax())
            if N0 < 2 * K or N1 < 2 * K:
                continue
            pre_pos = np.argpartition(-row0[:N0], K)[:K]       # prev-step top-K
            win_pos = np.argpartition(-row1[:N1], K)[:K]        # current top-K
            s = _pos_stats(pre_pos, win_pos, N1, K)
            if s:
                recs.append(s)
            ufracs.append(len(np.unique(row1[:N1])) / max(N1, 1))
            kread = (pre_pos + 1) % N1                          # V3.2 cr=1: kernel reads +1
            md = _miss_depths(row1[:N1], kread, win_pos, K)
            if md.size:
                mdepths.append(md)
        rec = _agg(recs)
        if rec:
            rec["unique_frac"] = float(np.mean(ufracs)) if ufracs else 1.0
            rec["miss_depth"] = _pool_depths(mdepths)
            out[int(L)] = rec
    return out


def save_model(model, rec):
    ASSETS.mkdir(exist_ok=True)
    layers = sorted(rec)
    z = {"layers": np.array(layers, dtype=np.int64),
         "nbin": np.array(NBIN),
         "meta_json": np.frombuffer(json.dumps(
             {"model": model, "K": MODEL_K[model], "layers": layers,
              "nbin": NBIN}).encode(), dtype=np.uint8)}
    for L in layers:
        r = rec[L]
        pre = f"L{L}__"
        z[pre + "mu_norm"] = r["mu_norm"]
        z[pre + "targets"] = np.array(
            [r["frac_adj"], r["gap_ratio"], r["recency"], r["sink"],
             r["hr"], r["unique_frac"], r["N_calib"]], dtype=np.float64)
        z[pre + "bands"] = np.array([r["sink_W"], r["rec_W"]], dtype=np.int64)
        z[pre + "miss_depth"] = r.get("miss_depth", np.array([1.0], np.float32))
    path = ASSETS / f"posz_{model}.npz"
    np.savez(path, **z)
    return path, layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v32,v4flash,v4pro")
    ap.add_argument("--nsteps", type=int, default=40)
    a = ap.parse_args()
    summary = {}
    for m in a.models.split(","):
        m = m.strip()
        if m == "v32":
            rec = calib_v32(a.nsteps)
        elif m == "v4flash":
            rec = calib_v4(m, V4FLASH_CAP, a.nsteps)
        elif m == "v4pro":
            rec = calib_v4(m, V4PRO_CAP, a.nsteps)
        else:
            raise SystemExit(f"unknown model {m}")
        path, layers = save_model(m, rec)
        print(f"[{m}] {len(layers)} layers -> {path}")
        for L in layers:
            r = rec[L]
            md = r.get("miss_depth", np.array([1.0]))
            print(f"   L{L:2d}: frac_adj={r['frac_adj']:.3f} gap_ratio={r['gap_ratio']:.3f} "
                  f"rec={r['recency']:.3f} hr={r['hr']:.3f} "
                  f"rec_W={r['rec_W']} uniq={r['unique_frac']:.3f} "
                  f"miss_depth med/p90={np.median(md):.2f}/{np.percentile(md,90):.2f} "
                  f"(n={md.size}) N={r['N_calib']}")
        summary[m] = {int(L): {k: (float(np.median(v)) if k == "miss_depth"
                                   else v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in rec[L].items() if k != "mu_norm"}
                      for L in layers}
    json.dump(summary, open(HERE.parent / "calib_positional_summary.json", "w"),
              indent=1)
    print("wrote calib_positional_summary.json")


if __name__ == "__main__":
    main()
