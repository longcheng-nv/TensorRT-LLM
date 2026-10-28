#!/usr/bin/env python
"""Weights-free numeric gate for DeepInfer-Mac's PRODUCTION expert quantization layouts.

The shipped tests never run the real expert record format at the real bit widths through the
engine's slab path (tiny_ckpt.py uses 4/4/4, test_engine.py 8/8/8, test_store.py only checks
byte round-trips of a 2/2/3 record). This script does, for every layout expert_layout_for() emits:
3/3/3 (layers 0-3, 8, 14, 20-38), 2/2/3 (the rest, and every DSpark/MTP layer), 4/4/4 (layer 39),
plus 8/8/8 as a near-lossless control of the harness itself.

Per layout:
  1. N random experts of the model's real per-expert shapes (dim 5120, moe_inter_dim 2304; see
     inference/config.json) are generated in the checkpoint's own fp4 format (e2m1 nibbles, low
     nibble first + one e8m0 scale per 32 along `in`), i.e. exactly what the converter reads.
  2. convert_expert() [deepinfer_mac/convert/remote.py] turns each into a page-aligned record;
     records are pwrite()n at e * layout.stride into experts/<key>.bin and layout.json is written,
     as Converter.write() / write_meta() do.
  3. ExpertStore [deepinfer_mac/store.py] loads them (preadv into the slab, F_NOCACHE).
  4. The production MoE code [deepinfer_mac/model/model.py] runs on the slab:
       prefill  MoE._slab_moe -> _experts_into -> _per_expert_ffn   (quantized_matmul per expert)
       decode   MoE._slab_moe -> _experts_into -> _slab_ffn         (gather_qmm, sorted slots), 1 token
       batched  the compiled _slab_ffn over all (token, expert) pairs (gather_qmm with many rows)
       get      ExpertStore.get + run_expert                        (per-expert path of tests/benchmarks)
  5. Reference A: mx.dequantize() of the SAME record bytes + plain fp32 matmuls, with the same
     act_quant / SwiGLU / bf16 dtype chain  -> isolates packing / slab / gather_qmm. Gate: cos > 0.9999.
     MLX's quantized kernels dequantize in two precisions depending on the row count (the
     matrix-vector kernels use s*q+b exactly, the tiled kernels round it to bf16 first); a probe
     at startup measures which regime each row count falls in, and the reference mirrors it per
     expert -- otherwise the fp8 act_quant between the projections amplifies 1-ulp bf16
     differences into a ~1e-4 cosine deficit that would mask real packing bugs.
  6. Reference B: the same reference on the ORIGINAL (fp4-exact) weights -> quantization damage,
     reported per layout, gated only by (a) the quantizer's own invariant -- the clipping search
     of quant.quantize_mse must not be worse than plain mx.quantize on any matrix -- and (b) loose
     floors that catch a broken quantizer (wrong bits, swapped scales/biases), not a mediocre one.

Exit status 0 = all gates pass, 1 = a gate failed (the table says which path / layout).

    /Users/loncheng/Models/DeepInfer-Mac/.venv/bin/python /Users/loncheng/Models/dsv41-mac-tests/gate_quant_layouts.py
    ... [--experts 8] [--tokens 16] [--layouts 3,3,3 2,2,3 4,4,4] [--no-control] [--dspark] [--keep] [--seed 0]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = "/Users/loncheng/Models/DeepInfer-Mac"
sys.path.insert(0, REPO)

import mlx.core as mx  # noqa: E402

from deepinfer_mac import format as F  # noqa: E402
from deepinfer_mac import quant as Q  # noqa: E402
from deepinfer_mac.convert.remote import convert_expert  # noqa: E402
from deepinfer_mac.model import model as M  # noqa: E402
from deepinfer_mac.model import ops  # noqa: E402
from deepinfer_mac.store import ExpertStore  # noqa: E402

SCRATCH = Path("/private/tmp/claude-502/-Users-loncheng-Models/60697234-ef74-4285-9fb6-da2f8011ce9b/scratchpad/quant-gate")
REF_CONFIG = Path("/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref/inference/config.json")

GATE_COS = 0.9999  # production path vs. dequantized reference of the same bytes
# quantization damage floors (cos of the routed-expert output vs. the fp4-exact original weights),
# keyed by the smallest bit width of the layout -- loose on purpose: they catch a broken quantizer
# (wrong bits, swapped scales/biases), not a slightly worse one. The values are reported either way.
# The 8/8/8 control measures the chain's own ceiling: weight-level cos 0.99996+, output cos ~0.9995,
# because three cascaded g64 affine projections plus the fp8 act_quant between them (whose rounding
# flips on any weight perturbation) cost ~1e-4..5e-4 even at 8 bits; measured 0.99950 (0.99984
# with the act_quant on h removed from the reference).
DAMAGE_FLOOR = {2: 0.70, 3: 0.85, 4: 0.95, 8: 0.999}
MSE_TOL = 1e-4  # quantize_mse must be <= plain mx.quantize in MSE on every matrix (ratio 1.0 is a candidate)

PROJ_SHAPES = lambda lay: (("w1", (lay.inter, lay.dim)), ("w3", (lay.inter, lay.dim)), ("w2", (lay.dim, lay.inter)))  # noqa: E731
BITS_OF = lambda lay: {"w1": lay.bits_w1, "w3": lay.bits_w3, "w2": lay.bits_w2}  # noqa: E731


# ------------------------------------------------------------------------------ synthetic checkpoint experts


def random_fp4_weight(out: int, inn: int, std: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """A random weight in the checkpoint's fp4 expert format (quant.py docstring): e2m1 codes packed
    two per byte, low nibble first [out, in/2], plus one e8m0 power-of-two scale per 32 along `in`
    [out, in/32]. Gaussian with a mild log-normal per-input-channel spread so 64-groups are not all
    alike. The fp4-exact values (what the converter dequantizes) are the 'original' weights."""
    col = mx.exp(0.5 * mx.random.normal((1, inn)))
    w = mx.random.normal((out, inn)) * std * col
    wg = w.reshape(out, inn // 32, 32)
    amax = mx.maximum(mx.abs(wg).max(axis=-1, keepdims=True), mx.array(1e-30, mx.float32))
    e = mx.ceil(mx.log2(amax / ops.FP4_MAX))  # smallest power of two with |w/s| <= 6
    s = mx.power(mx.array(2.0, mx.float32), e)
    q = ops.round_e2m1(mx.clip(wg / s, -ops.FP4_MAX, ops.FP4_MAX))
    mag = mx.abs(q)
    idx = sum((mag >= t).astype(mx.uint8) for t in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))  # index into Q.FP4_TABLE[:8]
    code = (idx + 8 * (q < 0).astype(mx.uint8)).reshape(out, inn)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    scale = (e + 127.0).astype(mx.uint8).reshape(out, inn // 32)
    mx.eval(packed, scale)
    # self-check of the encoder against the package's decoder (exact, by construction)
    back = Q.dequant_fp4(packed, scale)
    err = mx.abs(back - (q * s).reshape(out, inn)).max().item()
    assert err == 0.0, f"fp4 encoder/decoder mismatch: {err}"
    return np.array(packed), np.array(scale)


def make_experts(layout: F.ExpertLayout, n: int) -> list[dict[str, np.ndarray]]:
    experts = []
    for _ in range(n):
        w = {}
        for k, (o, i) in PROJ_SHAPES(layout):
            w[k + ".weight"], w[k + ".scale"] = random_fp4_weight(o, i)
        experts.append(w)
    return experts


def write_layer(root: Path, key: str, layout: F.ExpertLayout, experts: list[dict]) -> list[bytes]:
    """convert_expert() per expert, pwrite at e * stride, layout.json -- as Converter.write/write_meta."""
    (root / "experts").mkdir(parents=True, exist_ok=True)
    path = root / "experts" / f"{key}.bin"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    recs = []
    try:
        os.ftruncate(fd, len(experts) * layout.stride)
        for e, w in enumerate(experts):
            rec = convert_expert(w, layout)
            assert len(rec) == layout.stride, (len(rec), layout.stride)
            os.pwrite(fd, rec, e * layout.stride)
            recs.append(rec)
        os.fsync(fd)
    finally:
        os.close(fd)
    (root / "layout.json").write_text(json.dumps({"format": "deepinfer-mac/1", "group_size": F.GROUP, "experts": {key: layout.to_json()}}))
    return recs


# ------------------------------------------------------------------------------ references


def unpack_record(rec: bytes, layout: F.ExpertLayout) -> dict[str, mx.array]:
    """Record bytes -> arrays through the same ExpertLayout.parts() table the store's slab uses."""
    out = {}
    for name, shape, dtype, off, n in layout.parts():
        raw = np.frombuffer(rec, dtype=np.uint8, count=n, offset=off)
        if dtype == "uint32":
            out[name] = mx.array(raw.view(np.uint32).reshape(shape))
        else:
            out[name] = mx.array(raw.view(np.uint16).reshape(shape)).view(mx.bfloat16)
    return out


def dequant(q: mx.array, s: mx.array, b: mx.array, bits: int, mode: str) -> mx.array:
    """fp32 dequantized weight. mode 'exact': s*q+b in fp32 (what MLX's matrix-vector kernels use);
    'bf16': rounded to bf16 first (what the tiled kernels use, and mx.dequantize's own output dtype)."""
    if mode == "exact":
        return mx.dequantize(q, s.astype(mx.float32), b.astype(mx.float32), group_size=F.GROUP, bits=bits)
    return mx.dequantize(q, s, b, group_size=F.GROUP, bits=bits).astype(mx.float32)


def dequant_record(rec: bytes, layout: F.ExpertLayout, mode: str) -> tuple[mx.array, mx.array, mx.array]:
    p, bits = unpack_record(rec, layout), BITS_OF(layout)
    return tuple(dequant(p[f"{w}.q"], p[f"{w}.s"], p[f"{w}.b"], bits[w], mode) for w in ("w1", "w3", "w2"))


def dequant_original(w: dict[str, np.ndarray]) -> tuple[mx.array, mx.array, mx.array]:
    """The fp4-exact weights the converter started from (remote.py convert_expert, Q.dequant_fp4)."""
    return tuple(Q.dequant_fp4(mx.array(w[k + ".weight"]), mx.array(w[k + ".scale"])) for k in ("w1", "w3", "w2"))


def probe_kernel_modes(layout: F.ExpertLayout, max_rows: int) -> tuple[dict[int, str], str]:
    """Which dequantization precision the kernels effectively use, per row count M of
    quantized_matmul (1..max_rows), and for gather_qmm with one row per pair (the shape _slab_ffn
    uses): the mode whose plain-fp32-matmul reference the kernel output is bf16-identical with."""
    bits = layout.bits_w1
    w = (mx.random.normal((layout.inter, layout.dim)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=F.GROUP, bits=bits)
    refs = {m: dequant(q, s, b, bits, m) for m in ("exact", "bf16")}

    def pick(k, xf):
        frac = {m: (k == (xf @ W.T).astype(mx.bfloat16)).astype(mx.float32).mean().item() for m, W in refs.items()}
        best = max(frac, key=frac.get)
        return best, frac

    table = {}
    for Mrows in range(1, max_rows + 1):
        x = mx.random.normal((Mrows, layout.dim)).astype(mx.bfloat16)
        k = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=F.GROUP, bits=bits)
        table[Mrows], _ = pick(k, x.astype(mx.float32))
    x = mx.random.normal((8, 1, 1, layout.dim)).astype(mx.bfloat16)
    g = mx.gather_qmm(x, q[None], s[None], b[None], rhs_indices=mx.zeros((8, 1), mx.uint32), transpose=True, group_size=F.GROUP, bits=bits, sorted_indices=True)
    gmode, gfrac = pick(g.reshape(8, -1), x.reshape(8, -1).astype(mx.float32))
    return table, gmode


def describe_modes(table: dict[int, str], gmode: str) -> str:
    runs, start = [], 1
    for m in range(2, len(table) + 2):
        if m > len(table) or table[m] != table[start]:
            runs.append(f"M{start}-{m - 1}:{table[start]}" if m - 1 > start else f"M{start}:{table[start]}")
            start = m
    return "quantized_matmul " + " ".join(runs) + f"; gather_qmm(1 row/pair):{gmode}"


def reference(xq: mx.array, w: mx.array, idx_np: np.ndarray, mats_of, limit: float, dim: int, mode_of) -> mx.array:
    """Routed SwiGLU with plain fp32 matmuls on dequantized weights, mirroring the production dtype
    chain of model.py _slab_ffn / _per_expert_ffn: bf16 gate/up (the kernels' output dtype), bf16 h
    -> act_quant, bf16 per-pair output, fp32 accumulation over pairs. mats_of(e, mode) hands over
    the expert's (W1, W3, W2); mode_of(rows) says which dequant precision the kernel uses for that
    many rows."""
    T = xq.shape[0]
    y = mx.zeros((T, dim), dtype=mx.float32)
    lim = mx.array(limit, dtype=mx.float32)
    for e in np.unique(idx_np).tolist():
        if e < 0:
            continue
        rows, cols = np.nonzero(idx_np == e)
        r, c = mx.array(rows.astype(np.int32)), mx.array(cols.astype(np.int32))
        W1, W3, W2 = mats_of(e, mode_of(len(rows)))
        xe = xq[r].astype(mx.float32)
        g = (xe @ W1.T).astype(mx.bfloat16).astype(mx.float32)
        u = (xe @ W3.T).astype(mx.bfloat16).astype(mx.float32)
        wk = w[r, c].astype(mx.float32)[:, None]
        h = ops.act_quant((wk * M._swiglu_c(g, u, lim)).astype(mx.bfloat16))
        o = (h.astype(mx.float32) @ W2.T).astype(mx.bfloat16).astype(mx.float32)
        y = y.at[r].add(o)
        mx.eval(y)
        del W1, W3, W2
    return y


# ------------------------------------------------------------------------------ metrics


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def maxabs(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def f32(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def mse(a: mx.array, b: mx.array) -> float:
    return mx.square(a.astype(mx.float32) - b.astype(mx.float32)).mean().item()


# ------------------------------------------------------------------------------ production paths


def make_moe(store: ExpertStore, key: str, n_experts: int, topk: int) -> M.MoE:
    """A MoE bound to the store, without gate/shared-expert weights: _slab_moe/_experts_into only
    use a (swiglu_limit), key, experts and the class knobs (OVERLAP, EXPERT_BATCH)."""
    moe = object.__new__(M.MoE)
    moe.a, moe.key, moe.experts = M.Args(), key, store
    moe.n_routed, moe.topk = n_experts, topk
    return moe


def routing(rng: np.random.Generator, T: int, n_experts: int, topk: int, route_scale: float):
    idx = np.stack([rng.choice(n_experts, topk, replace=False) for _ in range(T)]).astype(np.int64)
    w = rng.random((T, topk)).astype(np.float32) + 0.05
    w = w / w.sum(1, keepdims=True) * route_scale  # as _route_shared returns: normalized scores * route_scale
    return idx, mx.array(w)


def batched_slab_ffn(store: ExpertStore, key: str, xq, w, idx_np, limit: float, dim: int) -> mx.array:
    """The compiled _slab_ffn (gather_qmm x3) over every (token, expert) pair at once, pairs sorted
    by slot as _experts_into does -- the decode kernel path with many rows."""
    ids = [e for e in np.unique(idx_np).tolist() if e >= 0]
    slab, slots = store.get_slots(key, ids)
    slot_of = dict(zip(ids, slots))
    rows, cols = np.nonzero(idx_np >= 0)
    pair_slots = np.array([slot_of[e] for e in idx_np[rows, cols]])
    order = np.argsort(pair_slots, kind="stable")
    rows, cols, pair_slots = rows[order], cols[order], pair_slots[order]
    rhs = mx.array(pair_slots.astype(np.uint32))[:, None]
    r = mx.array(rows.astype(np.int32))
    xr = xq[r][:, None, None, :]
    wk = w[r, mx.array(cols.astype(np.int32))].astype(mx.float32)[:, None, None, None]
    q1, q3, q2 = slab.proj("w1"), slab.proj("w3"), slab.proj("w2")
    f = M._ffn_fns.get((q1[3], q3[3], q2[3]))  # the production-compiled graph (built by the decode runs)
    if f is not None:
        o = f(xr, rhs, wk, mx.array(limit, dtype=mx.float32), *q1[:3], *q3[:3], *q2[:3])
    else:
        o = M._slab_ffn(xr, rhs, wk, mx.array(limit, dtype=mx.float32), q1, q3, q2)
    y = mx.zeros((xq.shape[0], dim), dtype=mx.float32).at[r].add(o)
    mx.eval(y)
    return y


def store_get_path(store: ExpertStore, key: str, xq, w, idx_np, limit: float, dim: int) -> mx.array:
    """ExpertStore.get + run_expert, as MoE.__call__'s non-slab prefill branch does (model.py 725-742)."""
    ids = [e for e in np.unique(idx_np).tolist() if e >= 0]
    y = mx.zeros((xq.shape[0], dim), dtype=mx.float32)
    for e, ew in zip(ids, store.get(key, ids)):
        rows, col = np.nonzero(idx_np == e)
        r = mx.array(rows.astype(np.int32))
        out = M.run_expert(xq[r], ew, w[r, mx.array(col.astype(np.int32))][:, None], limit)
        y = y.at[r].add(out.astype(mx.float32))
    mx.eval(y)
    return y


def check_slab_bytes(store: ExpertStore, key: str, layout: F.ExpertLayout, recs: list[bytes]) -> int:
    """Every byte of every loaded record, across the nine slab parts, equals the file (test_store._check)."""
    slab, slots = store.get_slots(key, list(range(len(recs))))
    bad = 0
    for e, sl in zip(range(len(recs)), slots):
        for name, shape, dtype, off, n in layout.parts():
            got = np.array(slab.parts[name][sl]).view(np.uint8).reshape(-1)[:n]
            if not np.array_equal(got, np.frombuffer(recs[e], np.uint8, n, off)):
                bad += 1
    return bad


# ------------------------------------------------------------------------------ one layout


def run_layout(layout: F.ExpertLayout, key: str, T: int, n_experts: int, topk: int, seed: int, keep: bool) -> dict:
    tag = f"{layout.bits_w1}/{layout.bits_w3}/{layout.bits_w2}"
    topk = min(topk, n_experts)
    root = SCRATCH / f"{key}-{layout.bits_w1}{layout.bits_w3}{layout.bits_w2}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    dim, limit = layout.dim, M.Args().swiglu_limit
    res = {"layout": tag, "key": key, "topk": topk, "paths": {}, "fail": []}
    t0 = time.time()

    experts = make_experts(layout, n_experts)
    recs = write_layer(root, key, layout, experts)
    t_conv = time.time() - t0
    size = (root / "experts" / f"{key}.bin").stat().st_size
    assert size == n_experts * layout.stride, (size, n_experts, layout.stride)

    # per-weight quantization damage: dequantized record vs fp4-exact original (cos, mean over
    # experts) and the quantizer invariant: quantize_mse's clipping search <= plain mx.quantize in MSE
    wcos, wcos_plain, mse_ratio = np.zeros(3), np.zeros(3), np.zeros(3)
    bits = BITS_OF(layout)
    for e in range(n_experts):
        for i, (name, a, orig) in enumerate(zip(("w1", "w3", "w2"), dequant_record(recs[e], layout, "bf16"), dequant_original(experts[e]))):
            pq, ps, pb = mx.quantize(orig.astype(mx.bfloat16), group_size=F.GROUP, bits=bits[name])
            plain = dequant(pq, ps, pb, bits[name], "bf16")
            m_rec, m_plain = mse(a, orig), mse(plain, orig)
            wcos[i] += cos(f32(a), f32(orig)) / n_experts
            wcos_plain[i] += cos(f32(plain), f32(orig)) / n_experts
            mse_ratio[i] = max(mse_ratio[i], m_rec / m_plain)
            if m_rec > m_plain * (1 + MSE_TOL):
                res["fail"].append(f"{tag} expert {e} {name}: quantize_mse MSE {m_rec:.4e} > plain mx.quantize {m_plain:.4e}")
    res["wcos"], res["wcos_plain"], res["mse_ratio"] = wcos, wcos_plain, mse_ratio

    # activations and routing, as the model would hand them to _slab_moe
    x = mx.random.normal((T, dim)).astype(mx.bfloat16)
    xq = ops.act_quant(x)  # _route_shared quantizes the MoE input once (model.py:600)
    idx_np, w = routing(rng, T, n_experts, topk, M.Args().route_scale)
    mx.eval(x, xq, w)

    # which dequant precision the kernels use per row count (see module docstring)
    modes, gmode = probe_kernel_modes(layout, T)
    res["modes"] = describe_modes(modes, gmode)
    rec_mats = lambda e, mode: dequant_record(recs[e], layout, mode)  # noqa: E731
    y_ref_mm = reference(xq, w, idx_np, rec_mats, limit, dim, lambda rows: modes[rows])  # quantized_matmul paths
    y_ref_g = reference(xq, w, idx_np, rec_mats, limit, dim, lambda rows: gmode)  # gather_qmm paths (1 row/pair)
    y_orig = reference(xq, w, idx_np, lambda e, mode: dequant_original(experts[e]), limit, dim, lambda rows: modes[rows])
    y_ref_mm, y_ref_g, y_orig = f32(y_ref_mm), f32(y_ref_g), f32(y_orig)
    res["ref_absmax"] = float(np.abs(y_ref_mm).max())
    res["ref_mm_vs_g"] = cos(y_ref_mm, y_ref_g)

    # the engine's store on the written file (minimum slab: 96 slots, as test_store does)
    store = ExpertStore(root, budget_bytes=0)
    res["slab_gb"] = store.used / 2**30
    try:
        moe = make_moe(store, key, n_experts, topk)

        # 1. decode on a cold store: token 0's experts are all misses (prefetch -> get_slots waits)
        y = moe._slab_moe(x[0:1], xq[0:1], w[0:1], idx_np[0:1], mx.zeros((1, dim), dtype=mx.float32))
        mx.eval(y)
        res["paths"]["decode cold (_slab_moe/_slab_ffn, misses)"] = (f32(y), y_ref_g[0:1], y_orig[0:1])

        # 2. prefill: all T tokens through _slab_moe -> _experts_into -> _per_expert_ffn
        y = moe._slab_moe(x, xq, w, idx_np, mx.zeros((T, dim), dtype=mx.float32))
        mx.eval(y)
        res["paths"]["prefill (_slab_moe/_per_expert_ffn)"] = (f32(y), y_ref_mm, y_orig)

        # 3. decode, every token separately (experts now resident: hits path), gather_qmm sorted slots
        outs = []
        for t in range(T):
            y = moe._slab_moe(x[t : t + 1], xq[t : t + 1], w[t : t + 1], idx_np[t : t + 1], mx.zeros((1, dim), dtype=mx.float32))
            outs.append(y)
        y_dec = mx.concatenate(outs, axis=0)
        mx.eval(y_dec)
        res["paths"][f"decode x{T} (_slab_moe/_slab_ffn, hits)"] = (f32(y_dec), y_ref_g, y_orig)

        # 4. compiled _slab_ffn over all T*topk pairs at once
        y_bat = batched_slab_ffn(store, key, xq, w, idx_np, limit, dim)
        res["paths"][f"batched _slab_ffn ({T * topk} pairs)"] = (f32(y_bat), y_ref_g, y_orig)

        # 5. ExpertStore.get + run_expert (per-expert quantized_matmul with bf16 slab views)
        y_get = store_get_path(store, key, xq, w, idx_np, limit, dim)
        res["paths"]["store.get + run_expert"] = (f32(y_get), y_ref_mm, y_orig)

        # bytes in the slab == bytes in the file
        res["bad_parts"] = check_slab_bytes(store, key, layout, recs)
        res["store_stats"] = dict(store.stats)
    finally:
        store.pool.shutdown(wait=True)
        store.spec_pool.shutdown(wait=True)
        for fd in store.fds.values():
            os.close(fd)
        del store, moe
        gc.collect()
        mx.clear_cache()

    # evaluate gates
    rows = []
    for name, (got, ref, orig) in res["paths"].items():
        c_ref, c_orig, err = cos(got, ref), cos(got, orig), maxabs(got, ref)
        ok = c_ref > GATE_COS
        rows.append((name, c_ref, err, c_orig, ok))
        if not ok:
            res["fail"].append(f"{tag} {name}: cos vs dequant reference {c_ref:.7f} <= {GATE_COS}")
    res["rows"] = rows
    floor = DAMAGE_FLOOR[min(layout.bits_w1, layout.bits_w3, layout.bits_w2)]
    res["damage"] = rows[1][3]  # prefill path vs the original weights
    res["floor"] = floor
    if res["damage"] <= floor:
        res["fail"].append(f"{tag} quantization damage: cos vs original {res['damage']:.5f} <= floor {floor}")
    if res["bad_parts"]:
        res["fail"].append(f"{tag} slab bytes != file bytes in {res['bad_parts']} parts")
    res["time"] = time.time() - t0
    res["t_conv"] = t_conv
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    return res


# ------------------------------------------------------------------------------ main


def production_layout_map(n_layers: int) -> dict[tuple[int, int, int], list[int]]:
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n_layers):
        lay = F.expert_layout_for(i, n_layers)
        groups.setdefault((lay.bits_w1, lay.bits_w3, lay.bits_w2), []).append(i)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--layouts", nargs="*", default=["3,3,3", "2,2,3", "4,4,4"], help="bits w1,w3,w2 per layout")
    ap.add_argument("--no-control", action="store_true", help="skip the 8/8/8 near-lossless control run")
    ap.add_argument("--dspark", action="store_true", help="also run the DSpark/MTP layout under an M-key with its top-3 routing")
    ap.add_argument("--keep", action="store_true", help="keep the written expert files in the scratchpad")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads(REF_CONFIG.read_text()) if REF_CONFIG.exists() else {}
    dim, inter = cfg.get("dim", F.DIM), cfg.get("moe_inter_dim", F.INTER)
    assert (dim, inter) == (F.DIM, F.INTER), f"format.py dims {(F.DIM, F.INTER)} != reference config {(dim, inter)}"
    n_layers = cfg.get("n_layers", 40)
    topk = cfg.get("n_activated_experts", 6)
    topk_mtp = cfg.get("dspark_n_activated_experts", 3)
    print(f"mlx {mx.__version__}  expert shapes: w1/w3 [{inter}, {dim}], w2 [{dim}, {inter}]  group {F.GROUP}  ALIGN {F.ALIGN}")
    groups = production_layout_map(n_layers)
    for bits, layers in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lay = F.ExpertLayout(*bits)
        print(f"  production layout {bits[0]}/{bits[1]}/{bits[2]}: {len(layers):2d} layers {layers}  record {lay.nbytes / 2**20:.2f} MiB, stride {lay.stride / 2**20:.2f} MiB")
    mtp = F.expert_layout_for(0, n_layers, is_mtp=True)
    print(f"  DSpark/MTP layout: {mtp.bits_w1}/{mtp.bits_w3}/{mtp.bits_w2} (same bits as the default lossy layout; same MoE class, key M*)")
    want = {tuple(int(b) for b in s.split(",")) for s in args.layouts}
    missing = set(groups) - want
    if missing and args.layouts == ap.get_default("layouts"):
        print(f"WARNING: expert_layout_for emits layouts not covered by the defaults: {missing}")

    runs = [(F.ExpertLayout(*b), "L00", topk) for b in (tuple(int(x) for x in s.split(",")) for s in args.layouts)]
    if args.dspark:
        runs.append((mtp, "M0", topk_mtp))
    if not args.no_control:
        runs.append((F.ExpertLayout(8, 8, 8), "L00", topk))

    results = []
    for lay, key, k in runs:
        print(f"\n== layout {lay.bits_w1}/{lay.bits_w3}/{lay.bits_w2} key {key} top-{min(k, args.experts)}: {args.experts} experts, {args.tokens} tokens ...", flush=True)
        r = run_layout(lay, key, args.tokens, args.experts, k, args.seed, args.keep)
        results.append(r)
        print(f"   converted in {r['t_conv']:.1f}s, slab {r['slab_gb']:.2f} GiB, total {r['time']:.1f}s, store {r['store_stats']}")
        print(f"   kernel dequant regimes: {r['modes']}   (cos of the two references: {r['ref_mm_vs_g']:.7f})")
        for name, c_ref, err, c_orig, ok in r["rows"]:
            print(f"   {'PASS' if ok else 'FAIL'}  {name:46s} cos_vs_dequant={c_ref:.7f}  maxabs={err:.3e}  cos_vs_orig={c_orig:.5f}")
        wc = "/".join(f"{v:.5f}" for v in r["wcos"])
        wp = "/".join(f"{v:.5f}" for v in r["wcos_plain"])
        mr = "/".join(f"{v:.3f}" for v in r["mse_ratio"])
        print(f"   weight cos vs original w1/w3/w2: quantize_mse {wc}  plain mx.quantize {wp}  worst MSE ratio mse/plain {mr}")
        print(f"   slab-bytes-vs-file mismatching parts: {r['bad_parts']}   ref|y|max={r['ref_absmax']:.3f}")

    print("\n" + "=" * 126)
    print(f"{'layout':7s} {'key':4s} {'min cos slab-vs-dequant':>24s} {'max abs err':>12s} {'cos quant-vs-orig':>18s} {'floor':>6s} {'weight cos w1/w3/w2':>26s} {'result':>7s}")
    print("-" * 126)
    fails = []
    for r in results:
        c_min = min(row[1] for row in r["rows"])
        err = max(row[2] for row in r["rows"])
        wc = "/".join(f"{v:.5f}" for v in r["wcos"])
        ok = not r["fail"]
        fails += r["fail"]
        print(f"{r['layout']:7s} {r['key']:4s} {c_min:24.7f} {err:12.3e} {r['damage']:18.5f} {r['floor']:6.3f} {wc:>26s} {'PASS' if ok else 'FAIL':>7s}")
    print("=" * 126)
    print(f"gates: cos(production path, dequantized reference of the same bytes) > {GATE_COS} on every path; slab bytes == file;")
    print(f"       quantize_mse MSE <= plain mx.quantize MSE on every matrix; damage floors {DAMAGE_FLOOR}")
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
