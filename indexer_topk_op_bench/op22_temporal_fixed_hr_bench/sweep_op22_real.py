# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22 REAL-capture chapter — nsys pure-kernel sweep of the report arm set
on PRODUCTION-CAPTURED indexer logits (REAL_DATA_INVENTORY B/C/D via
harness/real_data_v2.py), one batch per (model, dtype).

Data (per user spec, 2026-07-13):
  K=512  -> V4 Flash  SWE-100K capture (083843_swe100k), 21 layers, LAST
            decode step s=424, N=25154 (real valid length).
  K=1024 -> V4 Pro    SWE-64K native-K1024 capture (164146Z), 30 layers,
            LAST decode step s=304, N=14478.
  K=2048 -> V3.2      SWE_Bench_64K_decode_logits, 9 layers, LAST decode
            step row 2024, N=70690.
Every layer's last-step logits row is one BS=1 input; preIdx = the
PREVIOUS decode step's top-K (captured for V4, exact same-dtype torch.topk
recompute for V3.2 — production dsa.py conventions, see real_data_v2).
BS scaling replicates the SAME row to BS (report methodology; no VarLen).
16-bit cells = fp32 capture truncated; reference = torch.topk on the SAME
truncated dtype. Buffer rows are stride-padded to 64 elems (pad
finfo(dtype).min, seq_lens stay at the REAL N) — production buffers are
stride-padded too (17500/27500/20000 widths).

Arms = the REPORT.html arm set reproducible at HEAD (13 of 15):
  gvr_cutedsl [BASE] | op21_legacy (falsi=0,dist=0) | op27_hls (falsi=1,
  HLS lineage tip: K512/K1024 binaries bit-identical to op21_hls/op25_hls
  ship, K2048 adds the op27 tail ladder) | gvr_multicta_cutedsl |
  radix_cutedsl | radix_single_cuda | radix_multi_cuda | op26_1cta |
  op26_mc | op26_r0auto | sglang_streaming (fp32 & K<=1024) |
  sglang_v2 (fp32) | flashinfer_topk (fp32).
op21_hls / op25_hls are NOT separately re-measurable at HEAD (historical
binaries); op27_hls represents the lineage.

sglang_v2 / flashinfer_topk on real (stride-padded) rows: both get the
[BS, Npad] buffer. sglang_v2 receives the REAL seq_lens=N + max_seq_len=N
(production varlen semantics). flashinfer.top_k has no varlen arg -> it
scans the full Npad row; pad = finfo.min never enters the top-K (<=63
extra elems/row, documented in the report).

Timing protocol IDENTICAL to sweep_op22rr.py (measure_cell: 10 warmup,
warm-L2 "w|" reps, 512MB-evict cold-L2 "c|" reps, eager+sync in range,
cudaProfilerApi window). Exactness at BS=1 for EVERY (arm, layer, dtype):
value-equivalence vdiff/recall/n_neg vs the same-dtype torch.topk ref
(real_data_v2.value_metrics), recorded in the jsonl.

Run UNDER nsys via drive_nsys_op22real.sh.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "harness"))

from sweep import BS_GRID, DTYPES                  # noqa: E402
from sweep_nsys import build_call as _base_build_call, measure_cell  # noqa: E402
from sglang_v2_op import topk_v2, plan as sglv2_plan  # noqa: E402
import real_data_v2 as RD2                          # noqa: E402

DEV = "cuda"
KNOB_F, KNOB_D = "OP21_FB_LOGFALSI", "OP21_FB_DIST"

# arm name -> (harness op, falsi, dist); None = unset (kernel default)
ARMS = [
    ("gvr_cutedsl", "gvr_cutedsl", None, None),
    ("op21_legacy", "gvr_ms_auto", "0", "0"),
    ("op27_hls", "gvr_ms_auto", "1", None),
    ("gvr_multicta_cutedsl", "gvr_multicta_cutedsl", None, None),
    ("radix_cutedsl", "radix_cutedsl", None, None),
    ("radix_single_cuda", "radix_single_cuda", None, None),
    ("radix_multi_cuda", "radix_multi_cuda", None, None),
    ("op26_1cta", "op26_1cta", None, None),
    ("op26_mc", "op26_mc", None, None),
    ("op26_r0auto", "op26_r0auto", None, None),
    ("sglang_streaming", "sglang_streaming", None, None),
    ("sglang_v2", "sglang_v2", None, None),
    ("flashinfer_topk", "flashinfer_topk", None, None),
    # gvr29_hbe (op29 iter12 ship) — supplement batches only (added after
    # the concurrent op29 backfill landed in REPORT.html §1-2, 2026-07-13);
    # selected via OP22REAL_ARMS, NOT in the default 13-arm fleet.
    ("gvr29_hbe", "gvr29_hbe", None, None),
]
DEFAULT_SKIP = {"gvr29_hbe"}
FP32_ONLY = {"sglang_streaming", "sglang_v2", "flashinfer_topk",
             "gvr29_hbe"}

# OP22REAL_V32_NOSHIFT=1 -> v32 preIdx control experiment (pass preIdx-1 so the
# cr=1 kernel's internal +1 recovers RAW alignment; see run_batch note).
_V32_NOSHIFT = os.environ.get("OP22REAL_V32_NOSHIFT") == "1"

# OP22REAL_ARMS="gvr_cutedsl,op27_hls" -> arm subset (debug / split runs)
_ARM_FILTER = os.environ.get("OP22REAL_ARMS")
if _ARM_FILTER:
    _sel = [a.strip() for a in _ARM_FILTER.split(",") if a.strip()]
    _by = {a[0]: a for a in ARMS}
    unknown = [a for a in _sel if a not in _by]
    assert not unknown, f"OP22REAL_ARMS unknown arms: {unknown}"
    ARMS = [_by[a] for a in _sel]


def arms_for(model, dt_name):
    K = RD2.MODELS[model]["K"]
    out = []
    for a in ARMS:
        if a[0] in FP32_ONLY and dt_name != "fp32":
            continue
        if a[0] in DEFAULT_SKIP and not _ARM_FILTER:
            continue
        if a[0] == "sglang_streaming" and K > 1024:
            continue   # verified non-exact at K=2048 (same as the report)
        out.append(a)
    return out


def _pin_env(falsi, dist):
    for var, val in ((KNOB_F, falsi), (KNOB_D, dist)):
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def _build_ext_call(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    """sglang_v2 / flashinfer_topk on stride-padded real rows (Npad >= N).
    keep[3] = out, matching every harness builder (exactness hook)."""
    if dtype != torch.float32:
        raise ValueError(f"{op} is fp32-only in this bench")
    logits = logits_row.to(dtype).expand(BS, -1).contiguous()   # [BS, Npad]
    seq_nod = torch.full((BS,), N, dtype=torch.int32, device=DEV)
    out = torch.empty((BS, K), dtype=torch.int32, device=DEV)
    keep = [logits, seq_nod, None, out]
    if op == "sglang_v2":
        # plan is UNTIMED (production runs it once per step, reused across
        # all indexer layers); timed call = one transform launch.
        md = sglv2_plan(seq_nod)
        torch.cuda.synchronize()
        keep.append(md)
        topk_v2(logits, seq_nod, K, out=out, metadata=md, max_seq_len=N)
        return (lambda: topk_v2(logits, seq_nod, K, out=out, metadata=md,
                                max_seq_len=N)), keep, {}
    if op == "flashinfer_topk":
        import flashinfer
        flashinfer.top_k(logits, K)   # warm (JIT load + allocator)
        # timed call = fi.top_k bare (same contract as op28); exactness at
        # BS=1 re-runs it separately via _exact_fi (i64 indices).
        return (lambda: flashinfer.top_k(logits, K)), keep, {"fi_out": "i64"}
    if op == "gvr29_hbe":
        # op29 iter12 ship arm on stride-padded real rows: same padded-
        # buffer + real-seq_lens treatment as sglang_v2 (ops_ext29's own
        # builder asserts Npad==N, so we materialize here). plan + spill
        # buffer UNTIMED per the op29 report protocol.
        sys.path.insert(0, str(HERE.parents[0] / "op29_gvr_hbe" / "scripts"))
        from gvr29_op import gvr29_topk, plan as g29_plan, _spill_buf
        pre = preidx_row.to(torch.int32).expand(BS, -1).contiguous()
        md = g29_plan(seq_nod)
        spill = _spill_buf(BS, K, DEV, False)   # col_b=False ship default
        torch.cuda.synchronize()
        keep = [logits, seq_nod, pre, out, md, spill]
        gvr29_topk(logits, seq_nod, K, pre, out=out, metadata=md,
                   max_seq_len=N, spill=spill)   # warm (JIT + dispatch)
        return (lambda: gvr29_topk(logits, seq_nod, K, pre, out=out,
                                   metadata=md, max_seq_len=N,
                                   spill=spill)), keep, {}
    raise ValueError(op)


def build_call(op, K, dtype, N, BS, cr, logits_row, preidx_row):
    if op in ("sglang_v2", "flashinfer_topk", "gvr29_hbe"):
        return _build_ext_call(op, K, dtype, N, BS, cr, logits_row, preidx_row)
    return _base_build_call(op, K, dtype, N, BS, cr, logits_row, preidx_row)


def _exact_fi(logits_row, K, ref):
    """flashinfer returns (values, i64 indices); check via a fresh call."""
    import flashinfer
    idx = flashinfer.top_k(logits_row, K)[1][0]
    return RD2.value_metrics(idx.to(torch.int32), logits_row, ref, K)


def _load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["op"], r["model"], r["layer"], r["dtype"], r["BS"]))
            except Exception:
                pass
    return done


def run_batch(model, dt_name, out_path, reps_cold, reps_warm, bs_grid,
              layers=None):
    m = RD2.MODELS[model]
    K, cr = m["K"], m["cr"]
    arms = arms_for(model, dt_name)
    done = _load_done(out_path)
    f = open(out_path, "a")
    cells = [(L, BS) for L in (layers or m["layers"]) for BS in bs_grid]
    total = len(cells)
    prof.start()
    try:
        for i, (L, BS) in enumerate(cells):
            b = RD2.get_real_bundle_v2(model, L, dt_name)
            logits_row, preidx_row, N = b["logits"], b["preIdx"], b["N"]
            hit_rate = b["hit_rate"]
            # CONTROL EXPERIMENT (OP22REAL_V32_NOSHIFT=1): the v32 (cr=1) kernel
            # internally reads logits[(preIdx+1) mod N]; the report's preIdx is
            # the recomputed prev-step top-K (raw current-frame indices). Passing
            # (preIdx-1) mod N makes the kernel's +1 recover the RAW alignment,
            # lifting the kernel-read hit-rate from ~0.44 (+1) to ~0.66 (raw).
            # This isolates whether the GVR arms' real-row collapse is driven by
            # hit-rate/preIdx or by the short-N / traffic-regime structural wall.
            if _V32_NOSHIFT and model == "v32":
                raw = b["preIdx"][0].to(torch.int64) % N
                hit_rate = torch.isin(
                    raw, b["ref"].to(torch.int64)).float().mean().item()
                preidx_row = ((b["preIdx"].to(torch.int64) - 1) % N).to(
                    torch.int32).contiguous()
            for arm, op, falsi, dist in arms:
                if (arm, model, L, dt_name, BS) in done:
                    continue
                base = f"{arm}|{model}|L{L}|{dt_name}|{N}|{BS}"
                rec = {"sweep": "realcap", "op": arm, "harness_op": op,
                       "model": model, "K": K, "dtype": dt_name, "N": N,
                       "Npad": b["Npad"], "BS": BS, "cr": cr, "layer": L,
                       "s_last": b["s_last"],
                       "hit_rate": round(hit_rate, 4),
                       "preidx_variant": ("v32_noshift"
                                          if (_V32_NOSHIFT and model == "v32")
                                          else "default"),
                       "range_cold": f"c|{base}", "range_warm": f"w|{base}",
                       "reps_cold": reps_cold, "reps_warm": reps_warm}
                try:
                    _pin_env(falsi, dist)
                    call, keep, extra = build_call(
                        op, K, RD2.DTYPES[dt_name], N, BS, cr,
                        logits_row, preidx_row)
                    rec.update(extra)

                    def wrapped(_c=call, _f=falsi, _d=dist):
                        _pin_env(_f, _d)
                        _c()
                    if BS == 1:
                        wrapped()
                        torch.cuda.synchronize()
                        if arm == "flashinfer_topk":
                            vd, rc, nn = _exact_fi(logits_row, K, b["ref"])
                        else:
                            out = keep[3]
                            assert out.dtype == torch.int32 \
                                and out.shape == (BS, K), (arm, out.shape)
                            vd, rc, nn = RD2.value_metrics(
                                out[0], logits_row, b["ref"], K)
                        rec["vdiff"] = vd
                        rec["recall"] = round(rc, 5)
                        rec["n_neg"] = nn
                    measure_cell(wrapped, base, reps_cold, reps_warm)
                    del call, keep
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
                gc.collect()
                torch.cuda.empty_cache()
            if (i + 1) % 4 == 0 or i + 1 == total:
                print(f"[realcap {model} {dt_name}] {i+1}/{total} "
                      f"(L{L} BS={BS})", flush=True)
    finally:
        prof.stop()
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RD2.MODELS))
    ap.add_argument("--dtype", required=True, choices=list(RD2.DTYPES))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=20, help="cold-L2 reps")
    ap.add_argument("--reps-warm", type=int, default=50, help="warm-L2 reps")
    ap.add_argument("--bs", default=None, help="comma BS subset")
    ap.add_argument("--layers", default=None, help="comma layer subset")
    args = ap.parse_args()

    bs_grid = ([int(x) for x in args.bs.split(",")] if args.bs else BS_GRID)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    out_root = Path(args.out_root)
    (out_root / "realcap_sweep").mkdir(parents=True, exist_ok=True)
    out_path = (out_root / "realcap_sweep" /
                f"results_{args.model}_{args.dtype}.jsonl")

    m = RD2.MODELS[args.model]
    print(f"# op22real nsys batch: model={args.model} K={m['K']} "
          f"dt={args.dtype} layers={len(layers or m['layers'])} bs={bs_grid} "
          f"arms={[a[0] for a in arms_for(args.model, args.dtype)]} "
          f"reps_cold={args.reps} reps_warm={args.reps_warm}", flush=True)
    run_batch(args.model, args.dtype, out_path, args.reps, args.reps_warm,
              bs_grid, layers)
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
