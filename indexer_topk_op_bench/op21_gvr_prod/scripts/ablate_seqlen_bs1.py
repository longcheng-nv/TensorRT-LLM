#!/usr/bin/env python3
"""Seq-len sweep (BS=1) phase-time breakdown of gvr_ms_auto under the
PRODUCTION dispatch (fp32). Nested back-to-front no-op subclass chain
(iter4/iter6/iter7 pattern): each ablated phase's CONSUMERS are already
no-op'd, so no zero-publishing is needed.

Variant chain (increments):
  V0 full
  V1 = P4 off (rank-scatter)                     -> P4   = V0-V1
  V2 = V1 + P3 emit off (sandwich/from_slots/walk+push/gather/classic)
                                                 -> P3   = V1-V2
  V3 = V2 + P2 ladder off + remaining fallback consumers off
       (block_count_ge, snaps)                   -> P2   = V2-V3
  V4 = V3 + P1b off                              -> P1b  = V3-V4
  V5 = V4 + P1 stash off                         -> P1   = V4-V5; base = V5
No-op overrides are generated from the REAL method signatures via AST
(cute.jit cannot trace *args). Timing = paired same-process CUDA-graph
cold-L2 event medians (SCREENING axis; the relative split is the
deliverable). Sanity: V0>=V1>=...>=V5 monotone within jitter.
"""
import ast
import sys
import textwrap
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_BENCH / "ops"))
sys.path.insert(0, str(_BENCH / "op18_gvr_1cta_multithresh" / "src"))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as cr  # noqa: E402
import gvr_ms_op as MS  # noqa: E402
import gvr_msc_op as MC  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
CRMAP = {512: 4, 1024: 4, 2048: 1}


# ---------- signature-preserving no-op override generation ----------
def _sig_map(*mods):
    """method name -> parameter source string, harvested from module AST."""
    out = {}
    for mod in mods:
        tree = ast.parse(Path(mod.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef):
                        args = [a.arg for a in fn.args.args]
                        out.setdefault(fn.name, args)
    return out


import gvr_mt_op as MT18  # noqa: E402  op18 base (ladder defs)
_SIGS = {}
for m in (MC, MS, MT18):
    for k, v in _sig_map(m).items():
        _SIGS.setdefault(k, v)
# vendored base (block_count_ge, phase3_collect_candidates, snap defs)
import importlib  # noqa: E402
for vm in ("cute_vendored.blackwell.top_k.gvr_topk_decode",
           "cute_vendored.blackwell.top_k.gvr_topk_decode_cluster"):
    try:
        for k, v in _sig_map(importlib.import_module(vm)).items():
            _SIGS.setdefault(k, v)
    except Exception:
        pass


_ALL_NOOP_NAMES = None  # filled below after group defs


def _write_noop_module(names):
    """cute.jit's AST preprocessor needs real on-disk source — generate a
    module file with signature-exact no-op methods and import it."""
    lines = ["import cutlass  # noqa", "import cutlass.cute as cute", ""]
    for nm in sorted(set(names)):
        params = _SIGS.get(nm)
        assert params, f"no signature harvested for {nm}"
        lines += [f"@cute.jit",
                  f"def {nm}({', '.join(params)}):",
                  f"    pass", ""]
    p = _HERE / "_gen_noop_methods.py"
    p.write_text("\n".join(lines))
    sys.path.insert(0, str(_HERE))
    import importlib
    mod = importlib.import_module("_gen_noop_methods")
    importlib.reload(mod)
    return mod


_NOOP_MOD = None


def noop_cls(base, names, tag):
    body = {}
    for nm in names:
        if not hasattr(base, nm):
            continue
        body[nm] = getattr(_NOOP_MOD, nm)
    return type(f"{base.__name__}_{tag}", (base,), body)


P4_GRP = ["phase4_band_rank_scatter", "phase4_dist"]
P3_GRP = ["phase3_sandwich", "phase3_from_slots", "phase3_from_slots_mc",
          "_p3_leader_band_gather", "phase3_collect_candidates"]
P2_GRP = ["block_count_collect_multi", "block_count_collect_multi_smem",
          "block_count_collect_multi_base", "block_count_ge",
          "phase4_band_snap", "phase4_band_snap_hist"]
P1B_GRP = ["phase1b_rank_quantile"]
P1_GRP = ["phase1_stats_stash"]
_NOOP_MOD = _write_noop_module(P4_GRP + P3_GRP + P2_GRP + P1B_GRP + P1_GRP)


def make_variants(base, kind):
    # cluster: garbage-driven branch DIVERGENCE across CTAs breaks the
    # collective barriers once the slice ladder is no-op'd -> restrict to
    # the proven full/noP4/noP34 chain (iter6/iter7 pattern). single-CTA:
    # V5 (P1 off) showed negative-increment artifacts -> stop at V4.
    groups = ((("V1", P4_GRP), ("V2", P3_GRP)) if kind == "msc" else
              (("V1", P4_GRP), ("V2", P3_GRP), ("V3", P2_GRP),
               ("V4", P1B_GRP)))
    chain = []
    acc = []
    for tag, grp in groups:
        acc = acc + grp
        chain.append((tag, noop_cls(base, acc, tag)))
    return [("full", base)] + chain


# ---------- production-matched compiles (BS=1 fp32) ----------
def compile_ms(cls, n, K, crv):
    t = 1024 if n >= 65536 else 512
    use256 = (n >= 16384)
    fuse = (4 * K <= 5120)
    kobj = cls(dtype=MS._DT[torch.float32], top_k=K, next_n=1, num_threads=t,
               compress_ratio=crv, use_256bit_load=use256,
               enable_unroll_4=True, enable_phase3_unroll=True,
               min_blocks_per_mp=1, return_output_values=False,
               M_thr=4, R_rounds=1, band_accept=64, place_mode=5,
               kC_override=None, fracs=None, fuse_collect=fuse,
               smem_row_elems=0, p4_rank_scatter=True, qbins=256,
               p4_smallbin=True, p2_native=True)
    return _finish(kobj, use256, K)


def compile_mc(cls, n, K, crv, C):
    kobj = cls(dtype=MS._DT[torch.float32], top_k=K, next_n=1,
               num_threads=1024, compress_ratio=crv,
               use_256bit_load=(n >= 16384), enable_unroll_4=True,
               enable_phase3_unroll=True, min_blocks_per_mp=1,
               return_output_values=False, M_thr=4, R_rounds=1,
               band_accept=64, place_mode=5, fuse_collect=True, C_cta=C)
    return _finish(kobj, n >= 16384, K)


def _finish(kobj, use256, K):
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if use256 else 16
    in_f = cr.make_fake_compact_tensor(MS._DT[torch.float32], (nr, nc),
                                       stride_order=(1, 0), assumed_align=ia)
    pi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb, K),
                                       stride_order=(1, 0), assumed_align=16)
    sl_f = cr.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = cr.make_fake_compact_tensor(cutlass.Int32, (nr, K),
                                       stride_order=(1, 0), assumed_align=16)
    fs = cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, stream=fs,
                        options="--enable-tvm-ffi")


def cold_us(call, reps=40, warmup=5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    cold = []
    for _ in range(reps):
        _EVICT.uniform_(0, 1); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    cold.sort(); del g
    return cold[len(cold) // 2]


# ---------- cells: production dispatch at BS=1 fp32 ----------
CELLS = []
for K in (512, 1024, 2048):
    for N in (4096, 16384, 32768):
        CELLS.append((K, N, "ms", None))
    for N in (65536, 262144):
        C = 8 if (K >= 2048 and N >= 196608) else 4
        CELLS.append((K, N, "msc", C))

print(f"{'K':>5} {'N':>7} {'krn':>5} | raw variant cold-µs (event axis)")
sys.stdout.flush()
for K, N, kind, C in CELLS:
    crv = CRMAP[K]
    b = synth_data.get_bundle(K, torch.float32, N)
    lg = b["logits"][:1].contiguous()
    pre = b["preIdx"][:1].contiguous()
    sl = torch.full((1,), b["Npad"] * crv, dtype=torch.int32, device=DEV)
    out = torch.empty(1, K, dtype=torch.int32, device=DEV)
    base = MS.GvrSandwichKernel if kind == "ms" else MC.GvrMsClusterKernel
    res = {}
    for name, cls in make_variants(base, kind):
        comp = (compile_ms(cls, N, K, crv) if kind == "ms"
                else compile_mc(cls, N, K, crv, C))
        call = lambda: comp(lg, pre, sl, None, out)
        call(); torch.cuda.synchronize()
        res[name] = cold_us(call)
    lbl = f"{kind}{C or ''}"
    print(f"{K:>5} {N:>7} {lbl:>5} | " +
          " ".join(f"{n}={t:.2f}" for n, t in res.items()))
    sys.stdout.flush()
