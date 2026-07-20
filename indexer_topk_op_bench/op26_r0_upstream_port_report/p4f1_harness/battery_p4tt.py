# [p4tt] Battery for the tiny-tie collect+select fast path in gvrpkgprod2.
#
# Arms: fast = gvrpkgprod2 p4_tail_fast=True (default), slow = gvrpkgprod2
# p4_tail_fast=False (== shipped head), pristine = unmodified package copy.
#
# Run (GPU 3):
#   env -u GITHUB_TOKEN -u HF_TOKEN PYTHONNOUSERSITE=1 CUTE_DSL_KEEP_PTX=1 \
#     PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450 \
#     CUDA_VISIBLE_DEVICES=3 python3 battery_p4tt.py
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "harness"))
sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/"
                   "TensorRT-LLM/indexer_topk_op_bench/harness")

import numpy as np
import torch

from gvrpkgprod2.top_k.gvr_topk_decode import GvrParams as Params
from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as KMOD
from gvrpkgprod2_pristine.top_k.gvr_topk_decode import GvrTopKKernel as KPRI

DEV = "cuda"
CR = 4
KS = (512, 1024, 2048)
NS = (4096, 65536, 262144)
CAP = 128
NEGMAX = torch.finfo(torch.float32).min  # == -FLT_MAX

RESULTS = []


def record(case, sub, oks, note=""):
    npass = sum(oks)
    RESULTS.append((case, sub, npass, len(oks), note))
    tag = "PASS" if npass == len(oks) else "FAIL"
    print(f"[{tag}] case {case} :: {sub} :: {npass}/{len(oks)} {note}", flush=True)


def make_preidx(logits, top_k):
    noisy = logits + 0.05 * torch.randn_like(logits)
    return torch.topk(noisy, top_k, dim=1).indices.to(torch.int32).contiguous()


def run_kernel(kcls, logits, pre_idx, top_k, **overrides):
    num_rows, n = logits.shape
    seq_lens = torch.full((num_rows,), n * CR, dtype=torch.int32, device=DEV)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)
    kcls.launch(logits, pre_idx, seq_lens, out, top_k, compress_ratio=CR,
                **overrides)
    torch.cuda.synchronize()
    return out


def run_fast(logits, pre, k, **ov):
    return run_kernel(KMOD, logits, pre, k, p4_tail_fast=True, **ov)


def run_slow(logits, pre, k, **ov):
    return run_kernel(KMOD, logits, pre, k, p4_tail_fast=False, **ov)


def valueset_exact_rows(logits, out, top_k):
    oks = []
    ref = torch.topk(logits.float(), top_k, dim=1).values
    for r in range(logits.shape[0]):
        idx = out[r].long()
        if (idx < 0).any() or idx.unique().numel() != top_k:
            oks.append(False)
            continue
        vals = logits[r, idx].float().sort(descending=True).values
        oks.append(bool(torch.equal(vals, ref[r])))
    return oks


def agree_sorted(a, b):
    """Sorted-INDEX-set agreement (valid only on tie-free rows)."""
    return bool(torch.equal(a.sort(dim=1).values, b.sort(dim=1).values))


def agree_values(logits, a, b):
    """Sorted-VALUE-multiset agreement (ties are interchangeable, so on
    tie rows the index sets may legitimately differ between arms)."""
    va = logits[0, a[0].long().clamp(min=0)].clone()
    vb = logits[0, b[0].long().clamp(min=0)].clone()
    va[a[0] < 0] = float("nan")
    vb[b[0] < 0] = float("nan")
    return bool(torch.equal(va.sort().values.nan_to_num(-1.0),
                            vb.sort().values.nan_to_num(-1.0)))


def _norm_ptx(compiled):
    p = compiled.__ptx__
    if isinstance(p, bytes):
        p = p.decode()
    assert p, "empty PTX — CUTE_DSL_KEEP_PTX=1 not effective"
    return re.sub(r"kernel_cutlass_gvr_topk_kernel_\w+", "KNAME", p)


# ----------------------------------------------------------------------
def caseA():
    """p4_tail_fast=False PTX byte-identity vs the pristine package."""
    torch.manual_seed(101)
    n = 65536
    for k in KS:
        logits = torch.rand((2, n), dtype=torch.float32, device=DEV)
        pre = make_preidx(logits, k)
        np0, nm0 = len(KPRI._LAUNCH_CACHE), len(KMOD._LAUNCH_CACHE)
        out_p = run_kernel(KPRI, logits, pre, k)
        out_s = run_slow(logits, pre, k)
        ok_set = agree_sorted(out_p, out_s)
        cp = list(KPRI._LAUNCH_CACHE.values())[-1]
        cs = list(KMOD._LAUNCH_CACHE.values())[-1]
        ok_ptx = _norm_ptx(cp) == _norm_ptx(cs)
        assert len(KPRI._LAUNCH_CACHE) > np0 and len(KMOD._LAUNCH_CACHE) > nm0
        record("A", f"False==pristine K={k} N={n}", [ok_set, ok_ptx],
               note="idxset+PTX-biteq")


# ----------------------------------------------------------------------
def caseB():
    # B1 random
    torch.manual_seed(202)
    for k in KS:
        for n in NS:
            logits = torch.randn((4, n), dtype=torch.float32, device=DEV)
            pre = make_preidx(logits, k)
            of = run_fast(logits, pre, k)
            os_ = run_slow(logits, pre, k)
            oks = valueset_exact_rows(logits, of, k)
            oks.append(agree_sorted(of, os_))
            record("B", f"random K={k} N={n}", oks, note="exact+fast==slow-set")
    # B2 planted same-fine-bin pairs
    kbins = {512: 1024, 1024: 1024, 2048: 2048}
    rng = np.random.default_rng(42)
    n = 65536
    for k in KS:
        torch.manual_seed(3000 + k)
        kb = kbins[k]
        kc = Params.get("float32", k, CR).kC
        oks = []
        for fi, frac in enumerate((0.45, 0.55, 0.6499, 0.65, 0.6501, 0.75)):
            for jit in (-0.25, 0.0, 0.25, 0.5):
                row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
                row[0], row[1] = np.float32(0.0), np.float32(0.9999)
                pos = rng.choice(np.arange(2, n), size=k + 1, replace=False)
                row[pos[: k - 2]] = (0.9 + 0.0999 * rng.random(k - 2)).astype(
                    np.float32)
                cand_min_est = float(np.partition(row, n - kc)[n - kc])
                candrange = 0.9999 - cand_min_est
                coarse_w = candrange / kb
                fine_w = candrange / (kb * 255.99)
                gap = fine_w / 2.0 if fi % 2 == 0 else fine_w / 16.0
                v1 = np.float32(frac + jit * coarse_w)
                v2 = np.float32(v1 + gap)
                row[pos[k - 1]] = v2
                row[pos[k]] = v1
                logits = torch.from_numpy(row).view(1, n).to(DEV)
                pre = make_preidx(logits, k)
                of = run_fast(logits, pre, k)
                ok = valueset_exact_rows(logits, of, k)[0]
                if not ok:
                    # pre-existing admission miss iff slow fails identically
                    os_ = run_slow(logits, pre, k)
                    ok = (not valueset_exact_rows(logits, os_, k)[0]) and \
                        agree_values(logits, of, os_)
                oks.append(ok)
        record("B", f"planted pairs K={k} (24 rows)", oks,
               note="exact (or slow fails identically = admission)")
    # B3 64x 1-ULP ladder
    torch.manual_seed(4444)
    rng = np.random.default_rng(7)
    k = 1024
    for n in (4096, 65536):
        oks = []
        for rep in range(4):
            row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
            row[0], row[1] = np.float32(0.0), np.float32(0.9999)
            pos = rng.choice(np.arange(2, n), size=k + 32, replace=False)
            row[pos[: k - 32]] = (0.9 + 0.0999 * rng.random(k - 32)).astype(
                np.float32)
            ladder = np.empty(64, dtype=np.float32)
            v = np.float32(0.5)
            for i in range(64):
                ladder[i] = v
                v = np.nextafter(v, np.float32(1.0), dtype=np.float32)
            row[pos[k - 32: k + 32]] = ladder
            logits = torch.from_numpy(row).view(1, n).to(DEV)
            pre = make_preidx(logits, k)
            of = run_fast(logits, pre, k)
            oks.extend(valueset_exact_rows(logits, of, k))
        record("B", f"64x 1-ULP ladder K={k} N={n}", oks)
    # B4 all-equal rows
    k, n = 1024, 65536
    for const in (1.0, 0.0, -3.5):
        logits = torch.full((1, n), const, dtype=torch.float32, device=DEV)
        pre = torch.arange(k, dtype=torch.int32, device=DEV).view(1, k).contiguous()
        of = run_fast(logits, pre, k)
        record("B", f"all-equal const={const}", valueset_exact_rows(logits, of, k))
    # B5 cand==kK deterministic row: fast == slow bitwise
    rng2 = np.random.default_rng(11)
    row = (rng2.random(n, dtype=np.float32) * 0.35).astype(np.float32)
    posd = rng2.choice(n, size=k, replace=False)
    row[posd] = (10.0 + rng2.random(k)).astype(np.float32)
    logits = torch.from_numpy(row).view(1, n).to(DEV)
    pre = torch.from_numpy(posd.astype(np.int32)).view(1, k).to(DEV).contiguous()
    of = run_fast(logits, pre, k)
    os_ = run_slow(logits, pre, k)
    oks = valueset_exact_rows(logits, of, k)
    oks.append(bool(torch.equal(of, os_)))
    record("B", "cand==kK early path (bitwise fast==slow)", oks)


# ----------------------------------------------------------------------
def caseC():
    """CAP boundary: cnt_strad==128 (fast select) and 129 (radix fallback)."""
    torch.manual_seed(5555)
    rng = np.random.default_rng(55)
    k, n = 1024, 65536
    for extra, label in ((0, "cnt==128 fast"), (1, "cnt==129 radix-fb")):
        oks = []
        for rep in range(3):
            cn = CAP + extra
            row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
            row[0], row[1] = np.float32(0.0), np.float32(0.9999)
            pos = rng.choice(np.arange(2, n), size=(k - 65) + cn, replace=False)
            row[pos[: k - 65]] = (0.9 + 0.0999 * rng.random(k - 65)).astype(
                np.float32)
            base = np.float32(0.5 + 0.003 * rep)
            v8 = [base]
            for _ in range(7):
                v8.append(np.nextafter(v8[-1], np.float32(1.0), dtype=np.float32))
            clu = np.array([v8[i % 8] for i in range(cn)], dtype=np.float32)
            row[pos[k - 65:]] = clu
            logits = torch.from_numpy(row).view(1, n).to(DEV)
            pre = make_preidx(logits, k)
            of = run_fast(logits, pre, k)
            ok = valueset_exact_rows(logits, of, k)[0]
            if not ok:  # admission-miss guard (slow fails identically)
                os_ = run_slow(logits, pre, k)
                ok = (not valueset_exact_rows(logits, os_, k)[0]) and \
                    agree_values(logits, of, os_)
            oks.append(ok)
        record("C", f"{label} K={k} N={n}", oks)


# ----------------------------------------------------------------------
def caseD():
    """-FLT_MAX inside the straddle class (marker-free requirement teeth):
    row = (K-5) highs + (N-K+5) copies of -FLT_MAX; boundary tie class is
    the -FLT_MAX plateau, cnt = N-K+5, need = 5. N sized so cnt <= 128
    (fast path) and one variant with cnt > 128 (radix)."""
    k = 1024
    rng = np.random.default_rng(77)
    for n, label in ((1088, "cnt=69 fast"), (1216, "cnt=197 radix")):
        cnt = n - (k - 5)
        row = np.full(n, NEGMAX, dtype=np.float32)
        pos = rng.choice(n, size=k - 5, replace=False)
        row[pos] = (0.9 + 0.0999 * rng.random(k - 5)).astype(np.float32)
        logits = torch.from_numpy(row).view(1, n).to(DEV)
        pre = torch.arange(k, dtype=torch.int32, device=DEV).view(1, k).contiguous()
        of = run_fast(logits, pre, k)
        os_ = run_slow(logits, pre, k)
        oks = valueset_exact_rows(logits, of, k)
        oks.extend(valueset_exact_rows(logits, os_, k))
        oks.append(agree_values(logits, of, os_))
        record("D", f"-FLT_MAX straddle {label} N={n}", oks,
               note=f"class cnt={cnt} need=5")


# ----------------------------------------------------------------------
def caseF():
    """Launch-contract compile+exact smoke over ALL 25 real bench cells
    (flash/pro/v32 x every ISL rung, bench layer, fp32 BS=1) — the exact
    per-cell pick_config variants Gate C' exercises (run1-3 only covered
    the synthetic N grid, missing e.g. the cs=4/cs=8 switch cells at
    256k/512k). Both flags via the launch() path. PASS per arm-pair iff
    fast is value-exact, or slow fails identically (pre-existing
    admission/undershoot behavior) with agreeing value multisets."""
    import real_data_v4cap as RV4
    import real_data_v32 as RV32
    RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)

    bench_l = {"flash": 22, "pro": 30, "v32": 34}
    isls = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k",
                      "512k", "1024k"],
            "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k",
                      "512k", "1024k"],
            "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
    for model in ("flash", "pro", "v32"):
        RD = RV32 if model == "v32" else RV4
        L = bench_l[model]
        oks, notes = [], []
        for isl in isls[model]:
            bd = RD.get_bundle(model, isl, L, "fp32")
            k, n, cr = int(bd["K"]), int(bd["N"]), int(bd["cr"])
            lg = bd["logits"].contiguous()
            pre = bd["preIdx"].contiguous()
            sl = torch.full((1,), n * cr, dtype=torch.int32, device=DEV)
            outs = {}
            compile_fail = None
            for tag, tf in (("fast", True), ("slow", False)):
                ob = torch.empty((1, k), dtype=torch.int32, device=DEV)
                try:
                    KMOD.launch(lg, pre, sl, ob, k, compress_ratio=cr,
                                p4_tail_fast=tf)
                    torch.cuda.synchronize()
                    outs[tag] = ob
                except Exception as e:  # compile or launch failure = hard FAIL
                    compile_fail = f"{isl}/{tag}: {type(e).__name__}"
                    break
            if compile_fail is not None:
                oks.append(False)
                notes.append(compile_fail)
                continue
            ok = valueset_exact_rows(lg, outs["fast"], k)[0]
            if not ok:
                # baseline-gated: pre-existing iff slow fails identically
                ok = (not valueset_exact_rows(lg, outs["slow"], k)[0]) and \
                    agree_values(lg, outs["fast"], outs["slow"])
                if ok:
                    notes.append(f"{isl}: baseline-gated (slow non-exact too)")
            oks.append(ok)
        record("F", f"real-cell launch-contract smoke {model} "
               f"({len(isls[model])} ISLs, L{L})", oks,
               note="; ".join(notes) if notes else "all exact")


# ----------------------------------------------------------------------
def caseE():
    """Real cell pro/512k L30 (K1024, N=131075): exactness + CUDA-event
    warm timing fast vs slow (smoke; nsys verdict is the coordinator's)."""
    import real_data_v4cap as RD4

    b = RD4.get_bundle("pro", "512k", 30, "fp32")
    lg = b["logits"].contiguous()
    pre = b["preIdx"].contiguous()
    N = b["N"]
    K = 1024
    sl = torch.full((1,), N * CR, dtype=torch.int32, device=DEV)
    ov = dict(cluster_size=4, num_threads=1024,
              use_256bit_load=(lg.data_ptr() % 32 == 0), min_blocks_per_mp=1,
              enable_warp_parallel_reduce=True)

    L2 = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEV)

    def arm(fastflag):
        out = torch.empty((1, K), dtype=torch.int32, device=DEV)
        # prime the JIT cache, then time the PRECOMPILED callable directly
        # (KMOD.launch's python cfg/dict work would otherwise land inside
        # the event brackets and inflate the reading by ~15-20us).
        KMOD.launch(lg, pre, sl, out, K, compress_ratio=CR,
                    p4_tail_fast=fastflag, **ov)
        compiled = list(KMOD._LAUNCH_CACHE.values())[-1]

        def call():
            compiled(lg, pre, sl, None, out, None)
        call()
        torch.cuda.synchronize()
        idx = out[0].long()
        ok = bool((idx >= 0).all() and idx.unique().numel() == K)
        if ok:
            kv = lg[0, :N].gather(0, idx).sort().values
            rv = lg[0, :N].gather(0, b["ref"].long()).sort().values
            ok = bool(torch.equal(kv, rv))
        def bench(flush):
            for _ in range(30):
                call()
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
            for i in range(200):
                if flush:
                    L2.zero_()
                starts[i].record()
                call()
                ends[i].record()
            torch.cuda.synchronize()
            ts = sorted(starts[i].elapsed_time(ends[i]) * 1e3
                        for i in range(200))
            return ts[len(ts) // 2]
        return ok, bench(False), bench(True)

    ok_f, warm_f, cold_f = arm(True)
    ok_s, warm_s, cold_s = arm(False)
    record("E", f"pro/512k L30 N={N} exact fast/slow", [ok_f, ok_s],
           note=f"median us warm fast={warm_f:.2f} slow={warm_s:.2f} "
                f"(ratio {warm_s/warm_f:.3f}) | L2-flushed fast={cold_f:.2f} "
                f"slow={cold_s:.2f} (ratio {cold_s/cold_f:.3f})")


def main():
    assert os.environ.get("CUTE_DSL_KEEP_PTX") == "1"
    assert torch.cuda.is_available()
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    caseA()
    caseB()
    caseC()
    caseD()
    caseF()
    caseE()
    print("\n===== battery_p4tt summary =====")
    total_p = total_t = fail = 0
    for case, sub, npass, ntot, note in RESULTS:
        tag = "PASS" if npass == ntot else "FAIL"
        fail += int(npass != ntot)
        total_p += npass
        total_t += ntot
        print(f"  [{tag}] case{case} {sub}: {npass}/{ntot} {note}")
    print(f"TOTAL: {total_p}/{total_t} rows, {fail} failing subcases")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
