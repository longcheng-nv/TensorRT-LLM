# [p4f1] Battery for the F1 iterative fine-recursion fix (TRACKF1_DESIGN.md).
#
# Cases:
#   1. flag OFF == baseline snapshot: bit-equal outputs, 3 K x 3 N, random.
#   2. flag ON random: value-set exact vs torch.topk (cs=1 and cs>1 via launch).
#   3. planted adversarial: K-1/K boundary pair inside one level-0 fine bin,
#      swept across coarse-bin edges.
#   4. deep-tie: 64-value 1-ULP ladder straddling K; 1-ULP pair; all-equal
#      row (ULP-floor path); cand_count == kK early path.
#   5. report: pass counts + fine levels exercised (host replay of the
#      published-level recurrence).
#
# Run (umbriel-b200-027):
#   env -u GITHUB_TOKEN -u HF_TOKEN PYTHONNOUSERSITE=1 \
#     PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450 \
#     CUDA_VISIBLE_DEVICES=3 python3 battery_f1.py

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)  # gvrpkgf1
sys.path.insert(0, os.path.join(_REPORT, "gvrpkg_snapshot"))  # gvrpkg (baseline)

import numpy as np
import torch

from gvrpkgf1.top_k.gvr_topk_decode import GvrParams as ParamsF1
from gvrpkgf1.top_k.gvr_topk_decode import GvrTopKKernel as KF1
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel as KBase

DEV = "cuda"
CR = 4  # V4 semantics: preIdxOffset = 0
KS = (512, 1024, 2048)
NS = (4096, 65536, 262144)

RESULTS = []  # (case, subcase, npass, ntot, note)


def kbins_of(k):
    return ParamsF1.get("float32", k, CR).kNumBins


def make_preidx(logits, top_k):
    # hint = topk of a noised copy (realistic partial hit-rate); valid indices
    noisy = logits + 0.05 * torch.randn_like(logits)
    return torch.topk(noisy, top_k, dim=1).indices.to(torch.int32).contiguous()


def run_kernel(kcls, logits, pre_idx, top_k, **overrides):
    num_rows, n = logits.shape
    # seq_lens is uncompressed-token space: N = seq_len // cr (next_n=1)
    seq_lens = torch.full((num_rows,), n * CR, dtype=torch.int32, device=DEV)
    out = torch.empty((num_rows, top_k), dtype=torch.int32, device=DEV)
    kcls.launch(logits, pre_idx, seq_lens, out, top_k, compress_ratio=CR, **overrides)
    torch.cuda.synchronize()
    return out


def valueset_exact_rows(logits, out, top_k):
    """Per-row: no -1, no dup index, value multiset == torch.topk values."""
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


def record(case, sub, oks, note=""):
    npass = sum(oks)
    RESULTS.append((case, sub, npass, len(oks), note))
    tag = "PASS" if npass == len(oks) else "FAIL"
    print(f"[{tag}] case {case} :: {sub} :: {npass}/{len(oks)} {note}", flush=True)
    return npass == len(oks)


# ----------------------------------------------------------------------
# host replay of the level recurrence — reports L for a constructed row
# (mirrors kernel exprs in fp32; used for the §5 levels-exercised report)
# ----------------------------------------------------------------------
def host_levels(row_np, top_k, kbins, kc):
    v = np.asarray(row_np, dtype=np.float32)
    # candidate set approximation: kernel guarantees cand ⊇ topK with
    # kK <= cand_count <= kC; use the top-kC values as a proxy chain input.
    cand = np.sort(v)[::-1][: min(kc, v.size)].astype(np.float32)
    if cand.size <= top_k:
        return 0  # cand_count == kK early path
    bmin, bmax = np.float32(cand.min()), np.float32(cand.max())
    if bmax <= bmin:
        bmax = np.float32(bmin + np.float32(1e-6))
    inv1 = np.float32((np.float32(kbins - 1) + np.float32(0.99)) / (bmax - bmin))
    cb = np.clip(((cand - bmin) * inv1).astype(np.int32), 0, kbins - 1)
    # coarse straddle bin
    order = np.argsort(-cb, kind="stable")
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
    seed = rank_above
    for lvl in range(4):
        sb = np.clip(((chain - f_lo) * finv).astype(np.int32), 0, 255)
        cum = seed
        sb_star, ra = 255, seed
        for b in range(255, -1, -1):
            c = int((sb == b).sum())
            if cum + c >= top_k:
                sb_star, ra = b, cum
                break
            cum += c
        cnt_str = int((sb == sb_star).sum())
        width = np.float32(1.0) / finv
        ulp_floor = np.float32(max(abs(float(f_lo)), 1e-30) * 1.1920928955078125e-07)
        if not (ra + cnt_str > top_k and width > ulp_floor and lvl + 1 < 4):
            return lvl + 1
        chain = chain[sb == sb_star]
        f_lo = np.float32(f_lo + np.float32(sb_star) / finv)
        finv = np.float32(np.float32(255.99) * finv)
        seed = ra
    return 4


# ----------------------------------------------------------------------
# Case 1: flag OFF == baseline snapshot.
# NOTE (measured 2026-07-19): the snapshot baseline is itself run-to-run
# nondeterministic in output ORDER (atomicAdd scatter order), so raw
# output bit-equality fails even baseline-vs-baseline. The design's OFF
# contract is "same code emitted" — checked here as:
#   (a) per-row SORTED index-set bit-equality vs the snapshot (the
#       deterministic part of the output), and
#   (b) byte-equality of the emitted PTX (modulo the mangled kernel name,
#       which embeds the python package name gvrpkg vs gvrpkgf1).
# Requires CUTE_DSL_KEEP_PTX=1 (set below before kernel compiles).
# ----------------------------------------------------------------------
import re


def _norm_ptx(compiled):
    p = compiled.__ptx__
    if isinstance(p, bytes):
        p = p.decode()
    assert p, "empty PTX — CUTE_DSL_KEEP_PTX=1 not effective"
    return re.sub(r"kernel_cutlass_gvr_topk_kernel_\w+", "KNAME", p)


def case1():
    torch.manual_seed(1234)
    for k in KS:
        for n in NS:
            bs = 4
            logits = torch.rand((bs, n), dtype=torch.float32, device=DEV)
            pre_idx = make_preidx(logits, k)
            nb0 = len(KBase._LAUNCH_CACHE)
            nf0 = len(KF1._LAUNCH_CACHE)
            out_base = run_kernel(KBase, logits, pre_idx, k)
            out_off = run_kernel(KF1, logits, pre_idx, k)  # flag absent = OFF
            ok_set = bool(
                torch.equal(
                    out_base.sort(dim=1).values, out_off.sort(dim=1).values
                )
            )
            ok_ptx = True
            if len(KBase._LAUNCH_CACHE) > nb0 and len(KF1._LAUNCH_CACHE) > nf0:
                cb = list(KBase._LAUNCH_CACHE.values())[-1]
                cf = list(KF1._LAUNCH_CACHE.values())[-1]
                ok_ptx = _norm_ptx(cb) == _norm_ptx(cf)
                note = "idxset+PTX-biteq"
            else:
                note = "idxset (variant PTX already checked)"
            record(1, f"OFF K={k} N={n}", [ok_set, ok_ptx], note=note)


def case1b():
    """Bit-identity worlds evidence (coordinator request):
    - random rows (cand>K): even baseline-vs-baseline reruns differ bitwise
      (atomicAdd scatter order) -> raw torch.equal is not a valid OFF check.
    - deterministic rows (cand_count == kK early path, no atomics):
      baseline == F1-OFF must be bitwise equal (torch.equal).
    """
    n, k = 65536, 1024
    torch.manual_seed(99)
    logits = torch.rand((4, n), dtype=torch.float32, device=DEV)
    pre_idx = make_preidx(logits, k)
    b1 = run_kernel(KBase, logits, pre_idx, k)
    b2 = run_kernel(KBase, logits, pre_idx, k)
    f1 = run_kernel(KF1, logits, pre_idx, k)
    base_nondet = not torch.equal(b1, b2)
    off_set_eq = torch.equal(b1.sort(dim=1).values, f1.sort(dim=1).values)
    record(1, "1b base-vs-base rerun differs bitwise (order nondet, expected)",
           [base_nondet])
    record(1, "1b base == OFF sorted-index-set", [off_set_eq])
    # v2 subcase: random rows resolve at fine level 0 (verified by host
    # replay) -> flag ON must take the ORIGINAL hot-path scatter and produce
    # the same selected index set as the baseline (order is nondet, above).
    on = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
    on_set_eq = torch.equal(b1.sort(dim=1).values, on.sort(dim=1).values)
    kb = kbins_of(k)
    kc = ParamsF1.get("float32", k, CR).kC
    lv0 = sorted(
        host_levels(logits[r].cpu().numpy(), k, kb, kc) for r in range(4)
    )
    record(1, "1b random level-0 rows: base == ON sorted-index-set",
           [on_set_eq], note=f"host-replay levels={lv0}")
    rng2 = np.random.default_rng(11)
    row = (rng2.random(n, dtype=np.float32) * 0.35).astype(np.float32)
    posd = rng2.choice(n, size=k, replace=False)
    row[posd] = (10.0 + rng2.random(k)).astype(np.float32)
    logits_d = torch.from_numpy(row).view(1, n).to(DEV)
    pre_d = torch.from_numpy(posd.astype(np.int32)).view(1, k).to(DEV).contiguous()
    db = run_kernel(KBase, logits_d, pre_d, k)
    df = run_kernel(KF1, logits_d, pre_d, k)
    dn = run_kernel(KF1, logits_d, pre_d, k, p4_finebin_loop=True)
    record(1, "1b deterministic row: base == OFF bitwise (torch.equal)",
           [bool(torch.equal(db, df))])
    record(1, "1b deterministic row: base == ON bitwise (torch.equal)",
           [bool(torch.equal(db, dn))])


# ----------------------------------------------------------------------
# Case 2: flag ON, random, value-set exact
# ----------------------------------------------------------------------
def case2():
    torch.manual_seed(5678)
    for k in KS:
        for n in NS:
            bs = 4
            logits = torch.randn((bs, n), dtype=torch.float32, device=DEV)
            pre_idx = make_preidx(logits, k)
            out = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
            oks = valueset_exact_rows(logits, out, k)
            record(2, f"ON-random K={k} N={n} (launch-picked cs)", oks)


# ----------------------------------------------------------------------
# Case 3: planted adversarial — same-fine-bin K-boundary pair
# ----------------------------------------------------------------------
def _dump_failure(tag, row, pre_idx, logits, out, k):
    """Diagnostics + npy dump for a failing planted row."""
    idx = out[0].long()
    neg = int((idx < 0).sum())
    uniq = idx.unique().numel()
    ref = torch.topk(logits[0], k).values
    vals = logits[0, idx.clamp(min=0)].sort(descending=True).values
    mism = (vals != ref).nonzero().flatten()
    print(f"  FAILDUMP {tag}: neg={neg} uniq={uniq}/{k} mism={mism.numel()}")
    if mism.numel():
        mm = mism[:4]
        print(f"    got  {vals[mm].tolist()}")
        print(f"    want {ref[mm].tolist()}")
    # repeat 5x ON + 1x OFF to classify: F1 bug vs pre-existing/flaky
    for t in range(5):
        o2 = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
        print(f"    ON rerun{t}: exact={valueset_exact_rows(logits, o2, k)[0]}")
    o_off = run_kernel(KF1, logits, pre_idx, k)
    print(f"    OFF same row: exact={valueset_exact_rows(logits, o_off, k)[0]}")
    np.save(os.path.join(_HERE, f"fail_{tag}.npy"), row)
    np.save(os.path.join(_HERE, f"fail_{tag}_preidx.npy"), pre_idx.cpu().numpy())


def _classify_planted(logits, pre_idx, out_on, k):
    """Classify a planted row's flag-ON result against the SNAPSHOT baseline.

    Returns (ok, cls):
      ON exact                                  -> (True,  "exact") ;
                                                   also runs baseline: if the
                                                   baseline one-shot is inexact
                                                   here, cls="base-defect-fixed"
                                                   (the F1 fix in action).
      ON inexact, baseline fails with the SAME
      written value-multiset (same -1 pads)     -> (True,  "admission-miss")
                                                   pre-existing P1-P3 undershoot,
                                                   candidate set excludes the
                                                   boundary value BEFORE P4.
      ON inexact, baseline exact or different   -> (False, "f1-defect")
    """
    ok_on = valueset_exact_rows(logits, out_on, k)[0]
    out_b = run_kernel(KBase, logits, pre_idx, k)
    ok_b = valueset_exact_rows(logits, out_b, k)[0]
    if ok_on:
        return True, ("base-defect-fixed" if not ok_b else "exact")
    if not ok_b:
        vs_on = logits[0, out_on[0].long().clamp(min=0)].clone()
        vs_on[out_on[0] < 0] = float("nan")
        vs_b = logits[0, out_b[0].long().clamp(min=0)].clone()
        vs_b[out_b[0] < 0] = float("nan")
        same = torch.equal(
            vs_on.sort().values.nan_to_num(-1.0), vs_b.sort().values.nan_to_num(-1.0)
        )
        if same:
            return True, "admission-miss"
    return False, "f1-defect"


def case3():
    rng = np.random.default_rng(42)
    n = 65536
    for k in KS:
        torch.manual_seed(3000 + k)  # deterministic pre_idx per K
        lvls = set()
        n_adm = 0
        n_fixed = 0
        kb = kbins_of(k)
        kc = ParamsF1.get("float32", k, CR).kC
        oks = []
        for fi, frac in enumerate((0.45, 0.55, 0.6499, 0.65, 0.6501, 0.75)):
            for jit in (-0.25, 0.0, 0.25, 0.5):
                row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
                row[0], row[1] = np.float32(0.0), np.float32(0.9999)
                pos = rng.choice(np.arange(2, n), size=k + 1, replace=False)
                # row[1]=0.9999 is itself a high value -> plant only k-2 more
                # so ranks K / K+1 land exactly on the v2/v1 pair.
                row[pos[: k - 2]] = (0.9 + 0.0999 * rng.random(k - 2)).astype(
                    np.float32
                )
                # candidates are the admitted top <=kC values, so the bin
                # grid range is (max - cand_min), NOT the full row range.
                cand_min_est = float(np.partition(row, n - kc)[n - kc])
                candrange = 0.9999 - cand_min_est
                coarse_w = candrange / kb
                fine_w = candrange / (kb * 255.99)
                # gap: design value fine_w/2 on even fracs; a deep fine_w/16
                # variant on odd fracs guarantees same-fine-bin coverage
                # (the /2 plants straddle bin edges ~50% of the time).
                gap = fine_w / 2.0 if fi % 2 == 0 else fine_w / 16.0
                v1 = np.float32(frac + jit * coarse_w)
                v2 = np.float32(v1 + gap)
                row[pos[k - 1]] = v2  # must be selected (rank K)
                row[pos[k]] = v1  # must be dropped (rank K+1)
                logits = torch.from_numpy(row).view(1, n).to(DEV)
                # plant sanity: rank-K value must be v2, rank-K+1 must be v1
                refk1 = torch.topk(logits[0], k + 1).values
                assert refk1[-2].item() == float(v2) and refk1[-1].item() == float(v1), (
                    k, frac, jit)
                pre_idx = make_preidx(logits, k)
                out = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
                ok, cls = _classify_planted(logits, pre_idx, out, k)
                if cls == "admission-miss":
                    n_adm += 1
                elif cls == "base-defect-fixed":
                    n_fixed += 1
                if not ok:
                    print(f"  case3 FAIL K={k} frac={frac} jit={jit}")
                    _dump_failure(f"c3_k{k}_f{fi}_j{jit}", row, pre_idx, logits, out, k)
                oks.append(ok)
                lvls.add(host_levels(row, k, kb, kc))
        record(3, f"planted same-fine-bin K={k} (24 rows)", oks,
               note=f"host-replay levels={sorted(lvls)} "
                    f"base-defect-fixed={n_fixed} admission-miss(pre-exist)={n_adm}")


# ----------------------------------------------------------------------
# Case 4: deep ties
# ----------------------------------------------------------------------
def case4():
    torch.manual_seed(4444)  # deterministic pre_idx
    rng = np.random.default_rng(7)
    k = 1024
    kb = kbins_of(k)
    kc = ParamsF1.get("float32", k, CR).kC

    # (a) 64-value 1-ULP ladder straddling K (forces >=2 extra levels)
    for n in (4096, 65536):
        oks, lv = [], set()
        for rep in range(4):
            row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
            row[0], row[1] = np.float32(0.0), np.float32(0.9999)
            pos = rng.choice(np.arange(2, n), size=k + 32, replace=False)
            row[pos[: k - 32]] = (0.9 + 0.0999 * rng.random(k - 32)).astype(
                np.float32
            )
            ladder = np.empty(64, dtype=np.float32)
            v = np.float32(0.5)
            for i in range(64):
                ladder[i] = v
                v = np.nextafter(v, np.float32(1.0), dtype=np.float32)
            row[pos[k - 32 : k + 32]] = ladder  # top 32 of ladder are in top-K
            logits = torch.from_numpy(row).view(1, n).to(DEV)
            pre_idx = make_preidx(logits, k)
            out = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
            oks.extend(valueset_exact_rows(logits, out, k))
            lv.add(host_levels(row, k, kb, kc))
        record(4, f"64x 1-ULP ladder K={k} N={n}", oks,
               note=f"host-replay levels={sorted(lv)}")

    # (b) 1-ULP boundary pair
    oks, lv = [], set()
    n_adm = 0
    n_fixed = 0
    n = 65536
    for rep in range(8):
        row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
        row[0], row[1] = np.float32(0.0), np.float32(0.9999)
        pos = rng.choice(np.arange(2, n), size=k + 1, replace=False)
        # row[1]=0.9999 counts as one high -> k-2 planted highs (see case3)
        row[pos[: k - 2]] = (0.9 + 0.0999 * rng.random(k - 2)).astype(np.float32)
        v1 = np.float32(0.4 + 0.3 * rng.random())
        v2 = np.nextafter(v1, np.float32(1.0), dtype=np.float32)
        row[pos[k - 1]] = v2
        row[pos[k]] = v1
        logits = torch.from_numpy(row).view(1, n).to(DEV)
        pre_idx = make_preidx(logits, k)
        out = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
        ok, cls = _classify_planted(logits, pre_idx, out, k)
        if cls == "admission-miss":
            n_adm += 1
        elif cls == "base-defect-fixed":
            n_fixed += 1
        if not ok:
            _dump_failure(f"c4b_rep{rep}", row, pre_idx, logits, out, k)
        oks.append(ok)
        lv.add(host_levels(row, k, kb, kc))
    record(4, f"1-ULP pair K={k} N={n}", oks,
           note=f"host-replay levels={sorted(lv)} "
                f"base-defect-fixed={n_fixed} admission-miss(pre-exist)={n_adm}")

    # (c) all-equal row (ULP-floor / degenerate-range path)
    for const in (1.0, 0.0, -3.5):
        n = 65536
        row = np.full(n, const, dtype=np.float32)
        logits = torch.from_numpy(row).view(1, n).to(DEV)
        pre_idx = (
            torch.arange(k, dtype=torch.int32, device=DEV).view(1, k).contiguous()
        )
        out = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
        oks = valueset_exact_rows(logits, out, k)
        record(4, f"all-equal row const={const}", oks)

    # (d) cand_count == kK early path: K clearly-separated highs, exact hint
    n = 65536
    row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
    pos = rng.choice(n, size=k, replace=False)
    row[pos] = (10.0 + rng.random(k)).astype(np.float32)
    logits = torch.from_numpy(row).view(1, n).to(DEV)
    pre_idx = torch.from_numpy(pos.astype(np.int32)).view(1, k).to(DEV).contiguous()
    out_on = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
    out_off = run_kernel(KF1, logits, pre_idx, k)
    oks = valueset_exact_rows(logits, out_on, k)
    oks.append(bool(torch.equal(out_on, out_off)))  # early path untouched
    record(4, f"cand==kK early path K={k}", oks, note="incl. ON==OFF biteq")


def host_l0_cnt(row_np, top_k, kbins, kc):
    """Host replay of level 0 only -> (ra_fine, cnt_straddle) estimate."""
    v = np.asarray(row_np, dtype=np.float32)
    cand = np.sort(v)[::-1][: min(kc, v.size)].astype(np.float32)
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
    return ra, int((sb == sb_star).sum())


def case5():
    """v3 CAP boundary: cnt_straddle == CAP (tail-select path) and CAP+1
    (deep-recursion fallback). Cluster = 8 distinct 1-ULP-spaced values
    repeated to 128/129 copies inside ONE level-0 fine bin straddling K."""
    torch.manual_seed(5555)
    rng = np.random.default_rng(55)
    k = 1024
    kb = kbins_of(k)
    kc = ParamsF1.get("float32", k, CR).kC
    CAP = 128
    for n in (4096, 65536):
        for extra, label in ((0, "cnt==CAP select"), (1, "cnt==CAP+1 deepfb")):
            oks = []
            cnts = []
            n_adm = 0
            n_fixed = 0
            for rep in range(3):
                cn = CAP + extra
                row = (rng.random(n, dtype=np.float32) * 0.35).astype(np.float32)
                row[0], row[1] = np.float32(0.0), np.float32(0.9999)
                pos = rng.choice(np.arange(2, n), size=(k - 65) + cn, replace=False)
                row[pos[: k - 65]] = (0.9 + 0.0999 * rng.random(k - 65)).astype(
                    np.float32
                )
                base = np.float32(0.5 + 0.003 * rep)
                v8 = [base]
                for _ in range(7):
                    v8.append(np.nextafter(v8[-1], np.float32(1.0),
                                           dtype=np.float32))
                clu = np.array([v8[i % 8] for i in range(cn)], dtype=np.float32)
                row[pos[k - 65:]] = clu
                logits = torch.from_numpy(row).view(1, n).to(DEV)
                pre_idx = make_preidx(logits, k)
                out = run_kernel(KF1, logits, pre_idx, k, p4_finebin_loop=True)
                ok, cls = _classify_planted(logits, pre_idx, out, k)
                if cls == "admission-miss":
                    n_adm += 1
                elif cls == "base-defect-fixed":
                    n_fixed += 1
                if not ok:
                    _dump_failure(f"c5_n{n}_e{extra}_r{rep}", row, pre_idx,
                                  logits, out, k)
                oks.append(ok)
                cnts.append(host_l0_cnt(row, k, kb, kc)[1])
            record(5, f"{label} K={k} N={n}", oks,
                   note=f"host cnt_straddle={cnts} "
                        f"base-defect-fixed={n_fixed} admission-miss={n_adm}")


def main():
    assert os.environ.get("CUTE_DSL_KEEP_PTX") == "1", (
        "run with CUTE_DSL_KEEP_PTX=1 (case-1 PTX byte-equality check)"
    )
    assert torch.cuda.is_available()
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    case1()
    case1b()
    case2()
    case3()
    case4()
    case5()
    print("\n===== battery_f1 summary =====")
    total_p = total_t = 0
    fail = 0
    for case, sub, npass, ntot, note in RESULTS:
        tag = "PASS" if npass == ntot else "FAIL"
        if npass != ntot:
            fail += 1
        total_p += npass
        total_t += ntot
        print(f"  [{tag}] case{case} {sub}: {npass}/{ntot} {note}")
    print(f"TOTAL: {total_p}/{total_t} rows, {fail} failing subcases")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
