# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op37-lj] tight_bracket (L-J multi-rung tight bracket) exactness battery.

Clone of battery_dp4.py adapted to the tight_bracket ctor flag. Sections:

  S1 grid      fp32 x K{512,1024,2048} x N{65536,131072,262144} x BS{2,8,64},
               cs from pick_config (NO forcing — tight_bracket works at any
               cs, unlike dist_p4). Per cell:
                 - tight_bracket=True  -> exact vs torch.topk (index validity
                   + uniqueness + value multiset)
                 - tight_bracket=False control -> identical sorted index SET
                   to the PRISTINE gvrpkgprod2 output on the same inputs +
                   exact. (Raw byte-equality is atomic-arrival nondeterminstic
                   even on the pristine kernel alone; the default-off byte
                   contract is proven at the PTX level, see prove_ptx_lj.py.)
  S2 tie       forced tie fixtures, per K, tight_bracket=True, with
               high-hit preIdx (exact top-K indices, -1 compensating the
               cr=1 kernel-side +1 shift) so the bracket admission FIRES
               with a real hi rung (sure set > 0):
                 (a) BIG tie plateau straddling the K boundary inside the
                     band (K+500 members > 128): band radix-select arm.
                 (b) SMALL tie plateau (12 members, need0=6): [p4tt] fast
                     arm at K>=1024 (K512 compiles radix-only).
                 (c) all-distinct shuffled linspace: non-firing tail gate.
                 (d) hi-boundary plateau: 400 bit-equal duplicates located
                     ABOVE the K boundary so rung thresholds land at/inside
                     the plateau — duplicates of the hi rung value must land
                     on ONE side of the sure/band split (split is by value,
                     >= thr_hi), exactness by value multiset.
  S3 launch    contract smoke: cluster_size overrides {1,2,4,8} at
               K=1024 N=131072, BS{1,4}, tight_bracket=True, exact.
  S4 realdata  all 12 (model,isl) cells from src/replay_band_lj.py CELLS at
               BS=2, tight_bracket=True, exact vs torch.topk over the valid
               N prefix. tb_debug=True prints the per-row bracket-fire map
               ([tb] fired/lo_m/hi_m/cnt_hi/band) between the cell markers.
"""
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH / "op26_r0_upstream_port_report" / "p4f1_harness"))
sys.path.insert(0, str(_BENCH / "harness"))

from gvrpkg37.top_k.gvr_topk_decode import GvrTopKKernel as Gvr37  # noqa: E402
from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as GvrRef  # noqa: E402

DEV = "cuda"
CR = 1  # synthetic sections use the v32 convention: N == seqlen


def run_kernel(cls, logits, pre, K, tb, cr=CR, cs_force=None, tb_debug=False,
               n_valid=None):
    BS, N = logits.shape
    if n_valid is not None:
        N = n_valid  # padded real-capture rows: scan only the valid prefix
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ovr = {}
    if tb:
        ovr["tight_bracket"] = True
        if tb_debug:
            ovr["tb_debug"] = True
    if cs_force is not None:
        ovr["cluster_size"] = cs_force
    cls.launch(logits, pre, sl, out, K, compress_ratio=cr, **ovr)
    torch.cuda.synchronize()
    return out


def check_exact(logits, out, K, n_valid=None):
    """Value-multiset exactness vs torch.topk + index validity/uniqueness."""
    BS, N = logits.shape
    if n_valid is not None:
        N = n_valid
        logits = logits[:, :N]
    ref = torch.sort(torch.topk(logits.float(), K, dim=1).values,
                     dim=1, descending=True).values
    for b in range(BS):
        idx = out[b].long()
        if (idx < 0).any() or (idx >= N).any():
            return False, f"row{b}: invalid index (pad/-1 or OOB)"
        if idx.unique().numel() != K:
            return False, f"row{b}: duplicate indices"
        got = torch.sort(logits[b, idx].float(), descending=True).values
        if not torch.equal(got, ref[b]):
            nbad = (got != ref[b]).sum().item()
            return False, f"row{b}: value multiset mismatch ({nbad} slots)"
    return True, ""


def make_inputs(K, N, BS, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    logits = torch.randn(BS, N, dtype=torch.float32, device=DEV, generator=g)
    noisy = logits + 0.8 * logits.std() * torch.randn(
        BS, N, dtype=torch.float32, device=DEV, generator=g)
    pre = torch.topk(noisy, K, dim=1).indices.int().contiguous()
    return logits, pre


def make_pre_exact(logits, K):
    """High-hit hints: true top-K indices, -1 to compensate the cr==1
    kernel-side +1 temporal shift (so P1 gathers the true top-K VALUES)."""
    idx = torch.topk(logits, K, dim=1).indices
    return (idx - 1).clamp(min=0).int().contiguous()


def main():
    torch.cuda.init()
    sec_pass = {"S1_grid": 0, "S2_tie": 0, "S3_launch": 0, "S4_real": 0}
    sec_tot = {"S1_grid": 0, "S2_tie": 0, "S3_launch": 0, "S4_real": 0}

    # ---------------- S1: full grid ----------------
    grid_K = (512, 1024, 2048)
    grid_N = (65536, 131072, 262144)
    grid_BS = (2, 8, 64)
    seed = 20260721

    for K in grid_K:
        for N in grid_N:
            for BS in grid_BS:
                seed += 1
                logits, pre = make_inputs(K, N, BS, seed)
                cfg = Gvr37.pick_config(torch.float32, BS, N)
                out_tb = run_kernel(Gvr37, logits, pre, K, tb=True)
                ok_t, why_t = check_exact(logits, out_tb, K)
                out_ctrl = run_kernel(Gvr37, logits, pre, K, tb=False)
                out_ref = run_kernel(GvrRef, logits, pre, K, tb=False)
                ok_c = torch.equal(out_ctrl.sort(dim=1).values,
                                   out_ref.sort(dim=1).values)
                why_c = "" if ok_c else "ctrl index-set != pristine"
                ok_ce, why_ce = check_exact(logits, out_ctrl, K)
                ok = ok_t and ok_c and ok_ce
                sec_tot["S1_grid"] += 1
                sec_pass["S1_grid"] += int(ok)
                tag = f"K={K} N={N} BS={BS} cs={cfg['cluster_size']} T={cfg['num_threads']}"
                print(f"[{'PASS' if ok else 'FAIL'}] S1 {tag} | "
                      f"tb={'OK' if ok_t else 'FAIL:' + why_t} "
                      f"ctrl_eq={'OK' if ok_c else 'FAIL:' + why_c} "
                      f"ctrl_exact={'OK' if ok_ce else 'FAIL:' + why_ce}",
                      flush=True)

    # ---------------- S2: forced tie fixtures (high-hit hints) ----------
    N, BS = 131072, 2
    for K in grid_K:
        cs = Gvr37.pick_config(torch.float32, BS, N)["cluster_size"]

        # (a) BIG tie plateau straddling K (> 128 members): band radix arm.
        g = torch.Generator(device=DEV).manual_seed(3 + K)
        logits = torch.full((BS, N), -1000.0, dtype=torch.float32, device=DEV)
        highs = 100.0 + torch.arange(K - 6, device=DEV, dtype=torch.float32) * 1e-3
        logits[:, : K - 6] = highs.flip(0)
        logits[:, K - 6: K - 6 + K + 500] = 50.0
        logits[:, K - 6 + K + 500: K - 6 + K + 500 + 3000] = (
            49.0 + 0.9 * torch.rand(BS, 3000, dtype=torch.float32,
                                    device=DEV, generator=g))
        pre = make_pre_exact(logits, K)
        out = run_kernel(Gvr37, logits, pre, K, tb=True)
        ok, why = check_exact(logits, out, K)
        sec_tot["S2_tie"] += 1
        sec_pass["S2_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] S2a big-tie K={K} N={N} BS={BS} "
              f"cs={cs} strad~{K + 500} | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

        # (b) SMALL tie plateau (12 members, need0<=6): [p4tt] fast arm at
        #     K>=1024, radix-only text at K512.
        need0, strad, band = 6, 12, 3000
        g = torch.Generator(device=DEV).manual_seed(13 + K)
        logits = torch.full((BS, N), -1000.0, dtype=torch.float32, device=DEV)
        highs = 100.0 + torch.arange(K - need0, device=DEV,
                                     dtype=torch.float32) * 1e-3
        logits[:, : K - need0] = highs.flip(0)
        logits[:, K - need0: K - need0 + strad] = 50.0
        logits[:, K - need0 + strad: K - need0 + strad + band] = (
            49.0 + 0.9 * torch.rand(BS, band, dtype=torch.float32,
                                    device=DEV, generator=g))
        pre = make_pre_exact(logits, K)
        out = run_kernel(Gvr37, logits, pre, K, tb=True)
        ok, why = check_exact(logits, out, K)
        sec_tot["S2_tie"] += 1
        sec_pass["S2_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] S2b small-tie K={K} N={N} BS={BS} "
              f"cs={cs} strad={strad} need0={need0} | "
              f"{'OK' if ok else 'FAIL:' + why}", flush=True)

        # (c) all-distinct near-boundary (shuffled linspace): non-firing
        #     exact-tail gate, random-hit hints.
        g = torch.Generator(device=DEV).manual_seed(11)
        row = torch.linspace(0.0, 1.0, N, dtype=torch.float32, device=DEV)
        logits = torch.stack(
            [row[torch.randperm(N, generator=g, device=DEV)] for _ in range(BS)]
        ).contiguous()
        noisy = logits + 0.8 * logits.std() * torch.randn(
            BS, N, dtype=torch.float32, device=DEV, generator=g)
        pre = torch.topk(noisy, K, dim=1).indices.int().contiguous()
        out = run_kernel(Gvr37, logits, pre, K, tb=True)
        ok, why = check_exact(logits, out, K)
        sec_tot["S2_tie"] += 1
        sec_pass["S2_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] S2c distinct-linspace K={K} "
              f"N={N} BS={BS} cs={cs} | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

        # (d) hi-boundary plateau: 400 duplicates ABOVE the K boundary (ranks
        #     [200, 600) at value 60.0, K >= 512 > 600 only for K1024/K2048;
        #     for K512 the plateau [200, 600) straddles K itself too). Rung
        #     thresholds from exact hints land at/inside the plateau; all
        #     bit-equal duplicates must land on ONE side of the sure/band
        #     split. Distinct values elsewhere.
        g = torch.Generator(device=DEV).manual_seed(29 + K)
        logits = torch.full((BS, N), -1000.0, dtype=torch.float32, device=DEV)
        highs = 100.0 + torch.arange(200, device=DEV, dtype=torch.float32) * 1e-3
        logits[:, :200] = highs.flip(0)
        logits[:, 200:600] = 60.0
        ntail = K + 2000 - 600
        tail = 10.0 + torch.arange(ntail, device=DEV, dtype=torch.float32) * 1e-3
        logits[:, 600: 600 + ntail] = tail.flip(0)
        pre = make_pre_exact(logits, K)
        out = run_kernel(Gvr37, logits, pre, K, tb=True)
        ok, why = check_exact(logits, out, K)
        sec_tot["S2_tie"] += 1
        sec_pass["S2_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] S2d hi-plateau K={K} N={N} BS={BS} "
              f"cs={cs} dup=400@[200,600) | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

    # ---------------- S3: launch-contract smoke over cs overrides ----------
    K, N = 1024, 131072
    for BS in (1, 4):
        for cs_ovr in (1, 2, 4, 8):
            seed += 1
            logits, pre = make_inputs(K, N, BS, seed)
            out = run_kernel(Gvr37, logits, pre, K, tb=True, cs_force=cs_ovr)
            ok, why = check_exact(logits, out, K)
            sec_tot["S3_launch"] += 1
            sec_pass["S3_launch"] += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] S3 K={K} N={N} BS={BS} "
                  f"cs={cs_ovr}(override) | {'OK' if ok else 'FAIL:' + why}",
                  flush=True)

    # ---------------- S4: real decode-capture cells (BS=2) ----------------
    import real_data_v4cap as RV4  # noqa: E402
    import real_data_v32 as RV32  # noqa: E402

    CELLS = [(m, isl) for m in ("flash", "pro") for isl in
             ("128k", "256k", "512k", "1024k")] + \
            [("v32", isl) for isl in ("32k", "64k", "128k", "256k")]
    LAYER = {"flash": 22, "pro": 30, "v32": 34}
    BS = 2
    for m, isl in CELLS:
        RD = RV32 if m == "v32" else RV4
        name = f"{m}/{isl}"
        try:
            b = RD.get_bundle(m, isl, LAYER[m], "fp32")
        except Exception as e:  # noqa: BLE001 — loader availability probe
            sec_tot["S4_real"] += 1
            print(f"[FAIL] S4 {name}: loader error {e}", flush=True)
            continue
        K, cr, Nv = b["K"], b["cr"], b["N"]
        logits = b["logits"].float().to(DEV).repeat(BS, 1).contiguous()
        pre = b["preIdx"].to(DEV).repeat(BS, 1).contiguous()
        print(f"--- S4 cell {name} K={K} cr={cr} N={Nv} "
              f"hit={b['hit_rate']:.3f} layer={b['layer']} ---", flush=True)
        out = run_kernel(Gvr37, logits, pre, K, tb=True, cr=cr, tb_debug=True,
                         n_valid=Nv)
        ok, why = check_exact(logits, out, K, n_valid=Nv)
        sec_tot["S4_real"] += 1
        sec_pass["S4_real"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] S4 {name} K={K} cr={cr} N={Nv} "
              f"BS={BS} hit={b['hit_rate']:.3f} | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

    npass = sum(sec_pass.values())
    ntot = sum(sec_tot.values())
    for s in sec_tot:
        print(f"SECTION {s}: {sec_pass[s]}/{sec_tot[s]} PASS", flush=True)
    print(f"BATTERY_LJ: {npass}/{ntot} PASS", flush=True)
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    sys.exit(main())
