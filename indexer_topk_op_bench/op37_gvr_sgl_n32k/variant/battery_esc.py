# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""[op37-esc] tb_escape (escape-only bracket) exactness battery.

Clone of battery_lj.py adapted to the tb_escape ctor flag. The escape path
only runs on a base-admission MISS, so besides the default-config grid
(escape mostly dormant — proves the traced-but-idle build is unchanged),
several sections FORCE the base admission to miss via ctor overrides
(r0_vseed=False, r0_qfracs=(0.001,): need=1 -> rung threshold ~ max hinted
value -> count ~ 1 << K, guaranteed miss) and pick escape ladders that
deterministically fire / cannot fire (see the OVR constants):

  E1 grid      fp32 x K{512,1024,2048} x N{65536,131072,262144} x BS{2,8,64},
               tb_escape=True default config. Per cell: escape arm exact +
               tb_escape=False control == pristine gvrpkgprod2 index set +
               control exact (same triple gate as battery_lj S1).
  E2 esc-fire  same grid at BS=2, forced base miss (overrides above) with
               GOOD hints -> the thin escape ladder brackets and FIRES the
               band P3/P4 through the ESCAPE columns (tb_debug prints
               [esc] fired=... at BS<=2 for eyeball confirmation); exact.
  E3 esc-miss  forced base miss AND a useless escape ladder
               (ESC_MISS_OVR: all rung counts << K) -> escape cannot
               fire; the M_esc-seeded log-falsi chain finishes the row;
               exact.
  E4 tie       battery_lj S2 fixtures (big tie / small tie / distinct
               linspace / hi-plateau) under the E2 forced-miss config so
               the tie plateaus are crossed by the ESCAPE band split.
  E5 launch    cs override smoke {1,2,4,8} x BS{1,4}, E2 forced-miss config
               (escape cluster count-merge + short-row degrade contract).
  E6 realdata  all 12 (model,isl) real cells at BS=2, tb_escape=True default
               config, tb_debug (cold cells exercise escape organically).
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
CR = 1

# Forced base-admission miss: qneeds = ceil(q*K), so a TINY qfrac makes
# need=1 -> the rung threshold ~ the MAX hinted value -> count(>=thr) ~ 1
# << K -> guaranteed miss. (v1 of this battery used (0.999,), which with
# exact hints lands count ~ K INSIDE the accept window — the base never
# missed and E2-E5 exercised only the accept path; 0 fired=1 in the log.)
MISS_OVR = dict(r0_vseed=False, r0_qfracs=(0.001,))
# [esc-lite] Escape ladder whose deep rung lands ~K with exact hints so
# the seeded falsi starts from a tight bracket (the lite escape has no
# fire/band concept — it always finishes through the falsi).
ESC_FIRE_OVR = dict(tb_esc_qfracs=(0.9999, 0.85, 0.35))
# Escape ladder that CANNOT fire (all needs ~ 1-2 -> counts << K -> no lo
# rung) -> the M_esc-seeded falsi must finish the row.
ESC_MISS_OVR = dict(tb_esc_qfracs=(0.002, 0.0015, 0.001))
# Single-rung escape ladder (M_esc=1): degenerate ladder shape gate.
ESC_NOHI_OVR = dict(tb_esc_qfracs=(0.9999,))


def run_kernel(cls, logits, pre, K, esc, cr=CR, cs_force=None, tb_debug=False,
               n_valid=None, ovr=None):
    BS, N = logits.shape
    if n_valid is not None:
        N = n_valid
    sl = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)
    out = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    kw = dict(ovr or {})
    if esc:
        kw["tb_escape"] = True
        if tb_debug:
            kw["tb_debug"] = True
    if cs_force is not None:
        kw["cluster_size"] = cs_force
    cls.launch(logits, pre, sl, out, K, compress_ratio=cr, **kw)
    torch.cuda.synchronize()
    return out


def check_exact(logits, out, K, n_valid=None):
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
    idx = torch.topk(logits, K, dim=1).indices
    return (idx - 1).clamp(min=0).int().contiguous()


def main():
    torch.cuda.init()
    secs = ("E1_grid", "E2_fire", "E3_miss", "E4_tie", "E5_launch", "E6_real")
    sec_pass = {s: 0 for s in secs}
    sec_tot = {s: 0 for s in secs}

    grid_K = (512, 1024, 2048)
    grid_N = (65536, 131072, 262144)
    grid_BS = (2, 8, 64)
    seed = 20260721

    # ---------------- E1: full grid, default config ----------------
    for K in grid_K:
        for N in grid_N:
            for BS in grid_BS:
                seed += 1
                logits, pre = make_inputs(K, N, BS, seed)
                cfg = Gvr37.pick_config(torch.float32, BS, N)
                out_e = run_kernel(Gvr37, logits, pre, K, esc=True)
                ok_e, why_e = check_exact(logits, out_e, K)
                out_ctrl = run_kernel(Gvr37, logits, pre, K, esc=False)
                out_ref = run_kernel(GvrRef, logits, pre, K, esc=False)
                ok_c = torch.equal(out_ctrl.sort(dim=1).values,
                                   out_ref.sort(dim=1).values)
                why_c = "" if ok_c else "ctrl index-set != pristine"
                ok_ce, why_ce = check_exact(logits, out_ctrl, K)
                ok = ok_e and ok_c and ok_ce
                sec_tot["E1_grid"] += 1
                sec_pass["E1_grid"] += int(ok)
                tag = f"K={K} N={N} BS={BS} cs={cfg['cluster_size']} T={cfg['num_threads']}"
                print(f"[{'PASS' if ok else 'FAIL'}] E1 {tag} | "
                      f"esc={'OK' if ok_e else 'FAIL:' + why_e} "
                      f"ctrl_eq={'OK' if ok_c else 'FAIL:' + why_c} "
                      f"ctrl_exact={'OK' if ok_ce else 'FAIL:' + why_ce}",
                      flush=True)

    # ---------------- E2: forced base miss -> escape fires ----------------
    for K in grid_K:
        for N in grid_N:
            seed += 1
            BS = 2
            logits, _ = make_inputs(K, N, BS, seed)
            pre = make_pre_exact(logits, K)  # good hints: escape hist is honest
            out = run_kernel(Gvr37, logits, pre, K, esc=True, tb_debug=True,
                             ovr={**MISS_OVR, **ESC_FIRE_OVR})
            ok, why = check_exact(logits, out, K)
            sec_tot["E2_fire"] += 1
            sec_pass["E2_fire"] += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] E2 K={K} N={N} BS={BS} "
                  f"forced-miss+esc | {'OK' if ok else 'FAIL:' + why}",
                  flush=True)
        # fire WITHOUT a hi rung (FLT_MAX sentinel, empty sure set)
        seed += 1
        N, BS = 131072, 2
        logits, _ = make_inputs(K, N, BS, seed)
        pre = make_pre_exact(logits, K)
        out = run_kernel(Gvr37, logits, pre, K, esc=True, tb_debug=True,
                         ovr={**MISS_OVR, **ESC_NOHI_OVR})
        ok, why = check_exact(logits, out, K)
        sec_tot["E2_fire"] += 1
        sec_pass["E2_fire"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] E2 K={K} N={N} BS={BS} "
              f"esc-single-rung | {'OK' if ok else 'FAIL:' + why}", flush=True)

    # ---------------- E3: forced base miss + useless escape -> falsi -------
    for K in grid_K:
        seed += 1
        N, BS = 131072, 2
        logits, _ = make_inputs(K, N, BS, seed)
        pre = make_pre_exact(logits, K)
        out = run_kernel(Gvr37, logits, pre, K, esc=True, tb_debug=True,
                         ovr={**MISS_OVR, **ESC_MISS_OVR})
        ok, why = check_exact(logits, out, K)
        sec_tot["E3_miss"] += 1
        sec_pass["E3_miss"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] E3 K={K} N={N} BS={BS} "
              f"esc-miss->falsi | {'OK' if ok else 'FAIL:' + why}", flush=True)

    # ---------------- E4: tie fixtures under escape fire ----------------
    N, BS = 131072, 2
    for K in grid_K:
        cs = Gvr37.pick_config(torch.float32, BS, N)["cluster_size"]

        g = torch.Generator(device=DEV).manual_seed(3 + K)
        logits = torch.full((BS, N), -1000.0, dtype=torch.float32, device=DEV)
        highs = 100.0 + torch.arange(K - 6, device=DEV, dtype=torch.float32) * 1e-3
        logits[:, : K - 6] = highs.flip(0)
        logits[:, K - 6: K - 6 + K + 500] = 50.0
        logits[:, K - 6 + K + 500: K - 6 + K + 500 + 3000] = (
            49.0 + 0.9 * torch.rand(BS, 3000, dtype=torch.float32,
                                    device=DEV, generator=g))
        pre = make_pre_exact(logits, K)
        out = run_kernel(Gvr37, logits, pre, K, esc=True, ovr={**MISS_OVR, **ESC_FIRE_OVR})
        ok, why = check_exact(logits, out, K)
        sec_tot["E4_tie"] += 1
        sec_pass["E4_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] E4a big-tie K={K} strad~{K + 500} "
              f"cs={cs} | {'OK' if ok else 'FAIL:' + why}", flush=True)

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
        out = run_kernel(Gvr37, logits, pre, K, esc=True, ovr={**MISS_OVR, **ESC_FIRE_OVR})
        ok, why = check_exact(logits, out, K)
        sec_tot["E4_tie"] += 1
        sec_pass["E4_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] E4b small-tie K={K} strad={strad} "
              f"need0={need0} cs={cs} | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

        g = torch.Generator(device=DEV).manual_seed(11)
        row = torch.linspace(0.0, 1.0, N, dtype=torch.float32, device=DEV)
        logits = torch.stack(
            [row[torch.randperm(N, generator=g, device=DEV)] for _ in range(BS)]
        ).contiguous()
        noisy = logits + 0.8 * logits.std() * torch.randn(
            BS, N, dtype=torch.float32, device=DEV, generator=g)
        pre = torch.topk(noisy, K, dim=1).indices.int().contiguous()
        out = run_kernel(Gvr37, logits, pre, K, esc=True, ovr={**MISS_OVR, **ESC_FIRE_OVR})
        ok, why = check_exact(logits, out, K)
        sec_tot["E4_tie"] += 1
        sec_pass["E4_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] E4c distinct-linspace K={K} "
              f"cs={cs} | {'OK' if ok else 'FAIL:' + why}", flush=True)

        g = torch.Generator(device=DEV).manual_seed(29 + K)
        logits = torch.full((BS, N), -1000.0, dtype=torch.float32, device=DEV)
        highs = 100.0 + torch.arange(200, device=DEV, dtype=torch.float32) * 1e-3
        logits[:, :200] = highs.flip(0)
        logits[:, 200:600] = 60.0
        ntail = K + 2000 - 600
        tail = 10.0 + torch.arange(ntail, device=DEV, dtype=torch.float32) * 1e-3
        logits[:, 600: 600 + ntail] = tail.flip(0)
        pre = make_pre_exact(logits, K)
        out = run_kernel(Gvr37, logits, pre, K, esc=True, ovr={**MISS_OVR, **ESC_FIRE_OVR})
        ok, why = check_exact(logits, out, K)
        sec_tot["E4_tie"] += 1
        sec_pass["E4_tie"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] E4d hi-plateau K={K} "
              f"dup=400@[200,600) cs={cs} | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

    # ---------------- E5: cs override smoke, forced miss ----------------
    K, N = 1024, 131072
    for BS in (1, 4):
        for cs_ovr in (1, 2, 4, 8):
            seed += 1
            logits, _ = make_inputs(K, N, BS, seed)
            pre = make_pre_exact(logits, K)
            out = run_kernel(Gvr37, logits, pre, K, esc=True, cs_force=cs_ovr,
                             ovr={**MISS_OVR, **ESC_FIRE_OVR})
            ok, why = check_exact(logits, out, K)
            sec_tot["E5_launch"] += 1
            sec_pass["E5_launch"] += int(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] E5 K={K} N={N} BS={BS} "
                  f"cs={cs_ovr}(override) forced-miss | "
                  f"{'OK' if ok else 'FAIL:' + why}", flush=True)

    # ---------------- E6: real decode-capture cells (BS=2) ----------------
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
            sec_tot["E6_real"] += 1
            print(f"[FAIL] E6 {name}: loader error {e}", flush=True)
            continue
        K, cr, Nv = b["K"], b["cr"], b["N"]
        logits = b["logits"].float().to(DEV).repeat(BS, 1).contiguous()
        pre = b["preIdx"].to(DEV).repeat(BS, 1).contiguous()
        print(f"--- E6 cell {name} K={K} cr={cr} N={Nv} "
              f"hit={b['hit_rate']:.3f} layer={b['layer']} ---", flush=True)
        out = run_kernel(Gvr37, logits, pre, K, esc=True, cr=cr, tb_debug=True,
                         n_valid=Nv)
        ok, why = check_exact(logits, out, K, n_valid=Nv)
        sec_tot["E6_real"] += 1
        sec_pass["E6_real"] += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] E6 {name} K={K} cr={cr} N={Nv} "
              f"BS={BS} hit={b['hit_rate']:.3f} | {'OK' if ok else 'FAIL:' + why}",
              flush=True)

    npass = sum(sec_pass.values())
    ntot = sum(sec_tot.values())
    for s in secs:
        print(f"SECTION {s}: {sec_pass[s]}/{sec_tot[s]} PASS", flush=True)
    print(f"BATTERY_ESC: {npass}/{ntot} PASS", flush=True)
    return 0 if npass == ntot else 1


if __name__ == "__main__":
    sys.exit(main())
