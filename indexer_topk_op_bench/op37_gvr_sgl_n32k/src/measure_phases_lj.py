#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op37 L-J tax diagnosis — clock64 per-phase A/B: tight_bracket OFF vs ON.

Clone of measure_phases_bs.py, but both arms come from the op37 variant
package (gvrpkg37 untimed anchors, gvrpkgtimed37 clock64-stamped clones)
and each cell runs tb=False and tb=True back-to-back on the same GPU.
Qualitative-fraction use ONLY (clock64 instrumentation slows the kernel
path-dependently; absolute us anchored to the matching untimed arm).

Cells: the L-J nsys-verdict loss/win representatives (see task brief).
Output: results/phase_lj.csv + stdout table. GPU via CUDA_VISIBLE_DEVICES.
"""
import csv
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OP37 = HERE.parent
OPBENCH = OP37.parent
P4F1 = OPBENCH / "op26_r0_upstream_port_report" / "p4f1_harness"
sys.path.insert(0, str(P4F1))
sys.path.insert(0, str(OPBENCH / "harness"))
sys.path.insert(0, str(OP37 / "variant"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0"), f"need cutlass 4.5.0, got {cutlass.__version__}"

from gvrpkg37.top_k.gvr_topk_decode import GvrTopKKernel as ProdK  # noqa: E402
from gvrpkgtimed37.top_k.gvr_topk_decode import GvrTopKKernel as TimedK  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
sys.path.insert(0, str(OPBENCH / "op26_r0_upstream_port_report" / "harness"))
import real_data_v32 as RV32  # noqa: E402

RV32.BENCH_LAYERS = list(getattr(RV32, "LAYERS_ALL", [14, 34, 54]))

DEV = "cuda"
WARMUP = 10
REPS = 30
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=DEV)

PHASES = [
    ("p1_gather_stats", 0, 1),
    ("smem_stage", 1, 2),
    ("p1b_rungs", 2, 3),
    ("p2_count_admission", 3, 4),
    ("p3_collect", 4, 5),
    ("p4_select", 5, 6),
    ("epilogue", 6, 7),
]

# (model, isl, layer, BS list) — L-J verdict representatives
import os as _os
_ONLY = _os.environ.get("CELLS_ONLY", "")
CELLS = [
    ("pro", "128k", 30, [8]),      # warm loss, N=32771, cs1
    ("pro", "512k", 30, [8]),      # warm loss, N=131075, cs4
    ("flash", "512k", 22, [512]),  # the L-J WIN cell (hit=.057)
    ("flash", "128k", 22, [2]),    # small-N loss, cs1
]
if _ONLY:  # e.g. "flash:512k:22:256,1024"
    m, isl, ly, bss = _ONLY.split(":")
    CELLS = [(m, isl, int(ly), [int(b) for b in bss.split(",")])]


ARMS = {
    # name -> extra ctor kwargs on top of pick_config
    "tb_off": {},
    "tb_on": {"tight_bracket": True},
    # ablation A: tight bracket with the BASE-width ladder + one deep rung
    # (attributes the P2 tax to ladder width M)
    "tb_thin": {"tight_bracket": True, "tb_qfracs": (0.85, 0.35, 0.05)},
    # ablation B: base kernel with the P3 4-way unroll disabled
    # (attributes the band-P3 delta to the missing unroll fast path)
    "off_nou3": {"enable_phase3_unroll": False},
}
ARM_LIST = [a for a in _os.environ.get("ARMS", "tb_off,tb_on").split(",") if a]


def make_kernel(cls, K, cr, cfg, timed, tb):
    kw = dict(cfg)
    kw.update(ARMS[tb] if isinstance(tb, str) else ({"tight_bracket": True} if tb else {}))
    kobj = cls(dtype=cutlass.Float32, top_k=K, next_n=1, compress_ratio=cr,
               return_output_values=False, **kw)
    nr, nc, nb = cute.sym_int(), cute.sym_int(), cute.sym_int()
    ia = 32 if cfg["use_256bit_load"] else 16
    in_f = crt.make_fake_compact_tensor(cutlass.Float32, (nr, nc), stride_order=(1, 0), assumed_align=ia)
    pi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb, K), stride_order=(1, 0), assumed_align=16)
    sl_f = crt.make_fake_compact_tensor(cutlass.Int32, (nb,), stride_order=(0,))
    oi_f = crt.make_fake_compact_tensor(cutlass.Int32, (nr, K), stride_order=(1, 0), assumed_align=16)
    fs = crt.make_fake_stream(use_tvm_ffi_env_stream=True)
    if timed:
        ts_f = crt.make_fake_compact_tensor(cutlass.Int64, (nr, 8), stride_order=(1, 0), assumed_align=16)
        return cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, ts_f, stream=fs, options="--enable-tvm-ffi")
    return cute.compile(kobj, in_f, pi_f, sl_f, None, oi_f, None, stream=fs, options="--enable-tvm-ffi")


def cold_launches(callf, ts=None):
    for _ in range(WARMUP):
        callf()
    torch.cuda.synchronize()
    walls, tss = [], []
    for _ in range(REPS):
        _EVICT.zero_()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        callf()
        e1.record()
        torch.cuda.synchronize()
        walls.append(e0.elapsed_time(e1) * 1e3)
        if ts is not None:
            tss.append(ts.cpu().tolist())
    return walls, tss


def value_set_exact(idx, logits_row, N, K, ref):
    idx = idx.to(torch.int64)
    if int((idx < 0).sum()) > 0 or torch.unique(idx).numel() != K:
        return False
    lg = logits_row[:N].float()
    return bool(torch.equal(lg[idx].sort().values,
                            lg[ref.to(torch.int64)].sort().values))


def run_arm(cls_prod, cls_timed, tb, K, cr, cfg, logits, pre, seq_lens, N, ref):
    BS = logits.shape[0]
    out_t = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    out_p = torch.empty(BS, K, dtype=torch.int32, device=DEV)
    ts = torch.zeros(BS, 8, dtype=torch.int64, device=DEV)
    prod = make_kernel(cls_prod, K, cr, cfg, timed=False, tb=tb)
    timed = make_kernel(cls_timed, K, cr, cfg, timed=True, tb=tb)
    call_p = lambda: prod(logits, pre, seq_lens, None, out_p, None)  # noqa: E731
    call_t = lambda: timed(logits, pre, seq_lens, None, out_t, None, ts)  # noqa: E731
    call_p(); call_t(); torch.cuda.synchronize()
    ex_p = value_set_exact(out_p[0], logits[0], N, K, ref)
    ex_t = value_set_exact(out_t[BS - 1], logits[0], N, K, ref)
    walls_p, _ = cold_launches(call_p)
    walls_t, tss = cold_launches(call_t, ts=ts)
    us_p = statistics.median(walls_p)
    us_t = statistics.median(walls_t)
    cyc = {n: statistics.median([r[0][b_] - r[0][a_] for r in tss]) for n, a_, b_ in PHASES}
    window0 = statistics.median([r[0][7] - r[0][0] for r in tss])
    window_max = statistics.median([max(row[7] - row[0] for row in r) for r in tss])
    tot = sum(cyc.values()) or 1.0
    frac = {k: v / tot for k, v in cyc.items()}
    return dict(exact=ex_p and ex_t, us_prod=us_p, us_timed=us_t,
                overhead=us_t / us_p - 1.0,
                straggle=window_max / window0 if window0 else float("nan"),
                cyc=cyc, frac=frac)


def run_cell(model, isl, layer, BS):
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, layer, "fp32")
    lg_row = b["logits"].to(torch.float32).contiguous()
    pre_row = b["preIdx"].contiguous()
    N, K, cr, ref = b["N"], b["K"], b["cr"], b["ref"]
    logits = lg_row.expand(BS, -1).contiguous()
    pre = pre_row.expand(BS, -1).contiguous()
    seq_lens = torch.full((BS,), N * cr, dtype=torch.int32, device=DEV)

    cfg = ProdK.pick_config(torch.float32, BS, N, max_seq_len=N * cr)
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    print(f"[{model}/{isl} L{layer} BS={BS}] K={K} N={N} cs={cfg['cluster_size']} "
          f"T{cfg['num_threads']} hit={b['hit_rate']:.3f}", flush=True)

    res = {}
    for arm in ARM_LIST:
        res[arm] = run_arm(ProdK, TimedK, arm, K, cr, cfg, logits, pre, seq_lens, N, ref)
    return dict(cell=f"{model}/{isl}/L{layer}", BS=BS, N=N, K=K,
                cs=cfg["cluster_size"], hit=b["hit_rate"], res=res)


def main():
    torch.manual_seed(0)
    rows = []
    for model, isl, layer, bss in CELLS:
        for BS in bss:
            r = run_cell(model, isl, layer, BS)
            rows.append(r)
            base = r["res"][ARM_LIST[0]]
            for arm in ARM_LIST:
                a = r["res"][arm]
                print(f"  {arm:<9s} wall={a['us_prod']:.2f}us timed(+{100 * a['overhead']:.0f}%) "
                      f"exact={a['exact']} straggle={a['straggle']:.3f} "
                      f"base/arm={base['us_prod'] / a['us_prod']:.3f}", flush=True)
            hdr = "    " + f"{'phase':<22s}" + "".join(f" {arm + '_us':>10s}" for arm in ARM_LIST)
            print(hdr + "  (cyc ratio vs " + ARM_LIST[0] + ")", flush=True)
            for n, _, _ in PHASES:
                line = f"    {n:<22s}"
                for arm in ARM_LIST:
                    a = r["res"][arm]
                    line += f" {a['frac'][n] * a['us_prod']:9.2f}u"
                cb = base["cyc"][n]
                line += "  " + "/".join(
                    f"{(r['res'][arm]['cyc'][n] / cb if cb else float('inf')):.2f}x"
                    for arm in ARM_LIST[1:])
                print(line, flush=True)
    out_csv = OP37 / "results" / _os.environ.get("OUT", "phase_lj.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["cell", "BS", "N", "cs", "hit", "arm", "exact", "us_prod", "straggle"]
        hdr += [n for n, _, _ in PHASES] + [n + "_cyc" for n, _, _ in PHASES]
        w.writerow(hdr)
        for r in rows:
            for arm in ARM_LIST:
                a = r["res"][arm]
                w.writerow([r["cell"], r["BS"], r["N"], r["cs"], f"{r['hit']:.3f}",
                            arm, a["exact"],
                            f"{a['us_prod']:.2f}", f"{a['straggle']:.3f}"] +
                           [f"{a['frac'][n]:.4f}" for n, _, _ in PHASES] +
                           [f"{a['cyc'][n]:.0f}" for n, _, _ in PHASES])
    print("CSV ->", out_csv, flush=True)


if __name__ == "__main__":
    main()
