#!/usr/bin/env python3
"""In-kernel per-phase clock64 breakdown for the production GVR top-K kernel
(PR#16457 head, gvrpkgprod2) on real captured decode cells, BS=1 fp32.

Timed arm = gvrpkgtimed (spliced copy, [ptime] markers, phase_ts int64[1,8]).
Untimed arm = gvrpkgprod2 (pristine PR head) — trusted absolute wall time.

Per cell:
  * cfg from GvrTopKKernel.pick_config(fp32, 1, N, max_seq_len=N*cr) (printed)
  * 10 warmup + 30 timed launches, cold-L2 (512MB evict buffer zeroed,
    untimed, before each timed launch)
  * per-phase MEDIAN cycles + fraction of the [t0,t7] window
  * absolute us: us_est = frac * untimed_prod_wall_us (cold-L2 CUDA-event
    median). This anchors the breakdown to the trusted production wall time;
    the implied SM GHz (= window_cycles / timed_wall_us) is reported as a
    consistency check.
  * validation: (a) timed vs untimed index value-set EXACT (+ vs torch.topk),
    (b) timed wall within ~7% of untimed, (c) t0<=t1<=...<=t7 every launch.

Outputs: phase_breakdown.csv + PHASE_BREAKDOWN.md next to this script.
"""
import csv
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
WORKDIR = HERE.parent
OPBENCH = WORKDIR.parent.parent  # indexer_topk_op_bench
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(OPBENCH / "harness"))

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute import runtime as crt  # noqa: E402

assert cutlass.__version__.startswith("4.5.0"), f"need cutlass 4.5.0, got {cutlass.__version__}"

from gvrpkgprod2.top_k.gvr_topk_decode import GvrTopKKernel as ProdK  # noqa: E402
from gvrpkgtimed.top_k.gvr_topk_decode import GvrTopKKernel as TimedK  # noqa: E402

import real_data_v4cap as RV4  # noqa: E402
import real_data_v32 as RV32  # noqa: E402

RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)

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
PHASE_LABEL = {
    "p1_gather_stats": "P1 gather/stats",
    "smem_stage": "smem-stage",
    "p1b_rungs": "P1b rungs",
    "p2_count_admission": "P2 count+admission(+refine)",
    "p3_collect": "P3 collect",
    "p4_select": "P4 select(+tail)",
    "epilogue": "epilogue",
}

# (model, isl, layer). All BS=1 fp32.
CELLS = [
    ("flash", "32k", 22),
    ("flash", "128k", 22),
    ("pro", "128k", 30),
    ("pro", "512k", 30),
    ("v32", "128k", 34),
    ("flash", "1024k", 22),
]


def make_kernel(cls, K, cr, cfg, timed):
    kobj = cls(
        dtype=cutlass.Float32,
        top_k=K,
        next_n=1,
        compress_ratio=cr,
        return_output_values=False,
        **cfg,
    )
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


def cold_launches(callf, collect_ts=None):
    """WARMUP untimed launches, then REPS cold-L2 event-timed launches.

    Returns (list of wall_us, list of ts-rows if collect_ts) — ts rows read
    back after each timed launch."""
    for _ in range(WARMUP):
        callf()
    torch.cuda.synchronize()
    walls, tss = [], []
    for _ in range(REPS):
        _EVICT.zero_()  # cold-L2 (untimed: before e0.record on same stream)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        callf()
        e1.record()
        torch.cuda.synchronize()
        walls.append(e0.elapsed_time(e1) * 1e3)  # us
        if collect_ts is not None:
            tss.append(collect_ts[0].cpu().tolist())
    return walls, tss


def value_set_exact(idx, logits_row, N, K, ref):
    idx = idx.to(torch.int64)
    if int((idx < 0).sum()) > 0:
        return False
    if torch.unique(idx).numel() != K:
        return False
    lg = logits_row[:N].float()
    sel = lg[idx].sort().values
    rv = lg[ref.to(torch.int64)].sort().values
    return bool(torch.equal(sel, rv))


def run_cell(model, isl, layer):
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, layer, "fp32")
    logits = b["logits"].to(torch.float32).contiguous()
    pre = b["preIdx"].contiguous()
    N, K, cr, ref = b["N"], b["K"], b["cr"], b["ref"]
    seq_lens = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)

    cfg = ProdK.pick_config(torch.float32, 1, N, max_seq_len=N * cr)
    if cfg.get("use_256bit_load") and logits.data_ptr() % 32 != 0:
        cfg["use_256bit_load"] = False
    print(f"[{model}/{isl} L{layer}] K={K} N={N} cr={cr} hit={b['hit_rate']:.3f} cfg={cfg}", flush=True)

    out_t = torch.empty(1, K, dtype=torch.int32, device=DEV)
    out_p = torch.empty(1, K, dtype=torch.int32, device=DEV)
    ts = torch.zeros(1, 8, dtype=torch.int64, device=DEV)

    prod = make_kernel(ProdK, K, cr, cfg, timed=False)
    timed = make_kernel(TimedK, K, cr, cfg, timed=True)

    call_p = lambda: prod(logits, pre, seq_lens, None, out_p, None)  # noqa: E731
    call_t = lambda: timed(logits, pre, seq_lens, None, out_t, None, ts)  # noqa: E731

    # correctness first
    call_p()
    call_t()
    torch.cuda.synchronize()
    ex_p = value_set_exact(out_p[0], logits[0], N, K, ref)
    ex_t = value_set_exact(out_t[0], logits[0], N, K, ref)

    walls_p, _ = cold_launches(call_p)
    walls_t, tss = cold_launches(call_t, collect_ts=ts)

    us_p = statistics.median(walls_p)
    us_t = statistics.median(walls_t)

    mono = all(all(r[i] <= r[i + 1] for i in range(7)) for r in tss)
    cyc = {name: statistics.median([r[b_] - r[a_] for r in tss]) for name, a_, b_ in PHASES}
    window = statistics.median([r[7] - r[0] for r in tss])
    tot = sum(cyc.values())
    frac = {k: (v / tot if tot else 0.0) for k, v in cyc.items()}
    us_est = {k: frac[k] * us_p for k in cyc}
    ghz = window / us_t / 1e3  # implied SM clock consistency check

    return dict(
        cell=f"{model}/{isl}/L{layer}", model=model, isl=isl, layer=layer,
        K=K, N=N, cr=cr, hit=b["hit_rate"], cfg=cfg,
        exact_prod=ex_p, exact_timed=ex_t, mono=mono,
        us_prod=us_p, us_timed=us_t, overhead=us_t / us_p - 1.0,
        window_cyc=window, ghz=ghz, cyc=cyc, frac=frac, us_est=us_est,
    )


def main():
    torch.manual_seed(0)
    print(f"device={torch.cuda.get_device_name(0)} cutlass={cutlass.__version__}", flush=True)
    results = []
    for model, isl, layer in CELLS:
        results.append(run_cell(model, isl, layer))
        r = results[-1]
        print(f"  wall prod={r['us_prod']:.2f}us timed={r['us_timed']:.2f}us "
              f"(+{100 * r['overhead']:.1f}%) exact p/t={r['exact_prod']}/{r['exact_timed']} "
              f"mono={r['mono']} ghz~{r['ghz']:.2f}", flush=True)
        for name, _, _ in PHASES:
            print(f"    {PHASE_LABEL[name]:<30s} {r['cyc'][name]:>9.0f} cyc "
                  f"{100 * r['frac'][name]:5.1f}%  {r['us_est'][name]:7.2f} us", flush=True)

    # ---- CSV ----
    with open(HERE / "phase_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "phase", "cycles_med", "frac", "us_est"])
        for r in results:
            for name, _, _ in PHASES:
                w.writerow([r["cell"], name, f"{r['cyc'][name]:.0f}",
                            f"{r['frac'][name]:.4f}", f"{r['us_est'][name]:.3f}"])
            w.writerow([r["cell"], "total", f"{r['window_cyc']:.0f}", "1.0000", f"{r['us_prod']:.3f}"])

    # ---- Markdown ----
    lines = [
        "# GVR prod kernel (PR#16457 head) — in-kernel phase breakdown",
        "",
        "clock64 stamps by leader CTA tid0 in `_run_phases` (gvrpkgtimed spliced copy,",
        "`# [ptime]` markers; gvrpkgprod2 untouched). BS=1 fp32, real captured cells,",
        f"cold-L2 (512MB evict), {WARMUP} warmup + {REPS} timed launches, per-phase MEDIAN cycles.",
        "Absolute us = phase fraction x UNTIMED production cold-L2 wall-us (CUDA events);",
        "implied SM GHz (window_cyc / timed wall) shown as consistency check.",
        "",
        "| cell | K | N | cs | T | hit | " + " | ".join(PHASE_LABEL[n] for n, _, _ in PHASES) +
        " | total us (prod) | timed vs prod | exact | mono |",
        "|" + "---|" * (6 + len(PHASES) + 4),
    ]
    for r in results:
        cells = [r["cell"], str(r["K"]), str(r["N"]), str(r["cfg"]["cluster_size"]),
                 str(r["cfg"]["num_threads"]), f"{r['hit']:.2f}"]
        for name, _, _ in PHASES:
            cells.append(f"{r['us_est'][name]:.2f}us ({100 * r['frac'][name]:.0f}%)")
        cells += [f"{r['us_prod']:.2f}", f"{100 * r['overhead']:+.1f}%",
                  "Y" if (r["exact_prod"] and r["exact_timed"]) else "N",
                  "Y" if r["mono"] else "N"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Per-cell config + notes", ""]
    for r in results:
        lines.append(f"### {r['cell']} (K={r['K']}, N={r['N']}, cr={r['cr']}, hit={r['hit']:.3f})")
        lines.append(f"- cfg: `{r['cfg']}`; implied SM clock ~{r['ghz']:.2f} GHz; "
                     f"window {r['window_cyc']:.0f} cyc; prod {r['us_prod']:.2f}us / timed {r['us_timed']:.2f}us "
                     f"({100 * r['overhead']:+.1f}%).")
        lines.append("")
    (HERE / "PHASE_BREAKDOWN.md").write_text("\n".join(lines) + "\n")
    print("wrote", HERE / "phase_breakdown.csv", "and", HERE / "PHASE_BREAKDOWN.md", flush=True)

    bad = [r["cell"] for r in results if not (r["exact_prod"] and r["exact_timed"] and r["mono"])]
    slow = [r["cell"] for r in results if abs(r["overhead"]) > 0.07]
    print(f"VALIDATION: exact+mono fail={bad or 'none'}; >7% wall delta={slow or 'none'}", flush=True)


if __name__ == "__main__":
    main()
