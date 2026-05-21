"""Direct SQLite join of NVTX_EVENTS × CUPTI_ACTIVITY_KIND_KERNEL.
Bypasses `nsys stats -r nvtx_gpu_proj_trace` which silently drops some
NVTX-range events for unknown reasons (Pro family was missing in our case).

Usage:
  parse_sqlite.py <sweep_A_sqlite> <sweep_B_sqlite> <out_dir>
"""
from __future__ import annotations
import bisect, json, os, re, sqlite3, statistics, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(
    r"^(?P<variant>GVR_(?P<dtype>fp32|bf16|fp16)|RADIX_fp32)"
    r"\|(?P<family>flash|pro)\|(?P<cfg>beta_\w+?)"
    r"\|N(?P<N>\d+)\|BS(?P<BS>\d+)\|rep(?P<rep>\d+)$"
)


def kernel_time_per_nvtx(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(s.value, n.text) AS label, n.start, n.end
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE COALESCE(s.value, n.text) LIKE 'GVR_%|%|beta_%' OR COALESCE(s.value, n.text) LIKE 'RADIX_%|%|beta_%'
    """)
    nvtx_rows = cur.fetchall()
    # Get kernels
    cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    kernels = cur.fetchall()
    starts = [k[0] for k in kernels]
    conn.close()

    out = []
    for label, nstart, nend in nvtx_rows:
        m = PAT.match(label)
        if not m:
            continue
        lo = bisect.bisect_left(starts, nstart)
        total_ns = 0; n_kern = 0
        i = lo
        while i < len(kernels) and kernels[i][0] < nend:
            ks, ke = kernels[i]
            if ke <= nend:
                total_ns += (ke - ks); n_kern += 1
            i += 1
        out.append(dict(
            family=m.group("family"), cfg=m.group("cfg"),
            N=int(m.group("N")), BS=int(m.group("BS")),
            dtype=m.group("dtype") or "fp32",
            mode="heuristic" if m.group("variant").startswith("GVR_") else "radix",
            total_ns=total_ns, n_kern=n_kern,
        ))
    return out


def aggregate(rows, sweep):
    # Bucket by (family, cfg, N, BS, dtype, mode) → list of total_ns per range
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["family"], r["cfg"], r["N"], r["BS"], r["dtype"], r["mode"])].append(r["total_ns"])
    # Median per cell (each NVTX range = 1 measure iteration; total_ns = sum of kernel time per iter)
    med_us = {k: statistics.median(v) / 1000.0 for k, v in bucket.items()}
    # R/H per (family, cfg, N, BS, dtype): R(fp32 radix) / H(dt heuristic)
    out = defaultdict(lambda: defaultdict(dict))
    fams = sorted({k[0] for k in med_us})
    if sweep == "A":
        Ns = sorted({k[2] for k in med_us if k[3] == 1})
        for fam in fams:
            for dt in ("bf16", "fp16", "fp32"):
                for N in Ns:
                    sps = []
                    for cfg in ("beta_shallow", "beta_moderate", "beta_deep"):
                        h = med_us.get((fam, cfg, N, 1, dt, "heuristic"))
                        r = med_us.get((fam, cfg, N, 1, "fp32", "radix"))
                        if h and r:
                            sps.append(r / h)
                    if sps:
                        out[fam][dt][N] = (statistics.mean(sps), min(sps), max(sps))
    else:
        BSs = sorted({k[3] for k in med_us if k[2] == 65536})
        for fam in fams:
            for dt in ("bf16", "fp16", "fp32"):
                for BS in BSs:
                    sps = []
                    for cfg in ("beta_shallow", "beta_moderate", "beta_deep"):
                        h = med_us.get((fam, cfg, 65536, BS, dt, "heuristic"))
                        r = med_us.get((fam, cfg, 65536, BS, "fp32", "radix"))
                        if h and r:
                            sps.append(r / h)
                    if sps:
                        out[fam][dt][BS] = (statistics.mean(sps), min(sps), max(sps))
    return dict(out)


def plot_sweep(agg, sweep, png_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    if sweep == "A":
        xlabel = "N (numColumns, post-compress)"
        title_suffix = "BS=1, seq-len scaling, N 2K→256K"
    else:
        xlabel = "BS (row replicated)"
        title_suffix = "N=65536, BS scaling, BS 1→1024"
    colors = {"bf16": "tab:blue", "fp16": "tab:orange", "fp32": "tab:green"}
    markers = {"bf16": "o", "fp16": "s", "fp32": "^"}
    fam_titles = {"flash": "V4 Flash K=512", "pro": "V4 Pro K=1024"}
    for ax, fam in zip(axes, ("flash", "pro")):
        d = agg.get(fam, {})
        for dt in ("bf16", "fp16", "fp32"):
            xs, ys, lo, hi = [], [], [], []
            for x in sorted(d.get(dt, {})):
                m, mn, mx = d[dt][x]
                xs.append(x); ys.append(m); lo.append(mn); hi.append(mx)
            if xs:
                ax.plot(xs, ys, color=colors[dt], marker=markers[dt],
                        label=f"GVR {dt}", linewidth=1.6, markersize=6)
                ax.fill_between(xs, lo, hi, color=colors[dt], alpha=0.15)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Radix-fp32 baseline (=1.0×)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{fam_titles[fam]} — {title_suffix}")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(bottom=0.5)
    axes[0].set_ylabel("Speedup R/H = Radix-fp32 / GVR")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"V4 GVR Top-K vs Radix-fp32  —  nsys per-call GPU kernel time\n"
        f"B200 sm_10.0  ·  synth temporal-coherence data  ·  mean over 3 beta cfgs  "
        f"·  shaded = min-max envelope",
        fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {png_path}")


def main():
    if len(sys.argv) != 4:
        print("usage: parse_sqlite.py <sweep_A.sqlite> <sweep_B.sqlite> <out_dir>")
        sys.exit(2)
    a_sql, b_sql, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    print("=== Sweep A ===")
    rows_a = kernel_time_per_nvtx(a_sql)
    print(f"  parsed {len(rows_a)} NVTX-bracketed kernel sums")
    fams = {r["family"] for r in rows_a}; print(f"  families: {fams}")
    agg_a = aggregate(rows_a, "A")
    plot_sweep(agg_a, "A", os.path.join(out_dir, "sweep_A_seqlen_scaling.png"))

    print("=== Sweep B ===")
    rows_b = kernel_time_per_nvtx(b_sql)
    print(f"  parsed {len(rows_b)} NVTX-bracketed kernel sums")
    fams = {r["family"] for r in rows_b}; print(f"  families: {fams}")
    agg_b = aggregate(rows_b, "B")
    plot_sweep(agg_b, "B", os.path.join(out_dir, "sweep_B_bs_scaling.png"))

    # Save tables
    def serialise(agg):
        return {fam: {dt: {x: list(v) for x, v in d.items()} for dt, d in fd.items()}
                for fam, fd in agg.items()}
    json.dump(serialise(agg_a), open(os.path.join(out_dir, "sweep_A_table.json"), "w"), indent=2)
    json.dump(serialise(agg_b), open(os.path.join(out_dir, "sweep_B_table.json"), "w"), indent=2)

    # Print summary tables
    print("\n=== Sweep A: mean R/H across 3 cfgs, BS=1 ===")
    Ns = sorted({N for fd in agg_a.values() for dd in fd.values() for N in dd})
    print(f"{'family':<7} {'dtype':<5} " + " ".join(f"{N:>8}" for N in Ns))
    for fam in ("flash", "pro"):
        for dt in ("bf16", "fp16", "fp32"):
            d = agg_a.get(fam, {}).get(dt, {})
            row = " ".join(f"{d.get(N, (float('nan'),))[0]:>8.3f}" for N in Ns)
            print(f"{fam:<7} {dt:<5} {row}")

    print("\n=== Sweep B: mean R/H across 3 cfgs, N=65536 ===")
    BSs = sorted({B for fd in agg_b.values() for dd in fd.values() for B in dd})
    print(f"{'family':<7} {'dtype':<5} " + " ".join(f"BS{B:>5}" for B in BSs))
    for fam in ("flash", "pro"):
        for dt in ("bf16", "fp16", "fp32"):
            d = agg_b.get(fam, {}).get(dt, {})
            row = " ".join(f"{d.get(B, (float('nan'),))[0]:>7.3f}" for B in BSs)
            print(f"{fam:<7} {dt:<5} {row}")


if __name__ == "__main__":
    main()
