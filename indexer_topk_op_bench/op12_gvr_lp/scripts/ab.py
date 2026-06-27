"""A/B: new op12 GVR-LP vs SGLang StreamingTopK, cold-L2 (report protocol).

Times both on identical synth bundles (beta_moderate, hit-rate 0.6, fp32,
K in {512,1024}) over a cell grid, prints sglang_us / new_us (>1 => new faster).
Also verifies exactness of the new op vs torch.topk on every cell.

Usage:
  python ab.py --cells battleground --configs rs_exact:512,snap:512,snap:1024
  python ab.py --cells full        # all 182 report cells (slow)
"""
import argparse
import gc
import sys
import statistics
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_OP = _HERE.parent
sys.path.insert(0, str(_OP))
sys.path.insert(0, str(_OP.parent / "harness"))

from op_lp import gvr_lp  # noqa: E402
from sglang_streaming_op import streaming_topk  # noqa: E402
from synth_data import get_bundle  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)


def _time_cold(call, cold_reps=40, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
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
    for _ in range(cold_reps):
        _EVICT.uniform_(0, 1)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    cold.sort()
    del g
    return cold[len(cold) // 2]


def _exact(out, logits):
    K = out.shape[1]
    idx = out[0].clamp(min=0).long()
    v = logits[0].float().gather(0, idx).sort(descending=True).values
    ref = torch.topk(logits[0].float(), K).values
    d = (v - ref).abs().max().item()
    nuniq = len(set(out[0].tolist()))
    return (d <= 1e-5 and nuniq == K), d, nuniq


# Battleground = where gvr_rs loses worst to sglang (small N) + a couple GVR-wins.
BATTLEGROUND = [
    (512, 4096, 1), (512, 8192, 1), (512, 16384, 1), (512, 16384, 4),
    (512, 4096, 16), (512, 65536, 1), (1024, 8192, 1), (1024, 16384, 1),
    (1024, 65536, 1), (512, 131072, 1), (1024, 262144, 1),
]


def full_cells():
    KS = [512, 1024]
    N_SEQ = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    cells = []
    seen = set()
    for K in KS:
        for N in [n for n in N_SEQ if n > 2 * K]:
            cells.append((K, N, 1))  # seqlen sweep
            for BS in BS_GRID:
                if (K, N, BS) in seen:
                    continue
                seen.add((K, N, BS))
                cells.append((K, N, BS))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="battleground")
    ap.add_argument("--configs", default="rs_exact:512",
                    help="comma list of p4mode:threads")
    ap.add_argument("--cr", type=int, default=4)
    args = ap.parse_args()

    if args.cells == "battleground":
        cells = BATTLEGROUND
    elif args.cells == "full":
        cells = full_cells()
    else:
        # parse "K,N,BS;K,N,BS"
        cells = []
        for c in args.cells.split(";"):
            k, n, bs = c.split(",")
            cells.append((int(k), int(n), int(bs)))

    configs = []
    for c in args.configs.split(","):
        parts = c.split(":")
        mode, thr = parts[0], int(parts[1])
        kca = int(parts[2]) if len(parts) > 2 else None
        configs.append((mode, thr, kca))

    cr_val = args.cr
    print(f"# cells={len(cells)} configs={configs} cr={cr_val}")
    def cfg_label(cfg):
        mode, thr, kca = cfg
        return f"{mode}/{thr}" + (f"/a{kca}" if kca else "")

    hdr = f"{'K':>5}{'N':>8}{'BS':>6} {'sglang':>8}"
    for cfg in configs:
        hdr += f" {cfg_label(cfg):>16}"
    print(hdr)

    ratios = {c: [] for c in configs}
    fails = {c: 0 for c in configs}
    for (K, N, BS) in cells:
        b = get_bundle(K, torch.float32, N, cfg="beta_moderate", seed=42)
        logits_row = b["logits"].to(torch.float32)
        preidx_row = b["preIdx"]
        logits = logits_row.expand(BS, -1).contiguous()
        pre = preidx_row.expand(BS, -1).contiguous()
        seq_div = torch.full((BS,), N * cr_val, dtype=torch.int32, device=DEV)
        seq_nod = torch.full((BS,), N, dtype=torch.int32, device=DEV)
        out = torch.empty((BS, K), dtype=torch.int32, device=DEV)

        # sglang (fp32, K in 512/1024)
        try:
            streaming_topk(logits, seq_nod, K, out=out)
            sg = _time_cold(lambda: streaming_topk(logits, seq_nod, K, out=out))
        except Exception as e:
            sg = float("nan")
        line = f"{K:>5}{N:>8}{BS:>6} {sg:>8.1f}"
        for cfg in configs:
            mode, thr, kca = cfg
            try:
                gvr_lp(logits, pre, seq_div, K, cr_val, out=out, num_threads=thr,
                       p4_mode=mode, kc_accept=kca)
                ok, d, nuniq = _exact(out, logits)
                t = _time_cold(lambda: gvr_lp(logits, pre, seq_div, K, cr_val, out=out,
                                              num_threads=thr, p4_mode=mode, kc_accept=kca))
                if not ok:
                    fails[cfg] += 1
                    line += f" {('X%.1f' % t):>16}"
                else:
                    r = sg / t
                    ratios[cfg].append(r)
                    line += f" {('%.1f(%.2fx)' % (t, r)):>16}"
            except Exception as e:
                line += f" {('ERR'):>16}"
        print(line, flush=True)
        del logits, pre, out
        gc.collect()
        torch.cuda.empty_cache()

    print("\n# SUMMARY (sglang/new, >1 => new faster):")
    for cfg in configs:
        rs = ratios[cfg]
        if rs:
            mn = min(rs)
            print(f"  {cfg_label(cfg):<18}: median={statistics.median(rs):.3f} "
                  f"mean={statistics.mean(rs):.3f} min={mn:.3f} "
                  f"win={sum(1 for x in rs if x>1)}/{len(rs)} fails={fails[cfg]}")
        else:
            print(f"  {cfg_label(cfg):<18}: no valid cells fails={fails[cfg]}")


if __name__ == "__main__":
    main()
