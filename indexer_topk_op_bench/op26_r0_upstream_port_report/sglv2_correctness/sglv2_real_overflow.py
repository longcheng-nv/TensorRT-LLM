#!/usr/bin/env python3
"""End-to-end sglang v2 kernel check on the REAL rows the extended tie-bin
sweep flagged: V3.2 256k layer 52 step 6 (tie10=2214 > cap) plus the worst
last-step rows (below cap) as controls."""
import sys
from pathlib import Path
import torch

BENCH = Path("/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench")
sys.path.insert(0, str(BENCH / "harness"))
import real_data_v32 as RV32
from sglang_v2_op import topk_v2, plan  # noqa: E402

DEV = "cuda"
K = 2048

def run_case(tag, row_f32, N):
    W = ((N + 63) // 64) * 64
    lg = torch.full((1, W), torch.finfo(torch.float32).min, dtype=torch.float32, device=DEV)
    lg[0, :N] = row_f32[:N].to(DEV)
    sl = torch.full((1,), N, dtype=torch.int32, device=DEV)
    out = torch.empty((1, K), dtype=torch.int32, device=DEV)
    md = plan(sl)
    torch.cuda.synchronize()
    topk_v2(lg, sl, K, out=out, metadata=md, max_seq_len=N)
    torch.cuda.synchronize()
    idx = out[0].long().cpu()
    row = row_f32[:N].float().cpu()
    ref = torch.topk(row, K).values.sort().values
    uniq = int(idx.unique().numel())
    got = row[idx.clamp(0, N - 1)].sort().values
    exact = bool(idx.min() >= 0 and idx.max() < N and uniq == K
                 and torch.equal(got, ref))
    nbad = int((got != ref).sum())
    maxerr = float((ref - got).abs().max())
    print(f"{tag:42s} N={N:<8d} exact={str(exact):5s} uniq={uniq:<5d} "
          f"mismatched_slots={nbad:<5d} max_val_err={maxerr:.6f} "
          f"true_thr={float(ref[0]):.6f} worst_pick={float(got.min()):.6f}")
    return exact

def load_step_row(isl, L, step):
    d = RV32._layer_dir(isl, L)
    lg = torch.load(d / "decode.logits.in.pt", map_location="cpu", weights_only=False)
    pk = torch.load(d / "decode.topk.out.pt", map_location="cpu", weights_only=False)
    Ns = int(pk[step].max()) + 1
    row = lg[step]
    return (row[0] if row.dim() == 2 else row).float()[:Ns].clone(), Ns

# 1) the flagged overflow row
r, N = load_step_row("256k", 52, 6)
run_case("real v32 256k L52 step6 (tie10=2214)", r, N)

# scan L52 all steps end-to-end for good measure
d = RV32._layer_dir("256k", 52)
lg = torch.load(d / "decode.logits.in.pt", map_location="cpu", weights_only=False)
pk = torch.load(d / "decode.topk.out.pt", map_location="cpu", weights_only=False)
nfail = 0
for st in sorted(lg.keys()):
    Ns = int(pk[st].max()) + 1
    row = lg[st]
    r = (row[0] if row.dim() == 2 else row).float()[:Ns]
    W = ((Ns + 63) // 64) * 64
    lgd = torch.full((1, W), torch.finfo(torch.float32).min, dtype=torch.float32, device=DEV)
    lgd[0, :Ns] = r.to(DEV)
    sl = torch.full((1,), Ns, dtype=torch.int32, device=DEV)
    out = torch.empty((1, K), dtype=torch.int32, device=DEV)
    md = plan(sl); torch.cuda.synchronize()
    topk_v2(lgd, sl, K, out=out, metadata=md, max_seq_len=Ns)
    torch.cuda.synchronize()
    idx = out[0].long().cpu()
    ref = torch.topk(r, K).values.sort().values
    got = r[idx.clamp(0, Ns - 1)].sort().values
    ok = bool(idx.unique().numel() == K and torch.equal(got, ref))
    if not ok:
        nfail += 1
        print(f"  L52 step {st}: exact=False  mismatched={int((got!=ref).sum())} "
              f"max_val_err={float((ref-got).abs().max()):.6f}")
print(f"L52 all-steps: {nfail} / {len(lg)} steps FAIL end-to-end")

# 2) worst LAST-STEP rows (controls, below cap)
for isl, L, tie in (("256k", 52, 1466), ("8k", 52, 1382), ("64k", 39, 1237)):
    s = RV32._slim(isl)
    r, N = load_step_row(isl, L, s["s_last"])
    run_case(f"real v32 {isl} L{L} last-step (tie={tie})", r, N)
