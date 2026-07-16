# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""op35 L1 triage harness — paired same-process cold-L2 CUDA-event A/B on the
77 §6 metric cells (synth 52 + real 25, BS=1 fp32).

  base arm    = gvrpkg_head (PR HEAD eae374554c snapshot), launch() defaults
  variant arm = variant/gvrpkg35 kernel, launch() + --var-flags ctor overrides

Interleaved paired timing (A,B,A,B ... x reps) with a 512MB L2 evict before
every timed launch; exactness = tie-aware value-multiset vs torch.topk.
CUDA-event inflates both arms equally (ratio-fair); nsys is the ship arbiter.

Usage:
  python3 ab_op35.py [--family synth|real|all] [--minN 0] [--maxN 1e9]
                     [--reps 30] [--var-flags '{"p3_skip_frac":0.2}']
                     [--cells k512_worst_262144,...]  [--out results/x.jsonl]
                     [--no-exact-var]   # oracle/diagnostic variants (wrong output OK)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_OP35 = _HERE.parent
_BENCH = _OP35.parent
sys.path.insert(0, str(_BENCH / "harness"))
sys.path.insert(0, str(_BENCH / "op22_temporal_fixed_hr_bench"))
sys.path.insert(0, str(_OP35 / "gvrpkg_head"))
sys.path.insert(0, str(_OP35 / "variant"))
os.environ.setdefault("SYNTH_POSITIONAL", "1")

import bundle_data_env as SYNTH                                   # noqa: E402
import real_data_v4cap as RV4                                     # noqa: E402
import real_data_v32 as RV32                                      # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel as BaseK   # noqa: E402
from gvrpkg35.top_k.gvr_topk_decode import GvrTopKKernel as VarK  # noqa: E402

DEV = "cuda"
N_SEQ = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
REAL_LAYER = {"flash": 22, "pro": 30, "v32": 34}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}


def iter_cells(fam):
    if fam in ("synth", "all"):
        for scen in ("best", "worst"):
            for K in (512, 1024, 2048):
                for N in N_SEQ:
                    if N > 2 * K:
                        yield ("synth", scen, K, N, "", "")
    if fam in ("real", "all"):
        for model in ("flash", "pro", "v32"):
            for isl in REAL_ISLS[model]:
                yield ("real", "", None, None, model, isl)


def load_cell(cell):
    fam, scen, K, N, model, isl = cell
    if fam == "synth":
        b = SYNTH.get_bundle(scen, K, torch.float32, N)
        cid = f"synth_{scen}_K{K}_N{N}"
        return cid, b["logits"][0, :N].float(), b["preIdx"], K, b["cr"], N
    RD = RV32 if model == "v32" else RV4
    b = RD.get_bundle(model, isl, REAL_LAYER[model], "fp32")
    N = b["N"]
    cid = f"real_{model}_{isl}"
    return cid, b["logits"][0, :N].float(), b["preIdx"], b["K"], b["cr"], N


def exact_check(out_idx, logits, N, K):
    idx = out_idx[0].long()
    if idx.numel() != K or idx.min() < 0 or idx.max() >= N or idx.unique().numel() != K:
        return False
    ref = torch.topk(logits[:N].float(), K).values.sort().values
    return torch.equal(logits[idx].float().sort().values, ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all")
    ap.add_argument("--minN", type=float, default=0)
    ap.add_argument("--maxN", type=float, default=1e9)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--var-flags", default="{}")
    ap.add_argument("--base-flags", default="{}")
    ap.add_argument("--cells", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-exact-var", action="store_true")
    ap.add_argument("--shard", default="")          # "i/n": run cells where idx%n==i
    args = ap.parse_args()
    vflags = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in json.loads(args.var_flags).items()}
    bflags = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in json.loads(args.base_flags).items()}
    want = set(args.cells.split(",")) if args.cells else None
    sh_i, sh_n = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)

    evict = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
    results = []
    fout = open(args.out, "a") if args.out else None
    for ci, cell in enumerate(iter_cells(args.family)):
        if ci % sh_n != sh_i:
            continue
        cid, lg_row, pre, K, cr, N = load_cell(cell)
        if not (args.minN <= N <= args.maxN):
            continue
        if want and cid not in want:
            continue
        lg = lg_row.unsqueeze(0).contiguous().to(DEV)
        pre = pre[:1].contiguous().to(DEV)
        sl = torch.full((1,), N * cr, dtype=torch.int32, device=DEV)
        out_b = torch.empty(1, K, dtype=torch.int32, device=DEV)
        out_v = torch.empty(1, K, dtype=torch.int32, device=DEV)
        vf = {k: v for k, v in vflags.items() if not k.endswith('_k2048')}
        if K == 2048:
            for k, v in vflags.items():
                if k.endswith('_k2048'):
                    vf[k[:-6]] = v
        call_b = lambda: BaseK.launch(lg, pre, sl, out_b, K, compress_ratio=cr, **bflags)
        call_v = lambda: VarK.launch(lg, pre, sl, out_v, K, compress_ratio=cr, **vf)
        try:
            call_b(); call_v()
        except Exception as e:
            print(f"{cid}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        torch.cuda.synchronize()
        ex_b = exact_check(out_b, lg[0], N, K)
        ex_v = True if args.no_exact_var else exact_check(out_v, lg[0], N, K)

        tb, tv = [], []
        for r in range(args.reps):
            for arm, call, acc in (("b", call_b, tb), ("v", call_v, tv)):
                evict.uniform_()
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
                e0.record(); call(); e1.record()
                torch.cuda.synchronize()
                acc.append(e0.elapsed_time(e1) * 1e3)
        tb.sort(); tv.sort()
        mb = tb[len(tb) // 2]; mv = tv[len(tv) // 2]
        rec = dict(cell=cid, N=N, K=K, base_us=round(mb, 2), var_us=round(mv, 2),
                   ratio=round(mb / mv, 4), exact_base=ex_b, exact_var=ex_v)
        results.append(rec)
        print(f"{cid}: base={mb:.2f} var={mv:.2f} b/v={mb/mv:.3f} exact(b/v)={ex_b}/{ex_v}", flush=True)
        if fout:
            fout.write(json.dumps(rec) + "\n"); fout.flush()
        del lg, pre, sl, out_b, out_v
        torch.cuda.empty_cache()

    if results:
        g = math.exp(sum(math.log(r["ratio"]) for r in results) / len(results))
        print(f"\ngeomean base/var = {g:.4f} over {len(results)} cells "
              f"(>1 = variant faster); exact_var fails: {sum(not r['exact_var'] for r in results)}")


if __name__ == "__main__":
    main()
