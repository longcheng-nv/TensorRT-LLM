# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""iter13: APEX v3 full-grid L1 nsys sweep on the op26 rival envelope (fp32).

Same protocol as op26 rival_harness/sweep_rival.py (measure_cell NVTX c|/w|
ranges, cold-L2 evict outside range) so per-cell us_span aligns with
rival_long.csv. One arm: apex_v3. Exactness folded in per cell (row0 & BS-1,
tie-robust). Run UNDER nsys; resumable via jsonl.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(BENCH / "op22_temporal_fixed_hr_bench"))

from sweep_nsys import measure_cell            # noqa: E402
import bundle_data_env as SYNTH                # noqa: E402
import real_data_v4cap as RV4                  # noqa: E402
import real_data_v32 as RV32                   # noqa: E402
from apex_op import apex_topk, pick_config, workspace  # noqa: E402

DEV = "cuda"
N_SEQ = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
BS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_BS_REP = [16384, 65536, 131072, 262144]
REAL_LAYER = {"flash": 22, "pro": 30, "v32": 34}
REAL_ISLS = {"flash": ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "pro":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1024k"],
             "v32":   ["4k", "8k", "16k", "32k", "64k", "128k", "256k"]}
REAL_BS_ISL = "128k"


DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def build_call_apex(K, N, BS, logits_row, dtype):
    W = ((N + 63) // 64) * 64
    x = torch.full((BS, W), torch.finfo(dtype).min, dtype=dtype, device=DEV)
    x[:, :N] = logits_row[:, :N].to(dtype)
    cfg = pick_config(BS, N, K)
    ws = workspace(BS, K, cfg, x.device)
    call = lambda: apex_topk(x, K, N=N, cfg=cfg, ws=ws)  # noqa: E731
    call()
    torch.cuda.synchronize()
    return call, [x], (lambda: ws["out"])


def exact_check(getter, logits_full, N, K, BS):
    idx_t = getter()
    ref = torch.topk(logits_full[:N].float(), K).values.sort().values
    for r in ((0, BS - 1) if BS > 1 else (0,)):
        idx = idx_t[r].long()
        if idx.numel() != K or idx.min() < 0 or idx.max() >= N or idx.unique().numel() != K:
            return False
        got = logits_full[idx].float().sort().values
        if not torch.equal(got, ref):
            return False
    return True


def load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["family"], r.get("scenario", ""), r.get("model", ""),
                          r["K"], r["N"], r["BS"], r.get("isl", "")))
            except Exception:
                pass
    return done


def run_cell(f, rec_base, K, N, BS, logits_row, rc, rw, dtype):
    base = rec_base["range_cold"][2:]
    try:
        call, keep, getter = build_call_apex(K, N, BS, logits_row, dtype)
        ref_row = logits_row[0, :N].to(dtype).float()
        rec_base["exact"] = bool(exact_check(getter, ref_row, N, K, BS))
        measure_cell(call, base, rc, rw)
        del call, keep
    except Exception as e:
        rec_base["error"] = f"{type(e).__name__}: {str(e)[:140]}"
    f.write(json.dumps(rec_base) + "\n")
    f.flush()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--reps-warm", type=int, default=10)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)
    out = Path(a.out or str(HERE.parent / f"results/iter13/apex_{a.dtype}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    f = open(out, "a")
    prof.start()
    try:
        # ---- synth: best/worst x K x (seqlen BS=1 + bs grid) ----
        for scen in ("best", "worst"):
            for K in (512, 1024, 2048):
                cells = [(N, 1) for N in N_SEQ if N > 2 * K] + \
                        [(N, BS) for N in N_BS_REP if N > 2 * K for BS in BS_GRID if BS > 1]
                for N, BS in cells:
                    if ("synth", scen, "", K, N, BS, "") in done:
                        continue
                    b = SYNTH.get_bundle(scen, K, torch.float32, N)
                    lg = b["logits"]
                    base = f"apex_v3|{scen}|{K}|{a.dtype}|{N}|{BS}"
                    rec = dict(family="synth", scenario=scen, op="apex_v3", K=K,
                               dtype=a.dtype, N=N, BS=BS, isl="",
                               range_cold=f"c|{base}", range_warm=f"w|{base}",
                               reps_cold=a.reps, reps_warm=a.reps_warm)
                    run_cell(f, rec, K, N, BS, lg, a.reps, a.reps_warm, DT[a.dtype])
                print(f"[synth {scen} K{K}] done", flush=True)
        # ---- real: seqlen (BS1) + bs grid at 128k ----
        for model in ("flash", "pro", "v32"):
            RD = RV32 if model == "v32" else RV4
            L = REAL_LAYER[model]
            cells = [(isl, 1) for isl in REAL_ISLS[model]] + \
                    [(REAL_BS_ISL, BS) for BS in BS_GRID if BS > 1]
            for isl, BS in cells:
                try:
                    b = RD.get_bundle(model, isl, L, "fp32")
                except Exception as e:
                    print(f"  SKIP real {model} {isl}: {str(e)[:80]}", flush=True)
                    continue
                K, N = b["K"], b["N"]
                if ("real", "", model, K, N, BS, isl) in done:
                    continue
                base = f"apex_v3|{model}|{isl}|{a.dtype}|{N}|{BS}"
                rec = dict(family="real", model=model, op="apex_v3", K=K,
                           dtype=a.dtype, N=N, BS=BS, isl=isl,
                           range_cold=f"c|{base}", range_warm=f"w|{base}",
                           reps_cold=a.reps, reps_warm=a.reps_warm)
                run_cell(f, rec, K, N, BS, b["logits"], a.reps, a.reps_warm, DT[a.dtype])
            print(f"[real {model}] done", flush=True)
    finally:
        prof.stop()
    f.close()
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
