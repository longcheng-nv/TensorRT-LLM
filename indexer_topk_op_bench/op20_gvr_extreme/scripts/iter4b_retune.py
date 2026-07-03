# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op20 iter4b: dispatch re-tune of the M*R1p4-family keys under the fused
# P2+P3 option. Fusion changes the M tradeoff (P3 cost no longer ~N-scan but
# ~collected-candidates), so per-key best (M, fuse) must be re-probed at the
# exact (K, N, BS) bucket (protocol red line). Writes results/dispatch_table_
# fp32.json in place (backup .pre_iter4b) only where a variant beats the
# CURRENT cfg by >3%.
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
import synth_data  # noqa: E402
from gvr_x_op import gvr_sw  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
TBL = _HERE.parent / "results" / "dispatch_table_fp32.json"


def cold_us(call, reps=30, warmup=5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
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
    for _ in range(reps):
        _EVICT.uniform_(0, 1); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        cold.append(e0.elapsed_time(e1) * 1e3)
    cold.sort(); del g
    return cold[len(cold) // 2]


def exact(out, logits, K):
    lf = logits.float()
    ref = torch.topk(lf, K, dim=1).values
    idx = out.clamp(min=0).long()
    v = lf.gather(1, idx).sort(dim=1, descending=True).values
    if (v - ref).abs().max().item() != 0.0:
        return False
    return all(len(set(out[r].tolist())) == K for r in range(out.shape[0]))


if __name__ == "__main__":
    import re
    tbl = json.load(open(TBL))
    changes = {}
    for K in (512, 1024):
        for N in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
            for BS in (1, 4, 16, 64):
                key = f"{K}_{N}_{BS}"
                ent = tbl.get(key)
                if not ent or not re.match(r"M\d+R1p4$", ent["cfg"]):
                    continue
                cr = 4
                b = synth_data.get_bundle(K, torch.float32, N)
                logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
                pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
                seq = torch.full((BS,), b["Npad"] * cr, dtype=torch.int32, device=DEV)
                o = torch.empty(BS, K, dtype=torch.int32, device=DEV)
                res = {}
                for M in (2, 4, 6):
                    for fuse in (False, True):
                        tag = f"M{M}R1p4" + ("f" if fuse else "nf")
                        try:
                            call = (lambda M=M, fuse=fuse: gvr_sw(
                                logits, pre, seq, K, cr, out=o, M=M, R=1,
                                place_mode=4, fuse=fuse))
                            call(); torch.cuda.synchronize()
                            if not exact(o, logits, K):
                                continue
                            res[tag] = cold_us(call)
                        except Exception:
                            pass
                if not res:
                    continue
                cur_tag = ent["cfg"] + "nf"  # current = classic (pre-iter4)
                cur_us = res.get(cur_tag)
                best_tag = min(res, key=res.get)
                line = f"{key:>16} cur={ent['cfg']}({cur_us if cur_us else -1:6.1f}) best={best_tag}({res[best_tag]:6.1f})"
                if cur_us and res[best_tag] < cur_us * 0.97 and best_tag != cur_tag:
                    # keep the f/nf suffix explicit — a bare M*R1p4 would let
                    # the gvr_sw auto-gate re-enable fusion where nf won
                    tbl[key] = {"cfg": best_tag, "speedup": ent.get("speedup", 0)}
                    changes[key] = (ent["cfg"], best_tag, round(cur_us / res[best_tag], 3))
                    line += f"  -> UPDATE {tbl[key]['cfg']} ({cur_us/res[best_tag]:.3f}x)"
                print(line, flush=True)
    if changes:
        bak = TBL.with_suffix(".json.pre_iter4b")
        if not bak.exists():
            bak.write_text(TBL.read_text())
        json.dump(tbl, open(TBL, "w"), indent=1, sort_keys=True)
        print(f"updated {len(changes)} keys; backup at {bak.name}")
    else:
        print("no changes")
