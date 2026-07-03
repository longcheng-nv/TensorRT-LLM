# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# op20 tier3 probe: driven by results/tier3_iter0.jsonl losses (rival/x<0.97).
# Per (dtype,K,N,BS) bucket, time a region-specific candidate set vs the
# CURRENT table cfg (gvr_sw_auto, live table); write the per-dtype dispatch
# table update (backup .pre_tier3) only where best beats current by >3% AND
# is exact. 16-bit tie-stepped CCDFs make count>kC likelier — every variant
# is exactness-checked (value-equality, tie-tolerant).
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parents[1]
for p in ("harness", "ops"):
    sys.path.insert(0, str(_BENCH / p))
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_BENCH / "op17_gvr_portfolio" / "v2"))
import synth_data  # noqa: E402
from gvr_multicta_cutedsl_op import gvr_multicta_cutedsl  # noqa: E402
from gvr_portfolio_fusion_op import gvr_portfolio_fusion  # noqa: E402
from gvr_cutedsl_op import gvr_cutedsl  # noqa: E402
from gvr_x_op import gvr_sw, gvr_sw_auto  # noqa: E402

DEV = "cuda"
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}
CRMAP = {512: 4, 1024: 4, 2048: 1}


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


def candidates(N, BS):
    if N >= 131072 and BS <= 4:
        return ["mc", "mcC8", "fusP4T4", "M2R1p4f"]
    if N >= 131072 and BS <= 16:
        return ["mc", "mcC8", "M2R1p4f", "M4R1p4f"]
    if BS <= 64:
        return ["M2R1p4f", "M4R1p4f", "M6R1p4f", "M2R1p4nf", "M4R1p4nf"]
    return ["M2R1p4nf", "M4R1p4nf", "M6R1p4nf", "baseline"]


def make_call(cfg, logits, pre, seq, K, cr, o):
    if cfg == "baseline":
        return lambda: gvr_cutedsl(logits, pre, seq, K, cr, out=o)
    if cfg == "mc":
        return lambda: gvr_multicta_cutedsl(logits, pre, seq, K, cr, out=o)
    if cfg.startswith("mcC"):
        return lambda: gvr_multicta_cutedsl(logits, pre, seq, K, cr, out=o,
                                            cluster_size=int(cfg[3:]))
    if cfg.startswith("fusP"):
        import re
        m = re.match(r"fusP(\d+)T(\d+)$", cfg)
        return lambda: gvr_portfolio_fusion(logits, pre, seq, K, cr, out=o,
                                            P=int(m.group(1)), T=int(m.group(2)))
    import re
    m = re.match(r"M(\d+)R(\d+)p(\d+)(f|nf)$", cfg)
    fuse = m.group(4) == "f"
    return lambda: gvr_sw(logits, pre, seq, K, cr, out=o, M=int(m.group(1)),
                          R=int(m.group(2)), place_mode=int(m.group(3)), fuse=fuse)


if __name__ == "__main__":
    losses = []
    for l in open(_HERE.parent / "results" / "tier3_iter0.jsonl"):
        r = json.loads(l)
        if r["rival_us"] / r["x_us"] < 0.97:
            losses.append(r)
    losses.sort(key=lambda r: (r["dtype"], r["K"], r["N"], r["BS"]))
    print(f"{len(losses)} loss buckets to probe", flush=True)
    tbls = {dt: json.load(open(_HERE.parent / "results" / f"dispatch_table_{dt}.json"))
            for dt in ("bf16", "fp16")}
    changes = {dt: {} for dt in ("bf16", "fp16")}
    for r in losses:
        dt, K, N, BS = r["dtype"], r["K"], r["N"], r["BS"]
        cr = CRMAP[K]
        key = f"{K}_{N}_{BS}"
        cur_cfg = tbls[dt].get(key, {}).get("cfg", "?")
        b = synth_data.get_bundle(K, _DT[dt], N)
        logits = b["logits"].to(DEV).expand(BS, -1).contiguous()
        pre = b["preIdx"].to(DEV).expand(BS, -1).contiguous()
        seq = torch.full((BS,), b["Npad"] * cr, dtype=torch.int32, device=DEV)
        o = torch.empty(BS, K, dtype=torch.int32, device=DEV)
        try:
            cur_call = lambda: gvr_sw_auto(logits, pre, seq, K, cr, out=o)
            cur_call(); torch.cuda.synchronize()
            cur_us = cold_us(cur_call)
        except Exception:
            cur_us = None
        line = f"{dt} {key:>16} cur={cur_cfg:>9}({cur_us if cur_us else -1:6.1f})"
        best_cfg, best_us = None, None
        for cfg in candidates(N, BS):
            if cfg == cur_cfg:
                continue
            try:
                call = make_call(cfg, logits, pre, seq, K, cr, o)
                call(); torch.cuda.synchronize()
                if not exact(o, logits, K):
                    line += f" {cfg}=!EX"
                    continue
                t = cold_us(call)
                line += f" {cfg}={t:6.1f}"
                if best_us is None or t < best_us:
                    best_cfg, best_us = cfg, t
            except Exception as e:
                line += f" {cfg}=ERR"
        if cur_us and best_us and best_us < cur_us * 0.97:
            tbls[dt][key] = {"cfg": best_cfg,
                             "speedup": tbls[dt].get(key, {}).get("speedup", 0)}
            changes[dt][key] = (cur_cfg, best_cfg, round(cur_us / best_us, 3))
            line += f"  -> UPDATE {best_cfg} ({cur_us/best_us:.3f}x)"
        print(line, flush=True)
        del logits, pre, o
        torch.cuda.empty_cache()
    for dt in ("bf16", "fp16"):
        if changes[dt]:
            p = _HERE.parent / "results" / f"dispatch_table_{dt}.json"
            bak = p.with_suffix(".json.pre_tier3")
            if not bak.exists():
                bak.write_text(p.read_text())
            json.dump(tbls[dt], open(p, "w"), indent=1, sort_keys=True)
            print(f"{dt}: updated {len(changes[dt])} keys (backup {bak.name})", flush=True)
        else:
            print(f"{dt}: no changes", flush=True)
