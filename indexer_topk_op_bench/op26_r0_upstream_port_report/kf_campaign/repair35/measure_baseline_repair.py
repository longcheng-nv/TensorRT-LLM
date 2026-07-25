# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""repair35: external nsys cold-L2 baseline — PR#16457 head (e612, op40
gvrpkg40b) run natively batched on the 29 HETEROGENEOUS workloads, with the
production per-(BS, N) launch config (pick_config). NVTX c|pr|<uuid>|fp32.
Run under nsys; then parse_baseline_repair.py writes baselines.jsonl."""
import json
import sys
from pathlib import Path

import torch
import torch.cuda.profiler as prof
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
KF = HERE.parent
BENCH = KF.parent.parent
OP40 = BENCH / "op40_omni_gvr"
sys.path.insert(0, str(BENCH / "harness"))
sys.path.insert(0, str(OP40 / "src"))
sys.path.insert(0, str(OP40 / "scripts"))

import cutlass  # noqa: E402

assert cutlass.__version__.startswith("4.5.0")
from ab40 import compile_arm  # noqa: E402
import gvrpkg40b.top_k.gvr_topk_decode as B  # noqa: E402

WARMUP, REPS = 10, 15
_EVICT = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
CR = {"flash": 4, "pro": 4, "v32": 1}


def main():
    wls = [json.loads(l) for l in open(HERE / "workload_repair.jsonl")]
    files = sorted(set(json.load(open(HERE / "definition_repair.json"))
                       ["reference"].split("_GRP_FILES = ")[1]
                       .split("]")[0].strip("[").replace('"', "").split(", ")))
    prof.start()
    for wl in wls:
        u = wl["uuid"]
        gid = wl["axes"]["cell"]
        bs, n, K = wl["axes"]["b"], wl["axes"]["n"], wl["axes"]["k"]
        model = u.split("_")[0]
        isl = u.split("_")[1]
        d = load_file(str(HERE / "assets" / f"grp_{model}_{isl}.safetensors"))
        lg_all, pre_all = d["logits"].cuda(), d["pre_idx"].cuda()
        L = lg_all.shape[0]
        idx = torch.arange(bs) % L
        lg = lg_all[idx].contiguous().float()
        pre = pre_all[idx].contiguous().to(torch.int32)
        cr = CR[model]
        cfg = B.GvrTopKKernel.pick_config(torch.float32, bs, n)
        if cfg["cluster_size"] > 1:
            cfg["cluster_size"] = min(cfg["cluster_size"], 16)
        if cfg.get("use_256bit_load") and lg.data_ptr() % 32 != 0:
            cfg["use_256bit_load"] = False
        fn = compile_arm("base", K, cr, cfg)
        sl = torch.full((bs,), n * cr, dtype=torch.int32, device="cuda")
        oi = torch.full((bs, K), -7, dtype=torch.int32, device="cuda")
        fn(lg, pre, sl, None, oi, None)
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn(lg, pre, sl, None, oi, None)
        torch.cuda.synchronize()
        rname = f"c|pr|{u}|fp32"
        for _ in range(REPS):
            _EVICT.random_()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(rname)
            fn(lg, pre, sl, None, oi, None)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        print(f"[base] {u} done (cs={cfg['cluster_size']})", flush=True)
        del lg, pre, lg_all, pre_all
        torch.cuda.empty_cache()
    prof.stop()
    print("[base] ALL DONE")


if __name__ == "__main__":
    main()
