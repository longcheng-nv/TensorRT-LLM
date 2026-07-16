# vseed A/B sweep — base / pr / vseed on real decode captures, nsys cold-L2.
# One process per nsys run; NVTX ranges via harness measure_cell protocol.
# Regression cells (flash 1024k all dtypes, v32 256k) + guard cells (R0 wins).
import gc
import json
import sys

import torch
import torch.cuda.profiler as prof

sys.path.insert(0, "/tmp/gvrval1")   # edited gvrpkg (r0_vseed flag)
sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/harness")
import real_data_v4cap as RD4                       # noqa: E402
import real_data_v32 as RD32                        # noqa: E402
from sweep_nsys import measure_cell                 # noqa: E402
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402

DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
ARMS = [("base", {"enable_r0": False}), ("pr", {}), ("vseed", {"r0_vseed": True}), ("vs2", {"r0_qfracs": (0.85,), "r0_vseed": True})]

# (model, isl, layer, dtype, [BS...])
CELLS = [
    ("flash", "1024k", 22, "fp32", [1, 64, 128, 256, 512, 1024]),
    ("flash", "1024k", 22, "bf16", [1, 64, 1024]),
    ("flash", "1024k", 22, "fp16", [1, 64, 1024]),
    ("flash", "512k", 22, "fp32", [1, 1024]),
    ("flash", "128k", 22, "fp32", [1, 1024]),
    ("flash", "64k", 22, "fp32", [1]),
    ("pro", "128k", 30, "fp32", [1, 1024]),
    ("pro", "1024k", 30, "fp32", [1, 1024]),
    ("v32", "256k", 34, "fp32", [1, 128, 1024]),
    ("v32", "64k", 34, "fp32", [1]),
]


def bundle(model, isl, layer, dtn):
    if model == "v32":
        return RD32.get_bundle("v32", isl, layer, dtn)
    return RD4.get_bundle(model, isl, layer, dtn)


def main():
    torch.cuda.set_device(0)
    meta_f = open("/tmp/gvrval1/vseed_ab2/cells.jsonl", "a")
    prof.start()
    for model, isl, layer, dtn, BSs in CELLS:
        b = bundle(model, isl, layer, dtn)
        K, cr, N = b["K"], b["cr"], b["N"]
        lg1, pre1 = b["logits"], b["preIdx"]
        ref = b["ref"].long()
        ref_vals = torch.sort(lg1[0, :N].float()[ref]).values
        for BS in BSs:
            lg = lg1.to(DT[dtn]).expand(BS, -1).contiguous()
            pre = pre1.expand(BS, -1).contiguous()
            sl = torch.full((BS,), N * cr, dtype=torch.int32, device="cuda")
            for arm, ovr in ARMS:
                out = torch.empty(BS, K, dtype=torch.int32, device="cuda")

                def call(lg=lg, pre=pre, sl=sl, out=out, ovr=ovr, K=K, cr=cr):
                    GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr)

                call()
                torch.cuda.synchronize()
                o = out[BS - 1].long()
                exact = bool(o.unique().numel() == K and (o >= 0).all() and (o < N).all()
                             and (dtn != "fp32"
                                  or torch.equal(torch.sort(lg[BS - 1, o].float()).values, ref_vals)))
                tag = f"{model}.{isl}.{dtn}.BS{BS}.{arm}"
                measure_cell(call, tag, reps_cold=20, reps_warm=8, warmup=8)
                meta_f.write(json.dumps(dict(model=model, isl=isl, layer=layer, dtype=dtn,
                                             K=K, N=N, BS=BS, arm=arm, tag=tag,
                                             hit=b.get("hit_rate"), exact=exact)) + "\n")
                meta_f.flush()
                del out
            del lg, pre, sl
            gc.collect()
            torch.cuda.empty_cache()
        print(f"done {model}/{isl}/{dtn}", flush=True)
    prof.stop()
    meta_f.close()


if __name__ == "__main__":
    main()
