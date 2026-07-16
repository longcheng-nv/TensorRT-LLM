import sys, torch
sys.path.insert(0, "/tmp/gvrval1")   # edited gvrpkg (vseed)
sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/harness")
import real_data_v4cap as RD4
from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel
torch.cuda.set_device(0)

def run(model, isl, layer, BS, dt=torch.float32, reps=50):
    b = RD4.get_bundle(model, isl, layer, {torch.float32:"fp32", torch.bfloat16:"bf16", torch.float16:"fp16"}[dt])
    K, cr, N = b["K"], b["cr"], b["N"]
    lg = b["logits"].to(dt).expand(BS, -1).contiguous()
    pre = b["preIdx"].expand(BS, -1).contiguous()
    sl = torch.full((BS,), N*cr, dtype=torch.int32, device="cuda")
    ref_vals = torch.sort(b["logits"][0, :N].float()[b["ref"].long()]).values
    res = {}
    for arm, ovr in [("base", {"enable_r0": False}), ("pr", {}), ("vseed", {"r0_vseed": True})]:
        out = torch.empty(BS, K, dtype=torch.int32, device="cuda")
        call = lambda: GvrTopKKernel.launch(lg, pre, sl, out, K, compress_ratio=cr, **ovr)
        call(); torch.cuda.synchronize()
        # exactness: per-row unique==K and gathered value-set equal to ref
        ok = True
        for r in range(min(BS, 4)):
            o = out[r].long()
            if o.unique().numel() != K or int(o.max()) >= N or int(o.min()) < 0: ok = False; break
            gv = torch.sort(lg[r, o].float()).values
            if not torch.equal(gv, ref_vals) and dt is torch.float32: ok = False; break
        # CUDA-event rough timing (paired, same stream)
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        for _ in range(5): call()
        torch.cuda.synchronize(); s.record()
        for _ in range(reps): call()
        e.record(); torch.cuda.synchronize()
        res[arm] = (s.elapsed_time(e)/reps*1000, ok)
    return N, res

print(f"{'cell':>28} | {'base us':>8} {'pr us':>8} {'vseed us':>8} | pr/b  vs/b  vs/pr | exact")
for model, isl, L, BSs, dt in [
    ("flash","1024k",22,[1,128,1024],torch.float32),
    ("flash","1024k",22,[1,1024],torch.bfloat16),
    ("flash","512k",22,[1,1024],torch.float32),
    ("flash","128k",22,[1,1024],torch.float32),
    ("pro","1024k",30,[1,1024],torch.float32),
    ("pro","128k",30,[1],torch.float32),
]:
    for BS in BSs:
        N, r = run(model, isl, L, BS, dt)
        b,p,v = r["base"][0], r["pr"][0], r["vseed"][0]
        ex = all(r[a][1] for a in r)
        dtn = str(dt).split('.')[-1]
        print(f"{model}/{isl}/L{L}/{dtn}/BS{BS:>4} | {b:8.1f} {p:8.1f} {v:8.1f} | {b/p:.2f}  {b/v:.2f}  {p/v:.2f} | {'OK' if ex else 'FAIL'}")
