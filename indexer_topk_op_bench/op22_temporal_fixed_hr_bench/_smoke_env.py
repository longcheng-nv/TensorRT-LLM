import sys, os, torch
from pathlib import Path
HERE = Path.cwd()
sys.path.insert(0, str(HERE.parents[0] / "harness"))
sys.path.insert(0, str(HERE.parents[0] / "op28_ext_topk"))
from ops_ext import build_call_ext
import bundle_data_env as bd
ARMS9 = ["gvr_cutedsl", "radix_cutedsl", "gvr_multicta_cutedsl",
         "radix_single_cuda", "radix_multi_cuda", "gvr_ms_auto",
         "op26_r0auto", "sglang_v2", "flashinfer_topk"]


def pin(f):
    if f is None:
        os.environ.pop("OP21_FB_LOGFALSI", None)
    else:
        os.environ["OP21_FB_LOGFALSI"] = f
    os.environ.pop("OP21_FB_DIST", None)


for scen, K, N in [("best", 512, 8192), ("worst", 1024, 16384),
                   ("best", 2048, 16384)]:
    b = bd.get_bundle(scen, K, torch.float32, N)
    print(f"=== {scen} K={K} N={N} Npad={b['Npad']} cr={b['cr']} "
          f"hr={b['kernel_hit_rate']:.3f} cfg={b['cfg']} ===", flush=True)
    lg, pv, cr = b["logits"], b["preIdx"], b["cr"]
    ref = torch.topk(lg[0, :N].float(), K).values.sort().values
    for arm in ARMS9:
        try:
            pin("1" if arm == "gvr_ms_auto" else None)
            call, keep, extra = build_call_ext(arm, K, torch.float32, N, 1, cr,
                                               lg, pv)
            call()
            torch.cuda.synchronize()
            print(f"  {arm:22s} OK  extra={extra}", flush=True)
        except Exception as e:
            print(f"  {arm:22s} ERROR {type(e).__name__}: {str(e)[:110]}",
                  flush=True)
print("SMOKE DONE", flush=True)
