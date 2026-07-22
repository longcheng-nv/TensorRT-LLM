import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import build, bundle, make_batch, exact_rows
import real_data_v4cap as v4
import real_data_v32 as v32

mod = build("kernel_bs")
CELLS = [("flash", i, 22) for i in ["4k","8k","16k","32k","64k","128k","256k","512k","1024k"]] + \
        [("pro", i, 30) for i in ["4k","64k","128k","1024k"]] + \
        [("v32", i, 34) for i in ["4k","8k","16k","32k","64k","128k","256k"]]
bad_all = 0
for model, isl, L in CELLS:
    b = bundle(model, isl, L)
    for bs in (1, 2, 8, 16, 32, 64, 128, 256, 512, 1024):
        lg, pre = make_batch(b, bs)
        out = torch.empty(bs, b["K"], dtype=torch.int32, device="cuda")
        mod.run(lg, pre, b["N"], out)
        torch.cuda.synchronize()
        bad = exact_rows(b, out, bs)
        if bad:
            bad_all += 1
            print(f"INEXACT {model}_{isl} Npad={b['Npad']} BS{bs}: {bad}")
        del lg, pre, out
    print(f"{model}_{isl} Npad={b['Npad']:6d} K={b['K']} all-BS exact OK", flush=True)
    v4._bundle_cache.clear(); v32._bundle_cache.clear()
    torch.cuda.empty_cache()
print("TOTAL inexact:", bad_all)
