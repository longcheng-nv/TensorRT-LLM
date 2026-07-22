import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from probe import build, bundle, make_batch
mod = build("kernel_bs")
b = bundle("flash", "128k", 22)
lg, pre = make_batch(b, 1024)
out = torch.empty(1024, b["K"], dtype=torch.int32, device="cuda")
for _ in range(3):
    mod.run_cfg(lg, pre, b["N"], out, 1024, 1, 9, 4, 2)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
mod.run_cfg(lg, pre, b["N"], out, 1024, 1, 9, 4, 2)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
