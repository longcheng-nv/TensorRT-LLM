import os,sys
from pathlib import Path
os.environ.setdefault("OP21_FB_LOGFALSI","1"); os.environ.setdefault("OP27_K2048_TAIL","1")
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"harness")); sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
import bundle_data_rr
from sweep_nsys import build_call
K=int(os.environ["K"]); N=int(os.environ["N"]); scen=os.environ.get("SCEN","real"); REPS=int(os.environ.get("REPS","60"))
_F=torch.empty(256*1024*1024//4,dtype=torch.float32,device="cuda")
b=bundle_data_rr.get_bundle(scen,K,torch.float32,N); lg,pre,cr=b["logits"],b["preIdx"],b["cr"]
call,keep,extra=build_call("gvr_ms_auto",K,torch.float32,N,1,cr,lg,pre)
for _ in range(10): call()
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
for _ in range(REPS): _F.uniform_(); call()
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
