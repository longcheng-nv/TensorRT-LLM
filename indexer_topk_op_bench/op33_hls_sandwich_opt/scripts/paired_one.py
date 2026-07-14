import os,sys
from pathlib import Path
os.environ.setdefault("OP21_FB_LOGFALSI","1"); os.environ.setdefault("OP27_K2048_TAIL","1")
K=int(os.environ["K"]);N=int(os.environ["N"]);sc=os.environ["SCEN"];m3=int(os.environ["M3"])
if m3: os.environ["OP25_QFRACS"]="0.85,0.35"
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"harness")); sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
import bundle_data_rr
from sweep_nsys import build_call
_F=torch.empty(256*1024*1024//4,dtype=torch.float32,device="cuda")
b=bundle_data_rr.get_bundle(sc,K,torch.float32,N)
call,keep,extra=build_call("gvr_ms_auto",K,torch.float32,N,1,b["cr"],b["logits"],b["preIdx"])
for _ in range(10): call()
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
for _ in range(60): _F.uniform_(); call()
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
