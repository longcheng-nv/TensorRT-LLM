# clean PAIRED A/B: base vs M=3 in ONE process on ONE idle GPU, back-to-back nsys.
import os,sys
from pathlib import Path
os.environ.setdefault("OP21_FB_LOGFALSI","1"); os.environ.setdefault("OP27_K2048_TAIL","1")
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"harness")); sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
import bundle_data_rr
from sweep_nsys import build_call
_F=torch.empty(256*1024*1024//4,dtype=torch.float32,device="cuda")
def build(K,N,scen,m3):
    if m3: os.environ["OP25_QFRACS"]="0.85,0.35"
    else: os.environ.pop("OP25_QFRACS",None)
    b=bundle_data_rr.get_bundle(scen,K,torch.float32,N)
    call,keep,extra=build_call("gvr_ms_auto",K,torch.float32,N,1,b["cr"],b["logits"],b["preIdx"])
    os.environ.pop("OP25_QFRACS",None)
    return call
CELLS=[(512,8192,"worst"),(512,32768,"worst"),(1024,32768,"worst"),(1024,32768,"real"),
       (512,262144,"worst"),(1024,262144,"worst"),(512,32768,"real")]
built={}
for K,N,sc in CELLS:
    for m3 in (0,1):
        built[(K,N,sc,m3)]=build(K,N,sc,m3)
for _ in range(8):
    for c in built.values(): c()
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
# interleave base/m3 for each cell so both see identical thermal/contention
for K,N,sc in CELLS:
    for m3 in (0,1):
        for _ in range(50): _F.uniform_(); built[(K,N,sc,m3)]()
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
print("done")
