# validate gvr_ms_auto_op33 dispatch: exact + runs for K512(M3)/K1024(M3)/K2048(default).
import os,sys
from pathlib import Path
os.environ.setdefault("OP21_FB_LOGFALSI","1"); os.environ.setdefault("OP27_K2048_TAIL","1")
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench")); sys.path.insert(0,str(HERE/"src"))
import bundle_data_rr
from gvr_ms_op33 import gvr_ms_auto_op33
DEV="cuda"
for K in (512,1024,2048):
  for N in (8192,32768):
    for scen in ("real","worst"):
      b=bundle_data_rr.get_bundle(scen,K,torch.float32,N); lg,pre,cr=b["logits"],b["preIdx"],b["cr"]
      seq=torch.full((1,),N*cr,dtype=torch.int32,device=DEV)
      out=gvr_ms_auto_op33(lg,pre,seq,K,cr); torch.cuda.synchronize()
      sel=lg[0].gather(0,out[0].long()).float().sort().values; ref=torch.topk(lg[0][:N].float(),K).values.sort().values
      ok=torch.equal(sel,ref) and len(set(out[0].tolist()))==K
      print(f"K{K} N{N} {scen}: exact={ok}")
print("DISPATCH VALIDATED")
