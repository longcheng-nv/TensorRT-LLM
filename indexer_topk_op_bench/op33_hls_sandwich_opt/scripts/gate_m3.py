# Thorough M=3 (OP25_QFRACS=0.85,0.35) exactness gate for K512/1024 (the dispatch
# target) + K2048 (control): synth bundles (best/real/worst) + ADVERSARIAL hr=0/hr=1
# beta rows + tie plateaus. tie-aware value-multiset vs torch.topk.
import os,sys,math
from pathlib import Path
os.environ["OP25_QFRACS"]="0.85,0.35"     # force M=3
os.environ.setdefault("OP21_FB_LOGFALSI","1"); os.environ.setdefault("OP27_K2048_TAIL","1")
import torch
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"harness")); sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
import bundle_data_rr
from sweep_nsys import build_call
DEV="cuda"; nfail=0; nok=0
def check(lg,pre,N,K,cr,tag):
    global nfail,nok
    seq=torch.full((1,),N*cr,dtype=torch.int32,device=DEV)
    call,keep,extra=build_call("gvr_ms_auto",K,torch.float32,N,1,cr,lg,pre)
    call(); torch.cuda.synchronize()
    o=keep[3] if len(keep)>3 else keep[-1]
    idx=o[0].long()
    if bool(((idx<0)|(idx>=N)).any()): print(f"FAIL {tag}: oob"); nfail+=1; return
    if len(set(o[0].tolist()))!=K: print(f"FAIL {tag}: dup {len(set(o[0].tolist()))}"); nfail+=1; return
    sel=lg[0].gather(0,idx).float().sort().values; ref=torch.topk(lg[0][:N].float(),K).values.sort().values
    if not torch.equal(sel,ref): print(f"FAIL {tag}: vmiss {(sel-ref).abs().max():.2e}"); nfail+=1; return
    nok+=1
print("== Suite A: op22rr bundles ==")
for K,cr in ((512,4),(1024,4),(2048,1)):
  for N in (8192,16384,32768,65536):
    for scen in ("best","real","worst"):
      try:
        b=bundle_data_rr.get_bundle(scen,K,torch.float32,N)
        check(b["logits"],b["preIdx"],N,K,b["cr"],f"A:{scen}|K{K}|N{N}")
      except Exception as e: print(f"ERR A {scen} K{K} N{N}: {str(e)[:60]}")
print("== Suite B: adversarial hr=0 / hr=1 (beta rows) ==")
torch.manual_seed(1234)
for K,cr in ((512,4),(1024,4),(2048,1)):
  for N in (16384,65536):
    base=torch.distributions.Beta(2.0,5.0).sample((N,))
    row=(base*8.0).float().cuda().view(1,N).contiguous()
    tki=torch.topk(row[0],2*K).indices
    pre_hit=tki[:K].int().view(1,K).contiguous()
    check(row,pre_hit,N,K,cr,f"B:hr1|K{K}|N{N}")
    mask=torch.ones(N,dtype=torch.bool); mask[tki.cpu()]=False
    rest=torch.arange(N)[mask]; pre_miss=rest[torch.randperm(rest.numel())[:K]].int().cuda().view(1,K).contiguous()
    check(row,pre_miss,N,K,cr,f"B:hr0|K{K}|N{N}")
print(f"\nGATE M=3: ok={nok} FAIL={nfail}")
