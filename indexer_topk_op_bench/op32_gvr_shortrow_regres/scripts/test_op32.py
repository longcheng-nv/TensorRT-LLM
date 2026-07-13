import os,sys
from pathlib import Path
import torch, cutlass, cutlass.cute as cute
from cutlass.cute import runtime as cr
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
sys.path.insert(0,str(BENCH/"op26_gvr_logfalsi_rs"/"src"))
sys.path.insert(0,str(HERE/"src"))
import bundle_data_rr
from gvr_op26_op import dispatch_p2_op26, dispatch_rs_op26, _DT, gvr_cutedsl_op26
from gvr_op32_op import GvrOp32Kernel
DEV="cuda"
def build_op32(dt,K,N,cr_v,red):
    ul,kcc,kft,s2=dispatch_p2_op26(dt,K,N); rs=dispatch_rs_op26(dt,1); use256=(N>=16384)
    ko=GvrOp32Kernel(dtype=_DT[dt],top_k=K,next_n=1,num_threads=512,compress_ratio=cr_v,use_256bit_load=use256,
      enable_unroll_4=True,enable_phase3_unroll=True,min_blocks_per_mp=1,return_output_values=False,
      enable_p4_rank_scatter=rs,enable_p4_rank_scatter_exact=rs,p2_log=ul,kC_override=kcc,kFTarget_override=kft,
      p2_secant2=s2,fb_fix=True,redundant_secant=red)
    nr,nc,nb=cute.sym_int(),cute.sym_int(),cute.sym_int(); ia=32 if use256 else 16
    inf=cr.make_fake_compact_tensor(_DT[dt],(nr,nc),stride_order=(1,0),assumed_align=ia)
    pf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,K),stride_order=(1,0),assumed_align=16)
    sf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,),stride_order=(0,))
    of=cr.make_fake_compact_tensor(cutlass.Int32,(nr,K),stride_order=(1,0),assumed_align=16)
    return cute.compile(ko,inf,pf,sf,None,of,stream=cr.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi")
def exact(out,logits,N,K):
    o=out[0]; idx=o.long()
    if bool(((idx<0)|(idx>=N)).any()): return False,"oob"
    if len(set(o.tolist()))!=K: return False,f"dup{len(set(o.tolist()))}"
    sel=logits[0].gather(0,idx).float().sort().values
    ref=torch.topk(logits[0][:N].float(),K).values.sort().values
    return (torch.equal(sel,ref), "ok" if torch.equal(sel,ref) else f"d{(sel-ref).abs().max():.1e}")
dt=torch.float32
print("=== op32 redundant-secant EXACTNESS gate (fp32 BS=1) ===")
nfail=0
for K in [512,1024,2048]:
    for N in [8192,16384,32768]:
        for scen in ["best","real","worst"]:
            try:
                b=bundle_data_rr.get_bundle(scen,K,dt,N)
            except Exception:
                continue
            lg,pre,cr_v=b["logits"],b["preIdx"],b["cr"]
            seq=torch.full((1,),N*cr_v,dtype=torch.int32,device=DEV)
            comp=build_op32(dt,K,N,cr_v,True)
            o=torch.empty(1,K,dtype=torch.int32,device=DEV); comp(lg,pre,seq,None,o); torch.cuda.synchronize()
            ok,msg=exact(o,lg,N,K)
            if not ok: nfail+=1; print(f"  FAIL {scen} K{K} N{N}: {msg}")
print(f"exactness: {'ALL PASS' if nfail==0 else str(nfail)+' FAIL'}")
