import os,sys,time
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
DEV="cuda"; K=512; N=8192; scen="real"; dt=torch.float32
def build(red):
    ul,kcc,kft,s2=dispatch_p2_op26(dt,K,N); rs=dispatch_rs_op26(dt,1)
    ko=GvrOp32Kernel(dtype=_DT[dt],top_k=K,next_n=1,num_threads=512,compress_ratio=1,use_256bit_load=False,
      enable_unroll_4=True,enable_phase3_unroll=True,min_blocks_per_mp=1,return_output_values=False,
      enable_p4_rank_scatter=rs,enable_p4_rank_scatter_exact=rs,p2_log=ul,kC_override=kcc,kFTarget_override=kft,
      p2_secant2=s2,fb_fix=True,redundant_secant=red)
    nr,nc,nb=cute.sym_int(),cute.sym_int(),cute.sym_int()
    inf=cr.make_fake_compact_tensor(_DT[dt],(nr,nc),stride_order=(1,0),assumed_align=16)
    pf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,K),stride_order=(1,0),assumed_align=16)
    sf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,),stride_order=(0,))
    of=cr.make_fake_compact_tensor(cutlass.Int32,(nr,K),stride_order=(1,0),assumed_align=16)
    return cute.compile(ko,inf,pf,sf,None,of,stream=cr.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi")
b=bundle_data_rr.get_bundle(scen,K,dt,N); lg,pre=b["logits"],b["preIdx"]
seq=torch.full((1,),N,dtype=torch.int32,device=DEV)
print("compiling redundant=True...",flush=True); t=time.time()
comp=build(True); print(f"  compiled in {time.time()-t:.1f}s",flush=True)
o=torch.empty(1,K,dtype=torch.int32,device=DEV); comp(lg,pre,seq,None,o); torch.cuda.synchronize()
sel=lg[0].gather(0,o[0].long()).float().sort().values
ref=torch.topk(lg[0][:N].float(),K).values.sort().values
print(f"exact={torch.equal(sel,ref)} uniq={len(set(o[0].tolist()))} K={K}",flush=True)
