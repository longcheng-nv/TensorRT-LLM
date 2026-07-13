# phase decomposition: K-scaling at fixed N (P1+P4 ~K, count/P3/barrier ~N) + rs on/off.
import os,sys
from pathlib import Path
import torch, cutlass, cutlass.cute as cute
from cutlass.cute import runtime as cr
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
sys.path.insert(0,str(BENCH/"op26_gvr_logfalsi_rs"/"src"))
import bundle_data_rr
from gvr_op26_op import GvrOp26Kernel, dispatch_p2_op26, dispatch_rs_op26, _DT
DEV="cuda"; _F=torch.empty(256*1024*1024//4,dtype=torch.float32,device=DEV)
def flush(): _F.uniform_()
N=int(os.environ.get("N","8192")); K=int(os.environ.get("K","512"))
scen=os.environ.get("SCEN","real"); RS=int(os.environ.get("RS","1")); REPS=int(os.environ.get("REPS","50"))
dt=torch.float32
ul,kcc,kft,s2=dispatch_p2_op26(dt,K,N); rs=bool(RS) if os.environ.get("RS") else dispatch_rs_op26(dt,1)
use256=(N>=16384)
ko=GvrOp26Kernel(dtype=_DT[dt],top_k=K,next_n=1,num_threads=512,compress_ratio=1,use_256bit_load=use256,
  enable_unroll_4=True,enable_phase3_unroll=True,min_blocks_per_mp=1,return_output_values=False,
  enable_p4_rank_scatter=rs,enable_p4_rank_scatter_exact=rs,p2_log=ul,kC_override=kcc,kFTarget_override=kft,p2_secant2=s2,fb_fix=True)
nr,nc,nb=cute.sym_int(),cute.sym_int(),cute.sym_int(); ia=32 if use256 else 16
inf=cr.make_fake_compact_tensor(_DT[dt],(nr,nc),stride_order=(1,0),assumed_align=ia)
pf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,K),stride_order=(1,0),assumed_align=16)
sf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,),stride_order=(0,))
of=cr.make_fake_compact_tensor(cutlass.Int32,(nr,K),stride_order=(1,0),assumed_align=16)
comp=cute.compile(ko,inf,pf,sf,None,of,stream=cr.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi")
b=bundle_data_rr.get_bundle(scen,K,dt,N); lg,pre=b["logits"],b["preIdx"]
seq=torch.full((1,),N,dtype=torch.int32,device=DEV); out=torch.empty(1,K,dtype=torch.int32,device=DEV)
for _ in range(10): comp(lg,pre,seq,None,out)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for _ in range(REPS): flush(); comp(lg,pre,seq,None,out)
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
