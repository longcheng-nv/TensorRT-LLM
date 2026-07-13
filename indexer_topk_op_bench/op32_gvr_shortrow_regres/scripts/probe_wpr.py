# probe: enable_warp_parallel_reduce True vs False at 512 threads (cheapen the
# per-count-pass final aggregate from tid0-serial-16-sum to warp0-shuffle).
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
_c={}
def build(dt,K,N,cr_v,t,use256,wpr):
    key=(dt,K,N,cr_v,t,use256,wpr)
    if key in _c: return _c[key]
    ul,kcc,kft,s2=dispatch_p2_op26(dt,K,N); rs=dispatch_rs_op26(dt,1)
    ko=GvrOp26Kernel(dtype=_DT[dt],top_k=K,next_n=1,num_threads=t,compress_ratio=cr_v,
        use_256bit_load=use256,enable_unroll_4=True,enable_phase3_unroll=True,
        min_blocks_per_mp=1,return_output_values=False,enable_p4_rank_scatter=rs,
        enable_p4_rank_scatter_exact=rs,p2_log=ul,kC_override=kcc,kFTarget_override=kft,
        p2_secant2=s2,fb_fix=True,enable_warp_parallel_reduce=wpr)
    nr,nc,nb=cute.sym_int(),cute.sym_int(),cute.sym_int(); ia=32 if use256 else 16
    inf=cr.make_fake_compact_tensor(_DT[dt],(nr,nc),stride_order=(1,0),assumed_align=ia)
    pf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,K),stride_order=(1,0),assumed_align=16)
    sf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,),stride_order=(0,))
    of=cr.make_fake_compact_tensor(cutlass.Int32,(nr,K),stride_order=(1,0),assumed_align=16)
    comp=cute.compile(ko,inf,pf,sf,None,of,stream=cr.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi")
    _c[key]=comp; return comp
def exact(out,logits,N,K):
    o=out[0]; idx=o.long()
    if bool(((idx<0)|(idx>=N)).any()): return False
    if len(set(o.tolist()))!=K: return False
    return torch.equal(logits[0].gather(0,idx).float().sort().values,
                       torch.topk(logits[0][:N].float(),K).values.sort().values)
def tcold(fn,reps=40,wu=8):
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(wu): flush(); g.replay()
    torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        flush(); torch.cuda.synchronize()
        e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1)*1e3)
    ts.sort(); return ts[len(ts)//2]
dt=torch.float32; K=int(os.environ.get("K","512"))
Ns=[int(x) for x in os.environ.get("N_LIST","4096,8192,16384").split(",")]
print(f"# probe_wpr K={K} fp32 BS=1 t=512  (wpr False=baseline vs True)")
print(f"{'scen':5s}{'N':>7s}{'off':>9s}{'on':>9s}{'ok':>4s}{'ratio(off/on)':>14s}")
for N in Ns:
    use256=(N>=16384)
    for scen in ["best","real","worst"]:
        b=bundle_data_rr.get_bundle(scen,K,dt,N)
        lg,pre,cr_v=b["logits"],b["preIdx"],b["cr"]
        seq=torch.full((1,),N*cr_v,dtype=torch.int32,device=DEV)
        r={}
        for wpr in (False,True):
            comp=build(dt,K,N,cr_v,512,use256,wpr)
            o=torch.empty(1,K,dtype=torch.int32,device=DEV); comp(lg,pre,seq,None,o); torch.cuda.synchronize()
            r[wpr]=(tcold(lambda c=comp,oo=o:c(lg,pre,seq,None,oo)),exact(o,lg,N,K))
        print(f"{scen:5s}{N:>7d}{r[False][0]:>9.2f}{r[True][0]:>9.2f}{('Y' if r[True][1] else 'N!'):>4s}{r[False][0]/r[True][0]:>14.3f}")
