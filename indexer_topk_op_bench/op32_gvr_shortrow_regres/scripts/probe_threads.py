# rung-2 microbench: does raising threads/CTA (512->1024/768) at BS=1 short-N
# raise issue-rate enough to beat the 512 baseline? (reframed CRUX lever)
import os,sys
from pathlib import Path
import torch, cutlass, cutlass.cute as cute
from cutlass.cute import runtime as cr
HERE=Path(__file__).resolve().parents[1]; BENCH=HERE.parents[0]
sys.path.insert(0,str(BENCH/"op22_temporal_fixed_hr_bench"))
sys.path.insert(0,str(BENCH/"op26_gvr_logfalsi_rs"/"src"))
import bundle_data_rr
from gvr_op26_op import GvrOp26Kernel, dispatch_p2_op26, dispatch_rs_op26, _DT
DEV="cuda"
_FLUSH=torch.empty(256*1024*1024//4,dtype=torch.float32,device=DEV)
def flush(): _FLUSH.uniform_()
_cache={}
def build(dt,K,N,cr_v,t,use256,min_bpm):
    key=(dt,K,N,cr_v,t,use256,min_bpm)
    if key in _cache: return _cache[key]
    use_log,kcc,kft,sec2=dispatch_p2_op26(dt,K,N); rs=dispatch_rs_op26(dt,1)
    kobj=GvrOp26Kernel(dtype=_DT[dt],top_k=K,next_n=1,num_threads=t,
        compress_ratio=cr_v,use_256bit_load=use256,enable_unroll_4=True,
        enable_phase3_unroll=True,min_blocks_per_mp=min_bpm,return_output_values=False,
        enable_p4_rank_scatter=rs,enable_p4_rank_scatter_exact=rs,p2_log=use_log,
        kC_override=kcc,kFTarget_override=kft,p2_secant2=sec2,fb_fix=True)
    nr,nc,nb=cute.sym_int(),cute.sym_int(),cute.sym_int()
    ia=32 if use256 else 16
    inf=cr.make_fake_compact_tensor(_DT[dt],(nr,nc),stride_order=(1,0),assumed_align=ia)
    pf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,K),stride_order=(1,0),assumed_align=16)
    sf=cr.make_fake_compact_tensor(cutlass.Int32,(nb,),stride_order=(0,))
    of=cr.make_fake_compact_tensor(cutlass.Int32,(nr,K),stride_order=(1,0),assumed_align=16)
    fs=cr.make_fake_stream(use_tvm_ffi_env_stream=True)
    comp=cute.compile(kobj,inf,pf,sf,None,of,stream=fs,options="--enable-tvm-ffi")
    _cache[key]=comp; return comp
def exact(out,logits,N,K):
    o=out[0]; idx=o.long()
    if bool(((idx<0)|(idx>=N)).any()): return False
    if len(set(o.tolist()))!=K: return False
    sel=logits[0].gather(0,idx).float().sort().values
    ref=torch.topk(logits[0][:N].float(),K).values.sort().values
    return torch.equal(sel,ref)
def tcold(fn,reps=30,wu=5):
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
def main():
    dt=torch.float32; K=int(os.environ.get("K","512"))
    Ns=[int(x) for x in os.environ.get("N_LIST","4096,8192,16384").split(",")]
    THREADS=[int(x) for x in os.environ.get("THREADS","512,768,1024").split(",")]
    print(f"# probe_threads K={K} fp32 BS=1  threads={THREADS}")
    print(f"{'scen':5s}{'N':>7s}"+"".join(f"{'t'+str(t):>10s}" for t in THREADS)+f"{'best_t':>8s}{'ratio':>7s}")
    for N in Ns:
        use256=(N>=16384); min_bpm=1
        for scen in ["best","real","worst"]:
            b=bundle_data_rr.get_bundle(scen,K,dt,N)
            logits,pre,cr_v=b["logits"],b["preIdx"],b["cr"]
            seq=torch.full((1,),N*cr_v,dtype=torch.int32,device=DEV)
            res={}
            for t in THREADS:
                comp=build(dt,K,N,cr_v,t,use256,min_bpm)
                out=torch.empty(1,K,dtype=torch.int32,device=DEV)
                comp(logits,pre,seq,None,out); torch.cuda.synchronize()
                ok=exact(out,logits,N,K)
                fn=lambda c=comp,o=out: c(logits,pre,seq,None,o)
                res[t]=(tcold(fn),ok)
            base=res[512][0]; bt=min(THREADS,key=lambda t:res[t][0])
            cells="".join(f"{res[t][0]:>9.2f}{'' if res[t][1] else '!'}" for t in THREADS)
            print(f"{scen:5s}{N:>7d}{cells}{('t'+str(bt)):>8s}{base/res[bt][0]:>7.3f}")
main()
