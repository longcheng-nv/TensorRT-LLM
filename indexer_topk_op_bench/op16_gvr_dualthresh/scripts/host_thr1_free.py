# Free threshold_1 from the secant iteration PATH (single-CTA, user's clarified algo A):
# run baseline secant, record every (thr,count); threshold_1 = the largest thr with count<K
# (=> M = that count = definite winners peeled for free). Report M/K and band=(M0-M).
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path("op13_gvr_p2cand/src").resolve()))
sys.path.insert(0, str(Path("harness").resolve()))
from p2_replay import _prep_row, _count_ge, _init_thr, SecantCfg, F32, NEG_FLT_MAX, _DTYPE_NAME, MAX_REFINE_ITERS
from cute_vendored.blackwell.top_k.gvr_topk_decode import GvrParams
from synth_data import get_bundle

def run(lg, pi, N, K, cr, dtype):
    gp=GvrParams.get(_DTYPE_NAME[dtype],K,cr); kK=K; kCC=gp.kC; kFT=gp.kFTarget
    prep=_prep_row(lg,pi,N,K,cr,dtype); xs=prep["xs"]; pmin,pmax=prep["pmin"],prep["pmax"]
    if pmax<=NEG_FLT_MAX or pmin>=pmax: return None
    thr=_init_thr(prep,SecantCfg()); val_lo,val_hi=pmin,pmax; cnt_lo=kK+(kK>>2); cnt_hi=1
    path=[]  # (thr,count)
    def cls(t):
        c=_count_ge(xs,t); path.append((t,c)); return c
    done=0; c=cls(thr)
    if kK<=c<=kCC: done=1
    elif c>kCC: val_lo,cnt_lo=thr,c
    else: val_hi,cnt_hi=thr,c
    it=0
    while it<MAX_REFINE_ITERS and done==0:
        rng=F32(val_hi-val_lo)
        if cnt_lo>cnt_hi and rng>1e-10:
            f=F32((cnt_lo-kFT)/(cnt_lo-cnt_hi)); f=max(0.05,min(0.95,f))
            if it==0: f=min(f,0.5)
            nv=F32(val_lo+rng*f)
        else: nv=F32((val_lo+val_hi)*0.5)
        nv=max(val_lo+rng*0.05,min(val_hi-rng*0.05,nv)); thr=nv
        c=cls(thr)
        if kK<=c<=kCC: done=1
        elif c>kCC: val_lo,cnt_lo=thr,c
        else: val_hi,cnt_hi=thr,c
        it+=1
    if done==0: thr=val_lo if cnt_lo<=kCC*2 else val_hi
    M0=min(_count_ge(xs,thr),kCC)
    # threshold_1 = largest thr in path with count<K  => M = its count (definite winners)
    below=[(t,cc) for (t,cc) in path if cc<kK]
    if below:
        # pick the one with the LARGEST count (closest to K from below)
        t1,M = max(below, key=lambda p:p[1])
    else:
        t1,M = None,0
    return M0, M
    
dtmap={"fp32":torch.float32,"bf16":torch.bfloat16}
print(f"{'K':>4} {'dt':>4} {'N':>7} | {'M0(cand)':>8} {'M(free)':>7} {'M/K':>5} {'band=M0-M':>9} {'band/M0':>7}")
for K in [512,1024,2048]:
    cr=1 if K==2048 else 4
    for dts in ["fp32","bf16"]:
        for N in [16384,65536,131072,262144]:
            if N<=2*K: continue
            M0s=[];Ms=[]
            for cfg in ["beta_shallow","beta_moderate","beta_deep"]:
                for seed in [0,1,2]:
                    b=get_bundle(K,dtmap[dts],N,cfg=cfg,seed=seed)
                    r=run(b["logits"][0],b["preIdx"][0],N,K,cr,dtmap[dts])
                    if r: M0s.append(r[0]); Ms.append(r[1])
            import statistics as st
            M0=st.mean(M0s); M=st.mean(Ms)
            print(f"{K:>4} {dts:>4} {N:>7} | {M0:>8.0f} {M:>7.0f} {M/K:>5.2f} {M0-M:>9.0f} {(M0-M)/M0:>7.2f}")
